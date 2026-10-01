//! Row algorithms authored only in the existing public RUDA KernelDefinition IR.
//!
//! One logical 32-lane plane owns one row. Each lane handles columns lane+32*j.
//! Cross-chunk accumulation precedes Plane::Sum/Max. The Ascend backend must NOT
//! reinterpret a 32-lane operation as a reduction of an arbitrary-width vector.
//! Matrix/weight/statistic addressing is explicit in the IR, not a kernel-name
//! convention. Definitions also have ordinary PTX semantics with one plane/block.
use super::{Result, invalid};
use ruda_core::{ir::*, kernel::{KernelArg, KernelDefinition, KernelOptions, Visibility}, launch::RudaDim};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowProgram {
    Sum, Mean, Max, Softmax, LogSoftmax, RmsNorm,
    SoftmaxBackward, LogSoftmaxBackward, RmsNormInputBackward,
}
impl RowProgram {
    pub const ALL: [Self; 9] = [Self::Sum, Self::Mean, Self::Max, Self::Softmax,
        Self::LogSoftmax, Self::RmsNorm, Self::SoftmaxBackward,
        Self::LogSoftmaxBackward, Self::RmsNormInputBackward];
    pub fn name(self) -> &'static str { match self {
        Self::Sum => "row_sum", Self::Mean => "row_mean", Self::Max => "row_max",
        Self::Softmax => "softmax", Self::LogSoftmax => "log_softmax",
        Self::RmsNorm => "rms_norm", Self::SoftmaxBackward => "softmax_backward",
        Self::LogSoftmaxBackward => "log_softmax_backward",
        Self::RmsNormInputBackward => "rms_norm_input_backward",
    }}
    pub fn parse(s: &str) -> Option<Self> { Self::ALL.into_iter().find(|p| p.name() == s) }
}
fn f() -> Type { Type::scalar(ElemType::Float(FloatKind::F32)) }
fn u() -> Type { Type::scalar(ElemType::UInt(UIntKind::U32)) }
fn constant(x: f32) -> Variable { Variable::constant(ConstantValue::Float(x as f64), f()) }
struct Builder { k: KernelDefinition, next: u32, width: u32, row: Variable, lane: Variable }
impl Builder {
    fn new(name: String, width: u32) -> Self {
        Self { k: KernelDefinition { buffers: vec![], tensor_maps: vec![], scalars: vec![],
            ruda_dim: RudaDim::new_1d(32), body: Scope::root(false),
            options: KernelOptions { kernel_name: name, ..Default::default() } }, next: 0, width,
            row: Variable::builtin(Builtin::RudaPosX, UIntKind::U32.into()),
            lane: Variable::builtin(Builtin::UnitPosX, UIntKind::U32.into()) }
    }
    fn local(&mut self, ty: Type) -> Variable { let id = self.next; self.next += 1;
        Variable::new(VariableKind::LocalConst { id }, ty) }
    fn array(&mut self, writable: bool) -> Variable {
        let id = self.k.buffers.len() as u32;
        self.k.buffers.push(KernelArg { id, visibility: if writable { Visibility::ReadWrite } else { Visibility::Read },
            ty: f(), size: None, has_extended_meta: false });
        Variable::new(if writable { VariableKind::GlobalOutputArray(id) } else { VariableKind::GlobalInputArray(id) }, f())
    }
    fn arithmetic(&mut self, op: Arithmetic, ty: Type) -> Variable { let out = self.local(ty);
        self.k.body.instructions.push(Instruction::new(op, out)); out }
    fn binary(&mut self, op: fn(BinaryOperator) -> Arithmetic, a: Variable, b: Variable) -> Variable {
        self.arithmetic(op(BinaryOperator { lhs: a, rhs: b }), f())
    }
    fn add(&mut self, a: Variable, b: Variable) -> Variable { self.binary(Arithmetic::Add, a, b) }
    fn sub(&mut self, a: Variable, b: Variable) -> Variable { self.binary(Arithmetic::Sub, a, b) }
    fn mul(&mut self, a: Variable, b: Variable) -> Variable { self.binary(Arithmetic::Mul, a, b) }
    fn div(&mut self, a: Variable, b: Variable) -> Variable { self.binary(Arithmetic::Div, a, b) }
    fn unary(&mut self, op: fn(UnaryOperator) -> Arithmetic, x: Variable) -> Variable {
        self.arithmetic(op(UnaryOperator { input: x }), f())
    }
    fn index(&mut self, chunk: u32, shared: bool) -> Variable {
        let col = self.arithmetic(Arithmetic::Add(BinaryOperator { lhs: self.lane, rhs: (chunk * 32).into() }), u());
        if shared { return col; }
        let base = self.arithmetic(Arithmetic::Mul(BinaryOperator { lhs: self.row, rhs: self.width.into() }), u());
        self.arithmetic(Arithmetic::Add(BinaryOperator { lhs: base, rhs: col }), u())
    }
    fn read(&mut self, a: Variable, index: Variable) -> Variable {
        let out = self.local(f()); self.k.body.instructions.push(Instruction::new(
            Operator::Index(IndexOperator { list: a, index, vector_size: 0, unroll_factor: 1 }), out)); out
    }
    fn chunks(&mut self, a: Variable, shared: bool) -> Vec<Variable> {
        (0..self.width / 32).map(|j| { let i = self.index(j, shared); self.read(a, i) }).collect()
    }
    fn reduce(&mut self, xs: &[Variable], max: bool) -> Variable {
        let mut acc = xs[0];
        for &x in &xs[1..] { acc = self.binary(if max { Arithmetic::Max } else { Arithmetic::Add }, acc, x); }
        let out = self.local(f()); let input = UnaryOperator { input: acc };
        self.k.body.instructions.push(Instruction::new(if max { Plane::Max(input) } else { Plane::Sum(input) }, out)); out
    }
    fn write_row(&mut self, a: Variable, xs: &[Variable]) {
        for (j, &x) in xs.iter().enumerate() { let index = self.index(j as u32, false);
            self.k.body.instructions.push(Instruction::new(Operator::IndexAssign(IndexAssignOperator {
                index, value: x, vector_size: 0, unroll_factor: 1 }), a)); }
    }
    fn write_scalar(&mut self, a: Variable, value: Variable) {
        // Exactly lane zero writes a uniform row statistic. No racing stores.
        let cond = self.local(Type::scalar(ElemType::Bool));
        self.k.body.instructions.push(Instruction::new(Comparison::Equal(BinaryOperator { lhs: self.lane, rhs: 0u32.into() }), cond));
        let mut scope = Scope::root(false); scope.instructions.push(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator { index: self.row, value, vector_size: 0, unroll_factor: 1 }), a));
        self.k.body.instructions.push(Instruction::no_out(Branch::If(Box::new(If { cond, scope }))));
    }
}
/// Fixed FP32 row width: 32..4096 in multiples of 32. Epsilon is a finite,
/// positive compile-time value for RMSNorm; no runtime scalar or CPU fallback.
/// RMSNorm emits [Y, rstd]; input backward takes [X, dY, weight, rstd].
/// Affine weight gradient is deliberately not claimed by input-backward.
pub fn definition(op: RowProgram, width: u32, epsilon: f32) -> Result<KernelDefinition> {
    if !(32..=4096).contains(&width) || width % 32 != 0 { return Err(invalid("row width must be 32..4096 and divisible by 32")); }
    if !epsilon.is_finite() || epsilon <= 0.0 { return Err(invalid("epsilon must be finite and positive")); }
    let mut b = Builder::new(format!("ruda_cann_{}", op.name()), width);
    let count = match op { RowProgram::RmsNormInputBackward => 4,
        RowProgram::RmsNorm | RowProgram::SoftmaxBackward | RowProgram::LogSoftmaxBackward => 2, _ => 1 };
    let input: Vec<_> = (0..count).map(|_| b.array(false)).collect();
    let output = b.array(true);
    let stat = if op == RowProgram::RmsNorm { Some(b.array(true)) } else { None };
    let x = b.chunks(input[0], false);
    match op {
        RowProgram::Sum | RowProgram::Mean | RowProgram::Max => {
            let mut s = b.reduce(&x, op == RowProgram::Max);
            if op == RowProgram::Mean { s = b.div(s, constant(width as f32)); }
            b.write_scalar(output, s);
        }
        RowProgram::Softmax | RowProgram::LogSoftmax => {
            let max = b.reduce(&x, true);
            let shifted: Vec<_> = x.iter().map(|&v| b.sub(v, max)).collect();
            let exp: Vec<_> = shifted.iter().map(|&v| b.unary(Arithmetic::Exp, v)).collect();
            let sum = b.reduce(&exp, false);
            let y: Vec<_> = if op == RowProgram::Softmax { exp.iter().map(|&v| b.div(v, sum)).collect() }
                else { let log = b.unary(Arithmetic::Log, sum); shifted.iter().map(|&v| b.sub(v, log)).collect() };
            b.write_row(output, &y);
        }
        RowProgram::RmsNorm => {
            let weight = b.chunks(input[1], true);
            let square: Vec<_> = x.iter().map(|&v| b.mul(v, v)).collect();
            let sum = b.reduce(&square, false); let mean = b.div(sum, constant(width as f32));
            let shift = b.add(mean, constant(epsilon)); let r = b.unary(Arithmetic::InverseSqrt, shift);
            let y: Vec<_> = x.iter().zip(weight).map(|(&v, w)| { let z = b.mul(v, r); b.mul(z, w) }).collect();
            b.write_row(output, &y); b.write_scalar(stat.unwrap(), r);
        }
        RowProgram::SoftmaxBackward | RowProgram::LogSoftmaxBackward => {
            let dy = b.chunks(input[1], false);
            let y = if op == RowProgram::SoftmaxBackward { x.clone() }
                else { x.iter().map(|&v| b.unary(Arithmetic::Exp, v)).collect() };
            let dot = if op == RowProgram::SoftmaxBackward {
                let p: Vec<_> = y.iter().zip(&dy).map(|(&a, &g)| b.mul(a, g)).collect(); b.reduce(&p, false)
            } else { b.reduce(&dy, false) };
            let dx: Vec<_> = y.iter().zip(&dy).map(|(&a, &g)| {
                if op == RowProgram::SoftmaxBackward { let s = b.sub(g, dot); b.mul(a, s) }
                else { let s = b.mul(a, dot); b.sub(g, s) }
            }).collect(); b.write_row(output, &dx);
        }
        RowProgram::RmsNormInputBackward => {
            let dy = b.chunks(input[1], false); let w = b.chunks(input[2], true);
            let row = b.row; let r = b.read(input[3], row);
            let g: Vec<_> = dy.iter().zip(w).map(|(&v, w)| b.mul(v, w)).collect();
            let gx: Vec<_> = g.iter().zip(&x).map(|(&a, &x)| b.mul(a, x)).collect();
            let sum = b.reduce(&gx, false); let mean = b.div(sum, constant(width as f32));
            let rr = b.mul(r, r); let correction = b.mul(mean, rr);
            let dx: Vec<_> = x.iter().zip(g).map(|(&x, g)| { let c = b.mul(x, correction); let z = b.sub(g, c); b.mul(z, r) }).collect();
            b.write_row(output, &dx);
        }
    }
    Ok(b.k)
}
