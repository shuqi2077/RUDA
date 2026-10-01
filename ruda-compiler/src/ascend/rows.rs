//! Checked lowering of common IR with exactly one logical 32-lane plane per row.
//! This is compiler-local SSA, not a second public algorithm representation.
use super::{invalid, unsupported, Result};
use ruda_core::{ir::*, kernel::{KernelArg, KernelDefinition, Visibility}};
use std::collections::{BTreeSet, HashMap, HashSet};

pub(super) const LANES: u32 = 32;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BindingKind { Matrix, SharedRow, RowScalar }
impl BindingKind {
    pub fn count(self, rows: u64, width: u32) -> u64 { match self {
        Self::Matrix => rows * width as u64, Self::SharedRow => width as u64, Self::RowScalar => rows,
    }}
    pub fn local_count(self, width: u32) -> u32 { if self == Self::RowScalar { 8 } else { width } }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Unary { Neg, Abs, Exp, Log, Sqrt, Rsqrt, Recip }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Binary { Add, Sub, Mul, Div, Max }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reduction { Sum, Max }
#[derive(Clone, Copy, Debug)]
pub(super) enum Node {
    Input { binding: usize, offset: u32 }, Constant(u32),
    Unary(Unary, usize), Binary(Binary, usize, usize), Reduce(Reduction, usize),
}
impl Node {
    pub fn inputs(self) -> Vec<usize> { match self {
        Self::Unary(_, a) | Self::Reduce(_, a) => vec![a], Self::Binary(_, a, b) => vec![a, b], _ => vec![],
    }}
}
#[derive(Clone, Debug)]
pub(super) struct Binding { pub arg: KernelArg, pub kind: BindingKind }
#[derive(Debug)]
pub(super) struct Program {
    pub name: String, pub bindings: Vec<Binding>, pub nodes: Vec<Node>,
    /// Output binding, column offset (zero for a scalar), SSA value.
    pub stores: Vec<(usize, u32, usize)>, pub width: u32, pub rows: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Affine { row: u64, lane: u64, constant: u64 }
impl Affine {
    fn scalar(n: u64) -> Self { Self { row: 0, lane: 0, constant: n } }
    fn is_scalar(self) -> bool { self.row == 0 && self.lane == 0 }
    fn add(self, rhs: Self) -> Result<Self> { Ok(Self {
        row: self.row.checked_add(rhs.row).ok_or_else(|| invalid("row index overflow"))?,
        lane: self.lane.checked_add(rhs.lane).ok_or_else(|| invalid("lane index overflow"))?,
        constant: self.constant.checked_add(rhs.constant).ok_or_else(|| invalid("offset overflow"))?,
    }) }
    fn scale(self, n: u64) -> Result<Self> { Ok(Self {
        row: self.row.checked_mul(n).ok_or_else(|| invalid("row stride overflow"))?,
        lane: self.lane.checked_mul(n).ok_or_else(|| invalid("lane stride overflow"))?,
        constant: self.constant.checked_mul(n).ok_or_else(|| invalid("offset product overflow"))?,
    }) }
}
#[derive(Clone, Copy, Debug)]
enum Value { Index(Affine), LaneZero, Data(usize) }
fn fp32() -> Type { Type::scalar(ElemType::Float(FloatKind::F32)) }
fn uint32() -> Type { Type::scalar(ElemType::UInt(UIntKind::U32)) }
fn identifier(name: &str) -> bool {
    !name.is_empty() && name.len() <= 128 && name.as_bytes()[0].is_ascii_alphabetic()
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        && !["for", "if", "while", "return", "float", "int", "void", "class", "template", "auto", "const",
            "extern", "union", "struct", "namespace", "operator", "new", "delete"].contains(&name)
}
struct Lower {
    name: String, args: Vec<KernelArg>, kinds: Vec<Option<BindingKind>>, nodes: Vec<Node>,
    uniform: Vec<bool>, stores: Vec<(usize, u32, usize)>, writes: Vec<BTreeSet<u32>>,
    values: HashMap<Variable, Value>, loads: HashMap<(usize, u32), usize>, constants: HashMap<u32, usize>,
    width: u32, rows: u64,
}
impl Lower {
    fn resolve(&self, v: Variable) -> Result<Value> {
        if v.ty == uint32() { match v.kind {
            VariableKind::Builtin(Builtin::RudaPosX) => return Ok(Value::Index(Affine { row: 1, lane: 0, constant: 0 })),
            VariableKind::Builtin(Builtin::UnitPosX) => return Ok(Value::Index(Affine { row: 0, lane: 1, constant: 0 })),
            VariableKind::Constant(ConstantValue::UInt(n)) if n <= u32::MAX as u64 => return Ok(Value::Index(Affine::scalar(n))),
            _ => {},
        }}
        self.values.get(&v).copied().ok_or_else(|| invalid(format!("undefined/unsupported row operand {v:?}")))
    }
    fn index(&self, v: Variable) -> Result<Affine> { match self.resolve(v)? {
        Value::Index(i) => Ok(i), _ => Err(unsupported("row index must be affine integer arithmetic")),
    }}
    fn data(&mut self, v: Variable) -> Result<usize> {
        if v.ty != fp32() { return Err(unsupported("row arithmetic requires FP32; no implicit casts")); }
        if let VariableKind::Constant(ConstantValue::Float(x)) = v.kind {
            let f = x as f32; if !f.is_finite() { return Err(unsupported("non-finite row compile-time constant")); }
            let bits = f.to_bits(); if let Some(&n) = self.constants.get(&bits) { return Ok(n); }
            let n = self.push(Node::Constant(bits), true); self.constants.insert(bits, n); return Ok(n);
        }
        match self.resolve(v)? { Value::Data(n) => Ok(n), _ => Err(unsupported("indices cannot be row floating point values")) }
    }
    fn assign(&mut self, out: Variable, value: Value) -> Result<()> {
        if !matches!(out.kind, VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } | VariableKind::Versioned { .. }) {
            return Err(invalid("row result must be a local variable"));
        }
        let ty = match value { Value::Data(_) => fp32(), Value::Index(_) => uint32(), Value::LaneZero => Type::scalar(ElemType::Bool) };
        if ty != out.ty { return Err(invalid("row operation output type mismatch")); }
        if self.values.contains_key(&out) && !matches!(out.kind, VariableKind::LocalMut { .. }) {
            return Err(invalid("immutable row variable written twice"));
        }
        self.values.insert(out, value); Ok(())
    }
    fn push(&mut self, node: Node, uniform: bool) -> usize {
        let n = self.nodes.len(); self.nodes.push(node); self.uniform.push(uniform); n
    }
    fn write_node(&mut self, out: Variable, node: Node, uniform: bool) -> Result<()> {
        let n = self.push(node, uniform); self.assign(out, Value::Data(n))
    }
    fn array(&self, v: Variable, write: bool) -> Result<usize> {
        let id = match (v.kind, write) {
            (VariableKind::GlobalInputArray(id), false) | (VariableKind::GlobalOutputArray(id), true) => id,
            _ => return Err(unsupported("row loads require readonly inputs; stores require distinct outputs")),
        };
        let n = self.args.iter().position(|a| a.id == id).ok_or_else(|| invalid("unknown row binding id"))?;
        if v.ty != fp32() || (self.args[n].visibility == Visibility::ReadWrite) != write {
            return Err(invalid("row buffer type/visibility mismatch"));
        }
        Ok(n)
    }
    fn classify(&mut self, binding: usize, i: Affine) -> Result<(BindingKind, u32)> {
        let (kind, offset) = if i == (Affine { row: 1, lane: 0, constant: 0 }) {
            (BindingKind::RowScalar, 0)
        } else if i.lane == 1 && i.constant % LANES as u64 == 0
            && i.constant + LANES as u64 <= self.width as u64 {
            let kind = if i.row == self.width as u64 { BindingKind::Matrix }
                else if i.row == 0 { BindingKind::SharedRow }
                else { return Err(unsupported("row input has unsupported row stride")); };
            (kind, i.constant as u32)
        } else { return Err(unsupported("only row*width+lane+32*j, lane+32*j or row addresses are supported")); };
        if self.kinds[binding].is_some_and(|old| old != kind) { return Err(unsupported("one buffer cannot mix matrix/shared/scalar indexing")); }
        let count = kind.count(self.rows, self.width);
        if self.args[binding].size.is_some_and(|n| n as u64 != count) { return Err(invalid("row binding static size mismatch")); }
        self.kinds[binding] = Some(kind); Ok((kind, offset))
    }
    fn store(&mut self, dst: Variable, op: &IndexAssignOperator, lane_zero: bool) -> Result<()> {
        if op.vector_size != 0 || op.unroll_factor != 1 { return Err(unsupported("vectorized row store")); }
        let a = self.array(dst, true)?; let i = self.index(op.index)?; let (kind, offset) = self.classify(a, i)?;
        if kind == BindingKind::SharedRow || (kind == BindingKind::RowScalar) != lane_zero {
            return Err(unsupported("row output must be full row, or a scalar written only by lane zero"));
        }
        let v = self.data(op.value)?;
        if lane_zero && !self.uniform[v] { return Err(unsupported("scalar row store must be uniform across logical lanes")); }
        if !self.writes[a].insert(offset) { return Err(invalid("row output chunk written twice")); }
        self.stores.push((a, offset, v)); Ok(())
    }
    fn instruction(&mut self, i: &Instruction) -> Result<()> {
        if !i.modes.fp_math_mode.is_empty() { return Err(unsupported("row fast-math modes are not silently discarded")); }
        let dst = || i.out.ok_or_else(|| invalid("missing row operation result"));
        match &i.operation {
            Operation::NonSemantic(_) => Ok(()),
            Operation::Copy(v) => { let value = if v.ty == fp32() { Value::Data(self.data(*v)?) } else { self.resolve(*v)? };
                self.assign(dst()?, value) }
            Operation::Arithmetic(a) if i.out.is_some_and(|v| v.ty == uint32()) => {
                let index = match a {
                    Arithmetic::Add(op) => self.index(op.lhs)?.add(self.index(op.rhs)?)?,
                    Arithmetic::Mul(op) => { let l = self.index(op.lhs)?; let r = self.index(op.rhs)?;
                        if l.is_scalar() { r.scale(l.constant)? } else if r.is_scalar() { l.scale(r.constant)? }
                        else { return Err(unsupported("non-affine row index multiplication")); } },
                    _ => return Err(unsupported("unsupported row index operation")),
                };
                // The address calculation is UInt32 in the input IR. Never
                // silently turn wrapping arithmetic into wide host indexing.
                let max = index.row.checked_mul(self.rows.saturating_sub(1))
                    .and_then(|n| n.checked_add(index.lane.checked_mul(31)?))
                    .and_then(|n| n.checked_add(index.constant)).ok_or_else(|| invalid("row address overflow"))?;
                if max > u32::MAX as u64 { return Err(invalid("UInt32 row address would wrap")); }
                self.assign(dst()?, Value::Index(index))
            }
            Operation::Arithmetic(a) => {
                let binary = match a { Arithmetic::Add(op) => Some((Binary::Add, op)), Arithmetic::Sub(op) => Some((Binary::Sub, op)),
                    Arithmetic::Mul(op) => Some((Binary::Mul, op)), Arithmetic::Div(op) => Some((Binary::Div, op)),
                    Arithmetic::Max(op) => Some((Binary::Max, op)), _ => None };
                if let Some((kind, op)) = binary {
                    let l = self.data(op.lhs)?; let r = self.data(op.rhs)?;
                    return self.write_node(dst()?, Node::Binary(kind, l, r), self.uniform[l] && self.uniform[r]);
                }
                let (kind, op) = match a { Arithmetic::Neg(op) => (Unary::Neg, op), Arithmetic::Abs(op) => (Unary::Abs, op),
                    Arithmetic::Exp(op) => (Unary::Exp, op), Arithmetic::Log(op) => (Unary::Log, op),
                    Arithmetic::Sqrt(op) => (Unary::Sqrt, op), Arithmetic::InverseSqrt(op) => (Unary::Rsqrt, op),
                    Arithmetic::Recip(op) => (Unary::Recip, op), _ => return Err(unsupported("row arithmetic operation")) };
                let x = self.data(op.input)?; self.write_node(dst()?, Node::Unary(kind, x), self.uniform[x])
            }
            Operation::Plane(plane) => {
                let (kind, input) = match plane { Plane::Sum(op) => (Reduction::Sum, op.input),
                    Plane::Max(op) => (Reduction::Max, op.input), _ => return Err(unsupported("row compiler accepts only Plane::Sum/Max")) };
                let x = self.data(input)?; self.write_node(dst()?, Node::Reduce(kind, x), true)
            }
            Operation::Operator(Operator::Index(op)) => {
                if op.vector_size != 0 || op.unroll_factor != 1 { return Err(unsupported("vectorized row load")); }
                let a = self.array(op.list, false)?; let index = self.index(op.index)?;
                let (kind, offset) = self.classify(a, index)?;
                let n = if let Some(&n) = self.loads.get(&(a, offset)) { n }
                    else { let n = self.push(Node::Input { binding: a, offset }, kind == BindingKind::RowScalar);
                        self.loads.insert((a, offset), n); n };
                self.assign(dst()?, Value::Data(n))
            }
            Operation::Operator(Operator::IndexAssign(op)) => self.store(dst()?, op, false),
            Operation::Comparison(Comparison::Equal(op)) => {
                if self.index(op.lhs)? != (Affine { row: 0, lane: 1, constant: 0 }) || self.index(op.rhs)? != Affine::scalar(0) {
                    return Err(unsupported("only lane==0 scalar-store predicate"));
                }
                self.assign(dst()?, Value::LaneZero)
            }
            Operation::Branch(Branch::If(branch)) => {
                if !matches!(self.resolve(branch.cond)?, Value::LaneZero) || !branch.scope.const_arrays.is_empty() {
                    return Err(unsupported("only lane-zero scalar stores can be predicated"));
                }
                let mut nested = branch.scope.clone();
                let errors = nested.pop_errors();
                if !errors.is_empty() { return Err(invalid(errors.join("; "))); }
                for child in &branch.scope.instructions {
                    if !child.modes.fp_math_mode.is_empty() { return Err(unsupported("predicated row fast-math")); }
                    match &child.operation { Operation::NonSemantic(_) => {},
                        Operation::Operator(Operator::IndexAssign(op)) => self.store(child.out.ok_or_else(|| invalid("missing scalar output"))?, op, true)?,
                        _ => return Err(unsupported("no arithmetic, reduction or load in a lane-zero branch")),
                    }
                } Ok(())
            }
            other => Err(unsupported(format!("row instruction {other:?}; no backend fallback"))),
        }
    }
}

pub(super) fn lower(mut k: KernelDefinition, elements: u64, width: u32) -> Result<Program> {
    if !(32..=4096).contains(&width) || width % LANES != 0 || elements % width as u64 != 0 || elements > u32::MAX as u64 {
        return Err(invalid("row shape: FP32 width 32..4096, multiple of 32; exact integral rows within UInt32"));
    }
    if !identifier(&k.options.kernel_name) { return Err(invalid("invalid row entrypoint identifier")); }
    if k.ruda_dim.x != LANES || k.ruda_dim.y != 1 || k.ruda_dim.z != 1 {
        return Err(unsupported("row kernel must declare exactly one 32-lane logical plane per block"));
    }
    if k.options.debug_symbols || k.options.cluster_dim.is_some() || !k.tensor_maps.is_empty()
        || !k.scalars.is_empty() || !k.body.const_arrays.is_empty() {
        return Err(unsupported("row debug/cluster/tensor-map/scalar-array metadata"));
    }
    let errors = k.body.pop_errors(); if !errors.is_empty() { return Err(invalid(errors.join("; "))); }
    if k.buffers.is_empty() || k.buffers.len() > 8 || k.body.instructions.len() > 16384 {
        return Err(unsupported("row binding/instruction limit exceeded"));
    }
    let mut ids = HashSet::new(); let mut reads = 0; let mut writes = 0;
    for b in &k.buffers {
        if !ids.insert(b.id) { return Err(invalid("duplicate row binding id")); }
        if b.ty != fp32() || b.has_extended_meta { return Err(unsupported("row bindings must be scalar FP32 without extended metadata")); }
        if b.visibility == Visibility::Read { reads += 1; } else { writes += 1; }
    }
    if reads > 4 || writes == 0 || writes > 4 { return Err(unsupported("row compiler supports 1..4 outputs and at most 4 inputs")); }
    let count = k.buffers.len(); let mut l = Lower { name: k.options.kernel_name, args: k.buffers,
        kinds: vec![None; count], nodes: vec![], uniform: vec![], stores: vec![], writes: vec![BTreeSet::new(); count],
        values: HashMap::new(), loads: HashMap::new(), constants: HashMap::new(), width, rows: elements / width as u64 };
    for (n, i) in k.body.instructions.iter().enumerate() {
        l.instruction(i).map_err(|e| unsupported(format!("row instruction {n}: {e}")))?;
    }
    for (i, b) in l.args.iter().enumerate() {
        let kind = l.kinds[i].ok_or_else(|| invalid("unused row buffer must be removed"))?;
        if b.visibility == Visibility::ReadWrite {
            let expected: BTreeSet<_> = if kind == BindingKind::RowScalar { [0].into_iter().collect() }
                else { (0..width).step_by(LANES as usize).collect() };
            if l.writes[i] != expected { return Err(invalid("row output must be completely and exactly written")); }
        }
    }
    let bindings = l.args.into_iter().enumerate().map(|(i, arg)| Binding { arg, kind: l.kinds[i].unwrap() }).collect();
    Ok(Program { name: l.name, bindings, nodes: l.nodes, stores: l.stores, width, rows: l.rows })
}

#[derive(Debug)]
pub(super) struct Allocation { pub node_slots: Vec<Option<usize>>, pub slots: usize, pub ub_bytes: u32 }
pub(super) fn allocate(p: &Program, reuse: bool, limit: u32) -> Result<Allocation> {
    let n = p.nodes.len(); let mut last: Vec<_> = (0..n).collect();
    for (i, node) in p.nodes.iter().enumerate() { for src in node.inputs() {
        if src >= i { return Err(invalid("row SSA operand order")); } last[src] = last[src].max(i);
    }}
    for &(_, _, src) in &p.stores { if src >= n { return Err(invalid("row store SSA index")); } last[src] = n; }
    let mut ends = Vec::new(); let mut node_slots = vec![None; n];
    for (i, node) in p.nodes.iter().enumerate() {
        if let Node::Input { binding, .. } = node { if p.bindings[*binding].kind != BindingKind::RowScalar { continue; } }
        let slot = (if reuse { ends.iter().position(|&end| end < i) } else { None }).unwrap_or(ends.len());
        if slot == ends.len() { ends.push(last[i]); } else { ends[slot] = last[i]; }
        node_slots[i] = Some(slot);
    }
    let queue_bytes: usize = p.bindings.iter().map(|b| b.kind.local_count(p.width) as usize * 4).sum();
    // A dedicated 32-byte result plus 256-byte work buffer for 32 FP32 lanes.
    // Reduction scratch never aliases a live source or a destination vector.
    let ub = queue_bytes.checked_add(ends.len().checked_mul(128).ok_or_else(|| invalid("row temporary overflow"))?)
        .and_then(|n| n.checked_add(288)).ok_or_else(|| invalid("row UB size overflow"))?;
    if ub > limit as usize { return Err(unsupported(format!("row requires {ub} UB bytes, limit={limit}"))); }
    Ok(Allocation { node_slots, slots: ends.len(), ub_bytes: ub as u32 })
}
