use super::{Emitter, Result, Scalar, scalar};
use ruda_core::ir::{Arithmetic, FloatKind, Instruction, Operation, Operator, Variable};

// Pair only independent lanes of an existing vector. Never combine successive
// IR instructions (in particular, mul + add must not silently become fma).
impl Emitter {
    pub(super) fn packed_elementwise(
        &mut self,
        inst: &Instruction,
        out: Variable,
        args: &[Variable],
        dst: &[String],
        inputs: &[Vec<String>],
    ) -> Result<usize> {
        let element = scalar(out).ty;
        let ty = Scalar::of(element)?;
        let op = match &inst.operation {
            Operation::Operator(Operator::Cast(_)) => {
                return self.paired_cast(out, args[0], dst, &inputs[0]);
            }
            Operation::Arithmetic(Arithmetic::Add(_)) => "add",
            Operation::Arithmetic(Arithmetic::Sub(_)) => "sub",
            Operation::Arithmetic(Arithmetic::Mul(_)) => "mul",
            Operation::Arithmetic(Arithmetic::Fma(_)) => "fma",
            _ => return Ok(0),
        };
        if args.iter().any(|v| scalar(*v).ty != element) {
            return Ok(0); // Let the normal lowering report the type mismatch.
        }
        let bits = match ty {
            Scalar::F16 => {
                self.half_target(ty)?;
                32
            }
            Scalar::BF16 => {
                self.half_target(ty)?;
                32
            }
            Scalar::F32
                if element == FloatKind::F32.into()
                    && self.target.sm >= 100
                    && self.target.version >= (8, 6) =>
            {
                64
            }
            _ => return Ok(0),
        };
        // SM80 supports packed BF16 FMA but not BF16 add/sub/mul. The
        // constants mirror scalar lowering, including the sign of zero.
        let coefficient = if ty == Scalar::BF16
            && op != "fma"
            && (self.target.sm < 90 || self.target.version < (7, 8))
        {
            let literal = match op {
                "add" => "0x3f803f80",
                "sub" => "0xbf80bf80",
                "mul" => "0x80008000",
                _ => unreachable!(),
            };
            let register = self.reg_b32();
            self.line(format!("mov.b32 {register}, {literal};"));
            Some(register)
        } else {
            None
        };
        for (pair, destination) in dst.chunks_exact(2).enumerate() {
            let mut packed = Vec::with_capacity(inputs.len());
            for operand in inputs {
                let (lo, hi) = pair_lanes(operand, pair);
                let word = self.pack_float_pair(bits, ty, lo, hi);
                packed.push(word);
            }
            let opcode = if let Some(register) = &coefficient {
                if op == "sub" {
                    packed.swap(0, 1);
                }
                packed.insert(if op == "mul" { 2 } else { 1 }, register.clone());
                "fma"
            } else {
                op
            };
            let result = self.pair_word(bits);
            self.line(format!(
                "{opcode}.rn.{}x2 {result}, {};",
                ty.suffix(),
                packed.join(", ")
            ));
            self.line(format!(
                "mov.b{bits} {{{}, {}}}, {result};",
                destination[0], destination[1]
            ));
        }
        Ok(dst.len() & !1)
    }

    fn pair_word(&mut self, bits: usize) -> String {
        if bits == 32 {
            self.reg_b32()
        } else {
            self.reg_b64()
        }
    }

    fn float_register(&mut self, ty: Scalar, value: &str) -> String {
        // PTX vector operands to mov must be registers, not float literals.
        if value.starts_with('%') {
            return value.to_owned();
        }
        let reg = self.reg(ty);
        self.line(format!("mov.{} {reg}, {value};", ty.storage()));
        reg
    }

    fn pack_float_pair(&mut self, bits: usize, ty: Scalar, lo: &str, hi: &str) -> String {
        let word = self.pair_word(bits);
        let lo = self.float_register(ty, lo);
        let hi = self.float_register(ty, hi);
        self.line(format!("mov.b{bits} {word}, {{{lo}, {hi}}};"));
        word
    }

    fn paired_cast(
        &mut self,
        out: Variable,
        input: Variable,
        dst: &[String],
        src: &[String],
    ) -> Result<usize> {
        let to = Scalar::of(scalar(out).ty)?;
        let from = Scalar::of(scalar(input).ty)?;
        let f32_input = scalar(input).ty == FloatKind::F32.into();
        let f32_output = scalar(out).ty == FloatKind::F32.into();
        let half_pair =
            f32_input && to.half() && self.target.sm >= 80 && self.target.version >= (7, 0);
        let encode = (f32_input || from == Scalar::F16) && to.fp8();
        let decode = from.fp8() && f32_output;
        if !half_pair && !encode && !decode {
            return Ok(0);
        }
        if encode || decode {
            self.fp8_target()?;
        }
        for (pair, destination) in dst.chunks_exact(2).enumerate() {
            let (lo, hi) = pair_lanes(src, pair);
            if decode {
                // The IR keeps each FP8 byte in the low bits of a 32-bit register.
                let raw = self.reg_b32();
                let bytes = self.reg_b16();
                let halves = self.reg_b32();
                self.line(format!("bfi.b32 {raw}, {hi}, {lo}, 8, 8;"));
                self.line(format!("cvt.u16.u32 {bytes}, {raw};"));
                self.line(format!(
                    "cvt.rn.f16x2.{}x2 {halves}, {bytes};",
                    from.suffix()
                ));
                let low_half = self.reg_b16();
                let high_half = self.reg_b16();
                self.line(format!("mov.b32 {{{low_half}, {high_half}}}, {halves};"));
                for (d, h) in destination.iter().zip([low_half, high_half]) {
                    self.line(format!("cvt.f32.f16 {d}, {h};"));
                }
            } else {
                let result = if half_pair {
                    self.reg_b32()
                } else {
                    self.reg_b16()
                };
                if from == Scalar::F16 {
                    let halves = self.pack_float_pair(32, from, lo, hi);
                    self.line(format!(
                        "cvt.rn.satfinite.{}x2.f16x2 {result}, {halves};",
                        to.suffix()
                    ));
                } else {
                    // For two separate f32 operands PTX puts the first in the high lane.
                    let lo = self.float_register(from, lo);
                    let hi = self.float_register(from, hi);
                    let modifier = if encode { ".satfinite" } else { "" };
                    self.line(format!(
                        "cvt.rn{modifier}.{}x2.f32 {result}, {hi}, {lo};",
                        to.suffix()
                    ));
                }
                if half_pair {
                    self.line(format!(
                        "mov.b32 {{{}, {}}}, {result};",
                        destination[0], destination[1]
                    ));
                } else {
                    let word = self.reg(Scalar::U32);
                    self.line(format!("cvt.u32.u16 {word}, {result};"));
                    self.line(format!("and.b32 {}, {word}, 255;", destination[0]));
                    self.line(format!("shr.u32 {}, {word}, 8;", destination[1]));
                }
            }
        }
        Ok(dst.len() & !1)
    }
}

fn pair_lanes(values: &[String], pair: usize) -> (&str, &str) {
    if values.len() == 1 {
        (&values[0], &values[0])
    } else {
        (&values[2 * pair], &values[2 * pair + 1])
    }
}
