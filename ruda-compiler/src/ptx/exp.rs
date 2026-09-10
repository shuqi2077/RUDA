// expf algorithm and table adapted from Arm optimized-routines, under MIT.
// Copyright (c) 2017-2025, Arm Limited.
// See THIRD_PARTY_NOTICES.md for the source files and license.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

const EXPF_TABLE: [u64; 32] = [
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
    0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
    0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
    0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
    0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
    0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
    0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
];

impl Emitter {
    fn expf_table(&mut self) -> String {
        if let Some(base) = &self.expf_table_base {
            return base.clone();
        }
        let values = EXPF_TABLE.iter()
            .map(|value| format!("0x{value:016x}"))
            .collect::<Vec<_>>()
            .join(", ");
        self.module_declarations += &format!(
            ".const .align 8 .u64 {}[32] = {{ {values} }};\n",
            self.expf_table_symbol,
        );
        let base = self.reg(Scalar::U64);
        self.prologue += &format!("    mov.u64 {base}, {};\n", self.expf_table_symbol);
        self.expf_table_base = Some(base.clone());
        base
    }

    pub fn exponential(&mut self, out: Variable, input: Variable) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("exponential operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("exponential requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let source = self.half_to_f32(ty, &source)?;
            let result = self.reg(Scalar::F32);
            self.exponential_f32(&result, &source);
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.exponential_f32(&destination, &source);
        }
        Ok(())
    }

    pub(super) fn exponential_f32(&mut self, destination: &str, source: &str) {
        let predicate = self.reg(Scalar::Pred);
        let nan = self.label();
        let overflow = self.label();
        let underflow = self.label();
        let end = self.label();
        self.line(format!("setp.neu.f32 {predicate}, {source}, {source};"));
        self.line(format!("@{predicate} bra {nan};"));
        self.line(format!("setp.gt.f32 {predicate}, {source}, 0f42b17217;"));
        self.line(format!("@{predicate} bra {overflow};"));
        self.line(format!("setp.lt.f32 {predicate}, {source}, 0fc2cff1b4;"));
        self.line(format!("@{predicate} bra {underflow};"));

        self.exponential_finite_f32(destination, source, false);
        self.line(format!("bra {end};"));

        self.line(format!("{nan}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{overflow}:"));
        self.line(format!("mov.f32 {destination}, 0f7f800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{underflow}:"));
        self.line(format!("mov.f32 {destination}, 0f00000000;"));
        self.line(format!("{end}:"));
    }

    // Callers exclude non-finite values and bound x to [-104, 90].
    pub(super) fn exponential_finite_f32(&mut self, destination: &str, source: &str, halve: bool) {
        let x = self.reg(Scalar::F64);
        let z = self.reg(Scalar::F64);
        let kd = self.reg(Scalar::F64);
        let ki = self.reg(Scalar::U64);
        let r = self.reg(Scalar::F64);
        let r2 = self.reg(Scalar::F64);
        let y = self.reg(Scalar::F64);
        let s = self.reg(Scalar::F64);
        let offset = self.reg(Scalar::U64);
        let address = self.reg(Scalar::U64);
        let scale_bits = self.reg(Scalar::U64);
        let exponent_bits = self.reg(Scalar::U64);
        let table = self.expf_table();
        self.line(format!("cvt.f64.f32 {x}, {source};"));
        self.line(format!("mul.rn.f64 {z}, {x}, 0d40471547652b82fe;"));
        self.line(format!("add.rn.f64 {kd}, {z}, 0d4338000000000000;"));
        self.line(format!("mov.b64 {ki}, {kd};"));
        self.line(format!("sub.rn.f64 {kd}, {kd}, 0d4338000000000000;"));
        self.line(format!("sub.rn.f64 {r}, {z}, {kd};"));
        self.line(format!("and.b64 {offset}, {ki}, 31;"));
        self.line(format!("shl.b64 {offset}, {offset}, 3;"));
        self.line(format!("add.u64 {address}, {table}, {offset};"));
        self.line(format!("ld.const.u64 {scale_bits}, [{address}];"));
        self.line(format!("shl.b64 {exponent_bits}, {ki}, 47;"));
        self.line(format!("add.u64 {scale_bits}, {scale_bits}, {exponent_bits};"));
        if halve {
            self.line(format!("sub.u64 {scale_bits}, {scale_bits}, 0x0010000000000000;"));
        }
        self.line(format!("mov.b64 {s}, {scale_bits};"));
        self.line(format!("mul.rn.f64 {z}, {r}, 0d3ebc6af84b912394;"));
        self.line(format!("add.rn.f64 {z}, {z}, 0d3f2ebfce50fac4f3;"));
        self.line(format!("mul.rn.f64 {r2}, {r}, {r};"));
        self.line(format!("mul.rn.f64 {y}, {r}, 0d3f962e42ff0c52d6;"));
        self.line(format!("add.rn.f64 {y}, {y}, 0d3ff0000000000000;"));
        self.line(format!("mul.rn.f64 {z}, {z}, {r2};"));
        self.line(format!("add.rn.f64 {y}, {z}, {y};"));
        self.line(format!("mul.rn.f64 {y}, {y}, {s};"));
        self.line(format!("cvt.rn.f32.f64 {destination}, {y};"));
    }
}
