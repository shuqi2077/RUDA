// logf algorithm and table adapted from Arm optimized-routines, under MIT.
// Copyright (c) 2017-2025, Arm Limited.
// See THIRD_PARTY_NOTICES.md for the source files and license.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

const LOGF_TABLE: [(u64, u64); 16] = [
    (0x3ff661ec79f8f3be, 0xbfd57bf7808caade),
    (0x3ff571ed4aaf883d, 0xbfd2bef0a7c06ddb),
    (0x3ff49539f0f010b0, 0xbfd01eae7f513a67),
    (0x3ff3c995b0b80385, 0xbfcb31d8a68224e9),
    (0x3ff30d190c8864a5, 0xbfc6574f0ac07758),
    (0x3ff25e227b0b8ea0, 0xbfc1aa2bc79c8100),
    (0x3ff1bb4a4a1a343f, 0xbfba4e76ce8c0e5e),
    (0x3ff12358f08ae5ba, 0xbfb1973c5a611ccc),
    (0x3ff0953f419900a7, 0xbfa252f438e10c1e),
    (0x3ff0000000000000, 0x0000000000000000),
    (0x3fee608cfd9a47ac, 0x3faaa5aa5df25984),
    (0x3feca4b31f026aa0, 0x3fbc5e53aa362eb4),
    (0x3feb2036576afce6, 0x3fc526e57720db08),
    (0x3fe9c2d163a1aa2d, 0x3fcbc2860d224770),
    (0x3fe886e6037841ed, 0x3fd1058bc8a07ee1),
    (0x3fe767dcf5534862, 0x3fd4043057b6ee09),
];

impl Emitter {
    fn logf_table(&mut self) -> String {
        if let Some(base) = &self.logf_table_base {
            return base.clone();
        }
        let values = LOGF_TABLE.iter()
            .map(|(invc, logc)| format!("0x{invc:016x}, 0x{logc:016x}"))
            .collect::<Vec<_>>()
            .join(", ");
        self.module_declarations += &format!(
            ".const .align 8 .u64 {}[32] = {{ {values} }};\n",
            self.logf_table_symbol,
        );
        let base = self.reg(Scalar::U64);
        self.prologue += &format!("    mov.u64 {base}, {};\n", self.logf_table_symbol);
        self.logf_table_base = Some(base.clone());
        base
    }

    pub fn logarithm(&mut self, out: Variable, input: Variable) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("logarithm operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("logarithm requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let source = self.half_to_f32(ty, &source)?;
            let result = self.reg(Scalar::F32);
            self.logarithm_f32(&result, &source);
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.logarithm_f32(&destination, &source);
        }
        Ok(())
    }

    pub(super) fn logarithm_f32(&mut self, destination: &str, source: &str) {
        let predicate = self.reg(Scalar::Pred);
        let nan = self.label();
        let zero = self.label();
        let negative = self.label();
        let infinity = self.label();
        let one = self.label();
        let normalized = self.label();
        let end = self.label();
        self.line(format!("setp.neu.f32 {predicate}, {source}, {source};"));
        self.line(format!("@{predicate} bra {nan};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0f00000000;"));
        self.line(format!("@{predicate} bra {zero};"));
        self.line(format!("setp.lt.f32 {predicate}, {source}, 0f00000000;"));
        self.line(format!("@{predicate} bra {negative};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0f7f800000;"));
        self.line(format!("@{predicate} bra {infinity};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0f3f800000;"));
        self.line(format!("@{predicate} bra {one};"));

        let ix = self.reg(Scalar::U32);
        let normalized_x = self.reg(Scalar::F32);
        self.line(format!("mov.b32 {ix}, {source};"));
        self.line(format!("setp.ge.u32 {predicate}, {ix}, 0x00800000;"));
        self.line(format!("@{predicate} bra {normalized};"));
        self.line(format!("mul.rn.f32 {normalized_x}, {source}, 0f4b000000;"));
        self.line(format!("mov.b32 {ix}, {normalized_x};"));
        self.line(format!("sub.u32 {ix}, {ix}, 0x0b800000;"));
        self.line(format!("{normalized}:"));

        let tmp = self.reg(Scalar::U32);
        let index = self.reg(Scalar::U32);
        let k = self.reg(Scalar::I32);
        let iz = self.reg(Scalar::U32);
        let z_float = self.reg(Scalar::F32);
        let z = self.reg(Scalar::F64);
        let kd = self.reg(Scalar::F64);
        let offset = self.reg(Scalar::U64);
        let address = self.reg(Scalar::U64);
        let invc = self.reg(Scalar::F64);
        let logc = self.reg(Scalar::F64);
        let r = self.reg(Scalar::F64);
        let r2 = self.reg(Scalar::F64);
        let y0 = self.reg(Scalar::F64);
        let y = self.reg(Scalar::F64);
        let term = self.reg(Scalar::F64);
        let table = self.logf_table();
        self.line(format!("sub.u32 {tmp}, {ix}, 0x3f330000;"));
        self.line(format!("shr.u32 {index}, {tmp}, 19;"));
        self.line(format!("and.b32 {index}, {index}, 15;"));
        self.line(format!("mov.b32 {k}, {tmp};"));
        self.line(format!("shr.s32 {k}, {k}, 23;"));
        self.line(format!("and.b32 {iz}, {tmp}, 0xff800000;"));
        self.line(format!("sub.u32 {iz}, {ix}, {iz};"));
        self.line(format!("mov.b32 {z_float}, {iz};"));
        self.line(format!("cvt.f64.f32 {z}, {z_float};"));
        self.line(format!("cvt.rn.f64.s32 {kd}, {k};"));
        self.line(format!("cvt.u64.u32 {offset}, {index};"));
        self.line(format!("shl.b64 {offset}, {offset}, 4;"));
        self.line(format!("add.u64 {address}, {table}, {offset};"));
        self.line(format!("ld.const.f64 {invc}, [{address}];"));
        self.line(format!("ld.const.f64 {logc}, [{address}+8];"));
        self.line(format!("mul.rn.f64 {r}, {z}, {invc};"));
        self.line(format!("sub.rn.f64 {r}, {r}, 0d3ff0000000000000;"));
        self.line(format!("mul.rn.f64 {y0}, {kd}, 0d3fe62e42fefa39ef;"));
        self.line(format!("add.rn.f64 {y0}, {logc}, {y0};"));
        self.line(format!("mul.rn.f64 {r2}, {r}, {r};"));
        self.line(format!("mul.rn.f64 {y}, {r}, 0d3fd5575b0be00b6a;"));
        self.line(format!("add.rn.f64 {y}, {y}, 0dbfdffffef20a4123;"));
        self.line(format!("mul.rn.f64 {term}, {r2}, 0dbfd00ea348b88334;"));
        self.line(format!("add.rn.f64 {y}, {term}, {y};"));
        self.line(format!("mul.rn.f64 {y}, {y}, {r2};"));
        self.line(format!("add.rn.f64 {y0}, {y0}, {r};"));
        self.line(format!("add.rn.f64 {y}, {y}, {y0};"));
        self.line(format!("cvt.rn.f32.f64 {destination}, {y};"));
        self.line(format!("bra {end};"));

        self.line(format!("{nan}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{zero}:"));
        self.line(format!("mov.f32 {destination}, 0fff800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{negative}:"));
        self.line(format!("mov.f32 {destination}, 0f7fc00000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{infinity}:"));
        self.line(format!("mov.f32 {destination}, 0f7f800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{one}:"));
        self.line(format!("mov.f32 {destination}, 0f00000000;"));
        self.line(format!("{end}:"));
    }
}
