// Range reduction and polynomials adapted from Arm optimized-routines, under MIT.
// Copyright (c) 2018-2024, Arm Limited. See THIRD_PARTY_NOTICES.md.
use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

const INV_PIO4: [u32; 24] = [
    0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44,
    0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
    0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
];

impl Emitter {
    fn trig_table(&mut self) -> String {
        if let Some(base) = &self.trig_table_base { return base.clone(); }
        let symbol = format!("{}__ruda_inv_pio4", self.kernel_name);
        let values = INV_PIO4.iter().map(|x| format!("0x{x:08x}")).collect::<Vec<_>>().join(", ");
        self.module_declarations += &format!(".const .align 4 .u32 {symbol}[24] = {{ {values} }};\n");
        let base = self.reg(Scalar::U64);
        self.prologue += &format!("    mov.u64 {base}, {symbol};\n");
        self.trig_table_base = Some(base.clone());
        base
    }

    pub(super) fn trigonometry(&mut self, out: Variable, input: Variable, cosine: bool) -> Result<()> {
        if input.ty != out.ty { return Err(invalid("trigonometric operand types differ")); }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("trigonometry requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let source = self.half_to_f32(ty, &source)?;
            let result = self.reg(Scalar::F32);
            self.trigonometry_f32(&result, &source, cosine);
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.trigonometry_f32(&destination, &source, cosine);
        }
        Ok(())
    }

    fn trigonometry_f32(&mut self, destination: &str, source: &str, cosine: bool) {
        let bits = self.reg(Scalar::U32);
        let magnitude = self.reg(Scalar::F32);
        let predicate = self.reg(Scalar::Pred);
        let tiny = self.label();
        let nonfinite = self.label();
        let large = self.label();
        let polynomial = self.label();
        let end = self.label();
        let remainder = self.reg(Scalar::F64);
        let quadrant = self.reg(Scalar::U32);
        self.line(format!("mov.b32 {bits}, {source};"));
        self.line(format!("abs.f32 {magnitude}, {source};"));
        self.line(format!("setp.geu.f32 {predicate}, {magnitude}, 0f7f800000;"));
        self.line(format!("@{predicate} bra {nonfinite};"));
        self.line(format!("setp.lt.f32 {predicate}, {magnitude}, 0f39800000;"));
        self.line(format!("@{predicate} bra {tiny};"));
        self.line(format!("setp.ge.f32 {predicate}, {magnitude}, 0f42f00000;"));
        self.line(format!("@{predicate} bra {large};"));
        let x = self.reg(Scalar::F64);
        let n = self.reg(Scalar::F64);
        self.line(format!("cvt.f64.f32 {x}, {magnitude};"));
        self.line(format!("mul.rn.f64 {n}, {x}, 0d3fe45f306dc9c883;"));
        self.line(format!("cvt.rni.u32.f64 {quadrant}, {n};"));
        self.line(format!("cvt.rn.f64.u32 {n}, {quadrant};"));
        self.line(format!("mul.rn.f64 {n}, {n}, 0d3ff921fb54442d18;"));
        self.line(format!("sub.rn.f64 {remainder}, {x}, {n};"));
        self.line(format!("bra {polynomial};"));

        self.line(format!("{large}:"));
        self.trig_reduce_large(&bits, &remainder, &quadrant);
        self.line(format!("{polynomial}:"));
        let square = self.reg(Scalar::F64);
        self.line(format!("mul.rn.f64 {square}, {remainder}, {remainder};"));
        let sine = self.trig_polynomial(&remainder, &square, false);
        let cos = self.trig_polynomial(&remainder, &square, true);
        let parity = self.reg(Scalar::U32);
        let value = self.reg(Scalar::F64);
        self.line(format!("and.b32 {parity}, {quadrant}, 1;"));
        self.line(format!("setp.ne.u32 {predicate}, {parity}, 0;"));
        let (odd, even) = if cosine { (&sine, &cos) } else { (&cos, &sine) };
        self.line(format!("selp.f64 {value}, {odd}, {even}, {predicate};"));
        let sign = self.reg(Scalar::U32);
        if cosine {
            self.line(format!("add.u32 {sign}, {quadrant}, 1;"));
        } else {
            self.line(format!("mov.u32 {sign}, {quadrant};"));
        }
        self.line(format!("and.b32 {sign}, {sign}, 2;"));
        self.line(format!("shl.b32 {sign}, {sign}, 30;"));
        if !cosine {
            self.line(format!("and.b32 {bits}, {bits}, 0x80000000;"));
            self.line(format!("xor.b32 {sign}, {sign}, {bits};"));
        }
        self.line(format!("cvt.rn.f32.f64 {destination}, {value};"));
        self.line(format!("mov.b32 {bits}, {destination};"));
        self.line(format!("xor.b32 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b32 {destination}, {bits};"));
        self.line(format!("bra {end};"));
        self.line(format!("{tiny}:"));
        self.line(format!("mov.f32 {destination}, {};", if cosine { "0f3f800000" } else { source }));
        self.line(format!("bra {end};"));
        self.line(format!("{nonfinite}:"));
        self.line(format!("sub.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("{end}:"));
    }

    fn trig_reduce_large(&mut self, bits: &str, remainder: &str, quadrant: &str) {
        let table = self.trig_table();
        let index = self.reg(Scalar::U32);
        let shift = self.reg(Scalar::U32);
        let xi = self.reg(Scalar::U32);
        let address = self.reg(Scalar::U64);
        self.line(format!("shr.u32 {index}, {bits}, 26;"));
        self.line(format!("and.b32 {index}, {index}, 15;"));
        self.line(format!("shr.u32 {shift}, {bits}, 23;"));
        self.line(format!("and.b32 {shift}, {shift}, 7;"));
        self.line(format!("and.b32 {xi}, {bits}, 0xffffff;"));
        self.line(format!("or.b32 {xi}, {xi}, 0x800000;"));
        self.line(format!("shl.b32 {xi}, {xi}, {shift};"));
        self.line(format!("mul.wide.u32 {address}, {index}, 4;"));
        self.line(format!("add.u64 {address}, {table}, {address};"));
        let mut products = Vec::new();
        for offset in [0, 16, 32] {
            let digit = self.reg(Scalar::U32);
            let product = self.reg(Scalar::U64);
            self.line(format!("ld.const.u32 {digit}, [{address}+{offset}];"));
            self.line(format!("mul.wide.u32 {product}, {xi}, {digit};"));
            products.push(product);
        }
        let (p0, p1, p2) = (&products[0], &products[1], &products[2]);
        let n = self.reg(Scalar::U64);
        let signed = self.reg(Scalar::I64);
        self.line(format!("shr.u64 {p2}, {p2}, 32;"));
        self.line(format!("shl.b64 {p0}, {p0}, 32;"));
        self.line(format!("or.b64 {p0}, {p0}, {p2};"));
        self.line(format!("add.u64 {p0}, {p0}, {p1};"));
        self.line(format!("add.u64 {n}, {p0}, 0x2000000000000000;"));
        self.line(format!("shr.u64 {n}, {n}, 62;"));
        self.line(format!("cvt.u32.u64 {quadrant}, {n};"));
        self.line(format!("shl.b64 {n}, {n}, 62;"));
        self.line(format!("sub.u64 {p0}, {p0}, {n};"));
        self.line(format!("mov.b64 {signed}, {p0};"));
        self.line(format!("cvt.rn.f64.s64 {remainder}, {signed};"));
        self.line(format!("mul.rn.f64 {remainder}, {remainder}, 0d3c1921fb54442d18;"));
    }

    fn trig_polynomial(&mut self, x: &str, x2: &str, cosine: bool) -> String {
        let result = self.reg(Scalar::F64);
        let tail = self.reg(Scalar::F64);
        let power = self.reg(Scalar::F64);
        if cosine {
            self.line(format!("mul.rn.f64 {power}, {x2}, {x2};"));
            self.line(format!("mul.rn.f64 {tail}, {x2}, 0d3ef99343027bf8c3;"));
            self.line(format!("add.rn.f64 {tail}, {tail}, 0dbf56c087e89a359d;"));
            self.line(format!("mul.rn.f64 {result}, {x2}, 0dbfdffffffd0c621c;"));
            self.line(format!("add.rn.f64 {result}, {result}, 0d3ff0000000000000;"));
            let middle = self.reg(Scalar::F64);
            self.line(format!("mul.rn.f64 {middle}, {power}, 0d3fa55553e1068f19;"));
            self.line(format!("add.rn.f64 {result}, {result}, {middle};"));
        } else {
            self.line(format!("mul.rn.f64 {power}, {x}, {x2};"));
            self.line(format!("mul.rn.f64 {tail}, {x2}, 0dbf2994eb3774cf24;"));
            self.line(format!("add.rn.f64 {tail}, {tail}, 0d3f81107605230bc4;"));
            self.line(format!("mul.rn.f64 {result}, {power}, 0dbfc555545995a603;"));
            self.line(format!("add.rn.f64 {result}, {x}, {result};"));
        }
        self.line(format!("mul.rn.f64 {power}, {power}, {x2};"));
        self.line(format!("mul.rn.f64 {tail}, {power}, {tail};"));
        self.line(format!("add.rn.f64 {result}, {result}, {tail};"));
        result
    }
}
