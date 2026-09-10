// Adapted from OpenLibm src/s_log1pf.c, originally developed at SunPro.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn logarithm_one_plus(&mut self, out: Variable, input: Variable) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("log1p operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("log1p requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let source = self.half_to_f32(ty, &source)?;
            let result = self.reg(Scalar::F32);
            self.log1p_f32(&result, &source);
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.log1p_f32(&destination, &source);
        }
        Ok(())
    }

    pub(super) fn log1p_f32(&mut self, destination: &str, source: &str) {
        let predicate = self.reg(Scalar::Pred);
        let in_range = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U32);
        let hu = self.reg(Scalar::U32);
        let k = self.reg(Scalar::I32);
        let kf = self.reg(Scalar::F32);
        let f = self.reg(Scalar::F32);
        let c = self.reg(Scalar::F32);
        let u = self.reg(Scalar::F32);
        let hfsq = self.reg(Scalar::F32);
        let s = self.reg(Scalar::F32);
        let z = self.reg(Scalar::F32);
        let r = self.reg(Scalar::F32);
        let t = self.reg(Scalar::F32);
        let lo = self.reg(Scalar::F32);
        let hi = self.reg(Scalar::F32);
        let nan = self.label();
        let domain = self.label();
        let minus_one = self.label();
        let unchanged = self.label();
        let small_input = self.label();
        let large_input = self.label();
        let normalize = self.label();
        let normalize_half = self.label();
        let normalized = self.label();
        let primary = self.label();
        let small_f = self.label();
        let zero_f = self.label();
        let unscaled_small_f = self.label();
        let unscaled = self.label();
        let end = self.label();

        self.line(format!("setp.neu.f32 {predicate}, {source}, {source};"));
        self.line(format!("@{predicate} bra {nan};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0fbf800000;"));
        self.line(format!("@{predicate} bra {minus_one};"));
        self.line(format!("setp.lt.f32 {predicate}, {source}, 0fbf800000;"));
        self.line(format!("@{predicate} bra {domain};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0f7f800000;"));
        self.line(format!("@{predicate} bra {unchanged};"));
        self.line(format!("mov.b32 {bits}, {source};"));
        self.line(format!("and.b32 {bits}, {bits}, 0x7fffffff;"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x33800000;"));
        self.line(format!("@{predicate} bra {unchanged};"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x38000000;"));
        self.line(format!("@{predicate} bra {small_input};"));

        self.line(format!("mov.s32 {k}, 0;"));
        self.line(format!("mov.f32 {c}, 0f00000000;"));
        self.line(format!("mov.f32 {f}, {source};"));
        self.line(format!("mov.u32 {hu}, 1;"));
        self.line(format!("setp.lt.f32 {predicate}, {source}, 0f3ed413d0;"));
        self.line(format!("setp.ge.f32 {in_range}, {source}, 0fbe95f619;"));
        self.line(format!("and.pred {predicate}, {predicate}, {in_range};"));
        self.line(format!("@{predicate} bra {primary};"));
        self.line(format!("setp.ge.f32 {predicate}, {source}, 0f5a000000;"));
        self.line(format!("@{predicate} bra {large_input};"));

        self.line(format!("add.rn.f32 {u}, 0f3f800000, {source};"));
        self.line(format!("mov.b32 {hu}, {u};"));
        self.line(format!("shr.u32 {bits}, {hu}, 23;"));
        self.line(format!("mov.b32 {k}, {bits};"));
        self.line(format!("sub.s32 {k}, {k}, 127;"));
        self.line(format!("sub.rn.f32 {c}, {u}, {source};"));
        self.line(format!("sub.rn.f32 {c}, 0f3f800000, {c};"));
        self.line(format!("sub.rn.f32 {t}, {u}, 0f3f800000;"));
        self.line(format!("sub.rn.f32 {t}, {source}, {t};"));
        self.line(format!("setp.gt.s32 {predicate}, {k}, 0;"));
        self.line(format!("selp.f32 {c}, {c}, {t}, {predicate};"));
        self.line(format!("div.rn.f32 {c}, {c}, {u};"));
        self.line(format!("bra {normalize};"));

        self.line(format!("{large_input}:"));
        self.line(format!("mov.b32 {hu}, {source};"));
        self.line(format!("shr.u32 {bits}, {hu}, 23;"));
        self.line(format!("mov.b32 {k}, {bits};"));
        self.line(format!("sub.s32 {k}, {k}, 127;"));
        self.line(format!("{normalize}:"));
        self.line(format!("and.b32 {hu}, {hu}, 0x007fffff;"));
        self.line(format!("setp.ge.u32 {predicate}, {hu}, 0x003504f4;"));
        self.line(format!("@{predicate} bra {normalize_half};"));
        self.line(format!("or.b32 {bits}, {hu}, 0x3f800000;"));
        self.line(format!("bra {normalized};"));
        self.line(format!("{normalize_half}:"));
        self.line(format!("add.s32 {k}, {k}, 1;"));
        self.line(format!("or.b32 {bits}, {hu}, 0x3f000000;"));
        self.line(format!("sub.u32 {hu}, 0x00800000, {hu};"));
        self.line(format!("shr.u32 {hu}, {hu}, 2;"));
        self.line(format!("{normalized}:"));
        self.line(format!("mov.b32 {u}, {bits};"));
        self.line(format!("sub.rn.f32 {f}, {u}, 0f3f800000;"));

        self.line(format!("{primary}:"));
        self.line(format!("cvt.rn.f32.s32 {kf}, {k};"));
        self.line(format!("mul.rn.f32 {hi}, {kf}, 0f3f317180;"));
        self.line(format!("mul.rn.f32 {lo}, {kf}, 0f3717f7d1;"));
        self.line(format!("mul.rn.f32 {hfsq}, 0f3f000000, {f};"));
        self.line(format!("mul.rn.f32 {hfsq}, {hfsq}, {f};"));
        self.line(format!("setp.eq.u32 {predicate}, {hu}, 0;"));
        self.line(format!("@{predicate} bra {small_f};"));
        self.line(format!("add.rn.f32 {s}, 0f40000000, {f};"));
        self.line(format!("div.rn.f32 {s}, {f}, {s};"));
        self.line(format!("mul.rn.f32 {z}, {s}, {s};"));
        self.line(format!("mov.f32 {r}, 0f3e178897;"));
        for coefficient in [0x3e1cd04fu32, 0x3e3a3325, 0x3e638e29, 0x3e924925, 0x3ecccccd, 0x3f2aaaab] {
            self.line(format!("mul.rn.f32 {r}, {z}, {r};"));
            self.line(format!("add.rn.f32 {r}, 0f{coefficient:08x}, {r};"));
        }
        self.line(format!("mul.rn.f32 {r}, {z}, {r};"));
        self.line(format!("add.rn.f32 {r}, {hfsq}, {r};"));
        self.line(format!("mul.rn.f32 {r}, {s}, {r};"));
        self.line(format!("setp.eq.s32 {predicate}, {k}, 0;"));
        self.line(format!("@{predicate} bra {unscaled};"));
        self.line(format!("add.rn.f32 {lo}, {lo}, {c};"));
        self.line(format!("add.rn.f32 {r}, {r}, {lo};"));
        self.line(format!("sub.rn.f32 {r}, {hfsq}, {r};"));
        self.line(format!("sub.rn.f32 {r}, {r}, {f};"));
        self.line(format!("sub.rn.f32 {destination}, {hi}, {r};"));
        self.line(format!("bra {end};"));
        self.line(format!("{unscaled}:"));
        self.line(format!("sub.rn.f32 {r}, {hfsq}, {r};"));
        self.line(format!("sub.rn.f32 {destination}, {f}, {r};"));
        self.line(format!("bra {end};"));

        self.line(format!("{small_f}:"));
        self.line(format!("setp.eq.f32 {predicate}, {f}, 0f00000000;"));
        self.line(format!("@{predicate} bra {zero_f};"));
        self.line(format!("mul.rn.f32 {r}, 0f3f2aaaab, {f};"));
        self.line(format!("sub.rn.f32 {r}, 0f3f800000, {r};"));
        self.line(format!("mul.rn.f32 {r}, {hfsq}, {r};"));
        self.line(format!("setp.eq.s32 {predicate}, {k}, 0;"));
        self.line(format!("@{predicate} bra {unscaled_small_f};"));
        self.line(format!("add.rn.f32 {lo}, {lo}, {c};"));
        self.line(format!("sub.rn.f32 {r}, {r}, {lo};"));
        self.line(format!("sub.rn.f32 {r}, {r}, {f};"));
        self.line(format!("sub.rn.f32 {destination}, {hi}, {r};"));
        self.line(format!("bra {end};"));
        self.line(format!("{unscaled_small_f}:"));
        self.line(format!("sub.rn.f32 {destination}, {f}, {r};"));
        self.line(format!("bra {end};"));
        self.line(format!("{zero_f}:"));
        self.line(format!("add.rn.f32 {c}, {c}, {lo};"));
        self.line(format!("add.rn.f32 {destination}, {hi}, {c};"));
        self.line(format!("bra {end};"));

        self.line(format!("{small_input}:"));
        self.line(format!("mul.rn.f32 {t}, {source}, {source};"));
        self.line(format!("mul.rn.f32 {t}, {t}, 0f3f000000;"));
        self.line(format!("sub.rn.f32 {destination}, {source}, {t};"));
        self.line(format!("bra {end};"));
        self.line(format!("{nan}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{domain}:"));
        self.line(format!("mov.f32 {destination}, 0f7fc00000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{minus_one}:"));
        self.line(format!("mov.f32 {destination}, 0fff800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{unchanged}:"));
        self.line(format!("mov.f32 {destination}, {source};"));
        self.line(format!("{end}:"));
    }
}
