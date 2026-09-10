// Adapted from OpenLibm src/s_asinhf.c and src/e_acoshf.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn inverse_hyperbolic(
        &mut self,
        out: Variable,
        input: Variable,
        cosine: bool,
    ) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("inverse hyperbolic operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("asinh/acosh requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        let (source, result) = if ty.half() {
            (self.half_to_f32(ty, &source)?, self.reg(Scalar::F32))
        } else {
            (source, destination.clone())
        };
        if cosine {
            self.acosh_f32(&result, &source)?;
        } else {
            self.asinh_f32(&result, &source)?;
        }
        if ty.half() {
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        }
        Ok(())
    }

    fn asinh_f32(&mut self, destination: &str, source: &str) -> Result<()> {
        let predicate = self.reg(Scalar::Pred);
        let large = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U32);
        let sign = self.reg(Scalar::U32);
        let x = self.reg(Scalar::F32);
        let square = self.reg(Scalar::F32);
        let argument = self.reg(Scalar::F32);
        let t = self.reg(Scalar::F32);
        let result = self.reg(Scalar::F32);
        let special = self.label();
        let tiny = self.label();
        let medium = self.label();
        let ordinary_log = self.label();
        let apply_sign = self.label();
        let end = self.label();

        self.line(format!("mov.b32 {bits}, {source};"));
        self.line(format!("and.b32 {sign}, {bits}, 0x80000000;"));
        self.line(format!("and.b32 {bits}, {bits}, 0x7fffffff;"));
        self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x7f800000;"));
        self.line(format!("@{predicate} bra {special};"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x31800000;"));
        self.line(format!("@{predicate} bra {tiny};"));
        self.line(format!("mov.b32 {x}, {bits};"));
        self.line(format!("mov.f32 {argument}, {x};"));
        self.line(format!("setp.gt.u32 {large}, {bits}, 0x4d800000;"));
        self.line(format!("@{large} bra {ordinary_log};"));
        self.line(format!("mul.rn.f32 {square}, {x}, {x};"));
        self.line(format!("add.rn.f32 {t}, {square}, 0f3f800000;"));
        let root = self.square_root(Scalar::F32, &t, false)?;
        self.line(format!("setp.gt.u32 {predicate}, {bits}, 0x40000000;"));
        self.line(format!("@{predicate} bra {medium};"));

        self.line(format!("add.rn.f32 {t}, 0f3f800000, {root};"));
        self.line(format!("div.rn.f32 {t}, {square}, {t};"));
        self.line(format!("add.rn.f32 {argument}, {x}, {t};"));
        self.log1p_f32(&result, &argument);
        self.line(format!("bra {apply_sign};"));

        self.line(format!("{medium}:"));
        self.line(format!("add.rn.f32 {t}, {root}, {x};"));
        self.line(format!("div.rn.f32 {t}, 0f3f800000, {t};"));
        self.line(format!("mul.rn.f32 {argument}, 0f40000000, {x};"));
        self.line(format!("add.rn.f32 {argument}, {argument}, {t};"));
        self.line(format!("{ordinary_log}:"));
        self.logarithm_f32(&result, &argument);
        self.line(format!("@{large} add.rn.f32 {result}, {result}, 0f3f317218;"));
        self.line(format!("{apply_sign}:"));
        self.line(format!("mov.b32 {bits}, {result};"));
        self.line(format!("xor.b32 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b32 {destination}, {bits};"));
        self.line(format!("bra {end};"));
        self.line(format!("{special}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{tiny}:"));
        self.line(format!("mov.f32 {destination}, {source};"));
        self.line(format!("{end}:"));
        Ok(())
    }

    fn acosh_f32(&mut self, destination: &str, source: &str) -> Result<()> {
        let predicate = self.reg(Scalar::Pred);
        let large = self.reg(Scalar::Pred);
        let t = self.reg(Scalar::F32);
        let radicand = self.reg(Scalar::F32);
        let argument = self.reg(Scalar::F32);
        let result = self.reg(Scalar::F32);
        let special = self.label();
        let domain = self.label();
        let one = self.label();
        let medium = self.label();
        let ordinary_log = self.label();
        let end = self.label();

        self.line(format!("setp.neu.f32 {predicate}, {source}, {source};"));
        self.line(format!("@{predicate} bra {special};"));
        self.line(format!("setp.lt.f32 {predicate}, {source}, 0f3f800000;"));
        self.line(format!("@{predicate} bra {domain};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0f7f800000;"));
        self.line(format!("@{predicate} bra {special};"));
        self.line(format!("setp.eq.f32 {predicate}, {source}, 0f3f800000;"));
        self.line(format!("@{predicate} bra {one};"));
        self.line(format!("mov.f32 {argument}, {source};"));
        self.line(format!("setp.ge.f32 {large}, {source}, 0f4d800000;"));
        self.line(format!("@{large} bra {ordinary_log};"));
        self.line(format!("setp.gt.f32 {predicate}, {source}, 0f40000000;"));
        self.line(format!("@{predicate} bra {medium};"));

        self.line(format!("sub.rn.f32 {t}, {source}, 0f3f800000;"));
        self.line(format!("mul.rn.f32 {radicand}, {t}, {t};"));
        self.line(format!("mul.rn.f32 {argument}, 0f40000000, {t};"));
        self.line(format!("add.rn.f32 {radicand}, {argument}, {radicand};"));
        let small_root = self.square_root(Scalar::F32, &radicand, false)?;
        self.line(format!("add.rn.f32 {argument}, {t}, {small_root};"));
        self.log1p_f32(destination, &argument);
        self.line(format!("bra {end};"));

        self.line(format!("{medium}:"));
        self.line(format!("mul.rn.f32 {radicand}, {source}, {source};"));
        self.line(format!("sub.rn.f32 {radicand}, {radicand}, 0f3f800000;"));
        let root = self.square_root(Scalar::F32, &radicand, false)?;
        self.line(format!("add.rn.f32 {t}, {source}, {root};"));
        self.line(format!("div.rn.f32 {t}, 0f3f800000, {t};"));
        self.line(format!("mul.rn.f32 {argument}, 0f40000000, {source};"));
        self.line(format!("sub.rn.f32 {argument}, {argument}, {t};"));
        self.line(format!("{ordinary_log}:"));
        self.logarithm_f32(&result, &argument);
        self.line(format!("@{large} add.rn.f32 {result}, {result}, 0f3f317218;"));
        self.line(format!("mov.f32 {destination}, {result};"));
        self.line(format!("bra {end};"));

        self.line(format!("{special}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{domain}:"));
        self.line(format!("mov.f32 {destination}, 0f7fc00000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{one}:"));
        self.line(format!("mov.f32 {destination}, 0f00000000;"));
        self.line(format!("{end}:"));
        Ok(())
    }
}
