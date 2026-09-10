// Piecewise formulas adapted from OpenLibm src/e_sinhf.c and src/e_coshf.c.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn hyperbolic(&mut self, out: Variable, input: Variable, cosine: bool) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("hyperbolic operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("sinh/cosh requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        let (source, result) = if ty.half() {
            (self.half_to_f32(ty, &source)?, self.reg(Scalar::F32))
        } else {
            (source, destination.clone())
        };
        self.hyperbolic_f32(&result, &source, cosine);
        if ty.half() {
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        }
        Ok(())
    }

    fn hyperbolic_f32(&mut self, destination: &str, source: &str, cosine: bool) {
        let predicate = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U32);
        let sign = if cosine { None } else { Some(self.reg(Scalar::U32)) };
        let x = self.reg(Scalar::F32);
        let t = self.reg(Scalar::F32);
        let w = self.reg(Scalar::F32);
        let result = self.reg(Scalar::F32);
        let special = self.label();
        let tiny = self.label();
        let ordinary = self.label();
        let scaled = self.label();
        let overflow = self.label();
        let finish = self.label();
        let end = self.label();

        self.line(format!("mov.b32 {bits}, {source};"));
        if let Some(sign) = &sign {
            self.line(format!("and.b32 {sign}, {bits}, 0x80000000;"));
        }
        self.line(format!("and.b32 {bits}, {bits}, 0x7fffffff;"));
        self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x7f800000;"));
        self.line(format!("@{predicate} bra {special};"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x39800000;"));
        self.line(format!("@{predicate} bra {tiny};"));
        self.line(format!("mov.b32 {x}, {bits};"));
        let expm1_limit = if cosine { "0x3eb17218" } else { "0x41100000" };
        self.line(format!("setp.ge.u32 {predicate}, {bits}, {expm1_limit};"));
        self.line(format!("@{predicate} bra {ordinary};"));
        self.expm1_f32(&t, &x);
        self.line(format!("add.rn.f32 {w}, 0f3f800000, {t};"));

        if cosine {
            self.line(format!("add.rn.f32 {w}, {w}, {w};"));
            self.line(format!("mul.rn.f32 {result}, {t}, {t};"));
            self.line(format!("div.rn.f32 {result}, {result}, {w};"));
            self.line(format!("add.rn.f32 {result}, 0f3f800000, {result};"));
        } else {
            let above_one = self.label();
            let multiply_half = self.label();
            self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x3f800000;"));
            self.line(format!("@{predicate} bra {above_one};"));
            self.line(format!("mul.rn.f32 {result}, {t}, {t};"));
            self.line(format!("div.rn.f32 {result}, {result}, {w};"));
            self.line(format!("mul.rn.f32 {w}, 0f40000000, {t};"));
            self.line(format!("sub.rn.f32 {result}, {w}, {result};"));
            self.line(format!("bra {multiply_half};"));
            self.line(format!("{above_one}:"));
            self.line(format!("div.rn.f32 {result}, {t}, {w};"));
            self.line(format!("add.rn.f32 {result}, {t}, {result};"));
            self.line(format!("{multiply_half}:"));
            self.line(format!("mul.rn.f32 {result}, 0f3f000000, {result};"));
        }
        self.line(format!("bra {finish};"));

        self.line(format!("{ordinary}:"));
        self.line(format!("setp.gt.u32 {predicate}, {bits}, 0x42b2d4fc;"));
        self.line(format!("@{predicate} bra {overflow};"));
        self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x42b17217;"));
        self.line(format!("@{predicate} bra {scaled};"));
        self.exponential_f32(&t, &x);
        self.line(format!("mul.rn.f32 {result}, 0f3f000000, {t};"));
        if cosine {
            self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x41100000;"));
            self.line(format!("@{predicate} bra {finish};"));
            self.line(format!("div.rn.f32 {w}, 0f3f000000, {t};"));
            self.line(format!("add.rn.f32 {result}, {result}, {w};"));
        }
        self.line(format!("bra {finish};"));

        self.line(format!("{scaled}:"));
        self.exponential_finite_f32(&result, &x, true);
        self.line(format!("bra {finish};"));
        self.line(format!("{overflow}:"));
        self.line(format!("mov.f32 {result}, 0f7f800000;"));
        self.line(format!("{finish}:"));
        if let Some(sign) = &sign {
            self.line(format!("mov.b32 {bits}, {result};"));
            self.line(format!("xor.b32 {bits}, {bits}, {sign};"));
            self.line(format!("mov.b32 {destination}, {bits};"));
        } else {
            self.line(format!("mov.f32 {destination}, {result};"));
        }
        self.line(format!("bra {end};"));
        self.line(format!("{special}:"));
        if cosine {
            self.line(format!("mul.rn.f32 {destination}, {source}, {source};"));
        } else {
            self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        }
        self.line(format!("bra {end};"));
        self.line(format!("{tiny}:"));
        if cosine {
            self.line(format!("mov.f32 {destination}, 0f3f800000;"));
        } else {
            self.line(format!("mov.f32 {destination}, {source};"));
        }
        self.line(format!("{end}:"));
    }
}
