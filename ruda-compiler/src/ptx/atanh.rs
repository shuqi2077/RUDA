// Adapted from OpenLibm src/e_atanhf.c, originally developed at SunPro.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn inverse_hyperbolic_tangent(&mut self, out: Variable, input: Variable) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("atanh operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("atanh requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let source = self.half_to_f32(ty, &source)?;
            let result = self.reg(Scalar::F32);
            self.atanh_f32(&result, &source);
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.atanh_f32(&destination, &source);
        }
        Ok(())
    }

    fn atanh_f32(&mut self, destination: &str, source: &str) {
        let predicate = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U32);
        let sign = self.reg(Scalar::U32);
        let x = self.reg(Scalar::F32);
        let twice_x = self.reg(Scalar::F32);
        let denominator = self.reg(Scalar::F32);
        let argument = self.reg(Scalar::F32);
        let result = self.reg(Scalar::F32);
        let domain = self.label();
        let pole = self.label();
        let tiny = self.label();
        let large = self.label();
        let logarithm = self.label();
        let apply_sign = self.label();
        let end = self.label();

        self.line(format!("mov.b32 {bits}, {source};"));
        self.line(format!("and.b32 {sign}, {bits}, 0x80000000;"));
        self.line(format!("and.b32 {bits}, {bits}, 0x7fffffff;"));
        self.line(format!("setp.gt.u32 {predicate}, {bits}, 0x3f800000;"));
        self.line(format!("@{predicate} bra {domain};"));
        self.line(format!("setp.eq.u32 {predicate}, {bits}, 0x3f800000;"));
        self.line(format!("@{predicate} bra {pole};"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x31800000;"));
        self.line(format!("@{predicate} bra {tiny};"));
        self.line(format!("mov.b32 {x}, {bits};"));
        self.line(format!("add.rn.f32 {twice_x}, {x}, {x};"));
        self.line(format!("sub.rn.f32 {denominator}, 0f3f800000, {x};"));
        self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x3f000000;"));
        self.line(format!("@{predicate} bra {large};"));
        self.line(format!("mul.rn.f32 {argument}, {twice_x}, {x};"));
        self.line(format!("div.rn.f32 {argument}, {argument}, {denominator};"));
        self.line(format!("add.rn.f32 {argument}, {twice_x}, {argument};"));
        self.line(format!("bra {logarithm};"));
        self.line(format!("{large}:"));
        self.line(format!("div.rn.f32 {argument}, {twice_x}, {denominator};"));
        self.line(format!("{logarithm}:"));
        self.log1p_f32(&result, &argument);
        self.line(format!("mul.rn.f32 {result}, 0f3f000000, {result};"));
        self.line(format!("bra {apply_sign};"));

        self.line(format!("{pole}:"));
        self.line(format!("mov.f32 {result}, 0f7f800000;"));
        self.line(format!("{apply_sign}:"));
        self.line(format!("mov.b32 {bits}, {result};"));
        self.line(format!("xor.b32 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b32 {destination}, {bits};"));
        self.line(format!("bra {end};"));
        self.line(format!("{domain}:"));
        self.line(format!("mov.f32 {destination}, 0f7fc00000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{tiny}:"));
        self.line(format!("mov.f32 {destination}, {source};"));
        self.line(format!("{end}:"));
    }
}
