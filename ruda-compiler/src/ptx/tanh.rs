// Adapted from OpenLibm src/s_tanhf.c, originally developed at SunPro.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::Variable;

impl Emitter {
    pub fn hyperbolic_tangent(&mut self, out: Variable, input: Variable) -> Result<()> {
        if input.ty != out.ty {
            return Err(invalid("tanh operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        if ty != Scalar::F32 && !ty.half() {
            return Err(unsupported("tanh requires F32, F16 or BF16 storage"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if ty.half() {
            let source = self.half_to_f32(ty, &source)?;
            let result = self.reg(Scalar::F32);
            self.tanh_f32(&result, &source);
            self.line(format!("cvt.rn.{}.f32 {destination}, {result};", ty.suffix()));
        } else {
            self.tanh_f32(&destination, &source);
        }
        Ok(())
    }

    fn tanh_f32(&mut self, destination: &str, source: &str) {
        let predicate = self.reg(Scalar::Pred);
        let large = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U32);
        let sign = self.reg(Scalar::U32);
        let magnitude = self.reg(Scalar::F32);
        let argument = self.reg(Scalar::F32);
        let t = self.reg(Scalar::F32);
        let z = self.reg(Scalar::F32);
        let nan = self.label();
        let tiny = self.label();
        let saturated = self.label();
        let large_result = self.label();
        let apply_sign = self.label();
        let end = self.label();

        self.line(format!("mov.b32 {bits}, {source};"));
        self.line(format!("and.b32 {sign}, {bits}, 0x80000000;"));
        self.line(format!("and.b32 {bits}, {bits}, 0x7fffffff;"));
        self.line(format!("setp.gt.u32 {predicate}, {bits}, 0x7f800000;"));
        self.line(format!("@{predicate} bra {nan};"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x39800000;"));
        self.line(format!("@{predicate} bra {tiny};"));
        self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x41100000;"));
        self.line(format!("@{predicate} bra {saturated};"));
        self.line(format!("mov.b32 {magnitude}, {bits};"));
        self.line(format!("setp.ge.u32 {large}, {bits}, 0x3f800000;"));
        self.line(format!("mul.rn.f32 {argument}, {magnitude}, 0f40000000;"));
        self.line(format!("@!{large} neg.f32 {argument}, {argument};"));
        self.expm1_f32(&t, &argument);
        self.line(format!("add.rn.f32 {z}, {t}, 0f40000000;"));
        self.line(format!("@{large} bra {large_result};"));
        self.line(format!("neg.f32 {t}, {t};"));
        self.line(format!("div.rn.f32 {z}, {t}, {z};"));
        self.line(format!("bra {apply_sign};"));

        self.line(format!("{large_result}:"));
        self.line(format!("div.rn.f32 {z}, 0f40000000, {z};"));
        self.line(format!("sub.rn.f32 {z}, 0f3f800000, {z};"));
        self.line(format!("bra {apply_sign};"));

        self.line(format!("{saturated}:"));
        self.line(format!("mov.f32 {z}, 0f3f800000;"));
        self.line(format!("{apply_sign}:"));
        self.line(format!("mov.b32 {bits}, {z};"));
        self.line(format!("xor.b32 {bits}, {bits}, {sign};"));
        self.line(format!("mov.b32 {destination}, {bits};"));
        self.line(format!("bra {end};"));

        self.line(format!("{nan}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{tiny}:"));
        self.line(format!("mov.f32 {destination}, {source};"));
        self.line(format!("{end}:"));
    }
}
