// Adapted from OpenLibm src/s_expm1f.c, originally developed at SunPro.
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

use super::{emit::Emitter, types::Scalar};

impl Emitter {
    pub(super) fn expm1_f32(&mut self, destination: &str, source: &str) {
        let predicate = self.reg(Scalar::Pred);
        let negative = self.reg(Scalar::Pred);
        let bits = self.reg(Scalar::U32);
        let k = self.reg(Scalar::I32);
        let exponent = self.reg(Scalar::I32);
        let x = self.reg(Scalar::F32);
        let hi = self.reg(Scalar::F32);
        let lo = self.reg(Scalar::F32);
        let c = self.reg(Scalar::F32);
        let t = self.reg(Scalar::F32);
        let e = self.reg(Scalar::F32);
        let hfx = self.reg(Scalar::F32);
        let hxs = self.reg(Scalar::F32);
        let r1 = self.reg(Scalar::F32);
        let scale = self.reg(Scalar::F32);
        let y = self.reg(Scalar::F32);
        let nan = self.label();
        let overflow = self.label();
        let minus_one = self.label();
        let tiny = self.label();
        let primary = self.label();
        let general_reduction = self.label();
        let reduced = self.label();
        let zero_k = self.label();
        let minus_one_k = self.label();
        let one_k = self.label();
        let one_k_negative = self.label();
        let large_k = self.label();
        let max_k = self.label();
        let upper_k = self.label();
        let end = self.label();

        self.line(format!("mov.f32 {x}, {source};"));
        self.line(format!("mov.b32 {bits}, {source};"));
        self.line(format!("setp.ge.u32 {negative}, {bits}, 0x80000000;"));
        self.line(format!("and.b32 {bits}, {bits}, 0x7fffffff;"));
        self.line(format!("setp.gt.u32 {predicate}, {bits}, 0x7f800000;"));
        self.line(format!("@{predicate} bra {nan};"));
        self.line(format!("setp.ge.f32 {predicate}, {x}, 0f42b17218;"));
        self.line(format!("@{predicate} bra {overflow};"));
        self.line(format!("setp.le.f32 {predicate}, {x}, 0fc195b844;"));
        self.line(format!("@{predicate} bra {minus_one};"));
        self.line(format!("setp.lt.u32 {predicate}, {bits}, 0x33000000;"));
        self.line(format!("@{predicate} bra {tiny};"));
        self.line(format!("mov.s32 {k}, 0;"));
        self.line(format!("mov.f32 {c}, 0f00000000;"));
        self.line(format!("setp.le.u32 {predicate}, {bits}, 0x3eb17218;"));
        self.line(format!("@{predicate} bra {primary};"));
        self.line(format!("setp.ge.u32 {predicate}, {bits}, 0x3f851592;"));
        self.line(format!("@{predicate} bra {general_reduction};"));

        self.line(format!("selp.f32 {t}, 0fbf317180, 0f3f317180, {negative};"));
        self.line(format!("sub.rn.f32 {hi}, {x}, {t};"));
        self.line(format!("selp.f32 {lo}, 0fb717f7d1, 0f3717f7d1, {negative};"));
        self.line(format!("selp.s32 {k}, -1, 1, {negative};"));
        self.line(format!("bra {reduced};"));

        self.line(format!("{general_reduction}:"));
        self.line(format!("selp.f32 {t}, 0fbf000000, 0f3f000000, {negative};"));
        self.line(format!("mul.rn.f32 {hi}, {x}, 0f3fb8aa3b;"));
        self.line(format!("add.rn.f32 {t}, {hi}, {t};"));
        self.line(format!("cvt.rzi.s32.f32 {k}, {t};"));
        self.line(format!("cvt.rn.f32.s32 {t}, {k};"));
        self.line(format!("mul.rn.f32 {hi}, {t}, 0f3f317180;"));
        self.line(format!("sub.rn.f32 {hi}, {x}, {hi};"));
        self.line(format!("mul.rn.f32 {lo}, {t}, 0f3717f7d1;"));
        self.line(format!("{reduced}:"));
        self.line(format!("sub.rn.f32 {x}, {hi}, {lo};"));
        self.line(format!("sub.rn.f32 {c}, {hi}, {x};"));
        self.line(format!("sub.rn.f32 {c}, {c}, {lo};"));

        self.line(format!("{primary}:"));
        self.line(format!("mul.rn.f32 {hfx}, {x}, 0f3f000000;"));
        self.line(format!("mul.rn.f32 {hxs}, {x}, {hfx};"));
        self.line(format!("mul.rn.f32 {r1}, {hxs}, 0f3acf3010;"));
        self.line(format!("add.rn.f32 {r1}, {r1}, 0fbd088868;"));
        self.line(format!("mul.rn.f32 {r1}, {r1}, {hxs};"));
        self.line(format!("add.rn.f32 {r1}, {r1}, 0f3f800000;"));
        self.line(format!("mul.rn.f32 {t}, {r1}, {hfx};"));
        self.line(format!("sub.rn.f32 {t}, 0f40400000, {t};"));
        self.line(format!("mul.rn.f32 {e}, {x}, {t};"));
        self.line(format!("sub.rn.f32 {e}, 0f40c00000, {e};"));
        self.line(format!("sub.rn.f32 {y}, {r1}, {t};"));
        self.line(format!("div.rn.f32 {e}, {y}, {e};"));
        self.line(format!("mul.rn.f32 {e}, {hxs}, {e};"));
        self.line(format!("setp.eq.s32 {predicate}, {k}, 0;"));
        self.line(format!("@{predicate} bra {zero_k};"));
        self.line(format!("sub.rn.f32 {e}, {e}, {c};"));
        self.line(format!("mul.rn.f32 {e}, {x}, {e};"));
        self.line(format!("sub.rn.f32 {e}, {e}, {c};"));
        self.line(format!("sub.rn.f32 {e}, {e}, {hxs};"));
        self.line(format!("setp.eq.s32 {predicate}, {k}, -1;"));
        self.line(format!("@{predicate} bra {minus_one_k};"));
        self.line(format!("setp.eq.s32 {predicate}, {k}, 1;"));
        self.line(format!("@{predicate} bra {one_k};"));
        self.line(format!("shl.b32 {exponent}, {k}, 23;"));
        self.line(format!("add.s32 {exponent}, {exponent}, 0x3f800000;"));
        self.line(format!("mov.b32 {scale}, {exponent};"));
        self.line(format!("setp.le.s32 {predicate}, {k}, -2;"));
        self.line(format!("@{predicate} bra {large_k};"));
        self.line(format!("setp.gt.s32 {predicate}, {k}, 56;"));
        self.line(format!("@{predicate} bra {large_k};"));
        self.line(format!("setp.ge.s32 {predicate}, {k}, 23;"));
        self.line(format!("@{predicate} bra {upper_k};"));

        self.line(format!("shr.u32 {bits}, 0x01000000, {k};"));
        self.line(format!("sub.u32 {bits}, 0x3f800000, {bits};"));
        self.line(format!("mov.b32 {t}, {bits};"));
        self.line(format!("sub.rn.f32 {y}, {e}, {x};"));
        self.line(format!("sub.rn.f32 {y}, {t}, {y};"));
        self.line(format!("mul.rn.f32 {destination}, {y}, {scale};"));
        self.line(format!("bra {end};"));

        self.line(format!("{upper_k}:"));
        self.line(format!("sub.s32 {exponent}, 127, {k};"));
        self.line(format!("shl.b32 {exponent}, {exponent}, 23;"));
        self.line(format!("mov.b32 {t}, {exponent};"));
        self.line(format!("add.rn.f32 {y}, {e}, {t};"));
        self.line(format!("sub.rn.f32 {y}, {x}, {y};"));
        self.line(format!("add.rn.f32 {y}, {y}, 0f3f800000;"));
        self.line(format!("mul.rn.f32 {destination}, {y}, {scale};"));
        self.line(format!("bra {end};"));

        self.line(format!("{large_k}:"));
        self.line(format!("sub.rn.f32 {y}, {e}, {x};"));
        self.line(format!("sub.rn.f32 {y}, 0f3f800000, {y};"));
        self.line(format!("setp.eq.s32 {predicate}, {k}, 128;"));
        self.line(format!("@{predicate} bra {max_k};"));
        self.line(format!("mul.rn.f32 {y}, {y}, {scale};"));
        self.line(format!("sub.rn.f32 {destination}, {y}, 0f3f800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{max_k}:"));
        self.line(format!("mul.rn.f32 {y}, {y}, 0f40000000;"));
        self.line(format!("mul.rn.f32 {y}, {y}, 0f7f000000;"));
        self.line(format!("sub.rn.f32 {destination}, {y}, 0f3f800000;"));
        self.line(format!("bra {end};"));

        self.line(format!("{zero_k}:"));
        self.line(format!("mul.rn.f32 {e}, {x}, {e};"));
        self.line(format!("sub.rn.f32 {e}, {e}, {hxs};"));
        self.line(format!("sub.rn.f32 {destination}, {x}, {e};"));
        self.line(format!("bra {end};"));
        self.line(format!("{minus_one_k}:"));
        self.line(format!("sub.rn.f32 {y}, {x}, {e};"));
        self.line(format!("mul.rn.f32 {y}, {y}, 0f3f000000;"));
        self.line(format!("sub.rn.f32 {destination}, {y}, 0f3f000000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{one_k}:"));
        self.line(format!("setp.lt.f32 {predicate}, {x}, 0fbe800000;"));
        self.line(format!("@{predicate} bra {one_k_negative};"));
        self.line(format!("sub.rn.f32 {y}, {x}, {e};"));
        self.line(format!("mul.rn.f32 {y}, {y}, 0f40000000;"));
        self.line(format!("add.rn.f32 {destination}, {y}, 0f3f800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{one_k_negative}:"));
        self.line(format!("add.rn.f32 {y}, {x}, 0f3f000000;"));
        self.line(format!("sub.rn.f32 {y}, {e}, {y};"));
        self.line(format!("mul.rn.f32 {destination}, {y}, 0fc0000000;"));
        self.line(format!("bra {end};"));

        self.line(format!("{nan}:"));
        self.line(format!("add.rn.f32 {destination}, {source}, {source};"));
        self.line(format!("bra {end};"));
        self.line(format!("{overflow}:"));
        self.line(format!("mov.f32 {destination}, 0f7f800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{minus_one}:"));
        self.line(format!("mov.f32 {destination}, 0fbf800000;"));
        self.line(format!("bra {end};"));
        self.line(format!("{tiny}:"));
        self.line(format!("mov.f32 {destination}, {source};"));
        self.line(format!("{end}:"));
    }
}
