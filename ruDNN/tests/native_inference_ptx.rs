//! CPU-only native DSL -> PTX regression tests; no CUDA runtime is loaded.
use ruda_compiler::ptx::{PtxCompilationOptions, PtxCompiler, PtxTarget};
use ruda_core::{
    compiler::Compiler,
    ir::{FloatKind, IntKind, ManagedVariable, Scope, Type, UIntKind, Variable, VariableKind},
    kernel::{KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg, Visibility},
    launch::{ExecutionMode, RudaDim},
};
use rudnn::inference::{cascade, paged};

struct Export {
    kernel: KernelDefinition,
}
impl Export {
    fn new(name: String) -> Self {
        let mut kernel = KernelDefinition {
            buffers: vec![],
            tensor_maps: vec![],
            scalars: vec![],
            ruda_dim: RudaDim::new_1d(32),
            body: Scope::root(false),
            options: KernelOptions {
                kernel_name: name,
                ..Default::default()
            },
        };
        kernel.body.register_type::<usize>(UIntKind::U32.into());
        Self { kernel }
    }
    fn buffer(&mut self, ty: Type, write: bool) -> ManagedVariable {
        let id = self.kernel.buffers.len() as u32;
        self.kernel.buffers.push(KernelArg {
            id,
            ty,
            visibility: if write {
                Visibility::ReadWrite
            } else {
                Visibility::Read
            },
            size: None,
            has_extended_meta: false,
        });
        ManagedVariable::Plain(Variable::new(
            if write {
                VariableKind::GlobalOutputArray(id)
            } else {
                VariableKind::GlobalInputArray(id)
            },
            ty,
        ))
    }
    fn scalar(&mut self, ty: Type) -> ManagedVariable {
        let index = self
            .kernel
            .scalars
            .iter()
            .position(|v| v.ty == ty.storage_type())
            .unwrap_or_else(|| {
                self.kernel.scalars.push(ScalarKernelArg {
                    ty: ty.storage_type(),
                    count: 0,
                });
                self.kernel.scalars.len() - 1
            });
        let offset = self.kernel.scalars[index].count;
        self.kernel.scalars[index].count += 1;
        ManagedVariable::Plain(Variable::new(VariableKind::GlobalScalar(offset as u32), ty))
    }
    fn save(self) {
        let name = self.kernel.options.kernel_name.clone();
        let result = PtxCompiler
            .compile(
                self.kernel,
                &PtxCompilationOptions {
                    target: Some(PtxTarget {
                        version: (8, 0),
                        sm: 80,
                    }),
                },
                ExecutionMode::Unchecked,
                UIntKind::U32.into(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(result.source.contains(".entry"));
        assert_eq!(result.shared_memory_bytes, 0);
        if let Some(path) = std::env::var_os("RUDA_INFERENCE_PTX_OUTPUT") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join(format!("{name}.ptx")), result.source).unwrap();
        }
    }
}
#[test]
fn low_precision_attention_and_sampling_lower_to_ptx() {
    let f = Type::from(FloatKind::F32);
    let u = Type::from(UIntKind::U32);
    let i = Type::from(IntKind::I32);
    let b = Type::from(UIntKind::U8);
    for (kind, label) in [(FloatKind::F16, "f16"), (FloatKind::BF16, "bf16")] {
        let ty = Type::from(kind);
        let mut e = Export::new(format!("native_paged_{label}"));
        let q = e.buffer(ty, false);
        let k = e.buffer(ty, false);
        let v = e.buffer(ty, false);
        let m = e.buffer(u, false);
        let mask = e.buffer(b, false);
        let slopes = e.buffer(f, false);
        let out = e.buffer(ty, true);
        let lse = e.buffer(f, true);
        let scale = e.scalar(f);
        let cap = e.scalar(f);
        let window = e.scalar(i);
        let s: Vec<_> = (0..7).map(|_| e.scalar(u)).collect();
        macro_rules! expand {
            ($float:ty) => {
                paged::attention::expand::<$float>(
                    &mut e.kernel.body,
                    q.into(),
                    k.into(),
                    v.into(),
                    m.into(),
                    mask.into(),
                    slopes.into(),
                    out.into(),
                    lse.into(),
                    scale.into(),
                    cap.into(),
                    window.into(),
                    s[0].clone().into(),
                    s[1].clone().into(),
                    s[2].clone().into(),
                    s[3].clone().into(),
                    s[4].clone().into(),
                    s[5].clone().into(),
                    s[6].clone().into(),
                    64,
                    32,
                    32,
                    false,
                    true,
                    0,
                    false,
                    false,
                    ty.storage_type(),
                );
            };
        }
        if kind == FloatKind::F16 {
            expand!(half::f16);
        } else {
            expand!(half::bf16);
        }
        e.save();
        let mut e = Export::new(format!("native_weights_{label}"));
        let x = e.buffer(ty, false);
        let m = e.buffer(u, false);
        let p = e.buffer(f, false);
        let out = e.buffer(f, true);
        let valid = e.buffer(u, true);
        let width = e.scalar(u);
        macro_rules! expand {
            ($float:ty) => {
                rurand::tensor::sampling::weights::expand::<$float>(
                    &mut e.kernel.body,
                    x.into(),
                    m.into(),
                    p.into(),
                    out.into(),
                    valid.into(),
                    width.into(),
                    true,
                    ty.storage_type(),
                );
            };
        }
        if kind == FloatKind::F16 {
            expand!(half::f16);
        } else {
            expand!(half::bf16);
        }
        e.save();
        let mut e = Export::new(format!("native_merge_{label}"));
        let v = e.buffer(ty, false);
        let s = e.buffer(f, false);
        let out = e.buffer(ty, true);
        let lse = e.buffer(f, true);
        let n = e.scalar(u);
        let h = e.scalar(u);
        let d = e.scalar(u);
        macro_rules! expand {
            ($float:ty) => {
                cascade::merge::expand::<$float>(
                    &mut e.kernel.body,
                    v.into(),
                    s.into(),
                    out.into(),
                    lse.into(),
                    n.into(),
                    h.into(),
                    d.into(),
                    true,
                    ty.storage_type(),
                );
            };
        }
        if kind == FloatKind::F16 {
            expand!(half::f16);
        } else {
            expand!(half::bf16);
        }
        e.save();
    }
}
#[test]
fn cascade_and_cache_transfer_lower_to_ptx() {
    let f = Type::from(FloatKind::F32);
    let u = Type::from(UIntKind::U32);
    for base in [false, true] {
        let mut e = Export::new(format!("native_merge_{base}"));
        let v = e.buffer(f, false);
        let s = e.buffer(f, false);
        let out = e.buffer(f, true);
        let lse = e.buffer(f, true);
        let n = e.scalar(u);
        let h = e.scalar(u);
        let d = e.scalar(u);
        cascade::merge::expand::<f32>(
            &mut e.kernel.body,
            v.into(),
            s.into(),
            out.into(),
            lse.into(),
            n.into(),
            h.into(),
            d.into(),
            base,
            f.storage_type(),
        );
        e.save();
        let mut e = Export::new(format!("native_pair_{base}"));
        let a = e.buffer(f, false);
        let sa = e.buffer(f, false);
        let b = e.buffer(f, false);
        let sb = e.buffer(f, false);
        let out = e.buffer(f, true);
        let lse = e.buffer(f, true);
        let h = e.scalar(u);
        let d = e.scalar(u);
        cascade::pair::expand::<f32>(
            &mut e.kernel.body,
            a.into(),
            sa.into(),
            b.into(),
            sb.into(),
            out.into(),
            lse.into(),
            h.into(),
            d.into(),
            base,
            f.storage_type(),
        );
        e.save();
    }
    for hnd in [false, true] {
        for append in [false, true] {
            let mut e = Export::new(format!("native_transfer_{hnd}_{append}"));
            let t = e.buffer(f, true);
            let c = e.buffer(f, true);
            let m = e.buffer(u, false);
            let s: Vec<_> = (0..6).map(|_| e.scalar(u)).collect();
            paged::transfer::expand::<f32>(
                &mut e.kernel.body,
                t.into(),
                c.into(),
                m.into(),
                s[0].clone().into(),
                s[1].clone().into(),
                s[2].clone().into(),
                s[3].clone().into(),
                s[4].clone().into(),
                s[5].clone().into(),
                hnd,
                append,
                f.storage_type(),
            );
            e.save();
        }
    }
}
#[test]
fn masked_paged_attention_lowers_to_ptx() {
    let f = Type::from(FloatKind::F32);
    let u = Type::from(UIntKind::U32);
    let i = Type::from(IntKind::I32);
    let b = Type::from(UIntKind::U8);
    for hnd in [false, true] {
        for mask in 0u32..3 {
            for causal in [false, true] {
                let mut e = Export::new(format!("native_paged_{hnd}_{mask}_{causal}"));
                let q = e.buffer(f, false);
                let k = e.buffer(f, false);
                let v = e.buffer(f, false);
                let m = e.buffer(u, false);
                let mask_buf = e.buffer(b, false);
                let slopes = e.buffer(f, false);
                let out = e.buffer(f, true);
                let lse = e.buffer(f, true);
                let scale = e.scalar(f);
                let cap = e.scalar(f);
                let window = e.scalar(i);
                let s: Vec<_> = (0..7).map(|_| e.scalar(u)).collect();
                paged::attention::expand::<f32>(
                    &mut e.kernel.body,
                    q.into(),
                    k.into(),
                    v.into(),
                    m.into(),
                    mask_buf.into(),
                    slopes.into(),
                    out.into(),
                    lse.into(),
                    scale.into(),
                    cap.into(),
                    window.into(),
                    s[0].clone().into(),
                    s[1].clone().into(),
                    s[2].clone().into(),
                    s[3].clone().into(),
                    s[4].clone().into(),
                    s[5].clone().into(),
                    s[6].clone().into(),
                    33,
                    65,
                    32,
                    hnd,
                    causal,
                    mask,
                    true,
                    true,
                    f.storage_type(),
                );
                e.save();
            }
        }
    }
}
#[test]
fn filtering_sampling_and_bitpacking_lower_to_ptx() {
    let f = Type::from(FloatKind::F32);
    let u = Type::from(UIntKind::U32);
    let i = Type::from(IntKind::I32);
    let b = Type::from(UIntKind::U8);
    for logits in [false, true] {
        let mut e = Export::new(format!("native_weights_{logits}"));
        let x = e.buffer(f, false);
        let m = e.buffer(u, false);
        let p = e.buffer(f, false);
        let o = e.buffer(f, true);
        let valid = e.buffer(u, true);
        let w = e.scalar(u);
        rurand::tensor::sampling::weights::expand::<f32>(
            &mut e.kernel.body,
            x.into(),
            m.into(),
            p.into(),
            o.into(),
            valid.into(),
            w.into(),
            logits,
            f.storage_type(),
        );
        e.save();
    }
    for joint in [false, true] {
        let mut e = Export::new(format!("native_filter_{joint}"));
        let x = e.buffer(f, true);
        let valid = e.buffer(u, true);
        let m = e.buffer(u, false);
        let p = e.buffer(f, false);
        let w = e.scalar(u);
        rurand::tensor::sampling::filter::expand(
            &mut e.kernel.body,
            x.into(),
            valid.into(),
            m.into(),
            p.into(),
            w.into(),
            joint,
        );
        e.save();
    }
    let mut e = Export::new("native_draw".into());
    let x = e.buffer(f, false);
    let valid = e.buffer(u, false);
    let draws = e.buffer(f, false);
    let out = e.buffer(i, true);
    let good = e.buffer(u, true);
    let w = e.scalar(u);
    rurand::tensor::sampling::draw::expand(
        &mut e.kernel.body,
        x.into(),
        valid.into(),
        draws.into(),
        out.into(),
        good.into(),
        w.into(),
    );
    e.save();
    let mut e = Export::new("native_speculative_chain".into());
    let draft = e.buffer(f, false);
    let ids = e.buffer(i, false);
    let target = e.buffer(f, false);
    let au = e.buffer(f, false);
    let su = e.buffer(f, false);
    let out = e.buffer(i, true);
    let accepted = e.buffer(u, true);
    let emitted_draft = e.buffer(u, true);
    let emitted = e.buffer(u, true);
    let valid = e.buffer(u, true);
    let steps = e.scalar(u);
    let width = e.scalar(u);
    rurand::tensor::sampling::verify_chain::expand(
        &mut e.kernel.body,
        draft.into(),
        ids.into(),
        target.into(),
        au.into(),
        su.into(),
        out.into(),
        accepted.into(),
        emitted_draft.into(),
        emitted.into(),
        valid.into(),
        steps.into(),
        width.into(),
    );
    e.save();
    for big in [false, true] {
        let mut e = Export::new(format!("native_pack_{big}"));
        let x = e.buffer(b, false);
        let m = e.buffer(u, false);
        let o = e.buffer(b, true);
        ruprim::bitpack::pack::expand::<u8>(
            &mut e.kernel.body,
            x.into(),
            m.into(),
            o.into(),
            big,
            b.storage_type(),
        );
        e.save();
        let mut e = Export::new(format!("native_pack_u32_{big}"));
        let x = e.buffer(u, false);
        let m = e.buffer(u, false);
        let o = e.buffer(b, true);
        ruprim::bitpack::pack::expand::<u32>(
            &mut e.kernel.body,
            x.into(),
            m.into(),
            o.into(),
            big,
            u.storage_type(),
        );
        e.save();
    }
}
