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

        if let Some(path) = std::env::var_os("RUDA_INFERENCE_PTX_OUTPUT") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join(format!("{name}.ptx")), result.source).unwrap();
        }
    }
}

use rudnn::inference::*;

#[test]
fn activation_gate_cast_base() {
    let mut e = Export::new("activation_gate_cast_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::activation::gate_cast::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        0u32,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn activation_gate_cast_full() {
    let mut e = Export::new("activation_gate_cast_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::activation::gate_cast::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        1u32,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn activation_gate_base() {
    let mut e = Export::new("activation_gate_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::activation::gate::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        0u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn activation_gate_full() {
    let mut e = Export::new("activation_gate_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::activation::gate::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        1u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn attention_concat_mla_base() {
    let mut e = Export::new("attention_concat_mla_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::attention::concat_mla::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
    );
    e.save();
}

#[test]
fn attention_concat_mla_full() {
    let mut e = Export::new("attention_concat_mla_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::attention::concat_mla::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
    );
    e.save();
}

#[test]
fn attention_attend_base() {
    let mut e = Export::new("attention_attend_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(UIntKind::U8), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), false);
    let v9 = e.buffer(Type::from(FloatKind::F32), false);
    let v10 = e.buffer(Type::from(IntKind::I32), false);
    let v11 = e.buffer(Type::from(FloatKind::F32), true);
    let v12 = e.buffer(Type::from(FloatKind::F32), true);
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    let v16 = e.scalar(Type::from(UIntKind::U32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    let v18 = e.scalar(Type::from(UIntKind::U32));
    let v19 = e.scalar(Type::from(FloatKind::F32));
    let v20 = e.scalar(Type::from(FloatKind::F32));
    let v21 = e.scalar(Type::from(IntKind::I32));
    let v22 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::attention::attend::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        4usize,
        4usize,
        32usize,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn attention_attend_full() {
    let mut e = Export::new("attention_attend_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(UIntKind::U8), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), false);
    let v9 = e.buffer(Type::from(FloatKind::F32), false);
    let v10 = e.buffer(Type::from(IntKind::I32), false);
    let v11 = e.buffer(Type::from(FloatKind::F32), true);
    let v12 = e.buffer(Type::from(FloatKind::F32), true);
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    let v16 = e.scalar(Type::from(UIntKind::U32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    let v18 = e.scalar(Type::from(UIntKind::U32));
    let v19 = e.scalar(Type::from(FloatKind::F32));
    let v20 = e.scalar(Type::from(FloatKind::F32));
    let v21 = e.scalar(Type::from(IntKind::I32));
    let v22 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::attention::attend::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        4usize,
        4usize,
        32usize,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn cascade_merge_base() {
    let mut e = Export::new("cascade_merge_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::cascade::merge::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn cascade_merge_full() {
    let mut e = Export::new("cascade_merge_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::cascade::merge::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn cascade_pair_base() {
    let mut e = Export::new("cascade_pair_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::cascade::pair::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn cascade_pair_full() {
    let mut e = Export::new("cascade_pair_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::cascade::pair::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn cascade_masked_pair_base() {
    let mut e = Export::new("cascade_masked_pair_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::cascade::masked_pair::expand::<f32, u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn cascade_masked_pair_full() {
    let mut e = Export::new("cascade_masked_pair_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::cascade::masked_pair::expand::<f32, u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn experts_router_base() {
    let mut e = Export::new("experts_router_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::experts::router::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        4usize,
        1usize,
        1usize,
        1usize,
        4u32,
        false,
        false,
        false,
        false,
    );
    e.save();
}

#[test]
fn experts_router_full() {
    let mut e = Export::new("experts_router_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::experts::router::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        4usize,
        1usize,
        1usize,
        1usize,
        4u32,
        true,
        true,
        true,
        true,
    );
    e.save();
}

#[test]
fn experts_expert_first_base() {
    let mut e = Export::new("experts_expert_first_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::experts::expert_first::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        false,
        0u32,
        0u32,
        false,
        false,
    );
    e.save();
}

#[test]
fn experts_expert_first_full() {
    let mut e = Export::new("experts_expert_first_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::experts::expert_first::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        true,
        1u32,
        1u32,
        true,
        true,
    );
    e.save();
}

#[test]
fn experts_expert_second_base() {
    let mut e = Export::new("experts_expert_second_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::expert_second::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn experts_expert_second_full() {
    let mut e = Export::new("experts_expert_second_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::expert_second::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn experts_reorder_base() {
    let mut e = Export::new("experts_reorder_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::reorder::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn experts_reorder_full() {
    let mut e = Export::new("experts_reorder_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::reorder::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn experts_gather_scales_base() {
    let mut e = Export::new("experts_gather_scales_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::gather_scales::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn experts_gather_scales_full() {
    let mut e = Export::new("experts_gather_scales_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::gather_scales::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn experts_unpack_route_words_base() {
    let mut e = Export::new("experts_unpack_route_words_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::unpack_route_words::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn experts_unpack_route_words_full() {
    let mut e = Export::new("experts_unpack_route_words_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::experts::unpack_route_words::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn fused_transpose_kernel_base() {
    let mut e = Export::new("fused_transpose_kernel_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::fused::transpose_kernel::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn fused_transpose_kernel_full() {
    let mut e = Export::new("fused_transpose_kernel_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::fused::transpose_kernel::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn fused_gate_gradient_base() {
    let mut e = Export::new("fused_gate_gradient_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::fused::gate_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        0u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn fused_gate_gradient_full() {
    let mut e = Export::new("fused_gate_gradient_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::fused::gate_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        1u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn fused_mla_pack_base() {
    let mut e = Export::new("fused_mla_pack_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::fused::mla_pack::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
    );
    e.save();
}

#[test]
fn fused_mla_pack_full() {
    let mut e = Export::new("fused_mla_pack_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::fused::mla_pack::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
    );
    e.save();
}

#[test]
fn indexer_scores_base() {
    let mut e = Export::new("indexer_scores_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::indexer::scores::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn indexer_scores_full() {
    let mut e = Export::new("indexer_scores_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::indexer::scores::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn indexer_mqa_lengths_base() {
    let mut e = Export::new("indexer_mqa_lengths_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::indexer::mqa_lengths::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn indexer_mqa_lengths_full() {
    let mut e = Export::new("indexer_mqa_lengths_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::indexer::mqa_lengths::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn linear_fp8_blocks_base() {
    let mut e = Export::new("linear_fp8_blocks_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::fp8_blocks::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
    );
    e.save();
}

#[test]
fn linear_fp8_blocks_full() {
    let mut e = Export::new("linear_fp8_blocks_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::fp8_blocks::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
    );
    e.save();
}

#[test]
fn linear_encode_blocks_base() {
    let mut e = Export::new("linear_encode_blocks_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), true);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::encode_blocks::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
    );
    e.save();
}

#[test]
fn linear_encode_blocks_full() {
    let mut e = Export::new("linear_encode_blocks_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), true);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::encode_blocks::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
    );
    e.save();
}

#[test]
fn linear_fp8_values_base() {
    let mut e = Export::new("linear_fp8_values_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::fp8_values::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        false,
    );
    e.save();
}

#[test]
fn linear_fp8_values_full() {
    let mut e = Export::new("linear_fp8_values_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::fp8_values::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        true,
    );
    e.save();
}

#[test]
fn linear_fp8_pack_base() {
    let mut e = Export::new("linear_fp8_pack_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::fp8_pack::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        false,
    );
    e.save();
}

#[test]
fn linear_fp8_pack_full() {
    let mut e = Export::new("linear_fp8_pack_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::fp8_pack::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        true,
    );
    e.save();
}

#[test]
fn linear_linear_epilogue_base() {
    let mut e = Export::new("linear_linear_epilogue_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(FloatKind::F32));
    let v5 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::linear::linear_epilogue::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_linear_epilogue_full() {
    let mut e = Export::new("linear_linear_epilogue_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(FloatKind::F32));
    let v5 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::linear::linear_epilogue::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_blocked_weight_base() {
    let mut e = Export::new("linear_blocked_weight_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::blocked_weight::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        false,
    );
    e.save();
}

#[test]
fn linear_blocked_weight_full() {
    let mut e = Export::new("linear_blocked_weight_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::blocked_weight::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        true,
    );
    e.save();
}

#[test]
fn linear_affine_values_base() {
    let mut e = Export::new("linear_affine_values_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::linear::affine_values::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_affine_values_full() {
    let mut e = Export::new("linear_affine_values_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::linear::affine_values::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_segmented_base() {
    let mut e = Export::new("linear_segmented_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::segmented::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_segmented_full() {
    let mut e = Export::new("linear_segmented_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::linear::segmented::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_lora_epilogue_base() {
    let mut e = Export::new("linear_lora_epilogue_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::linear::lora_epilogue::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn linear_lora_epilogue_full() {
    let mut e = Export::new("linear_lora_epilogue_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::linear::lora_epilogue::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mamba_ids_to_boundaries_base() {
    let mut e = Export::new("mamba_ids_to_boundaries_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::ids_to_boundaries::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn mamba_ids_to_boundaries_full() {
    let mut e = Export::new("mamba_ids_to_boundaries_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::ids_to_boundaries::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn mamba_capture_states_base() {
    let mut e = Export::new("mamba_capture_states_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::capture_states::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
    );
    e.save();
}

#[test]
fn mamba_capture_states_full() {
    let mut e = Export::new("mamba_capture_states_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::capture_states::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
    );
    e.save();
}

#[test]
fn mamba_selective_scan_base() {
    let mut e = Export::new("mamba_selective_scan_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), false);
    let v9 = e.buffer(Type::from(IntKind::I32), false);
    let v10 = e.buffer(Type::from(IntKind::I32), false);
    let v11 = e.buffer(Type::from(IntKind::I32), false);
    let v12 = e.buffer(Type::from(IntKind::I32), false);
    let v13 = e.buffer(Type::from(FloatKind::F32), true);
    let v14 = e.buffer(Type::from(FloatKind::F32), true);
    let v15 = e.buffer(Type::from(FloatKind::F32), true);
    let v16 = e.buffer(Type::from(FloatKind::F32), true);
    let v17 = e.scalar(Type::from(UIntKind::U32));
    let v18 = e.scalar(Type::from(UIntKind::U32));
    let v19 = e.scalar(Type::from(UIntKind::U32));
    let v20 = e.scalar(Type::from(UIntKind::U32));
    let v21 = e.scalar(Type::from(UIntKind::U32));
    let v22 = e.scalar(Type::from(UIntKind::U32));
    let v23 = e.scalar(Type::from(IntKind::I32));
    let v24 = e.scalar(Type::from(FloatKind::F32));
    let v25 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::mamba::selective_scan::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        v23.into(),
        v24.into(),
        v25.into(),
        4usize,
        0u32,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mamba_selective_scan_full() {
    let mut e = Export::new("mamba_selective_scan_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), false);
    let v9 = e.buffer(Type::from(IntKind::I32), false);
    let v10 = e.buffer(Type::from(IntKind::I32), false);
    let v11 = e.buffer(Type::from(IntKind::I32), false);
    let v12 = e.buffer(Type::from(IntKind::I32), false);
    let v13 = e.buffer(Type::from(FloatKind::F32), true);
    let v14 = e.buffer(Type::from(FloatKind::F32), true);
    let v15 = e.buffer(Type::from(FloatKind::F32), true);
    let v16 = e.buffer(Type::from(FloatKind::F32), true);
    let v17 = e.scalar(Type::from(UIntKind::U32));
    let v18 = e.scalar(Type::from(UIntKind::U32));
    let v19 = e.scalar(Type::from(UIntKind::U32));
    let v20 = e.scalar(Type::from(UIntKind::U32));
    let v21 = e.scalar(Type::from(UIntKind::U32));
    let v22 = e.scalar(Type::from(UIntKind::U32));
    let v23 = e.scalar(Type::from(IntKind::I32));
    let v24 = e.scalar(Type::from(FloatKind::F32));
    let v25 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::mamba::selective_scan::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        v23.into(),
        v24.into(),
        v25.into(),
        4usize,
        1u32,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mamba_state_scatter_base() {
    let mut e = Export::new("mamba_state_scatter_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::state_scatter::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn mamba_state_scatter_full() {
    let mut e = Export::new("mamba_state_scatter_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::state_scatter::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn mamba_conv_history_base() {
    let mut e = Export::new("mamba_conv_history_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(IntKind::I32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::conv_history::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        4usize,
        false,
        false,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mamba_conv_history_full() {
    let mut e = Export::new("mamba_conv_history_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(IntKind::I32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mamba::conv_history::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        4usize,
        true,
        true,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn masks_ragged_rows_base() {
    let mut e = Export::new("masks_ragged_rows_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U64), false);
    let v3 = e.buffer(Type::from(UIntKind::U64), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::masks::ragged_rows::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
    );
    e.save();
}

#[test]
fn masks_ragged_rows_full() {
    let mut e = Export::new("masks_ragged_rows_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U64), false);
    let v3 = e.buffer(Type::from(UIntKind::U64), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::masks::ragged_rows::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
    );
    e.save();
}

#[test]
fn masks_convert_base() {
    let mut e = Export::new("masks_convert_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U64), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::masks::convert::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        4u32,
        false,
    );
    e.save();
}

#[test]
fn masks_convert_full() {
    let mut e = Export::new("masks_convert_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U64), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::masks::convert::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        4u32,
        true,
    );
    e.save();
}

#[test]
fn metadata_cast_base() {
    let mut e = Export::new("metadata_cast_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::metadata::cast::expand::<u32, u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        Type::from(UIntKind::U32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn metadata_cast_full() {
    let mut e = Export::new("metadata_cast_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::metadata::cast::expand::<u32, u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        Type::from(UIntKind::U32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_prepare_base() {
    let mut e = Export::new("mhc_prepare_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::prepare::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_prepare_full() {
    let mut e = Export::new("mhc_prepare_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::prepare::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_finish_base() {
    let mut e = Export::new("mhc_finish_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::finish::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_finish_full() {
    let mut e = Export::new("mhc_finish_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::finish::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_prepare_gradient_base() {
    let mut e = Export::new("mhc_prepare_gradient_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::prepare_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_prepare_gradient_full() {
    let mut e = Export::new("mhc_prepare_gradient_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::prepare_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_finish_gradient_base() {
    let mut e = Export::new("mhc_finish_gradient_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.buffer(Type::from(FloatKind::F32), true);
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(FloatKind::F32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::finish_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_finish_gradient_full() {
    let mut e = Export::new("mhc_finish_gradient_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.buffer(Type::from(FloatKind::F32), true);
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(FloatKind::F32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::finish_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_parameter_gradient_base() {
    let mut e = Export::new("mhc_parameter_gradient_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::parameter_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_parameter_gradient_full() {
    let mut e = Export::new("mhc_parameter_gradient_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::parameter_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_head_kernel_base() {
    let mut e = Export::new("mhc_head_kernel_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::head_kernel::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_head_kernel_full() {
    let mut e = Export::new("mhc_head_kernel_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::head_kernel::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_head_gradient_base() {
    let mut e = Export::new("mhc_head_gradient_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::head_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_head_gradient_full() {
    let mut e = Export::new("mhc_head_gradient_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::head_gradient::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_pre_external_base() {
    let mut e = Export::new("mhc_pre_external_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(FloatKind::F32));
    let v13 = e.scalar(Type::from(FloatKind::F32));
    let v14 = e.scalar(Type::from(FloatKind::F32));
    let v15 = e.scalar(Type::from(FloatKind::F32));
    let v16 = e.scalar(Type::from(FloatKind::F32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::pre_external::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_pre_external_full() {
    let mut e = Export::new("mhc_pre_external_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(FloatKind::F32));
    let v13 = e.scalar(Type::from(FloatKind::F32));
    let v14 = e.scalar(Type::from(FloatKind::F32));
    let v15 = e.scalar(Type::from(FloatKind::F32));
    let v16 = e.scalar(Type::from(FloatKind::F32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::pre_external::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_post_external_base() {
    let mut e = Export::new("mhc_post_external_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::post_external::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn mhc_post_external_full() {
    let mut e = Export::new("mhc_post_external_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::mhc::post_external::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn msa_ragged_metadata_base() {
    let mut e = Export::new("msa_ragged_metadata_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::ragged_metadata::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn msa_ragged_metadata_full() {
    let mut e = Export::new("msa_ragged_metadata_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::ragged_metadata::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn msa_proxy_base() {
    let mut e = Export::new("msa_proxy_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::proxy::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn msa_proxy_full() {
    let mut e = Export::new("msa_proxy_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::proxy::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn msa_expand_base() {
    let mut e = Export::new("msa_expand_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::expand::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn msa_expand_full() {
    let mut e = Export::new("msa_expand_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::expand::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn msa_select_base() {
    let mut e = Export::new("msa_select_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::select::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        4usize,
        v7.into(),
        v8.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn msa_select_full() {
    let mut e = Export::new("msa_select_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::select::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        4usize,
        v7.into(),
        v8.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn msa_live_page_counts_base() {
    let mut e = Export::new("msa_live_page_counts_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::live_page_counts::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        false,
    );
    e.save();
}

#[test]
fn msa_live_page_counts_full() {
    let mut e = Export::new("msa_live_page_counts_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::msa::live_page_counts::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        true,
    );
    e.save();
}

#[test]
fn norm_broadcast_numeric_base() {
    let mut e = Export::new("norm_broadcast_numeric_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::broadcast_numeric::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn norm_broadcast_numeric_full() {
    let mut e = Export::new("norm_broadcast_numeric_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::broadcast_numeric::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn norm_cast_values_base() {
    let mut e = Export::new("norm_cast_values_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::cast_values::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn norm_cast_values_full() {
    let mut e = Export::new("norm_cast_values_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::cast_values::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn norm_broadcast_base() {
    let mut e = Export::new("norm_broadcast_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::broadcast::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn norm_broadcast_full() {
    let mut e = Export::new("norm_broadcast_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::broadcast::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn norm_norm_kernel_base() {
    let mut e = Export::new("norm_norm_kernel_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::norm::norm_kernel::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
        false,
        false,
        false,
        false,
        false,
        4u32,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn norm_norm_kernel_full() {
    let mut e = Export::new("norm_norm_kernel_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::norm::norm_kernel::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
        true,
        true,
        true,
        true,
        true,
        4u32,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn norm_blend_kernel_base() {
    let mut e = Export::new("norm_blend_kernel_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::blend_kernel::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
    );
    e.save();
}

#[test]
fn norm_blend_kernel_full() {
    let mut e = Export::new("norm_blend_kernel_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::blend_kernel::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
    );
    e.save();
}

#[test]
fn norm_dit_stream_base() {
    let mut e = Export::new("norm_dit_stream_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::dit_stream::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        false,
        false,
    );
    e.save();
}

#[test]
fn norm_dit_stream_full() {
    let mut e = Export::new("norm_dit_stream_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::dit_stream::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        true,
        true,
    );
    e.save();
}

#[test]
fn norm_dit_affine_base() {
    let mut e = Export::new("norm_dit_affine_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::dit_affine::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
        false,
        false,
        false,
    );
    e.save();
}

#[test]
fn norm_dit_affine_full() {
    let mut e = Export::new("norm_dit_affine_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::norm::dit_affine::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
        true,
        true,
        true,
    );
    e.save();
}

#[test]
fn nvfp4_attention_means_base() {
    let mut e = Export::new("nvfp4_attention_means_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::means::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_means_full() {
    let mut e = Export::new("nvfp4_attention_means_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::means::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_center_base() {
    let mut e = Export::new("nvfp4_attention_center_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::center::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        0u32,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_center_full() {
    let mut e = Export::new("nvfp4_attention_center_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::center::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        1u32,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_quantize_vectors_base() {
    let mut e = Export::new("nvfp4_attention_quantize_vectors_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), true);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::quantize_vectors::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        0u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_quantize_vectors_full() {
    let mut e = Export::new("nvfp4_attention_quantize_vectors_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), true);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::quantize_vectors::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        1u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_correction_base() {
    let mut e = Export::new("nvfp4_attention_correction_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::correction::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_correction_full() {
    let mut e = Export::new("nvfp4_attention_correction_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::nvfp4_attention::correction::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_attend_base() {
    let mut e = Export::new("nvfp4_attention_attend_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), false);
    let v5 = e.buffer(Type::from(UIntKind::U8), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::nvfp4_attention::attend::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        4usize,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn nvfp4_attention_attend_full() {
    let mut e = Export::new("nvfp4_attention_attend_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), false);
    let v5 = e.buffer(Type::from(UIntKind::U8), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.buffer(Type::from(FloatKind::F32), true);
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::nvfp4_attention::attend::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        4usize,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_lengths_base() {
    let mut e = Export::new("page_lengths_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::lengths::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn page_lengths_full() {
    let mut e = Export::new("page_lengths_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::lengths::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn page_attention_schedule_base() {
    let mut e = Export::new("page_attention_schedule_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(UIntKind::U32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.buffer(Type::from(IntKind::I32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::attention_schedule::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
    );
    e.save();
}

#[test]
fn page_attention_schedule_full() {
    let mut e = Export::new("page_attention_schedule_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(UIntKind::U32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.buffer(Type::from(IntKind::I32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::attention_schedule::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
    );
    e.save();
}

#[test]
fn page_transfer_base() {
    let mut e = Export::new("page_transfer_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::transfer::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_transfer_full() {
    let mut e = Export::new("page_transfer_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::transfer::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_flatten_base() {
    let mut e = Export::new("page_flatten_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::flatten::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_flatten_full() {
    let mut e = Export::new("page_flatten_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::flatten::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_flatten_prepared_base() {
    let mut e = Export::new("page_flatten_prepared_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::flatten_prepared::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_flatten_prepared_full() {
    let mut e = Export::new("page_flatten_prepared_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::flatten_prepared::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_batch_positions_base() {
    let mut e = Export::new("page_batch_positions_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::batch_positions::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn page_batch_positions_full() {
    let mut e = Export::new("page_batch_positions_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::batch_positions::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn page_physical_rows_base() {
    let mut e = Export::new("page_physical_rows_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::physical_rows::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        false,
    );
    e.save();
}

#[test]
fn page_physical_rows_full() {
    let mut e = Export::new("page_physical_rows_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::physical_rows::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        true,
    );
    e.save();
}

#[test]
fn page_dense_pages_base() {
    let mut e = Export::new("page_dense_pages_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::dense_pages::expand::<f32, u32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn page_dense_pages_full() {
    let mut e = Export::new("page_dense_pages_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::page::dense_pages::expand::<f32, u32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn paged_transfer_base() {
    let mut e = Export::new("paged_transfer_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::paged::transfer::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn paged_transfer_full() {
    let mut e = Export::new("paged_transfer_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::paged::transfer::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn paged_attention_base() {
    let mut e = Export::new("paged_attention_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(FloatKind::F32));
    let v10 = e.scalar(Type::from(IntKind::I32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    let v16 = e.scalar(Type::from(UIntKind::U32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::paged::attention::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        4usize,
        4usize,
        32usize,
        false,
        false,
        0u32,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn paged_attention_full() {
    let mut e = Export::new("paged_attention_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(FloatKind::F32));
    let v10 = e.scalar(Type::from(IntKind::I32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    let v16 = e.scalar(Type::from(UIntKind::U32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::paged::attention::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        4usize,
        4usize,
        32usize,
        true,
        true,
        1u32,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn peer_candidates_initialize_base() {
    let mut e = Export::new("peer_candidates_initialize_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::peer_candidates::initialize::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
    );
    e.save();
}

#[test]
fn peer_candidates_initialize_full() {
    let mut e = Export::new("peer_candidates_initialize_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::peer_candidates::initialize::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
    );
    e.save();
}

#[test]
fn peer_candidates_publish_base() {
    let mut e = Export::new("peer_candidates_publish_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(UIntKind::U64), false);
    let v2 = e.buffer(
        Type::from(ruda_core::ir::StorageType::Atomic(
            (Type::from(UIntKind::U32)).elem_type(),
        )),
        true,
    );
    let v3 = e.buffer(
        Type::from(ruda_core::ir::StorageType::Atomic(
            (Type::from(UIntKind::U32)).elem_type(),
        )),
        true,
    );
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U64));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::peer_candidates::publish::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
    );
    e.save();
}

#[test]
fn peer_candidates_publish_full() {
    let mut e = Export::new("peer_candidates_publish_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(UIntKind::U64), false);
    let v2 = e.buffer(
        Type::from(ruda_core::ir::StorageType::Atomic(
            (Type::from(UIntKind::U32)).elem_type(),
        )),
        true,
    );
    let v3 = e.buffer(
        Type::from(ruda_core::ir::StorageType::Atomic(
            (Type::from(UIntKind::U32)).elem_type(),
        )),
        true,
    );
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U64));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::peer_candidates::publish::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
    );
    e.save();
}

#[test]
fn peer_candidates_receive_base() {
    let mut e = Export::new("peer_candidates_receive_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.buffer(
        Type::from(ruda_core::ir::StorageType::Atomic(
            (Type::from(UIntKind::U32)).elem_type(),
        )),
        true,
    );
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U64));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::peer_candidates::receive::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
    );
    e.save();
}

#[test]
fn peer_candidates_receive_full() {
    let mut e = Export::new("peer_candidates_receive_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.buffer(
        Type::from(ruda_core::ir::StorageType::Atomic(
            (Type::from(UIntKind::U32)).elem_type(),
        )),
        true,
    );
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U64));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::peer_candidates::receive::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
    );
    e.save();
}

#[test]
fn peer_moe_reset_base() {
    let mut e = Export::new("peer_moe_reset_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::reset::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn peer_moe_reset_full() {
    let mut e = Export::new("peer_moe_reset_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::reset::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn peer_moe_stage_inputs_base() {
    let mut e = Export::new("peer_moe_stage_inputs_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U64), false);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U64));
    let v12 = e.scalar(Type::from(UIntKind::U64));
    let v13 = e.scalar(Type::from(UIntKind::U64));
    let v14 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::stage_inputs::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
    );
    e.save();
}

#[test]
fn peer_moe_stage_inputs_full() {
    let mut e = Export::new("peer_moe_stage_inputs_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U64), false);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U64));
    let v12 = e.scalar(Type::from(UIntKind::U64));
    let v13 = e.scalar(Type::from(UIntKind::U64));
    let v14 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::stage_inputs::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
    );
    e.save();
}

#[test]
fn peer_moe_first_base() {
    let mut e = Export::new("peer_moe_first_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(FloatKind::F32));
    let v12 = e.scalar(Type::from(UIntKind::U64));
    let v13 = e.scalar(Type::from(UIntKind::U64));
    let v14 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::first::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
    );
    e.save();
}

#[test]
fn peer_moe_first_full() {
    let mut e = Export::new("peer_moe_first_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(FloatKind::F32));
    let v12 = e.scalar(Type::from(UIntKind::U64));
    let v13 = e.scalar(Type::from(UIntKind::U64));
    let v14 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::first::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
    );
    e.save();
}

#[test]
fn peer_moe_second_base() {
    let mut e = Export::new("peer_moe_second_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U64));
    let v13 = e.scalar(Type::from(UIntKind::U64));
    let v14 = e.scalar(Type::from(UIntKind::U64));
    let v15 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::second::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
    );
    e.save();
}

#[test]
fn peer_moe_second_full() {
    let mut e = Export::new("peer_moe_second_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U64));
    let v13 = e.scalar(Type::from(UIntKind::U64));
    let v14 = e.scalar(Type::from(UIntKind::U64));
    let v15 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::second::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
    );
    e.save();
}

#[test]
fn peer_moe_combine_base() {
    let mut e = Export::new("peer_moe_combine_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U64));
    let v9 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::combine::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn peer_moe_combine_full() {
    let mut e = Export::new("peer_moe_combine_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U64));
    let v9 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::combine::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn peer_moe_initialize_base() {
    let mut e = Export::new("peer_moe_initialize_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::initialize::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
    );
    e.save();
}

#[test]
fn peer_moe_initialize_full() {
    let mut e = Export::new("peer_moe_initialize_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::initialize::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
    );
    e.save();
}

#[test]
fn peer_moe_combine_empty_base() {
    let mut e = Export::new("peer_moe_combine_empty_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::combine_empty::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn peer_moe_combine_empty_full() {
    let mut e = Export::new("peer_moe_combine_empty_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U64), false);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U64));
    rudnn::inference::peer_moe::combine_empty::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn pre_attention_positions_base() {
    let mut e = Export::new("pre_attention_positions_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), true);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::positions::expand(&mut e.kernel.body, v0.into(), v1.into());
    e.save();
}

#[test]
fn pre_attention_positions_full() {
    let mut e = Export::new("pre_attention_positions_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), true);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::positions::expand(&mut e.kernel.body, v0.into(), v1.into());
    e.save();
}

#[test]
fn pre_attention_adaln_values_base() {
    let mut e = Export::new("pre_attention_adaln_values_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::adaln_values::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_adaln_values_full() {
    let mut e = Export::new("pre_attention_adaln_values_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::adaln_values::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_split_base() {
    let mut e = Export::new("pre_attention_split_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::split::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_split_full() {
    let mut e = Export::new("pre_attention_split_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::split::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_destination_pack_base() {
    let mut e = Export::new("pre_attention_destination_pack_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::destination_pack::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
    );
    e.save();
}

#[test]
fn pre_attention_destination_pack_full() {
    let mut e = Export::new("pre_attention_destination_pack_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::destination_pack::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
    );
    e.save();
}

#[test]
fn pre_attention_destination_pack_generic_base() {
    let mut e = Export::new("pre_attention_destination_pack_generic_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::destination_pack_generic::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_destination_pack_generic_full() {
    let mut e = Export::new("pre_attention_destination_pack_generic_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::destination_pack_generic::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_destination_fp8_base() {
    let mut e = Export::new("pre_attention_destination_fp8_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::destination_fp8::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn pre_attention_destination_fp8_full() {
    let mut e = Export::new("pre_attention_destination_fp8_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::pre_attention::destination_fp8::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn recurrence_state_slots_base() {
    let mut e = Export::new("recurrence_state_slots_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence::state_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn recurrence_state_slots_full() {
    let mut e = Export::new("recurrence_state_slots_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence::state_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn recurrence_delta_base() {
    let mut e = Export::new("recurrence_delta_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), false);
    let v9 = e.buffer(Type::from(UIntKind::U32), false);
    let v10 = e.buffer(Type::from(UIntKind::U32), false);
    let v11 = e.buffer(Type::from(IntKind::I32), false);
    let v12 = e.buffer(Type::from(IntKind::I32), false);
    let v13 = e.buffer(Type::from(IntKind::I32), false);
    let v14 = e.buffer(Type::from(IntKind::I32), false);
    let v15 = e.buffer(Type::from(IntKind::I32), false);
    let v16 = e.buffer(Type::from(FloatKind::F32), true);
    let v17 = e.buffer(Type::from(FloatKind::F32), true);
    let v18 = e.buffer(Type::from(FloatKind::F32), true);
    let v19 = e.buffer(Type::from(UIntKind::U32), true);
    let v20 = e.buffer(Type::from(FloatKind::F32), true);
    let v21 = e.scalar(Type::from(UIntKind::U32));
    let v22 = e.scalar(Type::from(UIntKind::U32));
    let v23 = e.scalar(Type::from(UIntKind::U32));
    let v24 = e.scalar(Type::from(UIntKind::U32));
    let v25 = e.scalar(Type::from(UIntKind::U32));
    let v26 = e.scalar(Type::from(UIntKind::U32));
    let v27 = e.scalar(Type::from(UIntKind::U32));
    let v28 = e.scalar(Type::from(UIntKind::U32));
    let v29 = e.scalar(Type::from(UIntKind::U32));
    let v30 = e.scalar(Type::from(FloatKind::F32));
    let v31 = e.scalar(Type::from(FloatKind::F32));
    let v32 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::recurrence::delta::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        v23.into(),
        v24.into(),
        v25.into(),
        v26.into(),
        v27.into(),
        v28.into(),
        v29.into(),
        v30.into(),
        v31.into(),
        v32.into(),
        4usize,
        false,
        false,
        false,
        false,
        0u32,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn recurrence_delta_full() {
    let mut e = Export::new("recurrence_delta_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), false);
    let v7 = e.buffer(Type::from(FloatKind::F32), false);
    let v8 = e.buffer(Type::from(FloatKind::F32), false);
    let v9 = e.buffer(Type::from(UIntKind::U32), false);
    let v10 = e.buffer(Type::from(UIntKind::U32), false);
    let v11 = e.buffer(Type::from(IntKind::I32), false);
    let v12 = e.buffer(Type::from(IntKind::I32), false);
    let v13 = e.buffer(Type::from(IntKind::I32), false);
    let v14 = e.buffer(Type::from(IntKind::I32), false);
    let v15 = e.buffer(Type::from(IntKind::I32), false);
    let v16 = e.buffer(Type::from(FloatKind::F32), true);
    let v17 = e.buffer(Type::from(FloatKind::F32), true);
    let v18 = e.buffer(Type::from(FloatKind::F32), true);
    let v19 = e.buffer(Type::from(UIntKind::U32), true);
    let v20 = e.buffer(Type::from(FloatKind::F32), true);
    let v21 = e.scalar(Type::from(UIntKind::U32));
    let v22 = e.scalar(Type::from(UIntKind::U32));
    let v23 = e.scalar(Type::from(UIntKind::U32));
    let v24 = e.scalar(Type::from(UIntKind::U32));
    let v25 = e.scalar(Type::from(UIntKind::U32));
    let v26 = e.scalar(Type::from(UIntKind::U32));
    let v27 = e.scalar(Type::from(UIntKind::U32));
    let v28 = e.scalar(Type::from(UIntKind::U32));
    let v29 = e.scalar(Type::from(UIntKind::U32));
    let v30 = e.scalar(Type::from(FloatKind::F32));
    let v31 = e.scalar(Type::from(FloatKind::F32));
    let v32 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::recurrence::delta::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        v23.into(),
        v24.into(),
        v25.into(),
        v26.into(),
        v27.into(),
        v28.into(),
        v29.into(),
        v30.into(),
        v31.into(),
        v32.into(),
        4usize,
        true,
        true,
        true,
        true,
        1u32,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn recurrence_paged_prefill_slots_base() {
    let mut e = Export::new("recurrence_paged_prefill_slots_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.buffer(Type::from(IntKind::I32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::prefill_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
    );
    e.save();
}

#[test]
fn recurrence_paged_prefill_slots_full() {
    let mut e = Export::new("recurrence_paged_prefill_slots_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.buffer(Type::from(IntKind::I32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::prefill_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
    );
    e.save();
}

#[test]
fn recurrence_paged_decode_slots_base() {
    let mut e = Export::new("recurrence_paged_decode_slots_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::decode_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn recurrence_paged_decode_slots_full() {
    let mut e = Export::new("recurrence_paged_decode_slots_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::decode_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn recurrence_paged_scatter_state_base() {
    let mut e = Export::new("recurrence_paged_scatter_state_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::scatter_state::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn recurrence_paged_scatter_state_full() {
    let mut e = Export::new("recurrence_paged_scatter_state_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::scatter_state::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn recurrence_paged_scatter_checkpoints_base() {
    let mut e = Export::new("recurrence_paged_scatter_checkpoints_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::scatter_checkpoints::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn recurrence_paged_scatter_checkpoints_full() {
    let mut e = Export::new("recurrence_paged_scatter_checkpoints_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::recurrence_paged::scatter_checkpoints::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn replay_materialize_requests_base() {
    let mut e = Export::new("replay_materialize_requests_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::replay::materialize_requests::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
    );
    e.save();
}

#[test]
fn replay_materialize_requests_full() {
    let mut e = Export::new("replay_materialize_requests_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::replay::materialize_requests::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
    );
    e.save();
}

#[test]
fn replay_copy_state_bits_base() {
    let mut e = Export::new("replay_copy_state_bits_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::replay::copy_state_bits::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn replay_copy_state_bits_full() {
    let mut e = Export::new("replay_copy_state_bits_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::replay::copy_state_bits::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn replay_request_metadata_base() {
    let mut e = Export::new("replay_request_metadata_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.buffer(Type::from(IntKind::I32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::replay::request_metadata::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
    );
    e.save();
}

#[test]
fn replay_request_metadata_full() {
    let mut e = Export::new("replay_request_metadata_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.buffer(Type::from(IntKind::I32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::replay::request_metadata::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
    );
    e.save();
}

#[test]
fn replay_reconstruct_base() {
    let mut e = Export::new("replay_reconstruct_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(IntKind::I32), false);
    let v8 = e.buffer(Type::from(IntKind::I32), false);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::replay::reconstruct::expand::<f32, f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        4usize,
        0u32,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn replay_reconstruct_full() {
    let mut e = Export::new("replay_reconstruct_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(IntKind::I32), false);
    let v8 = e.buffer(Type::from(IntKind::I32), false);
    let v9 = e.buffer(Type::from(FloatKind::F32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::replay::reconstruct::expand::<f32, f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        4usize,
        1u32,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn replay_record_base() {
    let mut e = Export::new("replay_record_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(IntKind::I32), false);
    let v8 = e.buffer(Type::from(IntKind::I32), false);
    let v9 = e.buffer(Type::from(IntKind::I32), false);
    let v10 = e.buffer(Type::from(FloatKind::F32), true);
    let v11 = e.buffer(Type::from(FloatKind::F32), true);
    let v12 = e.buffer(Type::from(FloatKind::F32), true);
    let v13 = e.buffer(Type::from(FloatKind::F32), true);
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    let v16 = e.scalar(Type::from(UIntKind::U32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    let v18 = e.scalar(Type::from(UIntKind::U32));
    let v19 = e.scalar(Type::from(UIntKind::U32));
    let v20 = e.scalar(Type::from(UIntKind::U32));
    let v21 = e.scalar(Type::from(FloatKind::F32));
    let v22 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::replay::record::expand::<f32, f32, f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        false,
        false,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn replay_record_full() {
    let mut e = Export::new("replay_record_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), false);
    let v6 = e.buffer(Type::from(IntKind::I32), false);
    let v7 = e.buffer(Type::from(IntKind::I32), false);
    let v8 = e.buffer(Type::from(IntKind::I32), false);
    let v9 = e.buffer(Type::from(IntKind::I32), false);
    let v10 = e.buffer(Type::from(FloatKind::F32), true);
    let v11 = e.buffer(Type::from(FloatKind::F32), true);
    let v12 = e.buffer(Type::from(FloatKind::F32), true);
    let v13 = e.buffer(Type::from(FloatKind::F32), true);
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    let v16 = e.scalar(Type::from(UIntKind::U32));
    let v17 = e.scalar(Type::from(UIntKind::U32));
    let v18 = e.scalar(Type::from(UIntKind::U32));
    let v19 = e.scalar(Type::from(UIntKind::U32));
    let v20 = e.scalar(Type::from(UIntKind::U32));
    let v21 = e.scalar(Type::from(FloatKind::F32));
    let v22 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::replay::record::expand::<f32, f32, f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        v20.into(),
        v21.into(),
        v22.into(),
        true,
        true,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn rope_rotate_base() {
    let mut e = Export::new("rope_rotate_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(FloatKind::F32));
    let v10 = e.scalar(Type::from(FloatKind::F32));
    let v11 = e.scalar(Type::from(FloatKind::F32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::rope::rotate::expand::<f32, u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        4usize,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn rope_rotate_full() {
    let mut e = Export::new("rope_rotate_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    let v9 = e.scalar(Type::from(FloatKind::F32));
    let v10 = e.scalar(Type::from(FloatKind::F32));
    let v11 = e.scalar(Type::from(FloatKind::F32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::rope::rotate::expand::<f32, u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        4usize,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn sage_block_offsets_base() {
    let mut e = Export::new("sage_block_offsets_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::block_offsets::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn sage_block_offsets_full() {
    let mut e = Export::new("sage_block_offsets_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::block_offsets::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn sage_quantize_blocks_base() {
    let mut e = Export::new("sage_quantize_blocks_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::quantize_blocks::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn sage_quantize_blocks_full() {
    let mut e = Export::new("sage_quantize_blocks_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::quantize_blocks::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn sage_decode_blocks_base() {
    let mut e = Export::new("sage_decode_blocks_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::decode_blocks::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn sage_decode_blocks_full() {
    let mut e = Export::new("sage_decode_blocks_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::decode_blocks::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn sage_channel_statistics_base() {
    let mut e = Export::new("sage_channel_statistics_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::channel_statistics::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
    );
    e.save();
}

#[test]
fn sage_channel_statistics_full() {
    let mut e = Export::new("sage_channel_statistics_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::channel_statistics::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
    );
    e.save();
}

#[test]
fn sage_quantize_value_base() {
    let mut e = Export::new("sage_quantize_value_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::quantize_value::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn sage_quantize_value_full() {
    let mut e = Export::new("sage_quantize_value_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sage::quantize_value::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn selective_select_slots_base() {
    let mut e = Export::new("selective_select_slots_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::selective::select_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn selective_select_slots_full() {
    let mut e = Export::new("selective_select_slots_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::selective::select_slots::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn selective_gather_state_base() {
    let mut e = Export::new("selective_gather_state_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::selective::gather_state::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        0u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn selective_gather_state_full() {
    let mut e = Export::new("selective_gather_state_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::selective::gather_state::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        1u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn selective_commit_maps_base() {
    let mut e = Export::new("selective_commit_maps_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::selective::commit_maps::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
        false,
        false,
        false,
    );
    e.save();
}

#[test]
fn selective_commit_maps_full() {
    let mut e = Export::new("selective_commit_maps_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(IntKind::I32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(IntKind::I32));
    rudnn::inference::selective::commit_maps::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
        true,
        true,
        true,
    );
    e.save();
}

#[test]
fn selective_scatter_state_base() {
    let mut e = Export::new("selective_scatter_state_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::selective::scatter_state::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn selective_scatter_state_full() {
    let mut e = Export::new("selective_scatter_state_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::selective::scatter_state::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn sharded_indexer_shard_scores_base() {
    let mut e = Export::new("sharded_indexer_shard_scores_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sharded_indexer::shard_scores::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        false,
    );
    e.save();
}

#[test]
fn sharded_indexer_shard_scores_full() {
    let mut e = Export::new("sharded_indexer_shard_scores_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(UIntKind::U32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(UIntKind::U32));
    let v14 = e.scalar(Type::from(UIntKind::U32));
    let v15 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sharded_indexer::shard_scores::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        true,
    );
    e.save();
}

#[test]
fn sharded_indexer_carrier_base() {
    let mut e = Export::new("sharded_indexer_carrier_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sharded_indexer::carrier::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
    );
    e.save();
}

#[test]
fn sharded_indexer_carrier_full() {
    let mut e = Export::new("sharded_indexer_carrier_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sharded_indexer::carrier::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
    );
    e.save();
}

#[test]
fn sharded_indexer_merge_carriers_base() {
    let mut e = Export::new("sharded_indexer_merge_carriers_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sharded_indexer::merge_carriers::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
    );
    e.save();
}

#[test]
fn sharded_indexer_merge_carriers_full() {
    let mut e = Export::new("sharded_indexer_merge_carriers_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sharded_indexer::merge_carriers::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
    );
    e.save();
}

#[test]
fn sparse_expand_blocks_base() {
    let mut e = Export::new("sparse_expand_blocks_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sparse::expand_blocks::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
    );
    e.save();
}

#[test]
fn sparse_expand_blocks_full() {
    let mut e = Export::new("sparse_expand_blocks_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sparse::expand_blocks::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
    );
    e.save();
}

#[test]
fn sparse_transpose_lse_base() {
    let mut e = Export::new("sparse_transpose_lse_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sparse::transpose_lse::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn sparse_transpose_lse_full() {
    let mut e = Export::new("sparse_transpose_lse_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::sparse::transpose_lse::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn state_storage_copy_rows_base() {
    let mut e = Export::new("state_storage_copy_rows_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(UIntKind::U32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::state_storage::copy_rows::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn state_storage_copy_rows_full() {
    let mut e = Export::new("state_storage_copy_rows_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), false);
    let v5 = e.buffer(Type::from(UIntKind::U32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::state_storage::copy_rows::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn state_storage_decode_base() {
    let mut e = Export::new("state_storage_decode_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::state_storage::decode::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        false,
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn state_storage_decode_full() {
    let mut e = Export::new("state_storage_decode_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::state_storage::decode::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        true,
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn state_storage_encode_rows_base() {
    let mut e = Export::new("state_storage_encode_rows_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U64), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U64));
    let v6 = e.scalar(Type::from(UIntKind::U64));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::state_storage::encode_rows::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        false,
        false,
        0usize,
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn state_storage_encode_rows_full() {
    let mut e = Export::new("state_storage_encode_rows_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U64), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), true);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U64));
    let v6 = e.scalar(Type::from(UIntKind::U64));
    let v7 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::state_storage::encode_rows::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        true,
        true,
        10usize,
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn training_rms_train_base() {
    let mut e = Export::new("training_rms_train_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::training::rms_train::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_rms_train_full() {
    let mut e = Export::new("training_rms_train_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::training::rms_train::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_rms_dx_base() {
    let mut e = Export::new("training_rms_dx_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::rms_dx::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn training_rms_dx_full() {
    let mut e = Export::new("training_rms_dx_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(FloatKind::F32), false);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.buffer(Type::from(FloatKind::F32), true);
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::rms_dx::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn training_rms_dw_base() {
    let mut e = Export::new("training_rms_dw_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::rms_dw::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        false,
    );
    e.save();
}

#[test]
fn training_rms_dw_full() {
    let mut e = Export::new("training_rms_dw_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::rms_dw::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        true,
    );
    e.save();
}

#[test]
fn training_conv_forward_base() {
    let mut e = Export::new("training_conv_forward_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_forward::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_forward_full() {
    let mut e = Export::new("training_conv_forward_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(FloatKind::F32), true);
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_forward::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_activation_backward_base() {
    let mut e = Export::new("training_conv_activation_backward_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_activation_backward::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_activation_backward_full() {
    let mut e = Export::new("training_conv_activation_backward_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_activation_backward::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_dx_base() {
    let mut e = Export::new("training_conv_dx_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_dx::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_dx_full() {
    let mut e = Export::new("training_conv_dx_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_dx::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_dw_base() {
    let mut e = Export::new("training_conv_dw_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_dw::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn training_conv_dw_full() {
    let mut e = Export::new("training_conv_dw_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    rudnn::inference::training::conv_dw::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn video_video_component_base() {
    let mut e = Export::new("video_video_component_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(FloatKind::F32));
    let v14 = e.scalar(Type::from(FloatKind::F32));
    let v15 = e.scalar(Type::from(FloatKind::F32));
    let v16 = e.scalar(Type::from(FloatKind::F32));
    let v17 = e.scalar(Type::from(FloatKind::F32));
    let v18 = e.scalar(Type::from(FloatKind::F32));
    let v19 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::video::video_component::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        0u32,
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn video_video_component_full() {
    let mut e = Export::new("video_video_component_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    let v12 = e.scalar(Type::from(UIntKind::U32));
    let v13 = e.scalar(Type::from(FloatKind::F32));
    let v14 = e.scalar(Type::from(FloatKind::F32));
    let v15 = e.scalar(Type::from(FloatKind::F32));
    let v16 = e.scalar(Type::from(FloatKind::F32));
    let v17 = e.scalar(Type::from(FloatKind::F32));
    let v18 = e.scalar(Type::from(FloatKind::F32));
    let v19 = e.scalar(Type::from(FloatKind::F32));
    rudnn::inference::video::video_component::expand::<f32, f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        v12.into(),
        v13.into(),
        v14.into(),
        v15.into(),
        v16.into(),
        v17.into(),
        v18.into(),
        v19.into(),
        1u32,
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_global_amax_base() {
    let mut e = Export::new("tensor_float_quant_global_amax_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(FloatKind::F32));
    rublas::tensor_float_quant::global_amax::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_global_amax_full() {
    let mut e = Export::new("tensor_float_quant_global_amax_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(FloatKind::F32));
    rublas::tensor_float_quant::global_amax::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_invert_base() {
    let mut e = Export::new("tensor_float_quant_invert_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::invert::expand(&mut e.kernel.body, v0.into(), v1.into(), v2.into());
    e.save();
}

#[test]
fn tensor_float_quant_invert_full() {
    let mut e = Export::new("tensor_float_quant_invert_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::invert::expand(&mut e.kernel.body, v0.into(), v1.into(), v2.into());
    e.save();
}

#[test]
fn tensor_float_quant_quantize_blocks_base() {
    let mut e = Export::new("tensor_float_quant_quantize_blocks_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::quantize_blocks::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        0u32,
        false,
        0u32,
        false,
        false,
        false,
        false,
        448u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_quantize_blocks_full() {
    let mut e = Export::new("tensor_float_quant_quantize_blocks_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::quantize_blocks::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        1u32,
        true,
        1u32,
        true,
        true,
        true,
        true,
        448u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_dequantize_values_base() {
    let mut e = Export::new("tensor_float_quant_dequantize_values_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::dequantize_values::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        0u32,
        false,
        0u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_dequantize_values_full() {
    let mut e = Export::new("tensor_float_quant_dequantize_values_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::dequantize_values::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        1u32,
        true,
        1u32,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_interleave_base() {
    let mut e = Export::new("tensor_float_quant_interleave_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::interleave::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        0u32,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_interleave_full() {
    let mut e = Export::new("tensor_float_quant_interleave_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::interleave::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        1u32,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_row_globals_base() {
    let mut e = Export::new("tensor_float_quant_row_globals_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::row_globals::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_row_globals_full() {
    let mut e = Export::new("tensor_float_quant_row_globals_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::row_globals::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_permute_quantized_rows_base() {
    let mut e = Export::new("tensor_float_quant_permute_quantized_rows_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.buffer(Type::from(UIntKind::U8), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::permute_quantized_rows::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        false,
        0u32,
    );
    e.save();
}

#[test]
fn tensor_float_quant_permute_quantized_rows_full() {
    let mut e = Export::new("tensor_float_quant_permute_quantized_rows_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.buffer(Type::from(UIntKind::U8), true);
    let v6 = e.buffer(Type::from(FloatKind::F32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::permute_quantized_rows::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        true,
        1u32,
    );
    e.save();
}

#[test]
fn tensor_float_quant_append_quantized_base() {
    let mut e = Export::new("tensor_float_quant_append_quantized_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::append_quantized::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        0u32,
    );
    e.save();
}

#[test]
fn tensor_float_quant_append_quantized_full() {
    let mut e = Export::new("tensor_float_quant_append_quantized_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(UIntKind::U8), true);
    let v4 = e.buffer(Type::from(UIntKind::U8), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::append_quantized::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
        1u32,
    );
    e.save();
}

#[test]
fn tensor_float_quant_mega_layout_base() {
    let mut e = Export::new("tensor_float_quant_mega_layout_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::mega_layout::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn tensor_float_quant_mega_layout_full() {
    let mut e = Export::new("tensor_float_quant_mega_layout_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U8), false);
    let v1 = e.buffer(Type::from(UIntKind::U8), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::mega_layout::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn tensor_float_quant_row_shuffle_base() {
    let mut e = Export::new("tensor_float_quant_row_shuffle_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::row_shuffle::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn tensor_float_quant_row_shuffle_full() {
    let mut e = Export::new("tensor_float_quant_row_shuffle_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(FloatKind::F32), true);
    let v2 = e.scalar(Type::from(UIntKind::U32));
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rublas::tensor_float_quant::row_shuffle::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn sampling_accumulate_counts_base() {
    let mut e = Export::new("sampling_accumulate_counts_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), true);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::accumulate_counts::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn sampling_accumulate_counts_full() {
    let mut e = Export::new("sampling_accumulate_counts_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), true);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::accumulate_counts::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn sampling_mask_logits_base() {
    let mut e = Export::new("sampling_mask_logits_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::mask_logits::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn sampling_mask_logits_full() {
    let mut e = Export::new("sampling_mask_logits_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::mask_logits::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn sampling_philox_uniforms_base() {
    let mut e = Export::new("sampling_philox_uniforms_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U64));
    let v3 = e.scalar(Type::from(UIntKind::U64));
    rurand::tensor::sampling::philox_uniforms::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn sampling_philox_uniforms_full() {
    let mut e = Export::new("sampling_philox_uniforms_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.scalar(Type::from(UIntKind::U32));
    let v2 = e.scalar(Type::from(UIntKind::U64));
    let v3 = e.scalar(Type::from(UIntKind::U64));
    rurand::tensor::sampling::philox_uniforms::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
    );
    e.save();
}

#[test]
fn sampling_weights_base() {
    let mut e = Export::new("sampling_weights_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::weights::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn sampling_weights_full() {
    let mut e = Export::new("sampling_weights_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), true);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::weights::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn sampling_filter_base() {
    let mut e = Export::new("sampling_filter_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::filter::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        false,
    );
    e.save();
}

#[test]
fn sampling_filter_full() {
    let mut e = Export::new("sampling_filter_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), true);
    let v1 = e.buffer(Type::from(UIntKind::U32), true);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::filter::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        true,
    );
    e.save();
}

#[test]
fn sampling_draw_base() {
    let mut e = Export::new("sampling_draw_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::draw::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn sampling_draw_full() {
    let mut e = Export::new("sampling_draw_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.buffer(Type::from(UIntKind::U32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::draw::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
    );
    e.save();
}

#[test]
fn sampling_verify_chain_base() {
    let mut e = Export::new("sampling_verify_chain_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(UIntKind::U32), true);
    let v7 = e.buffer(Type::from(UIntKind::U32), true);
    let v8 = e.buffer(Type::from(UIntKind::U32), true);
    let v9 = e.buffer(Type::from(UIntKind::U32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::verify_chain::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
    );
    e.save();
}

#[test]
fn sampling_verify_chain_full() {
    let mut e = Export::new("sampling_verify_chain_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(FloatKind::F32), false);
    let v3 = e.buffer(Type::from(FloatKind::F32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), false);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(UIntKind::U32), true);
    let v7 = e.buffer(Type::from(UIntKind::U32), true);
    let v8 = e.buffer(Type::from(UIntKind::U32), true);
    let v9 = e.buffer(Type::from(UIntKind::U32), true);
    let v10 = e.scalar(Type::from(UIntKind::U32));
    let v11 = e.scalar(Type::from(UIntKind::U32));
    rurand::tensor::sampling::verify_chain::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        v10.into(),
        v11.into(),
    );
    e.save();
}

#[test]
fn topk_varlen_schedule_base() {
    let mut e = Export::new("topk_varlen_schedule_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::varlen_schedule::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        false,
    );
    e.save();
}

#[test]
fn topk_varlen_schedule_full() {
    let mut e = Export::new("topk_varlen_schedule_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::varlen_schedule::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        true,
    );
    e.save();
}

#[test]
fn topk_absolute_indices_base() {
    let mut e = Export::new("topk_absolute_indices_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::absolute_indices::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn topk_absolute_indices_full() {
    let mut e = Export::new("topk_absolute_indices_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), true);
    let v3 = e.scalar(Type::from(UIntKind::U32));
    let v4 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::absolute_indices::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
    );
    e.save();
}

#[test]
fn topk_window_transform_base() {
    let mut e = Export::new("topk_window_transform_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::window_transform::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        false,
    );
    e.save();
}

#[test]
fn topk_window_transform_full() {
    let mut e = Export::new("topk_window_transform_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(IntKind::I32), false);
    let v2 = e.buffer(Type::from(IntKind::I32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), false);
    let v4 = e.buffer(Type::from(IntKind::I32), true);
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    let v9 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::window_transform::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        v9.into(),
        true,
    );
    e.save();
}

#[test]
fn topk_select_base() {
    let mut e = Export::new("topk_select_base".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(UIntKind::U32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::select::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        false,
        false,
        false,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn topk_select_full() {
    let mut e = Export::new("topk_select_full".into());
    let v0 = e.buffer(Type::from(FloatKind::F32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(UIntKind::U32), false);
    let v4 = e.buffer(Type::from(FloatKind::F32), true);
    let v5 = e.buffer(Type::from(IntKind::I32), true);
    let v6 = e.buffer(Type::from(UIntKind::U32), true);
    let v7 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::select::expand::<f32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        true,
        true,
        true,
        Type::from(FloatKind::F32).storage_type(),
    );
    e.save();
}

#[test]
fn topk_transform_indices_base() {
    let mut e = Export::new("topk_transform_indices_base".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::transform_indices::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        false,
        false,
    );
    e.save();
}

#[test]
fn topk_transform_indices_full() {
    let mut e = Export::new("topk_transform_indices_full".into());
    let v0 = e.buffer(Type::from(IntKind::I32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U32), false);
    let v3 = e.buffer(Type::from(IntKind::I32), true);
    let v4 = e.scalar(Type::from(UIntKind::U32));
    let v5 = e.scalar(Type::from(UIntKind::U32));
    let v6 = e.scalar(Type::from(UIntKind::U32));
    let v7 = e.scalar(Type::from(UIntKind::U32));
    let v8 = e.scalar(Type::from(UIntKind::U32));
    ruprim::topk::transform_indices::expand(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        v3.into(),
        v4.into(),
        v5.into(),
        v6.into(),
        v7.into(),
        v8.into(),
        true,
        true,
    );
    e.save();
}

#[test]
fn bitpack_pack_base() {
    let mut e = Export::new("bitpack_pack_base".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    ruprim::bitpack::pack::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        false,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}

#[test]
fn bitpack_pack_full() {
    let mut e = Export::new("bitpack_pack_full".into());
    let v0 = e.buffer(Type::from(UIntKind::U32), false);
    let v1 = e.buffer(Type::from(UIntKind::U32), false);
    let v2 = e.buffer(Type::from(UIntKind::U8), true);
    ruprim::bitpack::pack::expand::<u32>(
        &mut e.kernel.body,
        v0.into(),
        v1.into(),
        v2.into(),
        true,
        Type::from(UIntKind::U32).storage_type(),
    );
    e.save();
}
