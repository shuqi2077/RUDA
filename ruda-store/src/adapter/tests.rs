    use super::*;
    use alloc::rc::Rc;
    use alloc::sync::Arc;
    use ruda_tensor::api::{DType, Shape, TensorData};
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_module_names_match_ruda_nn() {
        // If these types are renamed or moved in `ruda-nn`, this test will fail to compile.
        #[allow(unused_imports)]
        use ruda_nn::{
            BatchNorm, Embedding, GroupNorm, InstanceNorm, LayerNorm, Linear, PRelu, RmsNorm,
            conv::{
                Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
                DeformConv2d,
            },
        };

        assert_eq!(module_names::LINEAR, "Struct:Linear");
        assert_eq!(module_names::BATCH_NORM, "Struct:BatchNorm");
        assert_eq!(module_names::LAYER_NORM, "Struct:LayerNorm");
        assert_eq!(module_names::GROUP_NORM, "Struct:GroupNorm");
        assert_eq!(module_names::EMBEDDING, "Struct:Embedding");
        assert_eq!(module_names::CONV1D, "Struct:Conv1d");
        assert_eq!(module_names::CONV2D, "Struct:Conv2d");
        assert_eq!(module_names::CONV3D, "Struct:Conv3d");
        assert_eq!(module_names::CONV_TRANSPOSE1D, "Struct:ConvTranspose1d");
        assert_eq!(module_names::CONV_TRANSPOSE2D, "Struct:ConvTranspose2d");
        assert_eq!(module_names::CONV_TRANSPOSE3D, "Struct:ConvTranspose3d");
        assert_eq!(module_names::DEFORM_CONV2D, "Struct:DeformConv2d");
        assert_eq!(module_names::INSTANCE_NORM, "Struct:InstanceNorm");
        assert_eq!(module_names::RMS_NORM, "Struct:RmsNorm");
        assert_eq!(module_names::PRELU, "Struct:PRelu");
    }

    fn create_test_snapshot(path: &str, shape: Shape, container_type: &str) -> TensorSnapshot {
        let path_parts: Vec<String> = path.split('.').map(|s| s.to_string()).collect();
        let values = vec![1.0f32; shape.iter().product()];
        let data = TensorData::new(values, shape.clone());

        TensorSnapshot::from_closure(
            Rc::new(move || Ok(data.clone())),
            DType::F32,
            shape,
            path_parts,
            vec![container_type.to_string()],
            ruda_model::module::ParamId::new(),
        )
    }

    #[test]
    fn test_pytorch_to_ruda_linear_weight() {
        let adapter = PyTorchToRudaAdapter;

        // Linear layer weight should be transposed
        let snapshot = create_test_snapshot("fc.weight", shape![10, 5], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![5, 10]);

        // Linear layer bias should not be transposed
        let snapshot = create_test_snapshot("fc.bias", shape![10], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![10]);
    }

    #[test]
    fn test_pytorch_to_ruda_norm_params() {
        let adapter = PyTorchToRudaAdapter;

        // BatchNorm weight -> gamma
        let snapshot = create_test_snapshot("norm.weight", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.gamma");

        // BatchNorm bias -> beta
        let snapshot = create_test_snapshot("norm.bias", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.beta");
    }

    #[test]
    fn test_ruda_to_pytorch_linear_weight() {
        let adapter = RudaToPyTorchAdapter;

        // Linear layer weight should be transposed
        let snapshot = create_test_snapshot("fc.weight", shape![5, 10], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![10, 5]);
    }

    #[test]
    fn test_ruda_to_pytorch_norm_params() {
        let adapter = RudaToPyTorchAdapter;

        // BatchNorm gamma -> weight
        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.weight");

        // BatchNorm beta -> bias
        let snapshot = create_test_snapshot("norm.beta", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.full_path(), "norm.bias");
    }

    #[test]
    fn test_transpose_different_dtypes() {
        // Test that transpose works for different data types

        // Test with F32
        let f32_data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = transpose_tensor_data(f32_data);
        assert_eq!(transposed.shape, shape![3, 2]);
        let values = transposed.to_vec::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Test with I32
        let i32_data = TensorData::new(vec![1i32, 2, 3, 4, 5, 6], [2, 3]);
        let transposed = transpose_tensor_data(i32_data);
        assert_eq!(transposed.shape, shape![3, 2]);
        let values = transposed.to_vec::<i32>().unwrap();
        assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);

        // Test with F64
        let f64_data = TensorData::new(vec![1.0f64, 2.0, 3.0, 4.0], [2, 2]);
        let transposed = transpose_tensor_data(f64_data);
        assert_eq!(transposed.shape, shape![2, 2]);
        let values = transposed.to_vec::<f64>().unwrap();
        assert_eq!(values, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_no_container_info() {
        let adapter = PyTorchToRudaAdapter;

        // Without container info, adapter returns unchanged for non-norm parameters
        let mut snapshot = create_test_snapshot("fc.weight", shape![10, 5], module_names::LINEAR);
        snapshot.container_stack = None;

        // Without container info, no transformation occurs for linear layers
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.shape, shape![10, 5]); // No transposition without container info

        // Test a non-linear, non-norm parameter - should pass through unchanged
        let mut snapshot2 = create_test_snapshot("other.weight", shape![10, 5], "Struct:Other");
        snapshot2.container_stack = None;
        let adapted2 = adapter.adapt(&snapshot2);
        assert_eq!(adapted2.shape, shape![10, 5]); // No transposition
    }

    #[derive(Clone)]
    struct RenameParamAdapter {
        from: &'static str,
        to: &'static str,
        called: Arc<AtomicUsize>,
    }

    impl ModuleAdapter for RenameParamAdapter {
        fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
            self.called.fetch_add(1, Ordering::Relaxed);

            let path_stack = match snapshot.path_stack.as_ref() {
                Some(stack) => stack,
                None => return snapshot.clone(),
            };
            let param = match path_stack.last() {
                Some(p) => p.as_str(),
                None => return snapshot.clone(),
            };
            if param != self.from {
                return snapshot.clone();
            }

            let mut new_path = path_stack.to_vec();
            *new_path.last_mut().unwrap() = self.to.to_string();

            TensorSnapshot::from_closure(
                snapshot.clone_data_fn(),
                snapshot.dtype,
                snapshot.shape.clone(),
                new_path,
                snapshot.container_stack.clone().unwrap_or_default(),
                snapshot.tensor_id.unwrap_or_default(),
            )
        }

        fn get_alternative_param_name(
            &self,
            _param_name: &str,
            _container_type: &str,
        ) -> Option<String> {
            None
        }

        fn clone_box(&self) -> Box<dyn ModuleAdapter> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct AltNameAdapter {
        from: &'static str,
        to: &'static str,
        called: Arc<AtomicUsize>,
    }

    impl ModuleAdapter for AltNameAdapter {
        fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
            TensorSnapshot::from_closure(
                snapshot.clone_data_fn(),
                snapshot.dtype,
                snapshot.shape.clone(),
                snapshot.path_stack.clone().unwrap_or_default(),
                snapshot.container_stack.clone().unwrap_or_default(),
                snapshot.tensor_id.unwrap_or_default(),
            )
        }

        fn get_alternative_param_name(
            &self,
            param_name: &str,
            _container_type: &str,
        ) -> Option<String> {
            self.called.fetch_add(1, Ordering::Relaxed);
            if param_name == self.from {
                Some(self.to.to_string())
            } else {
                None
            }
        }

        fn clone_box(&self) -> Box<dyn ModuleAdapter> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_chain_adapter_pipes_adapt() {
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));

        let a = RenameParamAdapter {
            from: "weight",
            to: "a",
            called: called1.clone(),
        };
        let b = RenameParamAdapter {
            from: "a",
            to: "b",
            called: called2.clone(),
        };

        let chain = a.chain(b);
        let snapshot = create_test_snapshot("fc.weight", shape![2, 2], module_names::LINEAR);
        let adapted = chain.adapt(&snapshot);

        assert_eq!(adapted.full_path(), "fc.b");
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_chain_adapter_alternative_name_pipes_and_fallbacks() {
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));

        let a = AltNameAdapter {
            from: "gamma",
            to: "weight",
            called: called1.clone(),
        };
        let b = AltNameAdapter {
            from: "weight",
            to: "scale",
            called: called2.clone(),
        };

        let chain = a.chain(b);
        let alt = chain.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("scale"));
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);

        // If the second adapter doesn't have a mapping for the first alternative,
        // fall back to the first alternative name.
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));
        let a = AltNameAdapter {
            from: "gamma",
            to: "weight",
            called: called1.clone(),
        };
        let b = AltNameAdapter {
            from: "something-else",
            to: "unused",
            called: called2.clone(),
        };
        let chain = a.chain(b);
        let alt = chain.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("weight"));
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);

        // If the first adapter doesn't provide an alternative, try the second with the original name.
        let called1 = Arc::new(AtomicUsize::new(0));
        let called2 = Arc::new(AtomicUsize::new(0));
        let a = AltNameAdapter {
            from: "something-else",
            to: "unused",
            called: called1.clone(),
        };
        let b = AltNameAdapter {
            from: "gamma",
            to: "weight",
            called: called2.clone(),
        };
        let chain = a.chain(b);
        let alt = chain.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("weight"));
        assert_eq!(called1.load(Ordering::Relaxed), 1);
        assert_eq!(called2.load(Ordering::Relaxed), 1);

        // clone_box must preserve behavior.
        let boxed = chain.clone_box();
        let alt = boxed.get_alternative_param_name("gamma", module_names::LAYER_NORM);
        assert_eq!(alt.as_deref(), Some("weight"));
    }

    #[test]
    fn test_half_precision_f32_to_f16() {
        let adapter = HalfPrecisionAdapter::new();
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);

        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F16);
        assert_eq!(adapted.shape, shape![2, 3]);

        let data = adapted.to_data().unwrap();
        assert_eq!(data.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_f16_to_f32() {
        let adapter = HalfPrecisionAdapter::new();

        // Create an F16 snapshot
        let values = vec![1.0f32; 6];
        let data = TensorData::new(values, shape![2, 3]).convert_dtype(DType::F16);
        let path_parts = vec!["fc".to_string(), "weight".to_string()];
        let snapshot = TensorSnapshot::from_closure(
            Rc::new(move || Ok(data.clone())),
            DType::F16,
            shape![2, 3],
            path_parts,
            vec![module_names::LINEAR.to_string()],
            ruda_model::module::ParamId::new(),
        );

        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_skips_batch_norm() {
        let adapter = HalfPrecisionAdapter::new();

        // BatchNorm is excluded by default
        let snapshot = create_test_snapshot("norm.weight", shape![10], module_names::BATCH_NORM);
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F32); // unchanged
    }

    #[test]
    fn test_half_precision_converts_default_modules() {
        let adapter = HalfPrecisionAdapter::new();

        // Linear
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // Embedding
        let snapshot = create_test_snapshot("emb.weight", shape![100, 64], module_names::EMBEDDING);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // Conv2d
        let snapshot =
            create_test_snapshot("conv.weight", shape![3, 3, 3, 3], module_names::CONV2D);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // LayerNorm (included by default)
        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // GroupNorm
        let snapshot = create_test_snapshot("gn.gamma", shape![10], module_names::GROUP_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);

        // RmsNorm
        let snapshot = create_test_snapshot("rms.weight", shape![10], module_names::RMS_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_without_module() {
        let adapter = HalfPrecisionAdapter::new().without_module("LayerNorm");

        // LayerNorm removed from conversion set
        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F32);

        // Linear still converted
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_with_module() {
        let adapter = HalfPrecisionAdapter::new().with_module("CustomLayer");

        // Custom module should now be converted
        let snapshot = create_test_snapshot("custom.weight", shape![5], "Struct:CustomLayer");
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_with_qualified_name() {
        let adapter = HalfPrecisionAdapter::new().with_module("Struct:CustomLayer");

        let snapshot = create_test_snapshot("custom.weight", shape![5], "Struct:CustomLayer");
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_chain() {
        let adapter = PyTorchToRudaAdapter.chain(HalfPrecisionAdapter::new());

        let snapshot = create_test_snapshot("fc.weight", shape![10, 5], module_names::LINEAR);
        let adapted = adapter.adapt(&snapshot);

        // Should be both transposed and cast
        assert_eq!(adapted.shape, shape![5, 10]);
        assert_eq!(adapted.dtype, DType::F16);
    }

    #[test]
    fn test_half_precision_skips_no_container() {
        let adapter = HalfPrecisionAdapter::new();
        let mut snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        snapshot.container_stack = None;

        // No module type info: skip
        let adapted = adapter.adapt(&snapshot);
        assert_eq!(adapted.dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_skips_non_float() {
        use ruda_tensor::api::quantization::QuantScheme;

        let adapter = HalfPrecisionAdapter::new();

        // QFloat source: skip
        let qfloat_dtype = DType::QFloat(QuantScheme::default());
        let snapshot = create_test_snapshot("fc.weight", shape![2, 3], module_names::LINEAR);
        let qfloat_snapshot = TensorSnapshot::from_closure(
            snapshot.clone_data_fn(),
            qfloat_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        );
        let adapted = adapter.adapt(&qfloat_snapshot);
        assert_eq!(adapted.dtype, qfloat_dtype);
    }

    #[test]
    fn test_half_precision_default_module_count() {
        let adapter = HalfPrecisionAdapter::new();
        // 14 modules: Linear, Embedding, Conv1d-3d, ConvTranspose1d-3d,
        // DeformConv2d, LayerNorm, GroupNorm, InstanceNorm, RmsNorm, PRelu
        assert_eq!(adapter.modules.len(), 14);
    }

    #[test]
    fn test_half_precision_without_module_qualified() {
        let adapter = HalfPrecisionAdapter::new().without_module("Struct:LayerNorm");

        let snapshot = create_test_snapshot("norm.gamma", shape![10], module_names::LAYER_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F32);
    }

    #[test]
    fn test_half_precision_with_module_batch_norm_opt_in() {
        let adapter = HalfPrecisionAdapter::new().with_module("BatchNorm");

        let snapshot = create_test_snapshot("bn.weight", shape![10], module_names::BATCH_NORM);
        assert_eq!(adapter.adapt(&snapshot).dtype, DType::F16);
    }
