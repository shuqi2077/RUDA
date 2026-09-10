use super::*;

#[test]
#[ignore = "requires RUDA_SILU_REFERENCE with pinned CUDA real gate and exhaustive 16-bit references"]
fn cuda_silu_reference_dtypes_and_layout() {
    type B = Cuda<bf16, i32>;
    let reference = PathBuf::from(std::env::var("RUDA_SILU_REFERENCE").unwrap());
    let device = CudaDevice::default();
    for transposed in [false, true] {
      for (name, dtype) in [("real-f32",DType::F32),("bf16",DType::BF16),("f16",DType::F16)] {
        let values = floats(&reference.join(format!("{name}-input.f32")));
        let expected = floats(&reference.join(format!("{name}-output.f32")));
        let len = values.len();
        let input = Tensor::<B,2>::from_data(TensorData::new(values,[len/16,16]),(&device,dtype));
        let input = if transposed { input.transpose() } else { input };
        let retained = input.clone();
        let output = silu(input);
        assert_eq!(output.dtype(),dtype);
        assert_eq!(output.dims(),if transposed { [16,len/16] } else { [len/16,16] });
        let output = if transposed { output.transpose() } else { output };
        let retained = if transposed { retained.transpose() } else { retained };
        let actual = output.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let original = retained.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let before = floats(&reference.join(format!("{name}-input.f32")));
        assert!(original.iter().zip(&before).all(|(a,b)| a.to_bits()==b.to_bits() || (a.is_nan() && b.is_nan())));
        let mut failures = 0;
        let mut maximum = 0f32;
        for (i, (&a,&b)) in actual.iter().zip(&expected).enumerate() {
            let valid = if b.is_nan() { a.is_nan() }
                else if b.is_infinite() || b == 0.0 || dtype != DType::F32 { a.to_bits() == b.to_bits() }
                else { (a-b).abs() <= b.abs()*1e-6 + 1e-38 };
            if a.is_finite() && b.is_finite() { maximum = maximum.max((a-b).abs()); }
            if !valid {
                failures += 1;
                if failures <= 8 { eprintln!("SiLU {name} index={i} x={} actual={a} expected={b}",before[i]); }
            }
        }
        eprintln!("SiLU {name} transposed={transposed}: count={len}, failures={failures}, max_abs={maximum}");
        assert_eq!(failures,0,"SiLU {name} reference mismatch");
      }
    }
}

#[test]
fn silu_low_precision_rounding_and_empty_input() {
    type B = Cuda<bf16, i32>;
    let device = CudaDevice::default();
    for dtype in [DType::F32,DType::F16,DType::BF16] {
        let empty = Tensor::<B,2>::empty([0,16],(&device,dtype));
        let output = silu(empty);
        assert_eq!(output.dims(),[0,16]);
        assert_eq!(output.dtype(),dtype);
        let values = vec![0.00555419921875f32,0.005584716796875,1.0,-1.0,0.0,-0.0,f32::INFINITY,f32::NEG_INFINITY];
        let input = Tensor::<B,1>::from_data(TensorData::new(values.clone(),[8]),(&device,dtype));
        let actual = silu(input).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let cuda_bits = [0x3b368163,0x3b3782d1,0x3f3b26a7,0xbe89b2b1,0,0x80000000,0x7f800000,0x7fffffff];
        for ((x,y), bits) in values.into_iter().zip(actual).zip(cuda_bits) {
            let expected = f32::from_bits(bits);
            let expected = match dtype { DType::F16 => half::f16::from_f32(expected).to_f32(),
                DType::BF16 => bf16::from_f32(expected).to_f32(), _ => expected };
            assert!(if expected.is_nan() { y.is_nan() } else { y.to_bits()==expected.to_bits() },
                "{dtype:?} SiLU({x}): actual={y} ({:08x}), CUDA expected={expected} ({:08x})", y.to_bits(), expected.to_bits());
        }
    }
}
