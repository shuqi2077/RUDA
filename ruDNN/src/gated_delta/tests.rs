use super::*;
use half::{bf16, f16};
use ruda_core::tensor::data::TensorData;
use ruda_kernel::tensor::{readback::into_data_sync, transfer::from_data};
use ruda_test_runtime::TestRuntime;

type Tensor = RudaTensor<TestRuntime>;

fn round(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => f16::from_f32(x).to_f32(),
        DType::BF16 => bf16::from_f32(x).to_f32(),
        _ => x,
    }
}

fn tensor(values: Vec<f32>, shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();
    if shape.num_elements() == 0 {
        let device = Default::default();
        return empty_device_contiguous_dtype(TestRuntime::client(&device), device, shape, dtype);
    }
    let data = match dtype {
        DType::F16 => TensorData::new(
            values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
            shape,
        ),
        DType::BF16 => TensorData::new(
            values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
            shape,
        ),
        _ => TensorData::new(values, shape),
    };
    from_data(data, &Default::default())
}

fn floats(tensor: Tensor) -> Vec<f32> {
    let dtype = tensor.dtype;
    let data = into_data_sync(tensor);
    match dtype {
        DType::F16 => data
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_f32)
            .collect(),
        DType::BF16 => data
            .to_vec::<bf16>()
            .unwrap()
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
        _ => data.to_vec::<f32>().unwrap(),
    }
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            (a - b).abs() <= tolerance * b.abs().max(1.0),
            "index {i}: {a} != {b}"
        );
    }
}

fn input(
    dtype: DType,
    start: usize,
    end: usize,
    initial_state: Tensor,
) -> GatedDeltaInput<TestRuntime> {
    let values = |width, shift| {
        (0..4)
            .flat_map(|h| {
                (start..end).flat_map(move |t| {
                    (0..width)
                        .map(move |d| (((h * 17 + t * 7 + d * 3 + shift) % 19) as f32 - 9.0) / 20.0)
                })
            })
            .collect::<Vec<_>>()
    };
    GatedDeltaInput {
        query: tensor(values(3, 0), [2, 2, end - start, 3], dtype),
        key: tensor(values(3, 1), [2, 2, end - start, 3], dtype),
        value: tensor(values(5, 2), [2, 2, end - start, 5], dtype),
        beta: tensor(vec![0.7; 4 * (end - start)], [2, 2, end - start], dtype),
        log_decay: tensor(
            vec![-0.3; 4 * (end - start)],
            [2, 2, end - start],
            DType::F32,
        ),
        initial_state,
        query_scale: 0.5,
    }
}

#[test]
fn recurrence_matches_reference_and_chunked_continuation() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let initial = vec![0.125; 60];
        let state = tensor(initial.clone(), [2, 2, 3, 5], DType::F32);
        let whole_input = input(dtype, 0, 3, state.clone());
        let q = floats(whole_input.query.clone());
        let k = floats(whole_input.key.clone());
        let v = floats(whole_input.value.clone());
        let mut expected_state = initial.clone();
        let mut expected_output = vec![0.0; 60];
        for head in 0..4 {
            for token in 0..3 {
                for column in 0..5 {
                    let mut prediction = 0.0;
                    for key_column in 0..3 {
                        let index = head * 15 + key_column * 5 + column;
                        expected_state[index] *= (-0.3f32).exp();
                        prediction +=
                            expected_state[index] * k[(head * 3 + token) * 3 + key_column];
                    }
                    let delta =
                        (v[(head * 3 + token) * 5 + column] - prediction) * round(0.7, dtype);
                    let mut output = 0.0;
                    for key_column in 0..3 {
                        let index = head * 15 + key_column * 5 + column;
                        expected_state[index] += k[(head * 3 + token) * 3 + key_column] * delta;
                        output +=
                            expected_state[index] * q[(head * 3 + token) * 3 + key_column] * 0.5;
                    }
                    expected_output[(head * 3 + token) * 5 + column] = round(output, dtype);
                }
            }
        }
        let whole = gated_delta_rule(whole_input).unwrap();
        let output = floats(whole.output);
        let final_state = floats(whole.final_state);
        close(&output, &expected_output, 1e-5);
        close(&final_state, &expected_state, 1e-6);
        assert_eq!(floats(state.clone()), initial);
        let prefill = gated_delta_rule(input(dtype, 0, 2, state)).unwrap();
        let decode = gated_delta_rule(input(dtype, 2, 3, prefill.final_state)).unwrap();
        let prefill_output = floats(prefill.output);
        let decode_output = floats(decode.output);
        for head in 0..4 {
            close(
                &prefill_output[head * 10..head * 10 + 10],
                &output[head * 15..head * 15 + 10],
                1e-6,
            );
            close(
                &decode_output[head * 5..head * 5 + 5],
                &output[head * 15 + 10..head * 15 + 15],
                1e-6,
            );
        }
        close(&floats(decode.final_state), &final_state, 1e-6);
    }
}

#[test]
fn empty_sequence_preserves_state_and_invalid_shapes_are_rejected() {
    let state = tensor(vec![0.125; 60], [2, 2, 3, 5], DType::F32);
    let empty = gated_delta_rule(input(DType::F32, 0, 0, state.clone())).unwrap();
    assert_eq!(&empty.output.meta.shape()[..], &[2, 2, 0, 5]);
    assert_eq!(floats(empty.final_state), vec![0.125; 60]);
    let mut invalid = input(DType::F32, 0, 1, state);
    invalid.beta = tensor(vec![0.7; 8], [2, 2, 2], DType::F32);
    assert!(gated_delta_rule(invalid).is_err());
}

#[test]
fn chunk_prefill_matches_recurrence_and_preserves_initial_state() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (length, chunk) in [(1, 1), (3, 4), (7, 3), (67, 33), (64, 64), (65, 64), (129, 64)] {
            let initial = tensor(vec![0.125; 60], [2, 2, 3, 5], DType::F32);
            let expected = gated_delta_rule(input(dtype, 0, length, initial.clone())).unwrap();
            let actual = chunk_gated_delta_rule(input(dtype, 0, length, initial.clone()), chunk).unwrap();
            close(&floats(actual.output), &floats(expected.output), match dtype {
                DType::BF16 => 0.001, DType::F16 => 0.0002, _ => 0.00001,
            });
            close(&floats(actual.final_state), &floats(expected.final_state), 0.00001);
            assert_eq!(floats(initial), vec![0.125; 60]);
        }
    }
}

#[test]
#[ignore = "requires RUDA_QWEN35_PREFIX_REFERENCE exported from the pinned real decay trace"]
fn real_decay_prefix_matches_cuda_reference() {
    use ruda_kernel::dsl::prelude::{RudaCount, RudaDim};
    let path = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_PREFIX_REFERENCE").unwrap());
    let read = |name| std::fs::read(path.join(name)).unwrap().chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>();
    let decay = read("decay.f32");
    let expected = read("cumulative.f32");
    let (heads, sequence, chunk) = (16, 148, 64);
    assert_eq!(decay.len(), heads * sequence);
    assert_eq!(expected.len(), heads * sequence.div_ceil(chunk) * chunk);
    let source = into_contiguous(tensor(decay, [heads, sequence], DType::F32));
    let tile = super::chunk::prefix_tile(heads, sequence, chunk);
    assert_eq!(tile, 32);
    for start in (0..sequence).step_by(chunk) {
        let output = empty_device_contiguous_dtype(source.client.clone(), source.device.clone(),
            [heads, chunk].into(), DType::F32);
        super::chunk_kernel::prefix::launch::<TestRuntime>(&source.client,
            RudaCount::Static(heads as u32, 1, 1), RudaDim::new_1d((tile / 2) as u32),
            source.clone().into_array_arg(), output.clone().into_array_arg(),
            sequence as u32, start as u32, chunk, tile, include_str!("chunk_kernel.rs").to_owned());
        let actual = floats(output);
        for head in 0..heads {
            let offset = head * sequence.div_ceil(chunk) * chunk + start;
            assert_eq!(&actual[head * chunk..(head + 1) * chunk], &expected[offset..offset + chunk]);
        }
    }
}

#[test]
#[ignore = "diagnostic only; requires RUDA_QWEN35_CHUNK_REFERENCE and RUDA_QWEN35_CHUNK_TRACE"]
fn real_chunk_intermediate_diagnosis() {
    let source = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_CHUNK_REFERENCE").unwrap());
    let trace = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_CHUNK_TRACE").unwrap());
    let read = |path: std::path::PathBuf| std::fs::read(path).unwrap().chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    let component = |name: &str, shape: &[usize], dtype| tensor(read(source.join(format!("{name}.f32"))), shape, dtype);
    let input = GatedDeltaInput {
        query: component("query", &[1,16,148,128], DType::BF16),
        key: component("key", &[1,16,148,128], DType::BF16),
        value: component("value", &[1,16,148,128], DType::BF16),
        beta: component("beta", &[1,16,148], DType::BF16),
        log_decay: component("decay", &[1,16,148], DType::F32),
        initial_state: tensor(vec![0.0;16*128*128], [1,16,128,128], DType::F32),
        query_scale: 128f32.sqrt().recip(),
    };
    let mut count = 0;
    let result = super::chunk::chunk_impl(input,64,Some(&mut |index,name,value| {
        let actual = floats(value);
        let expected = read(trace.join(format!("{index}-{name}.f32")));
        assert_eq!(actual.len(),expected.len());
        assert!(actual.iter().all(|x| x.is_finite()));
        let different = actual.iter().zip(&expected).filter(|(a,b)| a != b).count();
        let max = actual.iter().zip(&expected).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
        let err = actual.iter().zip(&expected).map(|(a,b)| (*a as f64-*b as f64).powi(2)).sum::<f64>();
        let norm = expected.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        eprintln!("chunk {index} {name}: different={different}/{} max_abs={max} relative_l2={}",actual.len(),(err/norm).sqrt());
        count += 1;
    })).unwrap();
    assert_eq!(count,36);
    close(&floats(result.output),&read(source.join("output.f32")),0.003);
    close(&floats(result.final_state),&read(source.join("state.f32")),0.0001);
}

#[test]
#[ignore = "requires RUDA_QWEN35_CHUNK_TRACE with original triangular inputs and inverses"]
fn real_triangular_inverse_matches_reference() {
    use ruda_kernel::dsl::prelude::{RudaCount, RudaDim};
    let path = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_CHUNK_TRACE").unwrap());
    let read = |name| std::fs::read(path.join(name)).unwrap().chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    let mut total_different = 0;
    for index in 0..3 {
        let input = into_contiguous(tensor(read(format!("{index}-triangular.f32")),[16,64,64],DType::F32));
        let output = empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),[16,64,64].into(),DType::F32);
        super::chunk_kernel::triangular_inverse::launch::<TestRuntime>(&input.client,
            RudaCount::Static(16,1,1),RudaDim::new_1d(32),input.clone().into_array_arg(),
            output.clone().into_array_arg(),64,include_str!("chunk_kernel.rs").to_owned());
        let actual = floats(output);
        let expected = read(format!("{index}-inverse.f32"));
        assert_eq!(actual.len(),expected.len());
        let different = actual.iter().zip(&expected).filter(|(a,b)| a != b).count();
        eprintln!("fixed triangular {index}: different={different}/{}",actual.len());
        total_different += different;
    }
    assert_eq!(total_different,0);
}

#[test]
#[ignore = "requires RUDA_QWEN35_CHUNK_TRACE and RUDA_QWEN35_CHUNK_PRODUCTS; matmul diagnosis and exact mask gate"]
fn real_products_and_decay_mask_isolation() {
    use ruda_kernel::dsl::prelude::RudaDim;
    use ruda_kernel::tensor::permutation::swap_dims;
    let trace = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_CHUNK_TRACE").unwrap());
    let products = std::path::PathBuf::from(std::env::var("RUDA_QWEN35_CHUNK_PRODUCTS").unwrap());
    let read = |path: std::path::PathBuf| std::fs::read(path).unwrap().chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    let load = |path, shape: &[usize]| into_contiguous(tensor(read(path),shape,DType::F32));
    for index in 0..3 {
        let key = load(trace.join(format!("{index}-key.f32")),&[16,64,128]);
        let cumulative = load(trace.join(format!("{index}-cumulative.f32")),&[16,64]);
        for (lhs, product_name, expected_name, strict) in [
            ("key-beta","key-products","triangular",true),
            ("query","query-products","attention",false),
        ] {
            let input = load(trace.join(format!("{index}-{lhs}.f32")),&[16,64,128]);
            let actual = super::chunk::product(input,swap_dims(key.clone(),1,2)).unwrap();
            let expected_products = read(products.join(format!("{index}-{product_name}.f32")));
            let actual = floats(actual);
            assert_eq!(actual.len(),expected_products.len());
            assert!(actual.iter().all(|v| v.is_finite()));
            let different = actual.iter().zip(&expected_products).filter(|(a,b)| a != b).count();
            let max = actual.iter().zip(&expected_products).map(|(a,b)| (a-b).abs()).fold(0f32,f32::max);
            eprintln!("raw matmul {index} {product_name}: different={different}/{} max_abs={max}",actual.len());
            assert_eq!(actual,expected_products);
            let fixed = into_contiguous(tensor(expected_products,[16,64,64],DType::F32));
            let output = empty_device_contiguous_dtype(key.client.clone(),key.device.clone(),[16,64,64].into(),DType::F32);
            let dim = RudaDim::new(key.client.properties(),16*64*64);
            let count = calculate_ruda_count_elemwise(&key.client,16*64*64,dim);
            super::chunk_kernel::decay_mask::launch::<TestRuntime>(&key.client,count,dim,
                fixed.into_array_arg(),cumulative.clone().into_array_arg(),output.clone().into_array_arg(),
                64,strict,include_str!("chunk_kernel.rs").to_owned());
            let actual = floats(output);
            let expected = read(trace.join(format!("{index}-{expected_name}.f32")));
            let different = actual.iter().zip(&expected).filter(|(a,b)| a != b).count();
            eprintln!("fixed mask {index} {expected_name}: different={different}/{}",actual.len());
            assert_eq!(actual,expected);
        }
    }
}

#[test]
fn matmul_fuses_float_accumulation_and_preserves_integers() {
    use ruda_kernel::tensor::permutation::swap_dims;
    use rublas::tensor_matmul::{matmul, MatmulStrategy};
    let mut lhs = vec![0f32;128];
    let mut rhs = vec![0f32;128];
    lhs[0] = -1.0;
    rhs[0] = 1.0;
    lhs[64] = 1.0 + 2f32.powi(-23);
    rhs[64] = 1.0 - 2f32.powi(-23);
    let actual = super::chunk::product(tensor(lhs,[1,1,128],DType::F32),
        swap_dims(tensor(rhs,[1,1,128],DType::F32),1,2)).unwrap();
    assert_eq!(floats(actual),vec![-2f32.powi(-46)]);
    let lhs = tensor(vec![1e20,-1e20,0.0,0.0,1.0,0.0,0.0,0.0],[1,1,8],DType::F32);
    let rhs = swap_dims(tensor(vec![1.0;8],[1,1,8],DType::F32),1,2);
    assert_eq!(floats(super::chunk::product(lhs,rhs).unwrap()),vec![1.0]);
    let integer = |values, shape: [usize;3]| from_data::<TestRuntime>(TensorData::new(values,shape),&Default::default());
    let lhs = integer(vec![1i32,2,3],[1,1,3]);
    let rhs = swap_dims(integer(vec![4i32,5,6],[1,1,3]),1,2);
    let output = matmul(lhs,rhs,None,MatmulStrategy::Naive,DType::I32).unwrap();
    assert_eq!(into_data_sync(output).to_vec::<i32>().unwrap(),vec![32]);
}

#[test]
fn chunk_prefill_then_recurrent_decode_and_validation() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let initial = tensor(vec![0.125; 60], [2, 2, 3, 5], DType::F32);
        let whole = chunk_gated_delta_rule(input(dtype, 0, 65, initial.clone()), 64).unwrap();
        let prefill = chunk_gated_delta_rule(input(dtype, 0, 64, initial.clone()), 64).unwrap();
        let decode = gated_delta_rule(input(dtype, 64, 65, prefill.final_state)).unwrap();
        let whole_output = floats(whole.output);
        let decode_output = floats(decode.output);
        for head in 0..4 {
            close(&whole_output[(head * 65 + 64) * 5..(head + 1) * 65 * 5],
                &decode_output[head * 5..(head + 1) * 5], 0.001);
        }
        close(&floats(whole.final_state), &floats(decode.final_state), 0.00001);
        let empty = chunk_gated_delta_rule(input(dtype, 0, 0, initial.clone()), 64).unwrap();
        assert_eq!(&empty.output.meta.shape()[..], &[2, 2, 0, 5]);
        assert_eq!(floats(empty.final_state), vec![0.125; 60]);
        assert!(chunk_gated_delta_rule(input(dtype, 0, 1, initial.clone()), 0).is_err());
        assert!(chunk_gated_delta_rule(input(dtype, 0, 1, initial.clone()), usize::MAX).is_err());
        let mut invalid = input(dtype, 0, 1, initial);
        invalid.log_decay = tensor(vec![-0.3; 4], [2, 2, 1], DType::F16);
        assert!(chunk_gated_delta_rule(invalid, 64).is_err());
    }
}

#[test]
fn chunk_prefill_accepts_strided_inputs_and_chunked_continuation() {
    use ruda_kernel::tensor::permutation::swap_dims;
    let strided = |value: Tensor, a, b| swap_dims(into_contiguous(swap_dims(value, a, b)), a, b);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let initial = tensor(vec![0.125; 60], [2, 2, 3, 5], DType::F32);
        let whole = chunk_gated_delta_rule(input(dtype, 0, 129, initial.clone()), 64).unwrap();
        let mut views = input(dtype, 0, 129, initial.clone());
        views.query = strided(views.query, 2, 3);
        views.key = strided(views.key, 2, 3);
        views.value = strided(views.value, 2, 3);
        views.beta = strided(views.beta, 1, 2);
        views.log_decay = strided(views.log_decay, 1, 2);
        views.initial_state = strided(views.initial_state, 2, 3);
        assert_ne!(views.query.meta.strides()[3], 1);
        let actual = chunk_gated_delta_rule(views, 64).unwrap();
        let output = floats(whole.output);
        let state = floats(whole.final_state);
        assert_eq!(floats(actual.output), output);
        assert_eq!(floats(actual.final_state), state);
        let first = chunk_gated_delta_rule(input(dtype, 0, 64, initial), 64).unwrap();
        let next = chunk_gated_delta_rule(input(dtype, 64, 129, first.final_state), 64).unwrap();
        let next_output = floats(next.output);
        for head in 0..4 {
            close(&next_output[head * 65 * 5..(head + 1) * 65 * 5],
                &output[(head * 129 + 64) * 5..(head + 1) * 129 * 5], 0.00001);
        }
        close(&floats(next.final_state), &state, 0.00001);
    }
}
