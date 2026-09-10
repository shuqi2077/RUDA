#![cfg(feature = "nvidia")]

use ruda_tensor::api::{FloatDType, Int, Tensor, TensorData, backend::Backend};
use ruda_tensor_device::cuda::{Cuda, CudaDevice};
use rullm::CausalModel;

struct SelectOnly;
impl<B: Backend> CausalModel<B> for SelectOnly {
    type Cache = ();
    fn new_cache(&self) {}
    fn forward_cached_last(&self, _: Tensor<B, 2, Int>, _: &mut ()) -> Tensor<B, 3> {
        unreachable!("selection regression does not invoke a model")
    }
}

fn check<B: Backend>(device: &B::Device, dtype: FloatDType) {
    let mut cases = vec![
        vec![3.0], vec![-3.0,-2.0,-2.0], vec![0.0,-0.0], vec![-0.0,0.0],
        vec![f32::NAN,3.0,3.0,f32::NAN], vec![f32::NAN], vec![f32::NAN;65],
        vec![f32::NEG_INFINITY;65], vec![f32::NAN,f32::NEG_INFINITY,f32::NAN,f32::NEG_INFINITY],
        vec![f32::NAN,f32::NEG_INFINITY,f32::MIN],
        vec![f32::NEG_INFINITY,f32::NAN,f32::INFINITY,f32::INFINITY],
    ];
    for length in [31,32,33,255,256,257,1025,248320] {
        let mut row = vec![-2.0;length];
        row[length/2] = 4.0;
        row[length-1] = 4.0;
        row[0] = f32::NAN;
        cases.push(row);
    }
    for row in cases {
        let round = |v: f32| match dtype {
            FloatDType::F16 => half::f16::from_f32(v).to_f32(),
            FloatDType::BF16 => half::bf16::from_f32(v).to_f32(),
            _ => v,
        };
        let mut expected = None;
        for (i,v) in row.iter().copied().map(round).enumerate() {
            if !v.is_nan() && expected.is_none_or(|(_,best)|v>best) { expected=Some((i,v)); }
        }
        let length = row.len();
        let mut values = vec![99.0;length];
        values.extend(row);
        let tensor = Tensor::<B,3>::from_data(TensorData::new(values,[1,2,length]),device).cast(dtype);
        let actual = SelectOnly.greedy_token(tensor);
        match expected {
            Some((i,_)) => assert_eq!(actual.unwrap(),i as i32,"{dtype:?} length={length}"),
            None => assert_eq!(actual.unwrap_err().0,"all vocabulary logits are NaN"),
        }
    }
    println!("PASS greedy selection {dtype:?}: NaN, infinities, ties, tail blocks, full Qwen vocabulary and last sequence row");
}

#[test]
fn greedy_device_preserves_selection() {
    let device=CudaDevice::default();
    for dtype in [FloatDType::F32,FloatDType::F16,FloatDType::BF16] {
        check::<Cuda>(&device,dtype);
    }
    check::<Cuda<f32,i64>>(&device,FloatDType::F32);
    check::<Cuda<f32,i8>>(&device,FloatDType::F32);
    check::<Cuda<f32,u32>>(&device,FloatDType::F32);
}
