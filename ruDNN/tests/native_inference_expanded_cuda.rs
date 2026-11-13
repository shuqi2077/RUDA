#![allow(unsafe_code)]
//! Whole-library numerical smoke/differential cases, compiled before hardware use.
use linear::{FloatFormat, Matrix, QuantOptions, ScaleLayout};
use ruda_core::tensor::{DType, Shape};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::{
    dsl::Runtime,
    tensor::{RudaTensor, readback::into_data_sync},
};
use rudnn::inference::*;
type T = RudaTensor<CudaRuntime>;
fn upload<V: bytemuck::Pod>(v: &[V], s: impl Into<Shape>, d: DType) -> T {
    let dev = CudaDevice::default();
    let c = CudaRuntime::client(&dev);
    RudaTensor::new_contiguous(
        c.clone(),
        dev,
        s.into(),
        c.create_from_slice(bytemuck::cast_slice(v)),
        d,
    )
}
fn f(v: &[f32], s: impl Into<Shape>) -> T {
    upload(v, s, DType::F32)
}
fn i(v: &[i32], s: impl Into<Shape>) -> T {
    upload(v, s, DType::I32)
}
fn vals(t: &T) -> Vec<f32> {
    into_data_sync(t.clone()).convert::<f32>().to_vec().unwrap()
}
fn close(t: &T, want: &[f32]) {
    let got = vals(t);
    assert_eq!(got.len(), want.len());
    for (j, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 4e-5 * (1.0 + b.abs()), "{j}: {a} != {b}");
    }
}
fn opts() -> attention::Options {
    attention::Options {
        scale: 1.0,
        causal: true,
        window_left: -1,
        window_right: -1,
        soft_cap: 0.0,
        lse_base: cascade::LogBase::Natural,
    }
}
fn ssm<'a>(dt: &'a T, a: &'a T, b: &'a T, c: &'a T) -> mamba::SsmInputs<'a, CudaRuntime> {
    mamba::SsmInputs {
        dt,
        a,
        b,
        c,
        d: None,
        z: None,
        dt_bias: None,
        checkpoint_boundaries: None,
        state: None,
        source_slots: None,
        accepted_tokens: None,
    }
}
#[test]
fn activation_normalization_rotary_and_training() {
    let x = f(&[1.; 8], [2, 4]);
    let w = f(&[1.; 4], [4]);
    let g = activation::silu_and_mul(&x).unwrap();
    close(&g, &[1.0 / (1.0 + (-1.0f32).exp()); 4]);
    let norm = norm::rmsnorm(&x, &w, 1e-5).unwrap();
    close(&norm, &[1.0 / (1.0f32 + 1e-5).sqrt(); 8]);
    let zero = f(&[0.; 4], [4]);
    let layer = norm::layernorm(&x, Some(&w), Some(&zero), 1e-5).unwrap();
    close(&layer, &[0.; 8]);
    let train = training::PreparedRmsTraining::prepare(&x, &w, None, None, 1e-5).unwrap();
    unsafe {
        train.forward().unwrap();
        let grads = train.backward(&x, None).unwrap();
        close(&grads.weight, &[2.0 / (1.0f32 + 1e-5).sqrt(); 4]);
        close(&grads.input, &[1e-5 / (1.0f32 + 1e-5).powf(1.5); 8]);
    }
    let q = f(&[1., 0.], [1, 1, 2]);
    let pos = i(&[1], [1]);
    let out = rope::apply_one(
        &q,
        &pos,
        rope::RopeOptions {
            rotary_dim: 2,
            interleaved: false,
            theta: 10000.,
            scale: 1.,
            llama31: None,
        },
        None,
    )
    .unwrap();
    close(&out, &[1.0f32.cos(), 1.0f32.sin()]);
}
#[test]
fn linear_segmented_low_rank_and_experts() {
    let x = f(&[1., 2., 3., 4.], [2, 2]);
    let eye = f(&[1., 0., 0., 1.], [2, 2]);
    let plan = linear::PreparedLinear::prepare(
        Matrix::Dense(&x),
        Matrix::Dense(&eye),
        None,
        linear::LinearOptions {
            output_dtype: DType::F32,
            ..Default::default()
        },
    )
    .unwrap();
    unsafe {
        close(plan.run().unwrap(), &[1., 2., 3., 4.]);
    }
    let weights = f(&[1., 0., 0., 1.], [2, 1, 2]);
    let cu = i(&[0, 1, 2], [3]);
    let grouped = linear::PreparedSegmented::prepare(
        Matrix::Dense(&x),
        Matrix::Dense(&weights),
        &cu,
        None,
        None,
        DType::F32,
    )
    .unwrap();
    unsafe {
        close(grouped.run().unwrap(), &[1., 4.]);
    }
    let down = f(&[1., 0.], [1, 2]);
    let up = f(&[1., 2.], [2, 1]);
    let adapter =
        linear::LowRankAdapter::prepare(Matrix::Dense(&down), Matrix::Dense(&up), 0.5, 7).unwrap();
    let lora = linear::PreparedLora::prepare(
        Matrix::Dense(&x),
        Matrix::Dense(&eye),
        None,
        &adapter,
        7,
        None,
        DType::F32,
    )
    .unwrap();
    unsafe {
        close(lora.run().unwrap(), &[1.5, 3., 4.5, 7.]);
    }
    let scores = f(&[3., 1., 0., 2.], [2, 2]);
    let router = experts::PreparedRoutes::prepare(
        &scores,
        None,
        experts::RoutingOptions::method(experts::RoutingMethod::TopK, 1),
    )
    .unwrap();
    let routes = unsafe { router.run().unwrap() };
    close(&routes.experts, &[0., 1.]);
    close(&routes.weights, &[3., 2.]);
    let w2 = f(&[2., 3.], [2, 1, 1]);
    let experts = experts::PreparedExperts::prepare(
        &x,
        Matrix::Dense(&weights),
        Matrix::Dense(&w2),
        routes,
        None,
        None,
        None,
        experts::ExpertOptions {
            gated: false,
            activation: activation::Activation::Identity,
            output_dtype: DType::F32,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();
    unsafe {
        close(experts.run().unwrap(), &[6., 24.]);
    }
    let packed = upload(&[0x00003c00u32, 0x00013c00], [2, 1], DType::U32);
    let unpacked = experts::unpack_routes(&packed, false).unwrap();
    close(&unpacked.experts, &[0., 1.]);
    close(&unpacked.weights, &[1., 1.]);
}
#[test]
fn dense_mla_sparse_msa_and_paged_attention() {
    let q = f(&[0.; 6], [3, 1, 2]);
    let k = f(&[0.; 6], [3, 1, 2]);
    let v = f(&[1., 1., 3., 3., 5., 5.], [3, 1, 2]);
    let plan = attention::AttentionPlan::ragged(&q, &[0, 3], &[0, 3]).unwrap();
    let prepared = attention::PreparedAttention::prepare(
        &plan,
        Matrix::Dense(&q),
        Matrix::Dense(&k),
        Matrix::Dense(&v),
        DType::F32,
    )
    .unwrap();
    let out = unsafe { prepared.run(opts(), Default::default()).unwrap() };
    close(&out.values, &[1., 1., 2., 2., 3., 3.]);
    close(&out.lse, &[0., 2.0f32.ln(), 3.0f32.ln()]);
    let q1 = f(&[0., 0.], [1, 1, 2]);
    let one = attention::AttentionPlan::ragged(&q1, &[0, 1], &[0, 3]).unwrap();
    let qp = f(&[0., 0.], [1, 1, 2]);
    let latent = f(&[1., 1., 3., 3., 5., 5.], [3, 2]);
    let pe = f(&[0.; 6], [3, 2]);
    let mla = attention::PreparedMla::prepare(
        &one,
        Matrix::Dense(&q1),
        Matrix::Dense(&qp),
        Matrix::Dense(&latent),
        Matrix::Dense(&pe),
        DType::F32,
    )
    .unwrap();
    unsafe {
        close(
            &mla.run(opts(), Default::default()).unwrap().values,
            &[3., 3.],
        );
    }
    let qsa = sparse::QsaPlan::new(&q1, 1, 3, 2).unwrap();
    let qsap = sparse::PreparedQsa::prepare(
        &qsa,
        Matrix::Dense(&q1),
        Matrix::Dense(&k),
        Matrix::Dense(&v),
        DType::F32,
    )
    .unwrap();
    let ids = i(&[0, 2], [1, 2]);
    unsafe {
        close(&qsap.indexed(&ids, 1.).unwrap().values, &[3., 3.]);
    }
    let qcu = i(&[0, 1], [2]);
    let kvcu = i(&[0, 3], [2]);
    let msa = msa::MsaPlan::new(&qcu, &kvcu, 1, 3, 3, 1, None).unwrap();
    let msap = msa::PreparedMsa::prepare(
        &msa,
        Matrix::Dense(&q1),
        Matrix::Dense(&k),
        Matrix::Dense(&v),
        2,
        true,
        0,
        0,
        DType::F32,
    )
    .unwrap();
    unsafe {
        close(&msap.run(opts(), None).unwrap().values, &[2., 2.]);
    }
    let indptr = i(&[0, 1], [2]);
    let pages = i(&[0], [1]);
    let last = i(&[2], [1]);
    let csr = page::CsrPages::new(&indptr, &pages, &last, 1, 2).unwrap();
    let batch = i(&[0], [1]);
    let position = i(&[1], [1]);
    let cache_k = f(&[0.; 4], [1, 2, 1, 2]);
    let cache_v = f(&[1., 1., 3., 3.], [1, 2, 1, 2]);
    let paged = page::PreparedPagedAttention::prepare(
        &csr,
        &q1,
        &cache_k,
        &cache_v,
        paged::CacheLayout::Nhd,
        &batch,
        &position,
        2,
    )
    .unwrap();
    unsafe {
        close(
            &paged.run(opts(), Default::default()).unwrap().values,
            &[2., 2.],
        );
    }
}
#[test]
fn floating_codecs_scale_layouts_and_paged_append() {
    let pattern = [
        6., -6., 4., -4., 3., -3., 2., -2., 1.5, -1.5, 1., -1., 0.5, -0.5, 0., 0.,
    ];
    let data: Vec<f32> = (0..256).map(|n| pattern[n % 16]).collect();
    let x = f(&data, [4, 64]);
    for layout in [
        ScaleLayout::Linear,
        ScaleLayout::Block128x4,
        ScaleLayout::Block8x4,
        ScaleLayout::PagedV4x4,
    ] {
        for mut recipe in [
            QuantOptions::nvfp4(),
            QuantOptions::mxfp4(),
            QuantOptions::mxfp8(),
        ] {
            recipe.scale_layout = layout;
            let q = rublas::tensor_float_quant::quantize(&x, None, None, recipe).unwrap();
            close(&q.dequantize(DType::F32).unwrap(), &data);
        }
    }
    let input = f(&[6.; 16], [1, 1, 16]);
    let cache_raw = f(&[0.; 32], [1, 1, 2, 16]);
    let cache = rublas::tensor_float_quant::quantize(&cache_raw, None, None, QuantOptions::nvfp4())
        .unwrap();
    let indptr = i(&[0, 1], [2]);
    let ids = i(&[0], [1]);
    let last = i(&[2], [1]);
    let pages = page::CsrPages::new(&indptr, &ids, &last, 1, 2).unwrap();
    let batch = i(&[0], [1]);
    let position = i(&[1], [1]);
    let append = page::PreparedQuantizedAppend::prepare(
        &pages,
        &input,
        &cache,
        &batch,
        &position,
        paged::CacheLayout::Hnd,
    )
    .unwrap();
    unsafe {
        append.run().unwrap();
    }
    let mut expected = vec![0.; 16];
    expected.extend([6.; 16]);
    close(&cache.dequantize(DType::F32).unwrap(), &expected);
    let a = f(&[6.; 16], [1, 16]);
    let aq = rublas::tensor_float_quant::quantize(&a, None, None, QuantOptions::nvfp4()).unwrap();
    let low = f(&[2.], [1, 1]);
    let up = f(&[3.], [1, 1]);
    let alpha = f(&[1.], [1]);
    let out = linear::nvfp4_svdquant(&aq, &aq, &low, &up, &alpha, None, DType::F32).unwrap();
    close(&out, &[582.]);
}
#[test]
fn delta_rules_and_paged_checkpoint_commits() {
    let q = f(&[1., 0., 1., 0.], [2, 1, 2]);
    let k = q.clone();
    let v = f(&[2., 4.], [2, 1, 1]);
    let beta = f(&[0.5; 2], [2, 1, 1]);
    let plan = recurrence::DeltaPlan::new(&q, &[0, 2], 1).unwrap();
    for o in [
        recurrence::DeltaOptions::gdn(1.),
        recurrence::DeltaOptions::kda(1.),
    ] {
        let p = recurrence::PreparedDelta::prepare(
            &plan,
            &q,
            &k,
            &v,
            recurrence::DeltaInputs {
                beta: Some(&beta),
                ..Default::default()
            },
            o,
        )
        .unwrap();
        unsafe {
            let out = p.run().unwrap();
            close(&out.values, &[1., 2.5]);
            close(&out.final_state, &[2.5, 0.]);
        }
    }
    let k4 = f(&[1., 0., 1., 0., 1., 0., 1., 0.], [4, 1, 2]);
    let v4 = f(&[2., 2., 4., 4.], [4, 1, 1]);
    let b4 = f(&[0.5; 4], [4, 1, 1]);
    let gdp = plan
        .run(
            &q,
            &k4,
            &v4,
            recurrence::DeltaInputs {
                beta: Some(&b4),
                ..Default::default()
            },
            recurrence::DeltaOptions::gdp(1., 2),
        )
        .unwrap();
    close(&gdp.values, &[1.5, 3.375]);
    let erase = f(&[0.5; 4], [2, 1, 2]);
    let two = plan
        .run(
            &q,
            &k,
            &v,
            recurrence::DeltaInputs {
                beta: Some(&erase),
                ..Default::default()
            },
            recurrence::DeltaOptions::gdn2(1.),
        )
        .unwrap();
    close(&two.values, &[2., 5.]);
    let cp = recurrence::DeltaPlan::new(&q, &[0, 2], 2).unwrap();
    let indptr = i(&[0, 1], [2]);
    let ids = i(&[0], [1]);
    let last = i(&[2], [1]);
    let pages = page::CsrPages::new(&indptr, &ids, &last, 1, 2).unwrap();
    let pool = f(&[0., 0.], [1, 1, 2, 1]);
    let paged = recurrence_paged::PreparedPagedDelta::prepare(
        &cp,
        &pages,
        &q,
        &k,
        &v,
        &pool,
        recurrence::DeltaInputs {
            beta: Some(&beta),
            ..Default::default()
        },
        recurrence::DeltaOptions::gdn(1.),
        true,
    )
    .unwrap();
    unsafe {
        close(&paged.run().unwrap().values, &[1., 2.5]);
    }
    close(&pool, &[2.5, 0.]);
}
#[test]
fn selective_ssd_quantized_state_and_replay() {
    let x = f(&[1.; 3], [3, 1, 1]);
    let dt = f(&[1.; 3], [3, 1]);
    let a = f(&[0.], [1]);
    let bc = f(&[1.; 3], [3, 1, 1]);
    let cu = i(&[0, 3], [2]);
    let plan = mamba::SsmPlan::packed(&cu, 3).unwrap();
    let out = plan
        .run(&x, ssm(&dt, &a, &bc, &bc), Default::default())
        .unwrap();
    close(&out.values, &[1., 2., 3.]);
    close(&out.final_states, &[3.]);
    let pool = f(&[0.], [1, 1, 1, 1]);
    let history = f(&[0.; 3], [1, 3, 1, 1, 1]);
    let slots = i(&[0], [1]);
    let prepared = selective::PreparedSelectiveUpdate::prepare(
        &plan,
        &x,
        ssm(&dt, &a, &bc, &bc),
        selective::StatePool::Float(&pool),
        None,
        Some(selective::StatePool::Float(&history)),
        Some(&slots),
        true,
        Default::default(),
        None,
        Default::default(),
    )
    .unwrap();
    unsafe {
        close(&prepared.run().unwrap().values, &[1., 2., 3.]);
    }
    close(&history, &[1., 2., 3.]);
    close(&pool, &[0.]);
    for format in [
        state_storage::StateFormat::I16,
        state_storage::StateFormat::I8,
        state_storage::StateFormat::E4m3,
        state_storage::StateFormat::E5m2,
    ] {
        let state = state_storage::encode(&pool, format, Default::default()).unwrap();
        let q = selective::PreparedSelectiveUpdate::prepare(
            &plan,
            &x,
            ssm(&dt, &a, &bc, &bc),
            selective::StatePool::Quantized(&state),
            None,
            None,
            None,
            false,
            Default::default(),
            None,
            Default::default(),
        )
        .unwrap();
        unsafe {
            close(&q.run().unwrap().values, &[1., 2., 3.]);
        }
        close(&state.decode().unwrap(), &[3.]);
    }
    let xd = f(&[1.; 3], [1, 3, 1, 1]);
    let dtd = f(&[1.; 3], [1, 3, 1]);
    let bcd = f(&[1.; 3], [1, 3, 1, 1]);
    let ssd = mamba::ssd_combined(
        &xd,
        &dtd,
        &a,
        &bcd,
        &bcd,
        None,
        None,
        None,
        None,
        Default::default(),
    )
    .unwrap();
    close(&ssd.values, &[1., 2., 3.]);
    let one = mamba::SsmPlan::uniform(&x, 1, 1).unwrap();
    let token = f(&[1.], [1, 1, 1]);
    let dt1 = f(&[1.], [1, 1]);
    let bc1 = f(&[1.], [1, 1, 1]);
    let cache = replay::ReplayCache {
        state: f(&[0.], [1, 1, 1, 1]),
        x: f(&[1.; 4], [1, 1, 4, 1]),
        b: f(&[1.; 4], [1, 1, 4, 1]),
        dt: f(&[1.; 4], [1, 1, 4]),
        ring_start: i(&[0], [1]),
        accepted: i(&[1], [1]),
    };
    let p = replay::PreparedReplay::prepare(
        &one,
        &token,
        ssm(&dt1, &a, &bc1, &bc1),
        &cache,
        &slots,
        1,
        Default::default(),
    )
    .unwrap();
    unsafe {
        close(&p.run().unwrap().values, &[2.]);
    }
    close(&cache.state, &[0.]);
    let starts = i(&[0], [1]);
    let prefix = i(&[2], [1]);
    close(
        &replay::materialize(&cache, &a, &slots, &starts, &prefix).unwrap(),
        &[2.],
    );
    let quantized = replay::QuantizedReplayCache {
        state: state_storage::encode(
            &cache.state,
            state_storage::StateFormat::I8,
            Default::default(),
        )
        .unwrap(),
        x: cache.x.clone(),
        b: cache.b.clone(),
        dt: cache.dt.clone(),
        ring_start: cache.ring_start.clone(),
        accepted: cache.accepted.clone(),
    };
    let materialized = replay::materialize_quantized(
        &quantized,
        &a,
        &slots,
        &starts,
        &prefix,
        Default::default(),
        None,
    )
    .unwrap();
    close(&materialized.decode().unwrap(), &[2.]);
}
#[test]
fn convolution_and_mhc_training() {
    let x = f(&[1., 2., 3.], [3, 1]);
    let w = f(&[1., 1.], [1, 2]);
    let cu = i(&[0, 3], [2]);
    let p = training::PreparedCausalConv::prepare(&x, &w, &cu, None, false, false).unwrap();
    let dy = f(&[1.; 3], [3, 1]);
    unsafe {
        close(&p.forward().unwrap().output, &[1., 3., 5.]);
        let g = p.backward(&dy).unwrap();
        close(&g.input, &[2., 2., 1.]);
        close(&g.weight, &[3., 6.]);
    }
    let residual = f(&[1., 1., 2., 2., 3., 3., 4., 4.], [1, 4, 2]);
    let mapping = f(&[0.; 192], [24, 8]);
    let scale = f(&[1.; 3], [3]);
    let base = f(&[0.; 24], [24]);
    let mhc = mhc::MhcPlan::new(&residual, &mapping, &scale, &base, Default::default()).unwrap();
    unsafe {
        close(&mhc.forward_prepare().collapsed, &[5., 5.]);
        let branch = f(&[4., 4.], [1, 2]);
        close(mhc.forward_finish(&branch).unwrap(), &[6.5; 8]);
    }
    close(&mhc::head(&residual).unwrap(), &[2.5, 2.5]);
    let grad = f(&[1., 1.], [1, 2]);
    close(&mhc::head_backward(&grad, None).unwrap(), &[0.25; 8]);
}
#[test]
fn video_preprojection_sage_masks_and_sampling_workspace() {
    let q = f(&[1.; 4], [2, 1, 2]);
    let cu = i(&[0, 2], [2]);
    let sage =
        sage::PreparedSage::prepare(&q, &q, &q, &cu, &cu, 4, 4, sage::SageFormat::Int8, false)
            .unwrap();
    unsafe {
        sage.run().unwrap();
    }
    close(&sage.query.decode().unwrap(), &[1.; 4]);
    let mask = upload(&[0b101101u8], [1], DType::U8);
    let offsets = upload(&[0u64, 3], [2], DType::U64);
    let lengths = i(&[3, 3], [2]);
    let mp = masks::PreparedMask::prepare(
        &mask,
        &offsets,
        &lengths,
        3,
        masks::MaskStorage::PackedLittle,
        true,
    )
    .unwrap();
    unsafe {
        close(mp.run().unwrap(), &[5., 5.]);
    }
    let logits = f(&[0.; 3], [1, 3]);
    let uniforms = f(&[0.5], [1]);
    let sampler = rurand::tensor::sampling::SamplingPlan::new(
        &logits,
        &[Default::default()],
        None,
        true,
        rurand::tensor::sampling::FilterOrder::TopKFirst,
    )
    .unwrap();
    let workspace = sampler.allocate_workspace(&logits).unwrap();
    unsafe {
        close(
            &sampler
                .sample_into(&logits, &uniforms, &workspace)
                .unwrap()
                .token_ids,
            &[1.],
        );
    }
    let qkv = f(&[0.; 18], [1, 18]);
    let weight = f(&[1.; 6], [6]);
    let video = video::PreparedVideo::prepare(
        &qkv,
        &weight,
        &weight,
        video::VideoOptions {
            frames: 1,
            height: 1,
            width: 1,
            frame_channels: 2,
            height_channels: 2,
            width_channels: 2,
            query_heads: 1,
            key_heads: 1,
            value_heads: 1,
            head_dim: 6,
            epsilon: 1e-5,
            base: 10000.,
            interleaved: false,
            factor: 1.,
            low: 0.,
            high: 1.,
            attention_factor: 1.,
            normalize: true,
            output_fp8: false,
            output_multiplier: 1.,
            value_multiplier: 1.,
        },
    )
    .unwrap();
    unsafe {
        close(&video.run().unwrap().query, &[0.; 6]);
    }
    let input = f(&[1., 1.], [1, 2]);
    let w = f(&[1., 1.], [2]);
    let zero = f(&[0., 0.], [1, 2]);
    let index = i(&[0], [1]);
    let proj = f(&[0.; 12], [6, 2]);
    let table = f(&[1., 0.], [1, 2]);
    let p = pre_attention::PreparedPreAttention::prepare_dense(
        &input,
        pre_attention::PreAttentionParameters {
            input_weight: &w,
            adaln_scale: &zero,
            adaln_shift: &zero,
            adaln_indices: &index,
            projection: Matrix::Dense(&proj),
            query_weight: &w,
            key_weight: &w,
            cos_sin: &table,
            positions: None,
            qk_epsilon: None,
            heads: 1,
            head_dim: 2,
            epsilon: 1e-5,
            order: pre_attention::ProjectionOrder::KindHeadFeature,
            activation_fp8: None,
            activation_recipe: None,
            activation_decoder_scale: None,
        },
        1,
    )
    .unwrap();
    let out = f(&[0.; 6], [1, 3, 2]);
    unsafe {
        p.run_dense_into(&out).unwrap();
    }
    close(&out, &[0.; 6]);
    let q = f(&[1., 1.], [1, 1, 2]);
    let k = f(&[2., 2.], [1, 1, 2]);
    let v = f(&[3., 3.], [1, 1, 2]);
    let scales = f(&[1., 2., 3.], [3]);
    let pack = pre_attention::PreparedTensorFp8Pack::prepare(
        &fused::Qkv {
            query: q,
            key: k,
            value: v,
        },
        1,
        &scales,
        FloatFormat::E4m3,
    )
    .unwrap();
    unsafe {
        close(pack.run().unwrap(), &[56.; 6]);
    }
}
#[test]
fn local_rank_peer_protocol_and_candidate_exchange() {
    use ruda_driver_cuda::peer::PeerBuffer;
    let event_like = upload(&[0u64], [1], DType::U64);
    let event = ruda_driver_cuda::peer::PeerEvent::new(&event_like.client).unwrap();
    let _ticket = event.export().unwrap();
    event.record().unwrap();
    event.wait().unwrap();
    event.synchronize().unwrap();
    let layout = peer_moe::PeerLayout::new(2, 1, 128, 1).unwrap();
    let like = upload(&[0u64], [1], DType::U64);
    let owners: Vec<_> = (0..2)
        .map(|_| PeerBuffer::allocate(&like.client, layout.bytes).unwrap())
        .collect();
    let addresses = upload(
        &owners.iter().map(PeerBuffer::address).collect::<Vec<_>>(),
        [2],
        DType::U64,
    );
    let w1 = f(&vec![0.; 256 * 128], [1, 256, 128]);
    let w2 = f(&vec![0.; 128 * 128], [1, 128, 128]);
    let a = unsafe {
        peer_moe::PeerMoe::prepare(
            &addresses,
            layout,
            0,
            Matrix::Dense(&w1),
            Matrix::Dense(&w2),
            10.,
            DType::F32,
        )
        .unwrap()
    };
    let b = unsafe {
        peer_moe::PeerMoe::prepare(
            &addresses,
            layout,
            1,
            Matrix::Dense(&w1),
            Matrix::Dense(&w2),
            10.,
            DType::F32,
        )
        .unwrap()
    };
    let x = upload(&[0u8; 128], [1, 128], DType::U8);
    let sf = f(&[1.], [1, 1]);
    let id0 = i(&[0], [1, 1]);
    let id1 = i(&[1], [1, 1]);
    let weights = f(&[1.], [1, 1]);
    unsafe {
        a.stage(&x, &sf, &id0, &weights, 1, 1).unwrap();
        b.stage(&x, &sf, &id1, &weights, 1, 1).unwrap();
        a.compute(1, 1).unwrap();
        b.compute(1, 1).unwrap();
        close(&a.combine(1, 1).unwrap(), &[0.; 128]);
        close(&b.combine(1, 1).unwrap(), &[0.; 128]);
    }
    let cl = peer_candidates::CandidateLayout::new(2, 1, 1).unwrap();
    let buffers: Vec<_> = (0..2)
        .map(|_| PeerBuffer::allocate(&like.client, cl.bytes).unwrap())
        .collect();
    let add = upload(
        &buffers.iter().map(PeerBuffer::address).collect::<Vec<_>>(),
        [2],
        DType::U64,
    );
    let c0 = unsafe { peer_candidates::CandidateComm::prepare(&add, cl, 0).unwrap() };
    let c1 = unsafe { peer_candidates::CandidateComm::prepare(&add, cl, 1).unwrap() };
    let mut data = vec![0i32; 66];
    for h in 0..2 {
        for j in 0..16 {
            data[h * 33 + j] = 1.0f32.to_bits() as i32;
            data[h * 33 + 16 + j] = j as i32;
        }
        data[h * 33 + 32] = 1;
    }
    let carrier = i(&data, [2, 1, 33]);
    unsafe {
        c0.publish(&carrier, 0, false, 1_000_000).unwrap();
        c1.publish(&carrier, 0, false, 1_000_000).unwrap();
        let r0 = c0.receive(1, 0, false, 1_000_000).unwrap();
        let values = vals(&r0);
        assert_eq!(values[16], 0.);
        let r1 = c1.receive(1, 0, false, 1_000_000).unwrap();
        assert_eq!(vals(&r1)[31], 15.);
    }
}

#[test]
fn indexers_topk_and_quantized_training_compositions() {
    let q = f(&[1., 1.], [1, 1, 2]);
    let k = f(&[1., 1., 2., 2.], [1, 2, 1, 2]);
    let weights = f(&[1.], [1, 1]);
    let host = rudnn::paged_attention::HostPlan::new(2, 1, &[vec![0]], &[2], &[0], &[1]).unwrap();
    let plan = paged::PagedPlan::upload(host, &q).unwrap();
    let idx = indexer::PreparedMqaIndexer::prepare(
        &plan,
        Matrix::Dense(&q),
        Matrix::Dense(&k),
        &weights,
        paged::CacheLayout::Nhd,
        2,
        1,
    )
    .unwrap();
    unsafe {
        close(&idx.run(1.).unwrap().indices, &[1.]);
    }
    close(&idx.logits, &[2., 4.]);
    let query = f(&[1.; 4], [1, 2, 2]);
    let key = f(&[1.; 128], [1, 64, 2]);
    let qcu = i(&[0, 1], [2]);
    let kvcu = i(&[0, 128], [2]);
    let mp = msa::MsaPlan::new(&qcu, &kvcu, 1, 128, 128, 128, None).unwrap();
    let table = i(&[0], [1, 1]);
    let score = sharded_indexer::score(
        &mp,
        Matrix::Dense(&query),
        Matrix::Dense(&key),
        &table,
        0,
        2,
        128,
        1,
        false,
    )
    .unwrap();
    close(&score, &[2., 2.]);
    let valid = i(&[1], [1]);
    let carrier = sharded_indexer::select_carrier(&score, &valid, 1, 0, 0, 0).unwrap();
    let c = vals(&carrier);
    assert_eq!(c[16], 0.);
    assert_eq!(c[17], -1.);
    let scores = f(&[1., 3., 2., 0., 4., 2., 1., 0.], [2, 4]);
    let lens = i(&[2, 3], [2]);
    let top = ruprim::topk::top_k_varlen(
        &scores,
        &lens,
        None,
        2,
        ruprim::topk::TieBreak::SmallestIndex,
        Default::default(),
    )
    .unwrap();
    close(&top.indices, &[1., 0., 0., 1.]);
    let input = f(&[6.; 16], [1, 16]);
    let w = f(&[1.; 16], [16]);
    let n = fused::PreparedNormQuant::prepare(
        &input,
        Some(&w),
        None,
        None,
        Default::default(),
        QuantOptions::nvfp4(),
        None,
    )
    .unwrap();
    unsafe {
        close(
            &n.run().unwrap().values.dequantize(DType::F32).unwrap(),
            &[1.03125; 16],
        );
    }
    let input = f(&[1.; 8], [2, 4]);
    let grad = f(&[1.; 4], [2, 2]);
    let gate = fused::PreparedGatedQuant::prepare(
        &input,
        Some(&grad),
        activation::Activation::Silu,
        true,
        true,
        QuantOptions::nvfp4(),
        None,
    )
    .unwrap();
    unsafe {
        let o = gate.run().unwrap();
        close(
            &o.rowwise.as_ref().unwrap().dequantize(DType::F32).unwrap(),
            &[0.9375, 0.9375, 0.625, 0.625, 0.9375, 0.9375, 0.625, 0.625],
        );
        close(
            &o.colwise.as_ref().unwrap().dequantize(DType::F32).unwrap(),
            &[0.9375, 0.9375, 0.9375, 0.9375, 0.75, 0.75, 0.75, 0.75],
        );
    }
}
#[test]
fn mhc_parameter_backward_and_convolution_state_history() {
    let residual = f(&[1., 1., 2., 2., 3., 3., 4., 4.], [1, 4, 2]);
    let mapping = f(&[0.; 192], [24, 8]);
    let scale = f(&[1.; 3], [3]);
    let base = f(&[0.; 24], [24]);
    let p = mhc::MhcPlan::new(&residual, &mapping, &scale, &base, Default::default()).unwrap();
    let branch = f(&[4., 4.], [1, 2]);
    let dy = f(&[1.; 8], [1, 4, 2]);
    let du = f(&[0., 0.], [1, 2]);
    unsafe {
        p.forward_prepare();
        p.forward_finish(&branch).unwrap();
        close(
            &p.backward_prepare(&branch, &dy).unwrap().branch_gradient,
            &[4., 4.],
        );
        let g = p.backward_finish(&dy, &du, None).unwrap();
        close(&g.input, &[1.; 8]);
        close(&g.parameters.scale, &[0.; 3]);
        let b = vals(&g.parameters.base);
        for j in 0..4 {
            assert!((b[4 + j] - 4.).abs() < 1e-4);
        }
    }
    let x = f(&[1., 2., 3.], [3, 1]);
    let cu = i(&[0, 3], [2]);
    let plan = mamba::SsmPlan::packed(&cu, 3).unwrap();
    let weight = f(&[1., 1.], [1, 2]);
    let history = f(&[0.], [1, 1, 1]);
    let slots = i(&[0], [1]);
    let out = plan
        .causal_conv1d(
            &x,
            mamba::ConvInputs {
                weight: &weight,
                bias: None,
                state: &history,
                slots: &slots,
                has_initial: None,
                accepted_tokens: None,
                speculative: false,
                silu: false,
            },
        )
        .unwrap();
    close(&out.values, &[1., 3., 5.]);
    close(&out.state, &[3.]);
}

#[test]
fn nvfp4_centered_attention_formats_and_trace_substitution() {
    // Nonconstant K exercises its token permutation and correction. Padding
    // participates in the Q mean; correction restores the omitted mean term.
    let q = f(&vec![1.; 2 * 129 * 64], [1, 2, 129, 64]);
    let mut keys = vec![0.; 4 * 64];
    keys[128..192].fill(0.5);
    let k = f(&keys, [1, 1, 4, 64]);
    let mut values = vec![1.; 4 * 64];
    values[128..192].fill(3.);
    let v = f(&values, [1, 1, 4, 64]);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let q = norm::cast(&q, dtype).unwrap();
        let k = norm::cast(&k, dtype).unwrap();
        let v = norm::cast(&v, dtype).unwrap();
        for per_block in [true, false] {
            let pack = nvfp4_attention::PreparedQuantize::prepare(&q, &k, &v, per_block).unwrap();
            unsafe {
                pack.run().unwrap();
            }
            // Input padding participates in Q's mean but not K's mean. Query mean
            // is block-dependent. K token 2, centered to 3/8, occupies physical token 8.
            let qp = vals(&pack.packed.query);
            assert_eq!(qp.len(), 2 * 256 * 32);
            let output = nvfp4_attention::forward(
                &pack.packed,
                nvfp4_attention::Options {
                    scale: Some(1.0),
                    unpadded_k_len: Some(4),
                    output_dtype: DType::F32,
                    ..Default::default()
                },
            )
            .unwrap();
            let got = vals(&output.values);
            // V=1,3 in one vector yields SF=1/2, so both values are exact.
            assert!((got[0] - 3.0).abs() < 1e-5);
            let lse = vals(output.lse.as_ref().unwrap());
            let expected_first = if per_block { 24.0 } else { 24.46875 };
            let expected_last = if per_block { 24.9375 } else { 24.46875 };
            assert!((lse[0] - expected_first).abs() < 1e-4);
            assert!((lse[128] - expected_last).abs() < 1e-4);
            assert!((lse[256] - expected_first).abs() < 1e-4);
            let no_lse = nvfp4_attention::forward(
                &pack.packed,
                nvfp4_attention::Options {
                    scale: Some(1.0),
                    causal: true,
                    unpadded_k_len: Some(4),
                    output_dtype: DType::F32,
                    return_lse: false,
                },
            )
            .unwrap();
            assert!(no_lse.lse.is_none());
            let tail = vals(&no_lse.values);
            // Bottom-right alignment on padded extents: row 0 has no keys,
            // while row 128 sees only logical K0 (not the permuted physical K).
            assert_eq!(tail[0], 0.0);
            assert!((tail[128 * 64] - 1.0).abs() < 1e-5);
        }
    }
    let mut definition = trace::Definition::new("test_linear", "linear");
    definition.axes.insert(
        "D".into(),
        trace::Axis {
            kind: "const".into(),
            value: Some(64),
        },
    );
    let mut registry = trace::ApplyRegistry::<u32, u32>::default();
    registry.register(&definition, 0, |x| x + 7).unwrap();
    assert_eq!(registry.apply(&definition, &3, |_| {}, |x| x + 1), 10);
    definition.axes.get_mut("D").unwrap().value = Some(128);
    assert_eq!(registry.apply(&definition, &3, |_| {}, |x| x + 1), 4);
}
