use ruda_core::tensor::{data::TensorData, host::HostTensor};
use rufft_host::{irfft_f32, irfft_f64, rfft_f32, rfft_f64};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn measure(mut run: impl FnMut()) -> f64 {
    for _ in 0..4 {
        run();
    }
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let mut count = 0;
        while start.elapsed() < Duration::from_millis(50) {
            run();
            count += 1;
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / count as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}

#[test]
#[ignore = "bounded CPU timing comparison; run with --ignored --nocapture"]
fn transform_timings() {
    for n in [64, 1024, 16384] {
        for batch in [1, 64] {
            let data: Vec<f64> = (0..n * batch).map(|i| (i as f64 * 0.13).sin()).collect();
            let f64_input = HostTensor::from_data(TensorData::new(data.clone(), [batch, n]));
            let f32_input = HostTensor::from_data(TensorData::new(
                data.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                [batch, n],
            ));
            let (re32, im32) = rfft_f32(f32_input.clone(), 1, None);
            let (re64, im64) = rfft_f64(f64_input.clone(), 1, None);
            let timings = [
                (
                    "rfft_f32",
                    measure(|| {
                        black_box(rfft_f32(f32_input.clone(), 1, None));
                    }),
                ),
                (
                    "irfft_f32",
                    measure(|| {
                        black_box(irfft_f32(re32.clone(), im32.clone(), 1, None));
                    }),
                ),
                (
                    "rfft_f64",
                    measure(|| {
                        black_box(rfft_f64(f64_input.clone(), 1, None));
                    }),
                ),
                (
                    "irfft_f64",
                    measure(|| {
                        black_box(irfft_f64(re64.clone(), im64.clone(), 1, None));
                    }),
                ),
            ];
            for (operation, us) in timings {
                println!("FFT_TIMING,{operation},{n},{batch},{us:.3}");
            }
        }
    }
}
