use std::path::{Path, PathBuf};

use ruccl::{CollectiveConfig, ReduceOperation, finish_collective, register};
use ruda_autodiff::Autodiff;
use ruda_model::{
    module::{Module, Param},
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{Device, Tensor, backend::Backend},
};
use ruda_nn::{
    Linear,
    loss::{MseLoss, Reduction},
};
use ruda_optim::{GradientsParams, Optimizer, SgdConfig, momentum::MomentumConfig};

#[cfg(feature = "cuda")]
type Inner = ruda_tensor_device::cuda::Cuda<f32>;
#[cfg(not(feature = "cuda"))]
type Inner = ruda_tensor_host::Host;
type Training = Autodiff<Inner>;

const INPUTS: [[f32; 2]; 2] = [[1.0, 2.0], [-1.0, 1.0]];
const TARGETS: [f32; 2] = [1.0, 0.0];
const INITIAL: [f32; 3] = [0.5, -0.25, 0.25];
const AFTER_ONE: [f32; 3] = [0.53125, 0.0, 0.40625];
const AFTER_TWO: [f32; 3] = [0.5390625, 0.15625, 0.5078125];

fn model(device: &Device<Inner>) -> Linear<Training> {
    Linear {
        weight: Param::initialized(
            1_u64.into(),
            Tensor::from_data([[INITIAL[0]], [INITIAL[1]]], device).require_grad(),
        ),
        bias: Some(Param::initialized(
            2_u64.into(),
            Tensor::from_data([INITIAL[2]], device).require_grad(),
        )),
    }
}

fn optimizer() -> impl Optimizer<Linear<Training>, Training> {
    SgdConfig::new()
        .with_momentum(Some(
            MomentumConfig::new().with_momentum(0.5).with_dampening(0.0),
        ))
        .init()
}

fn assert_model(layer: &Linear<Training>, expected: [f32; 3]) {
    assert_eq!(
        layer.weight.val().to_data().to_vec::<f32>().unwrap(),
        expected[..2]
    );
    assert_eq!(
        layer
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        expected[2..]
    );
    assert_eq!(layer.weight.id.val(), 1);
    assert_eq!(layer.bias.as_ref().unwrap().id.val(), 2);
}

fn gradient(parameters: [f32; 3], rank: usize) -> ([f32; 3], f32) {
    let [x, y] = INPUTS[rank];
    let residual = parameters[0] * x + parameters[1] * y + parameters[2] - TARGETS[rank];
    (
        [2.0 * residual * x, 2.0 * residual * y, 2.0 * residual],
        residual * residual,
    )
}

fn step(
    rank: usize,
    layer: Linear<Training>,
    optimizer: &mut impl Optimizer<Linear<Training>, Training>,
    expected_parameters: [f32; 3],
) -> Linear<Training> {
    let device = layer.weight.val().device();
    let loss = MseLoss::new().forward(
        layer.forward(Tensor::<Training, 2>::from_data([INPUTS[rank]], &device)),
        Tensor::<Training, 2>::from_data([[TARGETS[rank]]], &device),
        Reduction::Mean,
    );
    let (expected, expected_loss) = gradient(expected_parameters, rank);
    assert_eq!(loss.to_data().to_vec::<f32>().unwrap(), [expected_loss]);
    let mut extracted = GradientsParams::from_grads(loss.backward(), &layer);
    assert_eq!(extracted.len(), 2);
    let weight = extracted.remove::<Inner, 2>(layer.weight.id).unwrap();
    let bias_id = layer.bias.as_ref().unwrap().id;
    let bias = extracted.remove::<Inner, 1>(bias_id).unwrap();
    assert_eq!(weight.to_data().to_vec::<f32>().unwrap(), expected[..2]);
    assert_eq!(bias.to_data().to_vec::<f32>().unwrap(), expected[2..]);
    let mut gradients = GradientsParams::new();
    // Registration order differs; synchronization must still follow parameter IDs.
    if rank == 0 {
        gradients.register(layer.weight.id, weight);
        gradients.register(bias_id, bias);
    } else {
        gradients.register(bias_id, bias);
        gradients.register(layer.weight.id, weight);
    }
    let gradients = gradients
        .all_reduce::<Inner>(rank.into(), ReduceOperation::Mean)
        .unwrap();
    let a = gradient(expected_parameters, 0).0;
    let b = gradient(expected_parameters, 1).0;
    assert_eq!(
        gradients
            .get::<Inner, 2>(layer.weight.id)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
    );
    assert_eq!(
        gradients
            .get::<Inner, 1>(bias_id)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        [(a[2] + b[2]) * 0.5]
    );
    optimizer.step(0.125, layer, gradients)
}

fn checkpoint_path(directory: &Path, rank: usize, kind: &str) -> PathBuf {
    directory.join(format!("step1-rank{rank}-{kind}"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .expect("usage: collective_training <run|resume> <checkpoint-directory>");
    let directory = PathBuf::from(args.next().expect("checkpoint directory required"));
    assert!(args.next().is_none(), "unexpected extra argument");
    assert!(mode == "run" || mode == "resume", "expected run or resume");
    let resume = mode == "resume";
    if !resume {
        // A fresh directory preserves all previous checkpoints.
        std::fs::create_dir(&directory).expect("run requires a new checkpoint directory");
    }
    let device = Default::default();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    // Load every rank before registration so a missing checkpoint cannot strand a peer.
    let ranks: Vec<_> = (0..2)
        .map(|rank| {
            let mut layer = model(&device);
            let mut optim = optimizer();
            if resume {
                layer = layer.load_record(
                    recorder
                        .load(checkpoint_path(&directory, rank, "model"), &device)
                        .unwrap(),
                );
                optim = optim.load_record(
                    recorder
                        .load(checkpoint_path(&directory, rank, "optimizer"), &device)
                        .unwrap(),
                );
                assert_model(&layer, AFTER_ONE);
            }
            (rank, layer, optim)
        })
        .collect();
    let workers: Vec<_> = ranks
        .into_iter()
        .map(|(rank, mut layer, mut optim)| {
            let directory = directory.clone();
            std::thread::spawn(move || {
                let device = layer.weight.val().device();
                register::<Inner>(
                    rank.into(),
                    device,
                    CollectiveConfig::default().with_num_devices(2),
                )
                .unwrap();
                if !resume {
                    layer = step(rank, layer, &mut optim, INITIAL);
                    assert_model(&layer, AFTER_ONE);
                    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
                    let model_path = checkpoint_path(&directory, rank, "model");
                    let optim_path = checkpoint_path(&directory, rank, "optimizer");
                    recorder
                        .record(layer.clone().into_record(), model_path.clone())
                        .unwrap();
                    recorder
                        .record(optim.to_record(), optim_path.clone())
                        .unwrap();
                    for path in [model_path, optim_path] {
                        std::fs::File::open(path.with_extension("bin"))
                            .unwrap()
                            .sync_all()
                            .unwrap();
                    }
                }
                layer = step(rank, layer, &mut optim, AFTER_ONE);
                assert_model(&layer, AFTER_TWO);
                finish_collective::<Inner>(rank.into()).unwrap();
                println!("rank {rank}: step 2 parameters verified {AFTER_TWO:?}");
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    println!(
        "{mode}: backward, gradient mean, SGD momentum and step 2 verified on {}",
        Inner::name(&device)
    );
}
