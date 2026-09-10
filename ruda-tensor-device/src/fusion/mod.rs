use crate::BoolElement;
use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, RudaTensor};
use ruda_tensor::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use ruda_tensor::{DType, Shape, quantization::QuantScheme};
use ruda_fusion::device::optim::reduce::ReduceSettings;
use ruda_fusion::device::optim::reduce_broadcasted::ReduceBroadcastedFuser;
use ruda_fusion::device::{
    RudaFusionHandle, FallbackOperation,
    optim::{
        RudaOptimization, RudaOptimizationState,
        elemwise::{ElementWiseFuser, ElemwiseOptimization},
        matmul::{MatmulFuser, MatmulOptimization},
        reduce::{ReduceFuser, ReduceOptimization},
        reduce_broadcasted::ReduceBroadcastedOptimization,
    },
};
use ruda_fusion::UnfusedOp;
use ruda_fusion::{
    FusionBackend, FusionRuntime,
    stream::{Operation, OrderedExecution},
};
use ruda_tensor::graph::{BackendIr, TensorHandle};
use ruda_fusion::device::tensor::into_tensor;
use core::marker::PhantomData;
use std::sync::Arc;

impl<R> ruda_fusion::Optimization<DeviceFusionRuntime<R>> for RudaOptimization<R>
where
    R: DeviceRuntime,
{
    fn execute(
        &mut self,
        context: &mut ruda_fusion::stream::Context<
            <DeviceFusionRuntime<R> as FusionRuntime>::FusionHandle,
        >,
        execution: &OrderedExecution<DeviceFusionRuntime<R>>,
    ) {
        match self {
            Self::ElementWise(op) => op.execute(context),
            Self::Matmul(op) => op.execute(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
            Self::Reduce(op) => op.execute(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
            Self::ReduceBroadcasted(op) => op.execute(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
        }
    }

    fn to_state(&self) -> RudaOptimizationState {
        self.to_opt_state()
    }

    fn from_state(device: &R::Device, state: RudaOptimizationState) -> Self {
        match state {
            RudaOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElemwiseOptimization::from_state(device, state))
            }
            RudaOptimizationState::Matmul(state) => {
                Self::Matmul(MatmulOptimization::from_state(device, state))
            }
            RudaOptimizationState::Reduce(state) => {
                Self::Reduce(ReduceOptimization::from_state(device, state))
            }
            RudaOptimizationState::ReduceBroadcasted(state) => {
                Self::ReduceBroadcasted(ReduceBroadcastedOptimization::from_state(device, state))
            }
        }
    }
}

struct FallbackOperationWrapper<O: Clone> {
    operation: O,
}

impl<O: Clone> FallbackOperationWrapper<O> {
    fn new(op: O) -> Self {
        Self { operation: op }
    }
}

impl<R: DeviceRuntime> FallbackOperation<R>
    for FallbackOperationWrapper<Arc<dyn Operation<DeviceFusionRuntime<R>>>>
{
    fn run(&self, context: &mut ruda_fusion::stream::Context<RudaFusionHandle<R>>) {
        self.operation.as_ref().execute(&mut context.handles);
    }
}

impl<R: DeviceRuntime> FallbackOperation<R>
    for FallbackOperationWrapper<UnfusedOp<DeviceFusionRuntime<R>>>
{
    fn run(&self, context: &mut ruda_fusion::stream::Context<RudaFusionHandle<R>>) {
        self.operation.execute(&mut context.handles);
    }
}

impl<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> BackendIr
    for DeviceBackend<R, F, I, BT>
{
    type Handle = RudaFusionHandle<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor.into()
    }
}

impl<R: DeviceRuntime> FusionRuntime for DeviceFusionRuntime<R> {
    type OptimizationState = RudaOptimizationState;
    type Optimization = RudaOptimization<R>;
    type FusionHandle = RudaFusionHandle<R>;
    type FusionDevice = R::RudaDevice;

    fn fusers(device: R::Device) -> Vec<Box<dyn ruda_fusion::OperationFuser<Self::Optimization>>> {
        vec![
            Box::new(ElementWiseFuser::new(device.clone())),
            Box::new(MatmulFuser::new(device.clone())),
            Box::new(ReduceFuser::new(device.clone(), ReduceSettings::Always)),
            Box::new(ReduceBroadcastedFuser::new(device.clone())),
        ]
    }
}

/// Fusion runtime for JIT runtimes.
#[derive(Debug)]
pub struct DeviceFusionRuntime<R: DeviceRuntime> {
    _b: PhantomData<R>,
}

impl<R: DeviceRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusionBackend
    for DeviceBackend<R, F, I, BT>
{
    type FusionRuntime = DeviceFusionRuntime<R>;

    type FullPrecisionBackend = DeviceBackend<R, f32, i32, BT>;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        ruprim::elementwise::cast::cast(tensor, dtype).into()
    }

    fn q_swap_dims_scheme(
        scheme: QuantScheme,
        rank: usize,
        dim1: usize,
        dim2: usize,
    ) -> QuantScheme {
        ruda_kernel::tensor::permutation::swap_dims_scheme(scheme, rank, dim1, dim2)
    }

    fn q_permute_scheme(scheme: QuantScheme, axes: &[usize]) -> QuantScheme {
        ruda_kernel::tensor::permutation::permute_scheme(scheme, axes)
    }
}
