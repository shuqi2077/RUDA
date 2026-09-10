use ruda_tensor::{
    DType, ExecutionError, FloatDType, Shape, Slice, TensorData,
    graph::{BaseOperationIr, CastOpIr, DequantizeOpIr, FlipOpIr, FloatOperationIr,
        GatherOpIr, InitOperationIr, OperationIr, OperationOutput, PermuteOpIr,
        QuantizationParametersIr, QuantizeOpIr, SelectOpIr, ShapeOpIr, SliceOpIr, SwapDimsOpIr},
    ops::QTensorOps,
    quantization::{QuantScheme, QuantizationParametersPrimitive},
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};

use crate::{BackendRouter, RunnerChannel, RunnerClient, get_client};

impl<R: RunnerChannel> QTensorOps<Self> for BackendRouter<R> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        assert!(matches!(data.dtype, DType::QFloat(_)), "expected quantized tensor data");
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        client.register_op(OperationIr::Init(InitOperationIr { out: out.to_ir_out() }));
        out
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        assert_eq!(client.device(), qparams.scales.client.device(), "quantization parameters must be on the input device");
        let qparams = QuantizationParametersIr { scales: qparams.scales.into_ir() };
        let desc = QuantizeOpIr::create(tensor.into_ir(), qparams, *scheme, || client.create_empty_handle());
        client.register(OperationIr::Float(desc.tensor.dtype, FloatOperationIr::Quantize(desc))).output()
    }

    fn quantize_dynamic(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
    ) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), DType::QFloat(*scheme), || client.create_empty_handle());
        client.register(OperationIr::Float(desc.input.dtype, FloatOperationIr::QuantizeDynamic(desc))).output()
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DequantizeOpIr::create(tensor.into_ir(), dtype.into(), || client.create_empty_handle());
        client.register(OperationIr::Float(desc.out.dtype, FloatOperationIr::Dequantize(desc))).output()
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.client.device()
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        device: &Device<Self>,
    ) -> QuantizedTensor<Self> {
        if &tensor.client.device() == device { return tensor; }
        R::change_client_backend(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        if tensor.shape == shape { return tensor; }
        let client = tensor.client.clone();
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Reshape(desc))).output()
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.into_data().await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::SwapDims(desc))).output()
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let desc = PermuteOpIr::create(tensor.into_ir(), axes.into(), || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Permute(desc))).output()
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let desc = FlipOpIr::create(tensor.into_ir(), axes.into(), || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Flip(desc))).output()
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        assert_eq!(client.device(), indices.client.device(), "gather indices must be on the input device");
        let desc = GatherOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Gather(desc))).output()
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        assert_eq!(client.device(), indices.client.device(), "select indices must be on the input device");
        let desc = SelectOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Select(desc))).output()
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let desc = SliceOpIr::create(tensor.into_ir(), slices.into(), || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Slice(desc))).output()
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());
        client.register(OperationIr::BaseFloat(BaseOperationIr::Expand(desc))).output()
    }
}
