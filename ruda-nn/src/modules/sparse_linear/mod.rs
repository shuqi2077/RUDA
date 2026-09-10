mod record;

pub use record::SparseLinearRecord;

use ruda_model::module::{
    AutodiffModule, Content, Devices, DisplaySettings, HasAutodiffModule, Module,
    ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor, Param,
};
use ruda_model::record::CsrTensorRecord;
use ruda_model::tensor::{
    Tensor, TensorPrimitive, backend::AutodiffBackend, ops::SparseOps, read_sync,
    sparse::CsrTensor,
};
use serde::{Serialize, de::DeserializeOwned};

#[derive(Clone, Debug)]
pub struct SparseLinear<B: SparseOps> {
    handle: B::CsrHandle,
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: SparseOps> SparseLinear<B> {
    pub fn new(weight: CsrTensor<B>, bias: Option<Tensor<B, 1>>) -> Result<Self, B::SparseError> {
        let (handle, values) = weight.into_parts();
        Self::validate_bias(&handle, bias.as_ref())?;
        Ok(Self {
            handle,
            weight: Param::from_tensor(values),
            bias: bias.map(Param::from_tensor),
        })
    }

    pub fn shape(&self) -> [usize; 2] {
        B::csr_shape(&self.handle)
    }

    pub fn sparse_weight(&self) -> Result<CsrTensor<B>, B::SparseError> {
        CsrTensor::from_parts(self.handle.clone(), self.weight.val())
    }

    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Result<Tensor<B, D>, B::SparseError> {
        assert!(D > 0, "SparseLinear input must have a feature dimension");
        let mut dims = input.dims();
        let batch: usize = dims[..D - 1].iter().product();
        let rhs = input.reshape([batch, dims[D - 1]]).transpose();
        let output = self.sparse_weight()?.transpose_matmul(rhs)?.transpose();
        dims[D - 1] = self.shape()[1];
        let output = output.reshape(dims);
        match &self.bias {
            Some(bias) => {
                let bias = bias.val();
                Self::validate_bias(&self.handle, Some(&bias))?;
                Ok(output + bias.unsqueeze::<D>())
            }
            None => Ok(output),
        }
    }

    fn validate_bias(handle: &B::CsrHandle, bias: Option<&Tensor<B, 1>>) -> Result<(), B::SparseError> {
        if let Some(bias) = bias {
            let device = bias.device();
            let primitive = match bias.clone().into_primitive() {
                TensorPrimitive::Float(tensor) => tensor,
                TensorPrimitive::QFloat(_) => panic!("SparseLinear requires an unquantized bias"),
            };
            B::csr_validate_operand(handle, &primitive, &device, &[B::csr_shape(handle)[1]])?;
        }
        Ok(())
    }

    pub async fn into_record_async(self) -> Result<SparseLinearRecord<B>, B::SparseError> {
        let weight = self.weight.into_record();
        let handle = B::csr_to_device(&self.handle, &weight.val().device());
        let sparse = CsrTensor::from_parts(handle, weight.val())?;
        Ok(SparseLinearRecord {
            weight: CsrTensorRecord::capture(&sparse).await?,
            weight_id: weight.id.val(),
            bias: self.bias.into_record(),
        })
    }
}

impl<B: SparseOps> Module<B> for SparseLinear<B>
where
    B::CsrData: Serialize + DeserializeOwned,
{
    type Record = SparseLinearRecord<B>;

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        self.bias.collect_devices(self.weight.collect_devices(devices))
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            handle: B::csr_to_device(&self.handle, device),
            weight: self.weight.fork(device),
            bias: self.bias.fork(device),
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            handle: B::csr_to_device(&self.handle, device),
            weight: self.weight.to_device(device),
            bias: self.bias.to_device(device),
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.enter_module("weight", "Struct:SparseLinear");
        self.weight.visit(visitor);
        visitor.exit_module("weight", "Struct:SparseLinear");
        visitor.enter_module("bias", "Struct:SparseLinear");
        self.bias.visit(visitor);
        visitor.exit_module("bias", "Struct:SparseLinear");
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.enter_module("weight", "Struct:SparseLinear");
        let weight = Module::map(self.weight, mapper);
        mapper.exit_module("weight", "Struct:SparseLinear");
        mapper.enter_module("bias", "Struct:SparseLinear");
        let bias = Module::map(self.bias, mapper);
        mapper.exit_module("bias", "Struct:SparseLinear");
        Self {
            handle: B::csr_to_device(&self.handle, &weight.val().device()),
            weight,
            bias,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        let device = self.weight.val().device();
        let sparse = record.weight.restore(&device)
            .unwrap_or_else(|error| panic!("SparseLinear record restore failed: {error}"));
        let (handle, values) = sparse.into_parts();
        let weight = self.weight.load_record(Param::initialized(record.weight_id.into(), values));
        let bias = self.bias.load_record(record.bias);
        let handle = B::csr_to_device(&handle, &weight.val().device());
        CsrTensor::from_parts(handle.clone(), weight.val())
            .unwrap_or_else(|error| panic!("SparseLinear restored values are invalid: {error}"));
        Self::validate_bias(&handle, bias.as_ref().map(Param::val).as_ref())
            .unwrap_or_else(|error| panic!("SparseLinear restored bias is invalid: {error}"));
        Self { handle, weight, bias }
    }

    fn into_record(self) -> Self::Record {
        read_sync(self.into_record_async())
            .unwrap_or_else(|error| panic!("SparseLinear record capture failed: {error}"))
    }
}

impl<B> AutodiffModule<B> for SparseLinear<B>
where
    B: AutodiffBackend + SparseOps,
    B::InnerBackend: SparseOps<CsrHandle = B::CsrHandle, CsrData = B::CsrData>,
    B::CsrData: Serialize + DeserializeOwned,
{
    type InnerModule = SparseLinear<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        SparseLinear {
            handle: self.handle.clone(),
            weight: self.weight.valid(),
            bias: self.bias.valid(),
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Self {
            handle: module.handle,
            weight: AutodiffModule::from_inner(module.weight),
            bias: AutodiffModule::from_inner(module.bias),
        }
    }
}

impl<B, AB> HasAutodiffModule<AB> for SparseLinear<B>
where
    B: SparseOps,
    AB: AutodiffBackend<InnerBackend = B> + SparseOps<CsrHandle = B::CsrHandle, CsrData = B::CsrData>,
    B::CsrData: Serialize + DeserializeOwned,
{
    type TrainModule = SparseLinear<AB>;
}

impl<B: SparseOps> ModuleDisplayDefault for SparseLinear<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.shape();
        content.add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("nnz", &B::csr_nnz(&self.handle))
            .add("bias", &self.bias.is_some())
            .optional()
    }

    fn num_params(&self) -> usize {
        Module::num_params(&self.weight) + Module::num_params(&self.bias)
    }
}

impl<B: SparseOps> ModuleDisplay for SparseLinear<B> {}

impl<B: SparseOps> core::fmt::Display for SparseLinear<B> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(&self.format(DisplaySettings::default()))
    }
}
