mod kernel;

use ruda_core::device::Device;
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim};
use ruda_kernel::tensor::{
    RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous,
};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int4Error(pub &'static str);

impl Display for Int4Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for Int4Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwqGemmLayout {
    pub input_features: usize,
    pub output_features: usize,
    pub group_size: usize,
}

impl AwqGemmLayout {
    pub fn new(
        input_features: usize,
        output_features: usize,
        group_size: usize,
    ) -> Result<Self, Int4Error> {
        if input_features == 0
            || output_features == 0
            || group_size == 0
            || input_features % group_size != 0
            || output_features % 8 != 0
        {
            return Err(Int4Error(
                "AWQ GEMM requires nonzero dimensions, complete input groups and output channels divisible by eight",
            ));
        }
        if input_features
            .checked_mul(output_features)
            .is_none_or(|size| size > u32::MAX as usize)
        {
            return Err(Int4Error(
                "AWQ GEMM matrix exceeds the kernel's u32 indexing range",
            ));
        }
        Ok(Self {
            input_features,
            output_features,
            group_size,
        })
    }

    pub fn groups(self) -> usize {
        self.input_features / self.group_size
    }
}

#[derive(Debug, Clone)]
pub struct AwqGemm<R: Runtime> {
    qweight: RudaTensor<R>,
    qzeros: RudaTensor<R>,
    scales: RudaTensor<R>,
    bias: Option<RudaTensor<R>>,
    layout: AwqGemmLayout,
}

impl<R: Runtime> AwqGemm<R> {
    pub fn new(
        qweight: RudaTensor<R>,
        qzeros: RudaTensor<R>,
        scales: RudaTensor<R>,
        bias: Option<RudaTensor<R>>,
        group_size: usize,
    ) -> Result<Self, Int4Error> {
        if qweight.meta.num_dims() != 2 || scales.meta.num_dims() != 2 {
            return Err(Int4Error("AWQ GEMM weights and scales must be matrices"));
        }
        let layout =
            AwqGemmLayout::new(qweight.meta.shape()[0], scales.meta.shape()[1], group_size)?;
        let k = layout.input_features;
        let n = layout.output_features;
        if qweight.dtype != DType::I32
            || qzeros.dtype != DType::I32
            || qweight.meta.shape() != &Shape::from([k, n / 8])
            || qzeros.meta.shape() != &Shape::from([layout.groups(), n / 8])
            || scales.dtype != DType::F16
            || scales.meta.shape() != &Shape::from([layout.groups(), n])
        {
            return Err(Int4Error(
                "AWQ GEMM requires I32 packed weights/zeros and F16 per-group scales with matching shapes",
            ));
        }
        for tensor in [&qzeros, &scales].into_iter().chain(bias.iter()) {
            if tensor.device.to_id() != qweight.device.to_id() {
                return Err(Int4Error("AWQ GEMM operands must be on the same device"));
            }
        }
        if let Some(bias) = &bias {
            if bias.dtype != DType::F16 || bias.meta.shape() != &Shape::from([n]) {
                return Err(Int4Error(
                    "AWQ GEMM bias must be an F16 output-channel vector",
                ));
            }
        }
        Ok(Self {
            qweight: into_contiguous(qweight),
            qzeros: into_contiguous(qzeros),
            scales: into_contiguous(scales),
            bias: bias.map(into_contiguous),
            layout,
        })
    }

    pub fn layout(&self) -> AwqGemmLayout {
        self.layout
    }

    pub fn forward(&self, input: RudaTensor<R>) -> Result<RudaTensor<R>, Int4Error> {
        let rank = input.meta.num_dims();
        let k = self.layout.input_features;
        let n = self.layout.output_features;
        if rank == 0 || input.meta.shape()[rank - 1] != k || input.dtype != DType::F16 {
            return Err(Int4Error(
                "AWQ GEMM input must be F16 with matching last dimension",
            ));
        }
        if input.device.to_id() != self.qweight.device.to_id() {
            return Err(Int4Error(
                "AWQ GEMM input and weights must be on the same device",
            ));
        }
        let rows = input
            .meta
            .shape()
            .iter()
            .take(rank - 1)
            .try_fold(1usize, |size, &dim| size.checked_mul(dim))
            .ok_or(Int4Error("AWQ GEMM batch size overflow"))?;
        let elements = rows
            .checked_mul(n)
            .ok_or(Int4Error("AWQ GEMM output size overflow"))?;
        if elements > u32::MAX as usize
            || rows
                .checked_mul(k)
                .is_none_or(|size| size > u32::MAX as usize)
        {
            return Err(Int4Error(
                "AWQ GEMM batch exceeds the kernel's u32 indexing range",
            ));
        }
        let mut shape = input.meta.shape().clone();
        shape[rank - 1] = n;
        let output = empty_device_contiguous_dtype(
            input.client.clone(),
            input.device.clone(),
            shape,
            DType::F16,
        );
        if elements == 0 {
            return Ok(output);
        }
        let input = into_contiguous(input);
        let ruda_dim = RudaDim::new(input.client.properties(), elements);
        kernel::awq_gemm::launch::<R>(
            &input.client,
            calculate_ruda_count_elemwise(&input.client, elements, ruda_dim),
            ruda_dim,
            input.clone().into_array_arg(),
            self.qweight.clone().into_array_arg(),
            self.qzeros.clone().into_array_arg(),
            self.scales.clone().into_array_arg(),
            self.bias
                .as_ref()
                .unwrap_or(&self.scales)
                .clone()
                .into_array_arg(),
            output.clone().into_array_arg(),
            k as u32,
            n as u32,
            self.layout.group_size as u32,
            u32::from(self.bias.is_some()),
            DType::F16.into(),
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
