use super::HuggingFaceLoadError;
use rublas::tensor_int4::AwqGemmLayout;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AwqQuantizationConfig {
    pub quant_method: String,
    pub bits: usize,
    pub group_size: i64,
    pub zero_point: bool,
    pub version: String,
    pub modules_to_not_convert: Option<Vec<String>>,
}

impl AwqQuantizationConfig {
    pub fn validate(&self) -> Result<(), HuggingFaceLoadError> {
        if !self.quant_method.eq_ignore_ascii_case("awq")
            || self.bits != 4
            || !self.version.eq_ignore_ascii_case("gemm")
            || !self.zero_point
        {
            return Err(HuggingFaceLoadError(
                "expected AWQ GEMM with four bits and explicit zero points".into(),
            ));
        }
        if self.group_size != -1 && self.group_size <= 0 {
            return Err(HuggingFaceLoadError(
                "AWQ group_size must be positive or -1".into(),
            ));
        }
        Ok(())
    }

    pub fn layout(
        &self,
        input_features: usize,
        output_features: usize,
    ) -> Result<AwqGemmLayout, HuggingFaceLoadError> {
        self.validate()?;
        let group_size = if self.group_size == -1 {
            input_features
        } else {
            usize::try_from(self.group_size)
                .map_err(|_| HuggingFaceLoadError("AWQ group_size exceeds address range".into()))?
        };
        AwqGemmLayout::new(input_features, output_features, group_size)
            .map_err(|error| HuggingFaceLoadError(error.to_string()))
    }
}
