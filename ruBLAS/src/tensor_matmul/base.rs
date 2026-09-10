use super::init_matmul_output;
use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, dequantize::dequantize}};
use ruda_core::tensor::{DType, QTensorPrimitive};
use ruda_core::quant::scheme::QuantLevel;
use crate::kernel_ir::{
        definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
        launch::Strategy,
    };
use ruda_kernel::tiling::InputBinding;

#[cfg(feature = "tensor-matmul-autotune")]
use super::matmul_autotune_with_precision;

pub use crate::kernel_ir::definition::F32MathMode;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    #[cfg(feature = "tensor-matmul-autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Ruda implementation of matmul.
    Ruda,
    /// Tensor Core path with the partial K32 tile first; no materialized padding.
    CmmaResidueFirst,
    /// One output element per unit, used by runtimes without plane arithmetic.
    Naive,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "tensor-matmul-autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "tensor-matmul-autotune"))]
        MatmulStrategy::Ruda
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: Option<RudaTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    matmul_with_precision(lhs, rhs, out, strategy, out_dtype, F32MathMode::Strict)
}

/// Launch matmul with an explicit, per-call F32 arithmetic policy.
pub fn matmul_with_precision<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: Option<RudaTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
    f32_math: F32MathMode,
) -> Result<RudaTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::CmmaResidueFirst => {
            if !matches!(lhs.dtype, DType::BF16 | DType::F16) || lhs.dtype != rhs.dtype
                || lhs.qparams.is_some() || rhs.qparams.is_some()
            {
                return Err(MatmulSetupError::InvalidConfig(Box::new("residue-first matmul requires matching unquantized BF16 or F16 inputs".to_string())));
            }
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul(&Strategy::SimpleCyclicCmmaResidueFirst, lhs, rhs, out.clone())?;
            Ok(out)
        }
        MatmulStrategy::Ruda => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul_with_precision(&Default::default(), lhs, rhs, out.clone(), f32_math)?;
            Ok(out)
        }
        MatmulStrategy::Naive => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul_naive(&Strategy::Naive, lhs, rhs, out.clone())?;
            Ok(out)
        }
        #[cfg(feature = "tensor-matmul-autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune_with_precision(lhs, rhs, out, out_dtype, f32_math)),
    }
}

pub(crate) fn launch_matmul_naive<R: Runtime>(
    strategy: &Strategy,
    mut lhs: RudaTensor<R>,
    mut rhs: RudaTensor<R>,
    out: RudaTensor<R>,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<R: Runtime>(
    strategy: &Strategy,
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    out: RudaTensor<R>,
) -> Result<(), MatmulSetupError> {
    launch_matmul_with_precision(strategy, lhs, rhs, out, F32MathMode::Strict)
}

pub(crate) fn launch_matmul_with_precision<R: Runtime>(
    strategy: &Strategy,
    lhs: RudaTensor<R>,
    mut rhs: RudaTensor<R>,
    out: RudaTensor<R>,
    f32_math: F32MathMode,
) -> Result<(), MatmulSetupError> {
    let client = &out.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match lhs_quant_handles {
        None => {
            let lhs_dtype = lhs.dtype;
            (
                lhs_dtype,
                InputBinding::new(lhs.binding(), lhs_dtype.into()),
            )
        }
        Some((data, scale)) => {
            let scheme = *lhs.scheme();
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            (
                out_dtype,
                InputBinding::quantized(
                    data.binding(),
                    scale.binding(),
                    lhs.meta.shape().clone(),
                    scheme,
                    data_dtype.into(),
                    scale_dtype.into(),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match rhs_quant_handles {
        None => (
            lhs_dtype,
            InputBinding::new(rhs.binding(), lhs_dtype.into()),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(strategy, Strategy::Naive)
                && matches!(rhs.scheme().level, QuantLevel::Block(_))
            {
                rhs = dequantize(rhs.clone(), lhs_dtype);
                let rhs_dtype = rhs.dtype;
                (
                    lhs_dtype,
                    InputBinding::new(rhs.binding(), rhs_dtype.into()),
                )
            } else {
                let scheme = *rhs.scheme();
                let data_dtype = data.dtype;
                let scale_dtype = scale.dtype;
                (
                    out_dtype,
                    InputBinding::quantized(
                        data.binding(),
                        scale.binding(),
                        rhs.meta.shape().clone(),
                        scheme,
                        data_dtype.into(),
                        scale_dtype.into(),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        f32_math,
        lhs: lhs_dtype.into(),
        rhs: rhs_dtype.into(),
        out: out_dtype.into(),
    });

    crate::kernel_ir::launch::launch_ref(
        strategy,
        client,
        lhs_handle,
        rhs_handle,
        out.clone().binding(),
        &mut dtypes,
    )?;

    Ok(())
}
