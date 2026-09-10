use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::RudaCount;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::ir::AddressType;
use ruda_kernel::dsl::ir::DeviceProperties;
use ruda_kernel::dsl::ir::{ElemType, FloatKind, StorageType};
use ruda_kernel::dsl::quant::scheme::QuantLevel;
use ruda_kernel::dsl::server::LaunchError;
use ruda_kernel::tiling::MatrixLayout;

use crate::kernel_ir::{
    components::{
        RudaDimResource,
        batch::{
            BatchMatmulFamily,
            naive::{NaiveMatmul, NaiveMatmulConfig, matmul_entry},
        },
        global::memory::GlobalLayoutConfig,
        stage::NumStages,
    },
    definition::{
        Blueprint, RudaMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulTypes,
        MatmulVectorSizes, SwizzleModes, TilingScheme,
    },
    launch::*,
};

/// Simple partitioned batch matmul family for any precision
pub struct NaiveBatchMatmulFamily {}
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NaiveBlueprint {
    pub vector_size_out: u32,
    pub dtypes: MatmulElems,
}

impl Blueprint for NaiveBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        panic!("Naive Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("Naive Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for NaiveBatchMatmulFamily {
    type Matmul<MP: MatmulTypes> = NaiveMatmul<MP>;
    type Config = NaiveMatmulConfig;
    type Blueprint = NaiveBlueprint;

    fn expand_config(
        _device_props: &DeviceProperties,
        _blueprint: &Self::Blueprint,
        dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let f32_type = StorageType::Scalar(ElemType::Float(FloatKind::F32));
        Ok(NaiveMatmulConfig {
            fused_accumulation: dtypes.acc_register.is_float(),
            sequential_accumulation: dtypes.lhs_global == f32_type
                && dtypes.rhs_global == f32_type && dtypes.acc_register == f32_type,
        })
    }

    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    unsafe fn launch_unchecked<'a, MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        ruda_dim: RudaDim,
        ruda_count: RudaCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        _config: ConfigRuntimeArg<MA, R>,
        ruda_mapping: RudaMappingLaunch<R>,
        blueprint: NaiveBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, R>(
                client,
                ruda_count,
                ruda_dim,
                address_type,
                input,
                output,
                (),
                ruda_mapping,
                blueprint,
                include_str!("matmul.rs").to_owned(),
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn rudadim_resource(
        _blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<RudaDimResource, MatmulSetupError> {
        // Could be moved to blueprint to be less hard coded
        Ok(RudaDimResource::Planes(8))
    }

    fn validate_blueprint<R: Runtime>(
        _client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        if blueprint.vector_size_out > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Vector size on output not supported",
            )));
        }

        if let Some(scheme) = problem.lhs_scheme
            && let QuantLevel::Block(block_size) = scheme.level
        {
            let vector_size = vector_sizes.lhs * scheme.num_quants();
            let block_size = block_size.to_dim_vec(2.max(block_size.len()));
            let block_width = block_size[block_size.len() - 1] as usize;
            if !block_width.is_multiple_of(vector_size) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Block size isn't a multiple of load size on lhs",
                )));
            }
        }

        if let Some(scheme) = problem.rhs_scheme
            && let QuantLevel::Block(block_size) = scheme.level
        {
            let vector_size = vector_sizes.rhs * scheme.num_quants();
            let block_size = block_size.to_dim_vec(2.max(block_size.len()));
            let block_width = block_size[block_size.len() - 2] as usize;
            if !block_width.is_multiple_of(vector_size) {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Block size isn't a multiple of load size on rhs",
                )));
            }
        }

        Ok(())
    }
}
