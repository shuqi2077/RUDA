use ruda_kernel::dsl as kernel_dsl;
use std::marker::PhantomData;

use ruda_kernel::dsl::ir::AddressType;
use ruda_kernel::dsl::ir::DeviceProperties;
use ruda_kernel::dsl::server::LaunchError;

use crate::attention::kernel_ir::{
    components::{
        batch::{
            BatchAttentionFamily,
            entry_point::attention,
            simple::{SimpleBatchAttention, config::SimpleBatchConfig},
        },
        global::GlobalAttentionFamily,
    },
    definition::{
        AttentionBlueprint, AttentionElems, AttentionPrecision, AttentionSetupError,
        AttentionVectorSizes, RudaMappingLaunch, InputRuntimeArg, OutputRuntimeArg,
        launch_types::*,
    },
    launch::AttentionArgs,
};

pub struct SimpleBatchAttentionFamily<GA: GlobalAttentionFamily> {
    _phantom: PhantomData<GA>,
}

impl<GA: GlobalAttentionFamily> BatchAttentionFamily for SimpleBatchAttentionFamily<GA> {
    type Attention<AP: AttentionPrecision> = SimpleBatchAttention<AP, GA::Attention<AP>>;
    type Config = SimpleBatchConfig<GA::Config>;
    type Blueprint = AttentionBlueprint;

    unsafe fn launch_unchecked<'a, AA: AttentionArgs, R: ruda_kernel::dsl::Runtime>(
        client: &ruda_kernel::dsl::prelude::ComputeClient<R>,
        ruda_dim: ruda_kernel::dsl::RudaDim,
        ruda_count: ruda_kernel::dsl::RudaCount,
        address_type: AddressType,
        input: InputRuntimeArg<AA, R>,
        output: OutputRuntimeArg<AA, R>,
        ruda_mapping: RudaMappingLaunch<R>,
        dtypes: &AttentionElems,
        vector_sizes: &AttentionVectorSizes,
        blueprint: Self::Blueprint,
    ) -> Result<(), LaunchError> {
        unsafe {
            attention::launch_unchecked::<AA, QG, QGS, KG, KGS, VG, VGS, MSK, MSKS, OG, OGS, Self, R>(
                client,
                ruda_count,
                ruda_dim,
                address_type,
                input,
                output,
                ruda_mapping,
                blueprint,
                dtypes.clone(),
                dtypes.into(),
                vector_sizes.into(),
            )
        };

        Ok(())
    }

    fn expand_config(
        device_props: &DeviceProperties,
        blueprint: Self::Blueprint,
        dtypes: &AttentionElems,
    ) -> Result<Self::Config, AttentionSetupError> {
        let global_config = GA::expand_config(device_props, &blueprint, dtypes)?;

        Ok(SimpleBatchConfig::new(global_config))
    }
}
