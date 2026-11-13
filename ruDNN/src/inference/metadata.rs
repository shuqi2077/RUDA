//! Native scheduler/control casts, including physical U8/U32 boolean storage.
use super::{InferenceError, PreparedPack, alloc};
use ruda_core::tensor::{BoolStore, DType};
use ruda_kernel::dsl::prelude::Cast as DeviceCast;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
pub struct PreparedIndices<R: Runtime> {
    source: PreparedPack<R>,
    pub output: RudaTensor<R>,
    storage: DType,
}
impl<R: Runtime> PreparedIndices<R> {
    pub fn prepare(source: &RudaTensor<R>, dtype: DType) -> Result<Self, InferenceError> {
        let storage = match source.dtype {
            DType::Bool(BoolStore::U32) => DType::U32,
            DType::Bool(BoolStore::U8) | DType::Bool(BoolStore::Native) => DType::U8,
            t @ (DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64) => t,
            _ => return Err(InferenceError("metadata source must be integer/bool")),
        };
        if !matches!(dtype, DType::I32 | DType::U32 | DType::I64 | DType::U64) {
            return Err(InferenceError("metadata output must be I32/U32/I64/U64"));
        }
        let source = PreparedPack::prepare(source)?;
        let output = alloc(&source.output, source.output.meta.shape().clone(), dtype);
        Ok(Self {
            source,
            output,
            storage,
        })
    }
    /// Values must be representable in the chosen destination integer type.
    pub unsafe fn run(&self) {
        unsafe {
            self.source.run();
        }
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(self.output.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.output.client, n, dim);
            cast::launch::<R>(
                &self.output.client,
                count,
                dim,
                self.source.output.clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                n as u32,
                self.storage.into(),
                self.output.dtype.into(),
            );
        }
    }
}
pub fn cast_indices<R: Runtime>(
    source: &RudaTensor<R>,
    dtype: DType,
) -> Result<RudaTensor<R>, InferenceError> {
    let plan = PreparedIndices::prepare(source, dtype)?;
    unsafe {
        plan.run();
    }
    Ok(plan.output)
}
#[ruda(launch)]
pub fn cast<I: Int, O: Int>(
    source: &Array<I>,
    out: &mut Array<O>,
    n: u32,
    #[define(I)] _i: StorageType,
    #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = O::cast_from(source[i]);
    }
}
