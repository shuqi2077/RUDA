use tracel_llvm::mlir_rs::ExecutionEngine;

use crate::execution::compute_task::sync_ruda;

pub fn register_external_function(execution_engine: &ExecutionEngine) {
    unsafe {
        execution_engine.register_symbol("sync_ruda", sync_ruda as *mut ());
        // This is only there to fool the execution engine to generate .so for inspection even if symbol resolution will probably not work.
        execution_engine.register_symbol("_mlir_sync_ruda", sync_ruda as *mut ());
        execution_engine.register_symbol("rsqrtf", rsqrtf as *mut ());
        execution_engine.register_symbol("rsqrt", rsqrt as *mut ());
    }
}

extern "C" fn rsqrtf(input: f32) -> f32 {
    1.0f32 / input.sqrt()
}

extern "C" fn rsqrt(input: f64) -> f64 {
    1.0f64 / input.sqrt()
}
