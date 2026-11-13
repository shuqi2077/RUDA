use melior::{dialect::func, ir::attribute::FlatSymbolRefAttribute};
use ruda_core::ir::Synchronization;

use crate::mlir::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_synchronization(&mut self, synchronization: &Synchronization) {
        match synchronization {
            Synchronization::SyncRuda => {
                self.needs_parallelism = true;
                let func_name = FlatSymbolRefAttribute::new(self.context, "sync_ruda");
                self.block.append_operation(func::call(
                    self.context,
                    func_name,
                    &[],
                    &[],
                    self.location,
                ));
            }
            Synchronization::SyncPlane => {} // NOOP plane size is 1 on CPU
            Synchronization::SyncStorage => {
                panic!("SyncStorage is not supported")
            }
            Synchronization::FenceSystem => {
                panic!("native peer memory fence is not supported in MLIR")
            }
            Synchronization::SyncAsyncProxyShared => {
                panic!("SyncProxyShared is not supported")
            }
        }
    }
}
