//! Transaction operations for the Host backend.

use crate::Host;
use ruda_tensor::ops::TransactionOps;

// TransactionOps has default implementations.
impl TransactionOps<Host> for Host {}
