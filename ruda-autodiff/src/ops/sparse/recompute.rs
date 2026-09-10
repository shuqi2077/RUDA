use crate::{
    checkpoint::{retro_forward::RetroForward, state::BackwardStates},
    graph::NodeId,
};
use ruda_tensor::{ops::SparseOps, tensor::FloatTensor};

#[derive(Clone, Copy, Debug)]
pub(super) enum CsrUnaryReplay {
    Gather,
    ScatterAdd,
    ToDense,
    ToDenseBackward,
}

#[derive(Debug)]
pub(super) struct CsrReplay<B: SparseOps> {
    matrix: B::CsrHandle,
    input: NodeId,
    operation: CsrUnaryReplay,
}

impl<B: SparseOps> CsrReplay<B> {
    pub(super) fn new(matrix: B::CsrHandle, input: NodeId, operation: CsrUnaryReplay) -> Self {
        Self { matrix, input, operation }
    }
}

impl<B: SparseOps> RetroForward for CsrReplay<B> {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let input = states.get_state::<FloatTensor<B>>(&self.input);
        let output = match self.operation {
            CsrUnaryReplay::Gather => B::csr_gather(&self.matrix, input),
            CsrUnaryReplay::ScatterAdd => B::csr_scatter_add(&self.matrix, input),
            CsrUnaryReplay::ToDense => B::csr_to_dense(&self.matrix, input),
            CsrUnaryReplay::ToDenseBackward => B::csr_to_dense_backward(&self.matrix, input),
        }.unwrap_or_else(|error| panic!("CSR {:?} recomputation: {error}", self.operation));
        states.save(out_node, output);
    }
}
