use super::*;
use alloc::vec::Vec;
use ruda_tensor::collective::{VariableTensorCollective,VariableTensorExchange};

#[derive(Debug)]
struct RowExchange<C>(PhantomData<C>);
impl<B:Backend,C:VariableTensorCollective<B>> Backward<B,1> for RowExchange<C> {
    type State=(C,Vec<usize>,Vec<usize>,ruda_tensor::Shape,bool);
    fn ordered_backward(state:&Self::State) -> bool {state.4}
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (communicator,sent,received,shape,_)=ops.state;
        unary::<B,_>(ops.parents,ops.node,grads,|gradient| {
            let result=communicator.all_to_all_v_float(gradient,&received)
                .unwrap_or_else(|error|panic!("native variable row exchange backward failed: {error:?}"));
            assert_eq!(result.receive_counts,sent,"inverse row exchange changed original peer row counts");
            assert_eq!(result.value.shape(),shape,"inverse row exchange changed original source axes");result.value
        });
    }
}
fn exchange<B:Backend,S:CheckpointStrategy,C:VariableTensorCollective<B>,const D:usize>(
    input:Tensor<Autodiff<B,S>,D>,communicator:C,send_counts:&[usize],ordered:bool,
) -> Result<VariableTensorExchange<Tensor<Autodiff<B,S>,D>>,C::Error> {
    let scope=communicator.autodiff_context().and_then(|context|context.downcast_ref::<CollectiveScope<B,S>>()).cloned();
    let ordered=ordered || scope.is_some();let input=input.into_primitive().tensor();let shape=input.primitive.shape();
    let result=communicator.all_to_all_v_float(input.primitive,send_counts)?;let receive_counts=result.receive_counts;
    let output=match RowExchange::<C>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
        OpsKind::Tracked(prep)=>prep.finish((communicator,send_counts.to_vec(),receive_counts.clone(),shape,ordered),result.value),
        OpsKind::UnTracked(prep)=>prep.finish(result.value),
    };
    let value=Tensor::from_primitive(TensorPrimitive::Float(output));if let Some(scope)=scope {scope.capture(value.clone());}
    Ok(VariableTensorExchange {value,receive_counts})
}
/// Exchange actual destination-packed row blocks; backward is the exact inverse source-rank exchange.
/// Every rank must participate with matching trailing axes/dtype and tracking, even when local rows are empty.
pub fn all_to_all_v<B:Backend,S:CheckpointStrategy,C:VariableTensorCollective<B>,const D:usize>(
    input:Tensor<Autodiff<B,S>,D>,communicator:C,send_counts:&[usize],
) -> Result<VariableTensorExchange<Tensor<Autodiff<B,S>,D>>,C::Error> {exchange(input,communicator,send_counts,false)}
/// Same original exchange with rank-consistent tape order for graphs containing independent collectives.
pub fn all_to_all_v_ordered<B:Backend,S:CheckpointStrategy,C:VariableTensorCollective<B>,const D:usize>(
    input:Tensor<Autodiff<B,S>,D>,communicator:C,send_counts:&[usize],
) -> Result<VariableTensorExchange<Tensor<Autodiff<B,S>,D>>,C::Error> {exchange(input,communicator,send_counts,true)}
