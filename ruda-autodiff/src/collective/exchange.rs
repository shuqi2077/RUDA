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
            let result=communicator.all_to_all_v_float(grads.consume::<B>(&ops.node),&received)
                .unwrap_or_else(|error|panic!("native variable row exchange backward failed: {error:?}"));
            assert_eq!(result.receive_counts,sent,"inverse row exchange changed original peer row counts");
            assert_eq!(result.value.shape(),shape,"inverse row exchange changed original source axes");
            if let Some(parent)=ops.parents[0].as_ref() {grads.register::<B>(parent.id,result.value);}
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

/// Actual distributed row exchange with globally coordinated gradient dependence.
/// Different ranks may have frozen local inputs or empty expert partitions. The
/// output remains tracked wherever a peer input is tracked, while untracked
/// local inputs never acquire a parameter gradient. Backward enters the inverse
/// exchange on every participating rank; complete a scope when local losses can omit it.
pub fn all_to_all_v_coordinated<B:Backend,S:CheckpointStrategy,C:VariableTensorCollective<B>,const D:usize>(
    input:Tensor<Autodiff<B,S>,D>,communicator:C,send_counts:&[usize],
) -> Result<VariableTensorExchange<Tensor<Autodiff<B,S>,D>>,ScopedCollectiveError<C::Error>> {
    use crate::{graph::{ComputingProperty,Requirement},ops::OpsPrep,checkpoint::builder::CheckpointerBuilder};
    use ruda_tensor::{DType,TensorData,collective::IntegerTensorCollective};
    let scope=communicator.autodiff_context().and_then(|context|context.downcast_ref::<CollectiveScope<B,S>>()).cloned();
    let input=input.into_primitive().tensor();let world=communicator.world_size();
    if world==0 || communicator.rank()>=world {return Err(ScopedCollectiveError::Protocol("coordinated row exchange has invalid original topology"));}
    let tracked=if world==1 {input.is_tracked()} else {
        let device=B::float_device(&input.primitive);
        let flag=Tensor::<B,1,ruda_tensor::api::Int>::from_data(TensorData::new(alloc::vec![u8::from(input.is_tracked())],[1]),(&device,DType::U8));
        let flags=communicator.all_gather_int(flag.into_primitive()).map_err(ScopedCollectiveError::Collective)?;
        if flags.shape()[..]!=[world as usize] || flags.dtype()!=DType::U8 || B::int_device(&flags)!=device {
            return Err(ScopedCollectiveError::Protocol("row-exchange gradient-dependence metadata changed native shape/storage/device"));}
        let flags=ruda_tensor::read_sync(B::int_into_data(flags)).map_err(|_|ScopedCollectiveError::Protocol("row-exchange gradient-dependence metadata read failed"))?
            .to_vec::<u8>().map_err(|_|ScopedCollectiveError::Protocol("row-exchange gradient-dependence metadata storage mismatch"))?;
        if flags.iter().any(|&flag|flag>1) {return Err(ScopedCollectiveError::Protocol("invalid actual peer gradient-dependence flag"));}
        flags.iter().any(|&flag|flag!=0)
    };
    let shape=input.primitive.shape();let result=communicator.all_to_all_v_float(input.primitive,send_counts).map_err(ScopedCollectiveError::Collective)?;
    let receive_counts=result.receive_counts;
    let requirement=if tracked {Requirement::GradInBackward} else {Requirement::None};
    let prep=OpsPrep::<RowExchange<C>,B,<RowExchange<C> as Backward<B,1>>::State,S,1>::new(
        [input.node],requirement,RowExchange::<C>(PhantomData),ComputingProperty::Ambiguous,CheckpointerBuilder::default());
    let output=match prep.compute_bound().stateful() {
        OpsKind::Tracked(prep)=>prep.finish((communicator,send_counts.to_vec(),receive_counts.clone(),shape,true),result.value),
        OpsKind::UnTracked(prep)=>prep.finish(result.value),
    };
    let value=Tensor::from_primitive(TensorPrimitive::Float(output));if let Some(scope)=scope {scope.capture(value.clone());}
    Ok(VariableTensorExchange {value,receive_counts})
}
