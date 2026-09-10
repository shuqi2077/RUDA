use core::ops::{Index, IndexMut};

use ruda_core::ir::{
    IndexAssignOperator, Instruction, ManagedVariable, Operator, Scope, VariableKind, VectorSize,
};

use super::{RudaType, NativeExpand, index_expand, index_expand_no_vec};
use crate::dsl::{ir::Variable, prelude::RudaPrimitive, unexpanded};

/// Fake indexation so we can rewrite indexes into scalars as calls to this fake function in the
/// non-expanded function
pub trait RudaIndex:
    RudaType<
    ExpandType: RudaIndexExpand<
        Idx = <Self::Idx as RudaType>::ExpandType,
        Output = <Self::Output as RudaType>::ExpandType,
    >,
>
{
    type Output: RudaType;
    type Idx: RudaType;

    fn ruda_idx(&self, _i: Self::Idx) -> &Self::Output {
        unexpanded!()
    }

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: <Self::Idx as RudaType>::ExpandType,
    ) -> <Self::Output as RudaType>::ExpandType {
        array.expand_index(scope, index)
    }
    fn expand_index_unchecked(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: <Self::Idx as RudaType>::ExpandType,
    ) -> <Self::Output as RudaType>::ExpandType {
        array.expand_index_unchecked(scope, index)
    }
}

/// Workaround for comptime indexing, since the helper that replaces index operators doesn't know
/// about whether a variable is comptime. Has the same signature in unexpanded code, so it will
/// automatically dispatch the correct one.
pub trait ComptimeIndex<I>: Index<I> {
    fn ruda_idx(&self, i: I) -> &Self::Output {
        self.index(i)
    }
}

impl<I, T: Index<I>> ComptimeIndex<I> for T {}
impl<I, T: IndexMut<I>> ComptimeIndexMut<I> for T {}

pub trait ComptimeIndexMut<I>: ComptimeIndex<I> + IndexMut<I> {
    fn ruda_idx_mut(&mut self, i: I) -> &mut Self::Output {
        self.index_mut(i)
    }
}

pub trait RudaIndexExpand {
    type Output;
    type Idx;
    fn expand_index(self, scope: &mut Scope, index: Self::Idx) -> Self::Output;
    fn expand_index_unchecked(self, scope: &mut Scope, index: Self::Idx) -> Self::Output;
}

pub trait RudaIndexMut:
    RudaIndex
    + RudaType<ExpandType: RudaIndexMutExpand<Output = <Self::Output as RudaType>::ExpandType>>
{
    fn ruda_idx_mut(&mut self, _i: <Self as RudaIndex>::Idx) -> &mut <Self as RudaIndex>::Output {
        unexpanded!()
    }
    fn expand_index_mut(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: <Self::Idx as RudaType>::ExpandType,
        value: <Self::Output as RudaType>::ExpandType,
    ) {
        array.expand_index_mut(scope, index, value)
    }
}

pub trait RudaIndexMutExpand: RudaIndexExpand {
    fn expand_index_mut(
        self,
        scope: &mut Scope,
        index: <Self as RudaIndexExpand>::Idx,
        value: <Self as RudaIndexExpand>::Output,
    );
}

pub(crate) fn expand_index_native<A: RudaType + RudaIndex>(
    scope: &mut Scope,
    array: NativeExpand<A>,
    index: NativeExpand<usize>,
    vector_size: Option<VectorSize>,
    checked: bool,
) -> NativeExpand<A::Output>
where
    A::Output: RudaType + Sized,
{
    let index: ManagedVariable = index.into();
    let index_var: Variable = *index;
    let index = match index_var.kind {
        VariableKind::Constant(value) => {
            ManagedVariable::Plain(Variable::constant(value, usize::as_type(scope)))
        }
        _ => index,
    };
    let array: ManagedVariable = array.into();
    let var: Variable = *array;
    let var = if checked {
        match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } => {
                index_expand_no_vec(scope, array, index, Operator::Index)
            }
            _ => index_expand(scope, array, index, vector_size, Operator::Index),
        }
    } else {
        match var.kind {
            VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } => {
                index_expand_no_vec(scope, array, index, Operator::UncheckedIndex)
            }
            _ => index_expand(scope, array, index, vector_size, Operator::UncheckedIndex),
        }
    };

    NativeExpand::new(var)
}

pub(crate) fn expand_index_assign_native<A: RudaType<ExpandType = NativeExpand<A>> + RudaIndexMut>(
    scope: &mut Scope,
    array: A::ExpandType,
    index: NativeExpand<usize>,
    value: NativeExpand<<A as RudaIndex>::Output>,
    vector_size: Option<VectorSize>,
    checked: bool,
) where
    A::Output: RudaType + Sized,
{
    let index: Variable = index.expand.into();
    let index = match index.kind {
        VariableKind::Constant(value) => Variable::constant(value, usize::as_type(scope)),
        _ => index,
    };

    let vector_size = vector_size.unwrap_or(0);
    if checked {
        scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index,
                value: value.expand.into(),
                vector_size,
                unroll_factor: 1,
            }),
            array.expand.into(),
        ));
    } else {
        scope.register(Instruction::new(
            Operator::UncheckedIndexAssign(IndexAssignOperator {
                index,
                value: value.expand.into(),
                vector_size,
                unroll_factor: 1,
            }),
            array.expand.into(),
        ));
    }
}
