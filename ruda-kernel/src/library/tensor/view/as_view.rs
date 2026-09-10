use ruda_kernel::dsl::{prelude::*, unexpanded};

use crate::library::tensor::{View, ViewExpand, layout::*};

pub trait AsView<E: RudaPrimitive>:
    RudaType<ExpandType: AsViewExpand<E, SourceCoords = Self::SourceCoords>>
{
    type SourceCoords: Coordinates;

    #[allow(unused)]
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Self::SourceCoords>>,
    ) -> View<E, C, ReadOnly> {
        unexpanded!()
    }

    fn __expand_view<C: Coordinates + 'static>(
        scope: &mut Scope,
        this: Self::ExpandType,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly> {
        this.__expand_view_method(scope, layout)
    }
}

pub trait AsViewExpand<E: RudaPrimitive> {
    type SourceCoords: Coordinates;

    #[allow(unused)]
    fn __expand_view_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly>;
}

pub trait AsViewMut<E: RudaPrimitive>: AsView<E> {
    #[allow(unused)]
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Self::SourceCoords>>,
    ) -> View<E, C, ReadWrite> {
        unexpanded!()
    }
}

pub trait AsViewMutExpand<E: RudaPrimitive>: AsViewExpand<E> {
    #[allow(clippy::too_many_arguments)]
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadWrite>;
}

macro_rules! impl_as_view {
    ($ty: ident, $coords: ty) => {
        impl<E: RudaPrimitive> AsView<E> for $ty<E> {
            type SourceCoords = $coords;
        }
        impl<E: RudaPrimitive> AsViewExpand<E> for NativeExpand<$ty<E>> {
            type SourceCoords = $coords;
            fn __expand_view_method<C: Coordinates + 'static>(
                self,
                scope: &mut Scope,
                layout: VirtualLayoutExpand<C, $coords>,
            ) -> super::ViewExpand<E, C, ReadOnly> {
                View::__expand_new::<$ty<E>, $coords>(scope, self, layout)
            }
        }

        impl<E: RudaPrimitive> AsViewMut<E> for $ty<E> {}
        impl<E: RudaPrimitive> AsViewMutExpand<E> for NativeExpand<$ty<E>> {
            fn __expand_view_mut_method<C: Coordinates + 'static>(
                self,
                scope: &mut Scope,
                layout: VirtualLayoutExpand<C, $coords>,
            ) -> super::ViewExpand<E, C, ReadWrite> {
                View::__expand_new_mut::<$ty<E>, $coords>(scope, self, layout)
            }
        }
    };
}

impl_as_view!(Array, Coords1d);
impl_as_view!(Tensor, Coords1d);
impl_as_view!(SharedMemory, Coords1d);

impl<E: RudaPrimitive, IO: SliceVisibility + 'static> AsView<E> for Slice<E, IO> {
    type SourceCoords = Coords1d;
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<E, C, ReadOnly> {
        View::new::<Slice<E, IO>, Coords1d>(self, layout)
    }
}

impl<E: RudaPrimitive, IO: SliceVisibility + 'static> AsViewExpand<E> for SliceExpand<E, IO> {
    type SourceCoords = Coords1d;
    fn __expand_view_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly> {
        View::__expand_new::<Slice<E, IO>, Self::SourceCoords>(scope, self, layout)
    }
}

impl<E: RudaPrimitive> AsViewMut<E> for Slice<E, ReadWrite> {
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<E, C, ReadWrite> {
        View::new_mut::<Slice<E, ReadWrite>, Coords1d>(self, layout)
    }
}
impl<E: RudaPrimitive> AsViewMutExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut crate::dsl::prelude::Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadWrite> {
        View::__expand_new_mut::<Slice<E, ReadWrite>, Coords1d>(scope, self, layout)
    }
}

macro_rules! as_view_tensor_map {
    ($($dim: literal),*) => {
        paste::paste! {
            pub trait AsTensorView<E: RudaPrimitive>:
                RudaType<ExpandType: AsTensorViewExpand<E>>
            {
                $(
                    #[allow(unused)]
                    fn [<view_ $dim>]<C: Coordinates + 'static>(
                        &self,
                        layout: impl Into<VirtualLayout<C, [<Coords $dim>]>>,
                    ) -> View<E, C, ReadOnly> {
                        unexpanded!()
                    }

                    fn [<__expand_view_ $dim>]<C: Coordinates + 'static>(
                        scope: &mut Scope,
                        this: Self::ExpandType,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewExpand<E, C, ReadOnly> {
                        this.[<__expand_view_ $dim _method>](scope, layout)
                    }
                )*
            }

            pub trait AsTensorViewExpand<E: RudaPrimitive> {
                $(
                    #[allow(unused)]
                    fn [<__expand_view_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &mut Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewExpand<E, C, ReadOnly>;
                )*
            }

            pub trait AsTensorViewMut<E: RudaPrimitive>: AsTensorView<E> {
                $(
                    #[allow(unused)]
                    fn [<view_mut_ $dim>]<C: Coordinates + 'static>(
                        &mut self,
                        layout: impl Into<VirtualLayout<C, [<Coords $dim>]>>,
                    ) -> View<E, C, ReadWrite> {
                        unexpanded!()
                    }
                )*
            }

            pub trait AsTensorViewMutExpand<E: RudaPrimitive>: AsTensorViewExpand<E> {
                $(
                    #[allow(clippy::too_many_arguments)]
                    fn [<__expand_view_mut_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &mut Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewExpand<E, C, ReadWrite>;
                )*
            }

            impl<E: RudaPrimitive> AsTensorView<E> for TensorMap<E, Tiled> {}
            impl<E: RudaPrimitive> AsTensorViewExpand<E> for NativeExpand<TensorMap<E, Tiled>> {
                $(
                    fn [<__expand_view_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &mut Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> super::ViewExpand<E, C, ReadOnly> {
                        View::__expand_new::<TensorMap<E, Tiled>, [<Coords $dim>]>(scope, self, layout)
                    }
                )*
            }

            impl<E: RudaPrimitive> AsTensorViewMut<E> for TensorMap<E, Tiled> {}
            impl<E: RudaPrimitive> AsTensorViewMutExpand<E> for NativeExpand<TensorMap<E, Tiled>> {
                $(
                    fn [<__expand_view_mut_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &mut Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> super::ViewExpand<E, C, ReadWrite> {
                        View::__expand_new_mut::<TensorMap<E, Tiled>, [<Coords $dim>]>(scope, self, layout)
                    }
                )*
            }
        }
    };
}

as_view_tensor_map!(1d, 2d, 3d, 4d, 5d, 1i, 2i, 3i, 4i, 5i);
