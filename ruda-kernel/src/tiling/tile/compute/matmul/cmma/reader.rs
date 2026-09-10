use std::marker::PhantomData;

use crate::dsl::prelude::*;

use crate::tiling::tile::data::{Filled, Strided, StridedTile, TileKind};

/// Generic CMMA reader over any tile type
#[ruda]
pub trait CmmaFragmentReader {
    type TileKind: TileKind;

    /// Fill a fragment with data, with the implementation depending on the tile kind.
    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &<Self::TileKind as TileKind>::Tile<V, N>,
        fragment: &mut cmma::Matrix<E>,
        layout: ComptimeOption<cmma::MatrixLayout>,
    );
}

/// Reader using the cmma load/fill functions. Tile kind determines implementation.
#[derive(RudaType)]
pub struct CmmaStageReader<Kind: TileKind> {
    #[ruda(comptime)]
    _ty: PhantomData<Kind>,
}

#[ruda]
impl CmmaFragmentReader for CmmaStageReader<Strided> {
    type TileKind = Strided;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &StridedTile<V, N>,
        fragment: &mut cmma::Matrix<E>,
        layout: ComptimeOption<cmma::MatrixLayout>,
    ) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        #[comptime]
        match layout {
            ComptimeOption::None => cmma::load(fragment, &slice, stride),
            ComptimeOption::Some(layout) => {
                cmma::load_with_layout(fragment, &slice, stride, layout)
            }
        }
    }
}

#[ruda]
impl CmmaFragmentReader for CmmaStageReader<Filled> {
    type TileKind = Filled;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        value: &V,
        fragment: &mut cmma::Matrix<E>,
        _layout: ComptimeOption<cmma::MatrixLayout>,
    ) {
        cmma::fill(fragment, E::cast_from(*value));
    }
}

#[ruda]
impl<Inner: TileKind> CmmaFragmentReader for CmmaStageReader<Option<Inner>>
where
    CmmaStageReader<Inner>: CmmaFragmentReader<TileKind = Inner>,
{
    type TileKind = Option<Inner>;

    fn load_fragment<E: Numeric, V: Numeric, N: Size>(
        tile: &ComptimeOption<Inner::Tile<V, N>>,
        fragment: &mut cmma::Matrix<E>,
        layout: ComptimeOption<cmma::MatrixLayout>,
    ) {
        #[comptime]
        #[comptime]
        match tile {
            ComptimeOption::Some(tile) => {
                CmmaStageReader::<Inner>::load_fragment(tile, fragment, layout)
            }
            ComptimeOption::None => CmmaStageReader::<Filled>::load_fragment::<E, V, N>(
                &V::from_int(0),
                fragment,
                layout,
            ),
        }
    }
}
