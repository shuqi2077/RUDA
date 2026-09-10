use std::marker::PhantomData;

use crate::dsl::prelude::*;

use crate::tiling::RudaDimResource;

/// Identifies which compute primitive executes a tile matmul.
pub trait TileScope: Clone + Copy + Send + Sync + 'static {
    /// Compute resource a single instance of this scope occupies.
    fn default_resource() -> RudaDimResource;

    /// Comptime tag used at dispatch sites that need to assert a particular scope
    /// (e.g. variants that only make sense on a plane).
    const KIND: ScopeKind;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ScopeKind {
    Unit,
    Plane,
    Ruda,
}

#[derive(Clone, Copy)]
pub struct Unit;
#[derive(Clone, Copy)]
pub struct Plane;
#[derive(Clone, Copy)]
pub struct Ruda;

impl TileScope for Unit {
    fn default_resource() -> RudaDimResource {
        RudaDimResource::Units(1)
    }
    const KIND: ScopeKind = ScopeKind::Unit;
}
impl TileScope for Plane {
    fn default_resource() -> RudaDimResource {
        RudaDimResource::Planes(1)
    }
    const KIND: ScopeKind = ScopeKind::Plane;
}
impl TileScope for Ruda {
    fn default_resource() -> RudaDimResource {
        unimplemented!("Ruda scope does not have a default ruda-dim resource")
    }
    const KIND: ScopeKind = ScopeKind::Ruda;
}

/// Zero-sized comptime marker used to carry a [Scope] generic through [Tile].
#[derive(RudaType, Clone, Copy)]
pub struct ScopeMarker<Sc: TileScope> {
    #[ruda(comptime)]
    _phantom: PhantomData<Sc>,
}

/// Comptime assertion that a tile-scope generic resolves to `Plane`.
pub fn assert_plane_scope(kind: ScopeKind) {
    match kind {
        ScopeKind::Plane => {}
        _ => panic!("This Tile variant is only valid in Plane scope"),
    }
}

/// Comptime assertion that a tile-scope generic resolves to `Unit`.
pub fn assert_unit_scope(kind: ScopeKind) {
    match kind {
        ScopeKind::Unit => {}
        _ => panic!("This Tile variant is only valid in Unit scope"),
    }
}
