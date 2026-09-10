use ruda_core::ir::{self, Builtin, ElemType, UIntKind};
use rspirv::spirv::{BuiltIn, Word};

use crate::spirv::{
    SpirvCompiler, SpirvTarget,
    item::{Elem, Item},
    variable::Variable,
};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_builtin(&mut self, builtin: Builtin, ty: Item) -> Variable {
        match builtin {
            Builtin::UnitPos => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.load_builtin(BuiltIn::LocalInvocationIndex, &ty);
                    b.debug_name(id, "UNIT_POS");
                    id
                }),
                ty,
            ),
            Builtin::UnitPosX => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::LocalInvocationId, 0, &ty);
                    b.debug_name(id, "UNIT_POS_X");
                    id
                }),
                ty,
            ),
            Builtin::UnitPosY => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::LocalInvocationId, 1, &ty);
                    b.debug_name(id, "UNIT_POS_Y");
                    id
                }),
                ty,
            ),
            Builtin::UnitPosZ => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::LocalInvocationId, 2, &ty);
                    b.debug_name(id, "UNIT_POS_Z");
                    id
                }),
                ty,
            ),
            Builtin::RudaPosX => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::WorkgroupId, 0, &ty);
                    b.debug_name(id, "RUDA_POS_X");
                    id
                }),
                ty,
            ),
            Builtin::RudaPosY => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::WorkgroupId, 1, &ty);
                    b.debug_name(id, "RUDA_POS_Y");
                    id
                }),
                ty,
            ),
            Builtin::RudaPosZ => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::WorkgroupId, 2, &ty);
                    b.debug_name(id, "RUDA_POS_Z");
                    id
                }),
                ty,
            ),
            Builtin::RudaPosCluster
            | Builtin::RudaPosClusterX
            | Builtin::RudaPosClusterY
            | Builtin::RudaPosClusterZ => self.constant_var(0, ty),
            Builtin::RudaDim => Variable::Builtin(self.state.ruda_size, ty),
            Builtin::RudaDimX => Variable::Builtin(self.state.ruda_dims[0], ty),
            Builtin::RudaDimY => Variable::Builtin(self.state.ruda_dims[1], ty),
            Builtin::RudaDimZ => Variable::Builtin(self.state.ruda_dims[2], ty),
            Builtin::ClusterDim
            | Builtin::ClusterDimX
            | Builtin::ClusterDimY
            | Builtin::ClusterDimZ => self.constant_var(1, ty),
            Builtin::RudaCount => Variable::Builtin(
                self.insert_global(builtin, |b: &mut SpirvCompiler<T>| {
                    let ty_id = ty.id(b);
                    let x = b.compile_variable(builtin_u32(Builtin::RudaCountX)).id(b);
                    let y = b.compile_variable(builtin_u32(Builtin::RudaCountY)).id(b);
                    let z = b.compile_variable(builtin_u32(Builtin::RudaCountZ)).id(b);

                    let x = Item::builtin_u32().cast_to(b, None, x, &ty);
                    let y = Item::builtin_u32().cast_to(b, None, y, &ty);
                    let z = Item::builtin_u32().cast_to(b, None, z, &ty);

                    let count = b.i_mul(ty_id, None, x, y).unwrap();
                    let count = b.i_mul(ty_id, None, count, z).unwrap();
                    b.debug_name(count, "RUDA_COUNT");
                    count
                }),
                ty,
            ),
            Builtin::RudaCountX => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::NumWorkgroups, 0, &ty);
                    b.debug_name(id, "RUDA_COUNT_X");
                    id
                }),
                ty,
            ),
            Builtin::RudaCountY => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::NumWorkgroups, 1, &ty);
                    b.debug_name(id, "RUDA_COUNT_Y");
                    id
                }),
                ty,
            ),
            Builtin::RudaCountZ => Variable::Builtin(
                self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::NumWorkgroups, 2, &ty);
                    b.debug_name(id, "RUDA_COUNT_Z");
                    id
                }),
                ty,
            ),
            Builtin::PlaneDim => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.load_builtin(BuiltIn::SubgroupSize, &ty);
                    b.debug_name(id, "PLANE_DIM");
                    id
                });
                Variable::Builtin(id, ty)
            }
            Builtin::PlanePos => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.load_builtin(BuiltIn::SubgroupId, &ty);
                    b.debug_name(id, "PLANE_POS");
                    id
                });
                Variable::Builtin(id, ty)
            }
            Builtin::PlaneCount => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.load_builtin(BuiltIn::NumSubgroups, &ty);
                    b.debug_name(id, "PLANE_COUNT");
                    id
                });
                Variable::Builtin(id, ty)
            }
            Builtin::UnitPosPlane => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.load_builtin(BuiltIn::SubgroupLocalInvocationId, &ty);
                    b.debug_name(id, "UNIT_POS_PLANE");
                    id
                });
                Variable::Builtin(id, ty)
            }
            Builtin::RudaPos => {
                let id = self.insert_global(builtin, |b| {
                    let x = b.compile_variable(builtin_u32(Builtin::RudaPosX)).id(b);
                    let y = b.compile_variable(builtin_u32(Builtin::RudaPosY)).id(b);
                    let z = b.compile_variable(builtin_u32(Builtin::RudaPosZ)).id(b);

                    let x = Item::builtin_u32().cast_to(b, None, x, &ty);
                    let y = Item::builtin_u32().cast_to(b, None, y, &ty);
                    let z = Item::builtin_u32().cast_to(b, None, z, &ty);

                    let groups_x = b.compile_variable(builtin_u32(Builtin::RudaCountX)).id(b);
                    let groups_y = b.compile_variable(builtin_u32(Builtin::RudaCountY)).id(b);

                    let groups_x = Item::builtin_u32().cast_to(b, None, groups_x, &ty);
                    let groups_y = Item::builtin_u32().cast_to(b, None, groups_y, &ty);

                    let ty = ty.id(b);
                    let id = b.i_mul(ty, None, z, groups_y).unwrap();
                    let id = b.i_add(ty, None, id, y).unwrap();
                    let id = b.i_mul(ty, None, id, groups_x).unwrap();
                    let id = b.i_add(ty, None, id, x).unwrap();
                    b.debug_name(id, "RUDA_POS");
                    id
                });
                Variable::Builtin(id, ty)
            }
            Builtin::AbsolutePos => {
                let id = self.insert_global(builtin, |b| {
                    let x = b.compile_variable(builtin_u32(Builtin::AbsolutePosX)).id(b);
                    let y = b.compile_variable(builtin_u32(Builtin::AbsolutePosY)).id(b);
                    let z = b.compile_variable(builtin_u32(Builtin::AbsolutePosZ)).id(b);

                    let x = Item::builtin_u32().cast_to(b, None, x, &ty);
                    let y = Item::builtin_u32().cast_to(b, None, y, &ty);
                    let z = Item::builtin_u32().cast_to(b, None, z, &ty);

                    let groups_x = b.compile_variable(builtin_u32(Builtin::RudaCountX)).id(b);
                    let groups_y = b.compile_variable(builtin_u32(Builtin::RudaCountY)).id(b);

                    let groups_x = Item::builtin_u32().cast_to(b, None, groups_x, &ty);
                    let groups_y = Item::builtin_u32().cast_to(b, None, groups_y, &ty);

                    let size_x = ty.const_u32(b, b.ruda_dim.x);
                    let size_y = ty.const_u32(b, b.ruda_dim.y);

                    let ty = ty.id(b);
                    let size_x = b.i_mul(ty, None, groups_x, size_x).unwrap();
                    let size_y = b.i_mul(ty, None, groups_y, size_y).unwrap();
                    let id = b.i_mul(ty, None, z, size_y).unwrap();
                    let id = b.i_add(ty, None, id, y).unwrap();
                    let id = b.i_mul(ty, None, id, size_x).unwrap();
                    let id = b.i_add(ty, None, id, x).unwrap();
                    b.debug_name(id, "ABSOLUTE_POS");
                    id
                });
                Variable::Builtin(id, ty)
            }
            Builtin::AbsolutePosX => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::GlobalInvocationId, 0, &ty);
                    b.debug_name(id, "ABSOLUTE_POS_X");
                    id
                });

                Variable::Builtin(id, ty)
            }
            Builtin::AbsolutePosY => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::GlobalInvocationId, 1, &ty);
                    b.debug_name(id, "ABSOLUTE_POS_Y");
                    id
                });

                Variable::Builtin(id, ty)
            }
            Builtin::AbsolutePosZ => {
                let id = self.insert_global(builtin, |b| {
                    let id = b.extract(BuiltIn::GlobalInvocationId, 2, &ty);
                    b.debug_name(id, "ABSOLUTE_POS_Z");
                    id
                });

                Variable::Builtin(id, ty)
            }
        }
    }

    fn constant_var(&mut self, value: u32, ty: Item) -> Variable {
        let id = ty.const_u32(self, value);
        Variable::Builtin(id, ty.clone())
    }

    fn extract(&mut self, builtin: BuiltIn, idx: u32, ty: &Item) -> Word {
        let composite_id = self.vec_global(builtin);
        let ty = ty.id(self);
        self.composite_extract(ty, None, composite_id, vec![idx])
            .unwrap()
    }

    fn vec_global(&mut self, builtin: BuiltIn) -> Word {
        let item = Item::Vector(Elem::Int(32, false), 3);

        self.insert_builtin(builtin, |b| b.load_builtin(builtin, &item))
    }

    fn load_builtin(&mut self, builtin: BuiltIn, item: &Item) -> Word {
        let item_id = item.id(self);
        let id = self.builtin(builtin, item.clone());
        self.load(item_id, None, id, None, vec![]).unwrap()
    }
}

fn builtin_u32(builtin: Builtin) -> ir::Variable {
    ir::Variable::builtin(builtin, ElemType::UInt(UIntKind::U32).into())
}
