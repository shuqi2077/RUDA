use crate::{launch::RudaDim, ir::Builtin};

const NB_PASSED_BUILTIN: usize = 9;

#[derive(Default, Debug, Clone)]
pub struct BuiltinArray {
    pub dims: [u32; NB_PASSED_BUILTIN],
    //[
    //  ruda_dim_x
    //  ruda_dim_y
    //  ruda_dim_z
    //  ruda_count_x
    //  ruda_count_y
    //  ruda_count_z
    //  unit_pos_x
    //  unit_pos_y
    //  unit_pos_z
    //]
}

impl BuiltinArray {
    pub const fn builtin_order() -> [Builtin; 9] {
        [
            Builtin::RudaDimX,
            Builtin::RudaDimY,
            Builtin::RudaDimZ,
            Builtin::RudaCountX,
            Builtin::RudaCountY,
            Builtin::RudaCountZ,
            Builtin::UnitPosX,
            Builtin::UnitPosY,
            Builtin::UnitPosZ,
        ]
    }
    pub fn set_ruda_dim(&mut self, ruda_dim: RudaDim) {
        self.dims[0] = ruda_dim.x;
        self.dims[1] = ruda_dim.y;
        self.dims[2] = ruda_dim.z;
    }
    pub fn set_ruda_count(&mut self, ruda_count: [u32; 3]) {
        self.dims[3] = ruda_count[0];
        self.dims[4] = ruda_count[1];
        self.dims[5] = ruda_count[2];
    }
    pub fn set_unit_pos(&mut self, unit_pos: [u32; 3]) {
        self.dims[6] = unit_pos[0];
        self.dims[7] = unit_pos[1];
        self.dims[8] = unit_pos[2];
    }
    pub const fn len() -> usize {
        NB_PASSED_BUILTIN
    }
}
