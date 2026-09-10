use crate::ir::DeviceProperties;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[cfg_attr(compiler_std_io, derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
/// The number of units across all 3 axis totalling to the number of working units in a ruda.
pub struct RudaDim {
    /// The number of units in the x axis.
    pub x: u32,
    /// The number of units in the y axis.
    pub y: u32,
    /// The number of units in the z axis.
    pub z: u32,
}

impl RudaDim {
    /// Creates a new [`RudaDim`] based on the maximum number of tasks that can be parellalized by units, in other words,
    /// by the maximum number of working units.
    ///
    /// # Notes
    ///
    /// For complex problems, you probably want to have your own logic function to create the
    /// [`RudaDim`], but for simpler problems such as elemwise-operation, this is a great default.
    pub fn new(properties: &DeviceProperties, working_units: usize) -> Self {
        let plane_size = properties.hardware.plane_size_max;
        let plane_count = Self::calculate_plane_count_per_ruda(
            working_units as u32,
            plane_size,
            properties.hardware.num_cpu_cores,
        );

        // Make sure it respects the max units per ruda (especially on wasm).
        let unit_limit = properties.hardware.max_units_per_ruda / plane_size;

        // A runtime may expose only one-dimensional rudas. In that case, place its planes on X
        // instead of constructing a Y dimension that exceeds the advertised hardware topology.
        if properties.hardware.max_ruda_dim.1 == 1 {
            let x_limit = properties.hardware.max_ruda_dim.0 / plane_size;
            let planes = plane_count.min(unit_limit).min(x_limit).max(1);
            Self::new_1d(plane_size * planes)
        } else {
            // Ensure at least 1 plane so RudaDim is always valid (num_elems() > 0).
            Self::new_2d(plane_size, u32::min(unit_limit, plane_count).max(1))
        }
    }

    fn calculate_plane_count_per_ruda(
        working_units: u32,
        plane_dim: u32,
        num_cpu_cores: Option<u32>,
    ) -> u32 {
        match num_cpu_cores {
            Some(num_cores) => core::cmp::min(num_cores, working_units),
            None => {
                let plane_count_max = core::cmp::max(1, working_units / plane_dim);

                // Ensures `plane_count` is a power of 2.
                const NUM_PLANE_MAX: u32 = 8u32;
                const NUM_PLANE_MAX_LOG2: u32 = NUM_PLANE_MAX.ilog2();
                let plane_count_max_log2 =
                    core::cmp::min(NUM_PLANE_MAX_LOG2, u32::ilog2(plane_count_max));
                2u32.pow(plane_count_max_log2)
            }
        }
    }

    /// Create a new ruda dim with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new ruda dim with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new ruda dim with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new ruda dim with the given x, y and z.
    /// This is equivalent to the [new](RudaDim::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per ruda
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Whether this `RudaDim` can fully contain `other`
    pub const fn can_contain(&self, other: RudaDim) -> bool {
        self.x >= other.x && self.y >= other.y && self.z >= other.z
    }
}

impl From<(u32, u32, u32)> for RudaDim {
    fn from(value: (u32, u32, u32)) -> Self {
        RudaDim::new_3d(value.0, value.1, value.2)
    }
}

impl From<RudaDim> for (u32, u32, u32) {
    fn from(val: RudaDim) -> Self {
        (val.x, val.y, val.z)
    }
}

/// The kind of execution to be performed.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
#[cfg_attr(compiler_std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Validate OOB and alert if OOB access occurs
    Validate,
    /// Unchecked kernels are unsafe.
    Unchecked,
}
