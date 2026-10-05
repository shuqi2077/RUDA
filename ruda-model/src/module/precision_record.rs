use super::{Module, ModuleMapper, ModuleVisitor, Param, ParamId, precision::DtypeMapper};
use crate::record::{PrecisionSettings, Record, RecorderError};
use alloc::{format, vec::Vec};
use hashbrown::{HashMap, HashSet};
use ruda_tensor::{DType, FloatDType, api::Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

/// Floating-parameter storage dtypes, separate from recorder value precision.
///
/// Capture alongside the original module record and apply after loading it.
/// Parameter IDs and trainable/frozen configuration must match that record.
/// Recorder settings still determine saved value precision; this metadata
/// does not recover values narrowed during serialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleDTypeRecord {
    entries: Vec<(u64, bool, DType)>,
}

impl ModuleDTypeRecord {
    /// Capture floating-parameter dtype metadata without reading tensor values.
    pub fn capture<B: Backend, M: Module<B>>(module: &M) -> Result<Self, RecorderError> {
        let mut visitor = Capture {
            entries: HashMap::new(),
            error: None,
        };
        module.visit(&mut visitor);
        if let Some(error) = visitor.error {
            return Err(error);
        }
        let mut entries: Vec<_> = visitor
            .entries
            .into_iter()
            .map(|((id, trainable), dtype)| (id.val(), trainable, dtype))
            .collect();
        entries.sort_by_key(|(id, trainable, _)| (*id, *trainable));
        Ok(Self { entries })
    }

    /// Restore per-parameter storage dtypes using the shared leaf-preserving mapper.
    ///
    /// Tied parameters keep one converted leaf, frozen aliases remain independent,
    /// and already-recorded quantized tensors are checked rather than requantized.
    /// Integer and Bool parameters are not modified.
    pub fn apply<B: Backend, M: Module<B>>(self, module: M) -> Result<M, RecorderError> {
        let mut targets = HashMap::new();
        for (id, trainable, dtype) in self.entries {
            if !matches!(
                dtype,
                DType::F64
                    | DType::F32
                    | DType::Flex32
                    | DType::F16
                    | DType::BF16
                    | DType::QFloat(_)
            ) {
                return Err(invalid("non-floating dtype in module dtype record"));
            }
            if targets
                .insert((ParamId::from(id), trainable), dtype)
                .is_some()
            {
                return Err(invalid("duplicate parameter in module dtype record"));
            }
        }
        let mut mapper = Apply {
            targets,
            seen: HashSet::new(),
            mappers: HashMap::new(),
            error: None,
        };
        let module = module.map(&mut mapper);
        if let Some(error) = mapper.error {
            return Err(error);
        }
        if mapper.seen.len() != mapper.targets.len() {
            return Err(invalid("module dtype record includes an unknown parameter"));
        }
        Ok(module)
    }
}

impl<B: Backend> Record<B> for ModuleDTypeRecord {
    type Item<P: PrecisionSettings> = Self;
    fn into_item<P: PrecisionSettings>(self) -> Self {
        self
    }
    fn from_item<P: PrecisionSettings>(item: Self, _device: &B::Device) -> Self {
        item
    }
}

fn invalid(reason: &str) -> RecorderError {
    RecorderError::Unknown(format!("Invalid module dtype record: {reason}"))
}

struct Capture {
    entries: HashMap<(ParamId, bool), DType>,
    error: Option<RecorderError>,
}
impl<B: Backend> ModuleVisitor<B> for Capture {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        let key = (param.id, tensor.is_require_grad());
        if let Some(previous) = self.entries.insert(key, tensor.dtype())
            && previous != tensor.dtype()
        {
            self.error = Some(invalid("tied parameter has inconsistent storage dtypes"));
        }
    }
}

struct Apply {
    targets: HashMap<(ParamId, bool), DType>,
    seen: HashSet<(ParamId, bool)>,
    mappers: HashMap<FloatDType, DtypeMapper>,
    error: Option<RecorderError>,
}
impl<B: Backend> ModuleMapper<B> for Apply {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let key = (param.id, param.val().is_require_grad());
        let Some(&dtype) = self.targets.get(&key) else {
            self.error = Some(invalid("parameter ID or trainable configuration differs"));
            return param;
        };
        self.seen.insert(key);
        if matches!(dtype, DType::QFloat(_)) {
            if param.val().dtype() != dtype {
                self.error = Some(invalid("quantized parameter scheme differs"));
            }
            return param;
        }
        let dtype = FloatDType::from(dtype);
        let mapper = self
            .mappers
            .entry(dtype)
            .or_insert_with(|| DtypeMapper::new(dtype));
        <DtypeMapper as ModuleMapper<B>>::map_float(mapper, param)
    }
}
