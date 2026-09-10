use super::*;
use std::ffi::CStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    F64(f64),
    I64(i64),
    U64(u64),
    Bool(bool),
}

type BinaryPlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclTensor,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type UnaryPlan = unsafe extern "C" fn(
    *const AclTensor,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type MatmulPlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclTensor,
    *mut AclTensor,
    i8,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type AddPlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclTensor,
    *const AclScalar,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type SoftmaxPlan = unsafe extern "C" fn(
    *const AclTensor,
    i64,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type RmsPlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclTensor,
    f64,
    *const AclTensor,
    *const AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type PermutePlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclIntArray,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;
type SumPlan = unsafe extern "C" fn(
    *const AclTensor,
    *const AclIntArray,
    bool,
    i32,
    *mut AclTensor,
    *mut u64,
    *mut *mut AclOpExecutor,
) -> Status;

struct Scalar {
    session: Rc<CannSession>,
    handle: NonNull<AclScalar>,
    destroy: unsafe extern "C" fn(*const AclScalar) -> Status,
}
impl Drop for Scalar {
    fn drop(&mut self) {
        if self.session.releasable() {
            // SAFETY: the scalar is unique and no executor is using it.
            let _ = unsafe { (self.destroy)(self.handle.as_ptr()) };
        } else {
            std::mem::forget(self.session.clone());
        }
    }
}
struct IntArray {
    session: Rc<CannSession>,
    handle: NonNull<AclIntArray>,
    destroy: unsafe extern "C" fn(*const AclIntArray) -> Status,
}
impl Drop for IntArray {
    fn drop(&mut self) {
        if self.session.releasable() {
            // SAFETY: the array is unique and no executor is using it.
            let _ = unsafe { (self.destroy)(self.handle.as_ptr()) };
        } else {
            std::mem::forget(self.session.clone());
        }
    }
}

impl CannSession {
    fn int_array(self: &Rc<Self>, values: &[i64]) -> Result<IntArray, CannError> {
        type Create = unsafe extern "C" fn(*const i64, u64) -> *mut AclIntArray;
        // SAFETY: exact SDK signatures, valid host data for the creation call.
        unsafe {
            let destroy = self.ops.get(c"aclDestroyIntArray")?;
            let create: Create = self.ops.get(c"aclCreateIntArray")?;
            let handle = NonNull::new(create(values.as_ptr(), values.len() as u64))
                .ok_or(CannError::NullHandle("aclCreateIntArray"))?;
            Ok(IntArray {
                session: self.clone(),
                handle,
                destroy,
            })
        }
    }

    fn scalar(self: &Rc<Self>, value: &mut ScalarValue) -> Result<Scalar, CannError> {
        type Create = unsafe extern "C" fn(*mut c_void, i32) -> *mut AclScalar;
        // SAFETY: exact SDK signatures; the host scalar is copied by aclCreateScalar.
        unsafe {
            let destroy = self.ops.get(c"aclDestroyScalar")?;
            let create: Create = self.ops.get(c"aclCreateScalar")?;
            let (pointer, dtype) = match value {
                ScalarValue::F64(v) => ((v as *mut f64).cast(), DType::F64),
                ScalarValue::I64(v) => ((v as *mut i64).cast(), DType::I64),
                ScalarValue::U64(v) => ((v as *mut u64).cast(), DType::U64),
                ScalarValue::Bool(v) => ((v as *mut bool).cast(), DType::Bool),
            };
            let handle = NonNull::new(create(pointer, dtype as i32))
                .ok_or(CannError::NullHandle("aclCreateScalar"))?;
            Ok(Scalar {
                session: self.clone(),
                handle,
                destroy,
            })
        }
    }

    /// ND matmul including vector and batch broadcasting. Output dtype and Cube
    /// math policy are explicit; no precision-reducing mode is selected implicitly.
    pub fn matmul(
        self: &Rc<Self>,
        a: &CannTensor,
        b: &CannTensor,
        dtype: DType,
        cube_math_type: i8,
    ) -> Result<CannTensor, CannError> {
        self.same_session(&[a, b])?;
        let shape = layout::matmul_shape(a.layout.shape(), b.layout.shape())?;
        let output = self.allocate_tensor(&shape, dtype)?;
        // SAFETY: checked real tensors; vendor validation handles dtype/hardware restrictions.
        unsafe {
            let plan: MatmulPlan = self.ops.get(c"aclnnMatmulGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnMatmul")?;
            self.execute("aclnnMatmul", run, |size, exec| {
                plan(
                    a.descriptor.as_ptr(),
                    b.descriptor.as_ptr(),
                    output.descriptor.as_ptr(),
                    cube_math_type,
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }

    pub fn mul(
        self: &Rc<Self>,
        a: &CannTensor,
        b: &CannTensor,
        dtype: DType,
    ) -> Result<CannTensor, CannError> {
        self.same_session(&[a, b])?;
        let shape = layout::broadcast(a.layout.shape(), b.layout.shape())?;
        let output = self.allocate_tensor(&shape, dtype)?;
        // SAFETY: exact signatures; all descriptors and storage live through synchronization.
        unsafe {
            let plan: BinaryPlan = self.ops.get(c"aclnnMulGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnMul")?;
            self.execute("aclnnMul", run, |size, exec| {
                plan(
                    a.descriptor.as_ptr(),
                    b.descriptor.as_ptr(),
                    output.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }

    /// Computes a + alpha * b. ACLNN performs the requested output conversion.
    pub fn add(
        self: &Rc<Self>,
        a: &CannTensor,
        b: &CannTensor,
        mut alpha: ScalarValue,
        dtype: DType,
    ) -> Result<CannTensor, CannError> {
        self.same_session(&[a, b])?;
        let shape = layout::broadcast(a.layout.shape(), b.layout.shape())?;
        let output = self.allocate_tensor(&shape, dtype)?;
        let scalar = self.scalar(&mut alpha)?;
        // SAFETY: signatures and scalar storage match the SDK; no borrowed input escapes.
        unsafe {
            let plan: AddPlan = self.ops.get(c"aclnnAddGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnAdd")?;
            self.execute("aclnnAdd", run, |size, exec| {
                plan(
                    a.descriptor.as_ptr(),
                    b.descriptor.as_ptr(),
                    scalar.handle.as_ptr(),
                    output.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }

    fn unary(
        self: &Rc<Self>,
        input: &CannTensor,
        plan_name: &CStr,
        run_name: &CStr,
        operation: &'static str,
    ) -> Result<CannTensor, CannError> {
        self.same_session(&[input])?;
        let output = self.allocate_tensor(input.layout.shape(), input.layout.dtype())?;
        // SAFETY: private call sites only supply functions with UnaryPlan and Run signatures.
        unsafe {
            let plan: UnaryPlan = self.ops.get(plan_name)?;
            let run = self.ops.get(run_name)?;
            self.execute(operation, run, |size, exec| {
                plan(
                    input.descriptor.as_ptr(),
                    output.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }

    pub fn silu(self: &Rc<Self>, input: &CannTensor) -> Result<CannTensor, CannError> {
        self.unary(
            input,
            c"aclnnSiluGetWorkspaceSize",
            c"aclnnSilu",
            "aclnnSilu",
        )
    }

    pub fn softmax(self: &Rc<Self>, input: &CannTensor, dim: i64) -> Result<CannTensor, CannError> {
        self.same_session(&[input])?;
        let dim = layout::axis(dim, input.layout.shape.len())? as i64;
        let output = self.allocate_tensor(input.layout.shape(), input.layout.dtype())?;
        // SAFETY: exact SDK signature; axis and storage are valid.
        unsafe {
            let plan: SoftmaxPlan = self.ops.get(c"aclnnSoftmaxGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnSoftmax")?;
            self.execute("aclnnSoftmax", run, |size, exec| {
                plan(
                    input.descriptor.as_ptr(),
                    dim,
                    output.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }

    /// Returns (normalized output, FP32 reciprocal RMS with normalized axes retained).
    pub fn rms_norm(
        self: &Rc<Self>,
        input: &CannTensor,
        gamma: &CannTensor,
        epsilon: f64,
    ) -> Result<(CannTensor, CannTensor), CannError> {
        self.same_session(&[input, gamma])?;
        let rank = input.layout.shape.len();
        let norm_rank = gamma.layout.shape.len();
        if norm_rank == 0
            || norm_rank > rank
            || input.layout.shape[rank - norm_rank..] != gamma.layout.shape
        {
            return Err(invalid(
                "RMSNorm gamma must match nonempty trailing input dimensions",
            ));
        }
        let mut rstd_shape = input.layout.shape.clone();
        rstd_shape[rank - norm_rank..].fill(1);
        let output = self.allocate_tensor(input.layout.shape(), input.layout.dtype())?;
        let rstd = self.allocate_tensor(&rstd_shape, DType::F32)?;
        // SAFETY: SDK requires const descriptor pointers even for these output buffers.
        unsafe {
            let plan: RmsPlan = self.ops.get(c"aclnnRmsNormGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnRmsNorm")?;
            self.execute("aclnnRmsNorm", run, |size, exec| {
                plan(
                    input.descriptor.as_ptr(),
                    gamma.descriptor.as_ptr(),
                    epsilon,
                    output.descriptor.as_ptr(),
                    rstd.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok((output, rstd))
    }

    pub fn permute(
        self: &Rc<Self>,
        input: &CannTensor,
        dims: &[i64],
    ) -> Result<CannTensor, CannError> {
        self.same_session(&[input])?;
        if dims.len() != input.layout.shape.len() {
            return Err(invalid("permutation rank mismatch"));
        }
        let axes: Vec<usize> = dims
            .iter()
            .map(|&d| layout::axis(d, dims.len()))
            .collect::<Result<_, _>>()?;
        let unique: std::collections::BTreeSet<_> = axes.iter().collect();
        if unique.len() != axes.len() {
            return Err(invalid("duplicate permutation axis"));
        }
        let shape: Vec<i64> = axes.iter().map(|&d| input.layout.shape[d]).collect();
        let array = self.int_array(dims)?;
        let output = self.allocate_tensor(&shape, input.layout.dtype())?;
        // SAFETY: validated permutation and exact signatures; array retained across execution.
        unsafe {
            let plan: PermutePlan = self.ops.get(c"aclnnPermuteGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnPermute")?;
            self.execute("aclnnPermute", run, |size, exec| {
                plan(
                    input.descriptor.as_ptr(),
                    array.handle.as_ptr(),
                    output.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }

    pub fn sum(
        self: &Rc<Self>,
        input: &CannTensor,
        dims: &[i64],
        keep_dims: bool,
        dtype: DType,
    ) -> Result<CannTensor, CannError> {
        self.same_session(&[input])?;
        let rank = input.layout.shape.len();
        let axes: Vec<usize> = if dims.is_empty() {
            (0..rank).collect()
        } else {
            dims.iter()
                .map(|&d| layout::axis(d, rank))
                .collect::<Result<_, _>>()?
        };
        let unique: std::collections::BTreeSet<_> = axes.iter().copied().collect();
        if unique.len() != axes.len() {
            return Err(invalid("duplicate reduction axis"));
        }
        let shape: Vec<i64> = input
            .layout
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| {
                if !unique.contains(&i) {
                    Some(d)
                } else if keep_dims {
                    Some(1)
                } else {
                    None
                }
            })
            .collect();
        let canonical: Vec<i64> = axes.iter().map(|&d| d as i64).collect();
        let array = self.int_array(&canonical)?;
        let output = self.allocate_tensor(&shape, dtype)?;
        // SAFETY: exact SDK signature, including C bool and aclDataType integer representation.
        unsafe {
            let plan: SumPlan = self.ops.get(c"aclnnReduceSumGetWorkspaceSize")?;
            let run = self.ops.get(c"aclnnReduceSum")?;
            self.execute("aclnnReduceSum", run, |size, exec| {
                plan(
                    input.descriptor.as_ptr(),
                    array.handle.as_ptr(),
                    keep_dims,
                    dtype as i32,
                    output.descriptor.as_ptr(),
                    size,
                    exec,
                )
            })?;
        }
        Ok(output)
    }
}
