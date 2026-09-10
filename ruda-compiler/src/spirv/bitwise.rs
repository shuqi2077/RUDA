use ruda_core::ir::{self as core, ElemType, Operator, Bitwise};
use crate::spirv::{SpirvCompiler, SpirvTarget, item::Elem};


impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_bitwise(&mut self, op: Bitwise, out: Option<core::Variable>, uniform: bool) {
        if let Some(op) = bool_op(&op) {
            self.compile_operator(op, out, uniform);
            return;
        }

        let out = out.unwrap();
        match op {
            Bitwise::BitwiseAnd(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_and(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::BitwiseOr(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_or(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::BitwiseXor(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_xor(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::BitwiseNot(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, _, ty, input, out| {
                    b.not(ty, Some(out), input).unwrap();
                });
            }
            Bitwise::ShiftLeft(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.shift_left_logical(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::ShiftRight(op) => {
                self.compile_binary_op(op, out, uniform, |b, item, ty, lhs, rhs, out| {
                    match item.elem() {
                        // Match behaviour on most compilers
                        Elem::Int(_, true) => {
                            b.shift_right_arithmetic(ty, Some(out), lhs, rhs).unwrap()
                        }
                        _ => b.shift_right_logical(ty, Some(out), lhs, rhs).unwrap(),
                    };
                })
            }

            Bitwise::CountOnes(op) => {
                self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                    b.bit_count(ty, Some(out), input).unwrap();
                });
            }
            Bitwise::ReverseBits(op) => {
                self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                    b.bit_reverse(ty, Some(out), input).unwrap();
                });
            }
            Bitwise::LeadingZeros(op) => {
                let width = op.input.ty.storage_type().size() as u32 * 8;
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    // Indices are zero based, so subtract 1
                    let width = out_ty.const_u32(b, width - 1);
                    let msb = b.id();
                    T::find_msb(b, ty, input, msb);
                    b.mark_uniformity(msb, uniform);
                    b.i_sub(ty, Some(out), width, msb).unwrap();
                });
            }
            Bitwise::FindFirstSet(op) => {
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    let one = out_ty.const_u32(b, 1);
                    let lsb = b.id();
                    T::find_lsb(b, ty, input, lsb);
                    b.mark_uniformity(lsb, uniform);
                    // Normalize to CUDA/POSIX convention of 1 based index, with 0 meaning not found
                    b.i_add(ty, Some(out), lsb, one).unwrap();
                });
            }
            Bitwise::TrailingZeros(op) => {
                let width = op.input.ty.storage_type().size() as u32 * 8;
                self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
                    // find_lsb returns -1 (0xFFFFFFFF) for zero input
                    // trailing_zeros should return bit_width for zero input
                    let width_const = out_ty.const_u32(b, width);
                    let zero = out_ty.const_u32(b, 0);
                    let lsb = b.id();
                    T::find_lsb(b, ty, input, lsb);
                    b.mark_uniformity(lsb, uniform);
                    // Check if input is zero
                    let bool_ty = out_ty.same_vectorization(Elem::Bool).id(b);
                    let is_zero = b.id();
                    b.i_equal(bool_ty, Some(is_zero), input, zero).unwrap();
                    b.mark_uniformity(is_zero, uniform);
                    // Select width if zero, otherwise lsb
                    b.select(ty, Some(out), is_zero, width_const, lsb).unwrap();
                });
            }
        }
    }
}

/// Map bitwise on boolean to logical, since bitwise ops aren't allowed in Vulkan. This fixes the
/// case of
/// ```ignore
/// let a = true;
/// for shape in 0..dims {
///     a |= shape < width;
/// }
/// ```
///
/// Rust maps this to logical and/or internally, but the macro only sees the bitwise op.
fn bool_op(bitwise: &Bitwise) -> Option<Operator> {
    match bitwise {
        Bitwise::BitwiseAnd(op)
            if op.lhs.elem_type() == ElemType::Bool || op.rhs.elem_type() == ElemType::Bool =>
        {
            Some(Operator::And(op.clone()))
        }
        Bitwise::BitwiseOr(op)
            if op.lhs.elem_type() == ElemType::Bool || op.rhs.elem_type() == ElemType::Bool =>
        {
            Some(Operator::Or(op.clone()))
        }
        Bitwise::BitwiseNot(op) if op.input.elem_type() == ElemType::Bool => {
            Some(Operator::Not(op.clone()))
        }
        _ => None,
    }
}

