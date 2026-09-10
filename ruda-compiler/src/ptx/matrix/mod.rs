use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{CoopMma, Matrix, MatrixIdent, MatrixLayout, Type, Variable, VariableKind};

mod fragment;
mod memory;
mod execute;
mod registers;
mod manual;
mod indexing;
mod volta;

fn descriptor(variable: Variable) -> Result<Matrix> {
    let VariableKind::Matrix { mat, .. } = variable.kind else {
        return Err(invalid("expected a matrix fragment"));
    };
    if variable.ty != Type::new(mat.storage) { return Err(invalid("matrix fragment storage type mismatch")); }
    Ok(mat)
}

fn shape(matrix: Matrix) -> String {
    format!("m{}n{}k{}", matrix.m, matrix.n, matrix.k)
}

fn layout(matrix: MatrixLayout) -> Result<&'static str> {
    match matrix {
        MatrixLayout::RowMajor => Ok("row"),
        MatrixLayout::ColMajor => Ok("col"),
        MatrixLayout::Undefined => Err(invalid("matrix operation requires an explicit layout")),
    }
}

fn same_shape(a: Matrix, b: Matrix) -> bool {
    (a.m, a.n, a.k) == (b.m, b.n, b.k)
}

impl Emitter {
    pub fn matrix(&mut self, operation: CoopMma, out: Variable) -> Result<()> {
        match operation {
            CoopMma::Fill { value } => self.matrix_fill(out, value),
            CoopMma::Load { value, stride, offset, layout } => self.matrix_load(out, value, offset, stride, layout),
            CoopMma::Store { mat, stride, offset, layout } => self.matrix_store(out, mat, offset, stride, layout),
            CoopMma::Execute { mat_a, mat_b, mat_c } => self.matrix_execute(out, mat_a, mat_b, mat_c),
            CoopMma::Cast { input } => self.matrix_cast(out, input),
            CoopMma::LoadMatrix { buffer, offset, vector_size, factor, transpose } => self.matrix_transfer(out, buffer, offset, vector_size, factor, transpose, false),
            CoopMma::StoreMatrix { offset, vector_size, registers, factor, transpose } => self.matrix_transfer(registers, out, offset, vector_size, factor, transpose, true),
            CoopMma::ExecuteManual { matrix, registers_a, registers_b, registers_c } => self.matrix_manual(out, matrix, registers_a, registers_b, registers_c),
            CoopMma::RowIndex { lane_id, i, matrix } => self.matrix_index(out, lane_id, i, matrix, true),
            CoopMma::ColIndex { lane_id, i, matrix } => self.matrix_index(out, lane_id, i, matrix, false),
            other => Err(unsupported(format!("matrix operation {other:?}"))),
        }
    }
}
