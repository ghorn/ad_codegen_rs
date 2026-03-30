use std::collections::HashMap;

use anyhow::{Result as AnyResult, bail};
use approx::assert_abs_diff_eq;
use rstest::rstest;
use sx_codegen::lower_function;
use sx_codegen_llvm::{CompiledJitFunction, JitOptimizationLevel};
use sx_core::{CCS, NamedMatrix, NodeView, SX, SXFunction, SXMatrix, SxError};

#[path = "../../test_support/symbolic_eval.rs"]
mod symbolic_eval;

use symbolic_eval::eval;

fn eval_matrix(matrix: &SXMatrix, vars: &HashMap<u32, f64>) -> Vec<Vec<f64>> {
    let (nrow, ncol) = matrix.shape();
    let mut dense = vec![vec![0.0; ncol]; nrow];
    for col in 0..ncol {
        for (row, row_values) in dense.iter_mut().enumerate().take(nrow) {
            row_values[col] = eval(matrix.get(row, col), vars);
        }
    }
    dense
}

#[derive(Clone, Copy)]
enum MatrixBinaryOp {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    Hypot,
    MtimesRhsTranspose,
}

fn apply_matrix_binary_op(lhs: SX, rhs: SX, op: MatrixBinaryOp) -> SX {
    match op {
        MatrixBinaryOp::Add => lhs + rhs,
        MatrixBinaryOp::Sub => lhs - rhs,
        MatrixBinaryOp::Mul => lhs * rhs,
        MatrixBinaryOp::Max => lhs.max(rhs),
        MatrixBinaryOp::Min => lhs.min(rhs),
        MatrixBinaryOp::Hypot => lhs.hypot(rhs),
        MatrixBinaryOp::MtimesRhsTranspose => unreachable!("handled separately"),
    }
}

fn dense_matrix_binary_adaptor(lhs: &SXMatrix, rhs: &SXMatrix, op: MatrixBinaryOp) -> SXMatrix {
    match op {
        MatrixBinaryOp::MtimesRhsTranspose => {
            let (lhs_rows, lhs_cols) = lhs.shape();
            let (rhs_rows, rhs_cols) = rhs.shape();
            assert_eq!(lhs_cols, rhs_cols);
            let mut values = Vec::with_capacity(lhs_rows * rhs_rows);
            for col in 0..rhs_rows {
                for row in 0..lhs_rows {
                    let mut sum = SX::zero();
                    for k in 0..lhs_cols {
                        sum += lhs.get(row, k) * rhs.get(col, k);
                    }
                    values.push(sum);
                }
            }
            SXMatrix::dense(lhs_rows, rhs_rows, values).unwrap()
        }
        _ => {
            assert_eq!(lhs.shape(), rhs.shape());
            let (nrow, ncol) = lhs.shape();
            let mut values = Vec::with_capacity(nrow * ncol);
            for col in 0..ncol {
                for row in 0..nrow {
                    values.push(apply_matrix_binary_op(
                        lhs.get(row, col),
                        rhs.get(row, col),
                        op,
                    ));
                }
            }
            SXMatrix::dense(nrow, ncol, values).unwrap()
        }
    }
}

fn dense_matrix_multiply_adaptor(lhs: &SXMatrix, rhs: &SXMatrix) -> sx_core::Result<SXMatrix> {
    let (lhs_rows, lhs_cols) = lhs.shape();
    let (rhs_rows, rhs_cols) = rhs.shape();
    if lhs_cols != rhs_rows {
        return Err(sx_core::SxError::Shape(format!(
            "matrix multiply shape mismatch: {}x{} times {}x{}",
            lhs_rows, lhs_cols, rhs_rows, rhs_cols
        )));
    }
    let mut values = Vec::with_capacity(lhs_rows * rhs_cols);
    for col in 0..rhs_cols {
        for row in 0..lhs_rows {
            let mut sum = SX::zero();
            for k in 0..lhs_cols {
                sum += lhs.get(row, k) * rhs.get(k, col);
            }
            values.push(sum);
        }
    }
    SXMatrix::dense(lhs_rows, rhs_cols, values)
}

fn numeric_dense_matrix_binary(
    lhs: &[Vec<f64>],
    rhs: &[Vec<f64>],
    op: MatrixBinaryOp,
) -> Vec<Vec<f64>> {
    match op {
        MatrixBinaryOp::MtimesRhsTranspose => {
            let lhs_rows = lhs.len();
            let rhs_rows = rhs.len();
            let cols = lhs[0].len();
            let mut out = vec![vec![0.0; rhs_rows]; lhs_rows];
            for row in 0..lhs_rows {
                for col in 0..rhs_rows {
                    out[row][col] = (0..cols).map(|k| lhs[row][k] * rhs[col][k]).sum();
                }
            }
            out
        }
        _ => lhs
            .iter()
            .zip(rhs)
            .map(|(lhs_row, rhs_row)| {
                lhs_row
                    .iter()
                    .zip(rhs_row)
                    .map(|(&lhs_value, &rhs_value)| match op {
                        MatrixBinaryOp::Add => lhs_value + rhs_value,
                        MatrixBinaryOp::Sub => lhs_value - rhs_value,
                        MatrixBinaryOp::Mul => lhs_value * rhs_value,
                        MatrixBinaryOp::Max => lhs_value.max(rhs_value),
                        MatrixBinaryOp::Min => lhs_value.min(rhs_value),
                        MatrixBinaryOp::Hypot => lhs_value.hypot(rhs_value),
                        MatrixBinaryOp::MtimesRhsTranspose => unreachable!(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    }
}

fn assert_dense_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_dense_matrix_close_with_epsilon(actual, expected, 1e-9);
}

fn assert_dense_matrix_close_with_epsilon(
    actual: &[Vec<f64>],
    expected: &[Vec<f64>],
    epsilon: f64,
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.first().map(Vec::len), expected.first().map(Vec::len));
    for (actual_row, expected_row) in actual.iter().zip(expected) {
        for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
            assert_abs_diff_eq!(actual_value, expected_value, epsilon = epsilon);
        }
    }
}

fn dense_matrix_bindings(matrix: &SXMatrix, values: &[Vec<f64>]) -> HashMap<u32, f64> {
    let (nrow, ncol) = matrix.shape();
    let mut bindings = HashMap::new();
    for col in 0..ncol {
        for (row, row_values) in values.iter().enumerate().take(nrow) {
            let expr = matrix.get(row, col);
            if !expr.is_zero() {
                bindings.insert(expr.id(), row_values[col]);
            }
        }
    }
    bindings
}

fn dense_matrix_column_major(values: &[Vec<f64>]) -> Vec<f64> {
    let nrow = values.len();
    let ncol = values.first().map_or(0, Vec::len);
    let mut flattened = Vec::with_capacity(nrow * ncol);
    for col in 0..ncol {
        for row_values in values.iter().take(nrow) {
            flattened.push(row_values[col]);
        }
    }
    flattened
}

fn add_scaled(lhs: &[Vec<f64>], rhs: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs_row, rhs_row)| {
            lhs_row
                .iter()
                .zip(rhs_row)
                .map(|(&lhs_value, &rhs_value)| lhs_value + scale * rhs_value)
                .collect::<Vec<_>>()
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
struct DenseValue {
    nrow: usize,
    ncol: usize,
    values: Vec<f64>,
}

impl DenseValue {
    fn scalar(value: f64) -> Self {
        Self {
            nrow: 1,
            ncol: 1,
            values: vec![value],
        }
    }

    fn zeros(nrow: usize, ncol: usize) -> Self {
        Self {
            nrow,
            ncol,
            values: vec![0.0; nrow * ncol],
        }
    }

    fn empty(nrow: usize, ncol: usize) -> Self {
        Self {
            nrow,
            ncol,
            values: Vec::new(),
        }
    }

    fn from_rows(rows: &[&[f64]]) -> Self {
        let nrow = rows.len();
        let ncol = rows.first().map_or(0, |row| row.len());
        let mut values = Vec::with_capacity(nrow * ncol);
        for col in 0..ncol {
            for row in rows {
                values.push(row[col]);
            }
        }
        Self { nrow, ncol, values }
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    fn to_rows(&self) -> Vec<Vec<f64>> {
        let mut rows = vec![vec![0.0; self.ncol]; self.nrow];
        for col in 0..self.ncol {
            for (row, row_values) in rows.iter_mut().enumerate().take(self.nrow) {
                row_values[col] = self.values[row + col * self.nrow];
            }
        }
        rows
    }
}

fn assert_dense_value_close(actual: &DenseValue, expected: &DenseValue) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual_value, expected_value) in actual.values.iter().zip(&expected.values) {
        assert_abs_diff_eq!(actual_value, expected_value, epsilon = 1e-9);
    }
}

fn dense_value_to_slot(ccs: &CCS, dense: &DenseValue) -> AnyResult<Vec<f64>> {
    let expected_len = ccs.nrow() * ccs.ncol();
    if dense.shape() != (ccs.nrow(), ccs.ncol()) && dense.values.len() != expected_len {
        bail!(
            "input shape mismatch: expected {}x{} ({} values), got {}x{} ({} values)",
            ccs.nrow(),
            ccs.ncol(),
            expected_len,
            dense.nrow,
            dense.ncol,
            dense.values.len()
        );
    }
    Ok(ccs
        .positions()
        .into_iter()
        .map(|(row, col)| dense.values[row + col * ccs.nrow()])
        .collect())
}

fn dense_value_from_slot(ccs: &CCS, values: &[f64]) -> DenseValue {
    let mut dense = DenseValue::zeros(ccs.nrow(), ccs.ncol());
    for ((row, col), value) in ccs.positions().into_iter().zip(values.iter().copied()) {
        dense.values[row + col * ccs.nrow()] = value;
    }
    dense
}

fn jit_eval_function(function: &SXFunction, inputs: &[DenseValue]) -> AnyResult<Vec<DenseValue>> {
    if inputs.len() != function.n_in() {
        bail!(
            "input arity mismatch: expected {}, got {}",
            function.n_in(),
            inputs.len()
        );
    }

    let lowered = lower_function(function)?;
    let compiled = CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O0)?;
    let mut context = compiled.create_context();

    for (slot_idx, (slot, dense)) in function.inputs().iter().zip(inputs).enumerate() {
        let projected = dense_value_to_slot(slot.matrix().ccs(), dense)?;
        let input = context.input_mut(slot_idx);
        if input.len() != projected.len() {
            bail!(
                "input nnz mismatch at slot {}: expected {}, got {}",
                slot_idx,
                input.len(),
                projected.len()
            );
        }
        input.copy_from_slice(&projected);
    }

    compiled.eval(&mut context);

    Ok(lowered
        .outputs
        .iter()
        .enumerate()
        .map(|(slot_idx, slot)| dense_value_from_slot(&slot.ccs, context.output(slot_idx)))
        .collect())
}

fn named_matrix(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).unwrap()
}

fn dense_vcat_adaptor(parts: &[SXMatrix]) -> sx_core::Result<SXMatrix> {
    let Some(first) = parts.first() else {
        return Ok(SXMatrix::new(CCS::empty(0, 0), Vec::new()).unwrap());
    };
    let ncol = first.shape().1;
    let total_rows = parts.iter().map(|part| part.shape().0).sum();
    if parts.iter().any(|part| part.shape().1 != ncol) {
        return Err(SxError::Shape(
            "vertical concatenation requires a common column count".into(),
        ));
    }
    let mut values = Vec::with_capacity(total_rows * ncol);
    for col in 0..ncol {
        for part in parts {
            for row in 0..part.shape().0 {
                values.push(part.get(row, col));
            }
        }
    }
    SXMatrix::dense(total_rows, ncol, values)
}

fn dense_pattern_from_ccs(ccs: &CCS) -> DenseValue {
    dense_value_from_slot(ccs, &vec![1.0; ccs.nnz()])
}

fn normalize_index(index: isize, len: usize) -> usize {
    if index < 0 {
        usize::try_from(len as isize + index).unwrap()
    } else {
        usize::try_from(index).unwrap()
    }
}

fn dense_select_adaptor(matrix: &SXMatrix, rows: &[isize], cols: &[isize]) -> SXMatrix {
    let normalized_rows = rows
        .iter()
        .copied()
        .map(|row| normalize_index(row, matrix.shape().0))
        .collect::<Vec<_>>();
    let normalized_cols = cols
        .iter()
        .copied()
        .map(|col| normalize_index(col, matrix.shape().1))
        .collect::<Vec<_>>();
    let mut values = Vec::with_capacity(normalized_rows.len() * normalized_cols.len());
    for &col in &normalized_cols {
        for &row in &normalized_rows {
            values.push(matrix.get(row, col));
        }
    }
    SXMatrix::dense(normalized_rows.len(), normalized_cols.len(), values).unwrap()
}

fn nz_select_adaptor(matrix: &SXMatrix, indices: &[isize]) -> SXMatrix {
    let values = indices
        .iter()
        .copied()
        .map(|index| matrix.nz(normalize_index(index, matrix.nnz())))
        .collect::<Vec<_>>();
    SXMatrix::dense_column(values).unwrap()
}

fn copysign_adaptor(lhs: SX, rhs: SX) -> SX {
    lhs.copysign(rhs)
}

fn heaviside_adaptor(value: SX) -> SX {
    0.5 * (value.sign() + SX::one())
}

fn ramp_adaptor(value: SX) -> SX {
    0.5 * (value + value.abs())
}

fn rectangle_adaptor(value: SX) -> SX {
    heaviside_adaptor(value + 0.5) - heaviside_adaptor(value - 0.5)
}

fn triangle_adaptor(value: SX) -> SX {
    ramp_adaptor(value + 1.0) - 2.0 * ramp_adaptor(value) + ramp_adaptor(value - 1.0)
}

fn depends_on_adaptor(expr: &SXMatrix, wrt: &SXMatrix) -> bool {
    let wrt_symbols = wrt
        .nonzeros()
        .iter()
        .copied()
        .flat_map(SX::free_symbols)
        .collect::<Vec<_>>();
    expr.nonzeros().iter().copied().any(|out| {
        let free = out.free_symbols();
        wrt_symbols.iter().any(|symbol| free.contains(symbol))
    })
}

fn symvar_adaptor(expr: SX) -> Vec<SX> {
    expr.free_symbols().into_iter().collect()
}

fn has_nonfinite_constant(expr: SX) -> bool {
    match expr.inspect() {
        NodeView::Constant(value) => !value.is_finite(),
        NodeView::Symbol { .. } => false,
        NodeView::Unary { arg, .. } => has_nonfinite_constant(arg),
        NodeView::Binary { lhs, rhs, .. } => {
            has_nonfinite_constant(lhs) || has_nonfinite_constant(rhs)
        }
    }
}

fn is_regular_adaptor(matrix: &SXMatrix) -> sx_core::Result<bool> {
    let mut symbolic = false;
    for &expr in matrix.nonzeros() {
        if has_nonfinite_constant(expr) {
            return Ok(false);
        }
        if !expr.free_symbols().is_empty() {
            symbolic = true;
            continue;
        }
        if !eval(expr, &HashMap::new()).is_finite() {
            return Ok(false);
        }
    }

    if symbolic {
        return Err(SxError::Shape(
            "regularity is undefined for expressions with free symbols".into(),
        ));
    }
    Ok(true)
}

fn contains_adaptor(haystack: &[SXMatrix], needle: SX) -> sx_core::Result<bool> {
    haystack.iter().try_fold(false, |found, candidate| {
        let scalar = candidate
            .scalar_expr()
            .map_err(|_| SxError::Shape("Can only convert 1-by-1 matrices to scalars".into()))?;
        Ok(found || scalar == needle)
    })
}

fn contains_any_adaptor(lhs: &[SXMatrix], rhs: &[SXMatrix]) -> sx_core::Result<bool> {
    let rhs_scalars = rhs
        .iter()
        .map(|candidate| {
            candidate
                .scalar_expr()
                .map_err(|_| SxError::Shape("Can only convert 1-by-1 matrices to scalars".into()))
        })
        .collect::<sx_core::Result<Vec<_>>>()?;
    lhs.iter().try_fold(false, |found, candidate| {
        let scalar = candidate
            .scalar_expr()
            .map_err(|_| SxError::Shape("Can only convert 1-by-1 matrices to scalars".into()))?;
        Ok(found || rhs_scalars.contains(&scalar))
    })
}

fn contains_all_adaptor(lhs: &[SXMatrix], rhs: &[SXMatrix]) -> sx_core::Result<bool> {
    let rhs_scalars = rhs
        .iter()
        .map(|candidate| {
            candidate
                .scalar_expr()
                .map_err(|_| SxError::Shape("Can only convert 1-by-1 matrices to scalars".into()))
        })
        .collect::<sx_core::Result<Vec<_>>>()?;
    lhs.iter().try_fold(true, |all_found, candidate| {
        let scalar = candidate
            .scalar_expr()
            .map_err(|_| SxError::Shape("Can only convert 1-by-1 matrices to scalars".into()))?;
        Ok(all_found && rhs_scalars.contains(&scalar))
    })
}

#[derive(Clone, Copy)]
enum MatrixShapeKind {
    Column,
    Row,
    Matrix,
}

#[derive(Clone, Copy)]
enum MatrixDensityKind {
    Dense,
    Sparse,
}

fn ad_dense_column(values: Vec<SX>) -> SXMatrix {
    SXMatrix::dense_column(values).unwrap()
}

fn ad_sparse_column(values: Vec<SX>) -> SXMatrix {
    let ccs = CCS::from_positions(6, 1, &[(0, 0), (2, 0), (4, 0), (5, 0)]).unwrap();
    SXMatrix::new(ccs, values).unwrap()
}

fn ccs_union_adaptor(lhs: &CCS, rhs: &CCS) -> CCS {
    lhs.unite(rhs).unwrap()
}

fn ccs_intersection_adaptor(lhs: &CCS, rhs: &CCS) -> CCS {
    lhs.intersect(rhs).unwrap()
}

fn ccs_is_subset_adaptor(lhs: &CCS, rhs: &CCS) -> bool {
    lhs.is_subset_of(rhs)
}

fn ad_input_case(shape: MatrixShapeKind, density: MatrixDensityKind) -> SXMatrix {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let z = SX::sym("z");
    let w = SX::sym("w");
    let column = match density {
        MatrixDensityKind::Dense => ad_dense_column(vec![x, y, z, w]),
        MatrixDensityKind::Sparse => ad_sparse_column(vec![x, y, z, w]),
    };
    match shape {
        MatrixShapeKind::Column => column,
        MatrixShapeKind::Row => column.transpose(),
        MatrixShapeKind::Matrix => match density {
            MatrixDensityKind::Dense => column.reshape(2, 2).unwrap(),
            MatrixDensityKind::Sparse => column.reshape(3, 2).unwrap(),
        },
    }
}

fn ad_output_case(
    shape: MatrixShapeKind,
    density: MatrixDensityKind,
    input: &SXMatrix,
) -> SXMatrix {
    let x = input.nz(0);
    let y = input.nz(1);
    let z = input.nz(2);
    let w = input.nz(3);
    let column = match density {
        MatrixDensityKind::Dense => ad_dense_column(vec![
            x,
            x + 2.0 * y.powi(2),
            x + 2.0 * y.powi(3) + 3.0 * z.powi(4),
            w,
        ]),
        MatrixDensityKind::Sparse => ad_sparse_column(vec![
            x,
            x + 2.0 * y.powi(2),
            x + 2.0 * y.powi(3) + 3.0 * z.powi(4),
            w,
        ]),
    };
    match shape {
        MatrixShapeKind::Column => column,
        MatrixShapeKind::Row => column.transpose(),
        MatrixShapeKind::Matrix => match density {
            MatrixDensityKind::Dense => column.reshape(2, 2).unwrap(),
            MatrixDensityKind::Sparse => column.reshape(3, 2).unwrap(),
        },
    }
}

fn expected_ad_jacobian(
    input_density: MatrixDensityKind,
    output_density: MatrixDensityKind,
    y: f64,
    z: f64,
) -> Vec<Vec<f64>> {
    match (input_density, output_density) {
        (MatrixDensityKind::Dense, MatrixDensityKind::Dense) => vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 4.0 * y, 0.0, 0.0],
            vec![1.0, 6.0 * y.powi(2), 12.0 * z.powi(3), 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ],
        (MatrixDensityKind::Dense, MatrixDensityKind::Sparse) => vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 4.0 * y, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 6.0 * y.powi(2), 12.0 * z.powi(3), 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ],
        (MatrixDensityKind::Sparse, MatrixDensityKind::Dense) => vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 4.0 * y, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 6.0 * y.powi(2), 0.0, 12.0 * z.powi(3), 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        (MatrixDensityKind::Sparse, MatrixDensityKind::Sparse) => vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 4.0 * y, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 6.0 * y.powi(2), 0.0, 12.0 * z.powi(3), 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    }
}

fn expand_slot_jacobian(
    output: &SXMatrix,
    input: &SXMatrix,
    jacobian: &SXMatrix,
    vars: &HashMap<u32, f64>,
) -> Vec<Vec<f64>> {
    let (out_rows, out_cols) = output.shape();
    let (in_rows, in_cols) = input.shape();
    let out_numel = out_rows * out_cols;
    let in_numel = in_rows * in_cols;
    let mut dense = vec![vec![0.0; in_numel]; out_numel];

    for (out_slot, (out_row, out_col)) in output.ccs().positions().into_iter().enumerate() {
        let out_linear = out_row + out_col * out_rows;
        for (in_slot, (in_row, in_col)) in input.ccs().positions().into_iter().enumerate() {
            let in_linear = in_row + in_col * in_rows;
            dense[out_linear][in_linear] = eval(jacobian.get(out_slot, in_slot), vars);
        }
    }

    dense
}

fn expand_slot_jacobian_sparsity(
    output: &SXMatrix,
    input: &SXMatrix,
    jacobian_ccs: &CCS,
) -> Vec<Vec<u8>> {
    let (out_rows, out_cols) = output.shape();
    let (in_rows, in_cols) = input.shape();
    let out_numel = out_rows * out_cols;
    let in_numel = in_rows * in_cols;
    let mut dense = vec![vec![0_u8; in_numel]; out_numel];

    for (out_slot, (out_row, out_col)) in output.ccs().positions().into_iter().enumerate() {
        let out_linear = out_row + out_col * out_rows;
        for (in_slot, (in_row, in_col)) in input.ccs().positions().into_iter().enumerate() {
            if jacobian_ccs.nz_index(out_slot, in_slot).is_some() {
                let in_linear = in_row + in_col * in_rows;
                dense[out_linear][in_linear] = 1;
            }
        }
    }

    dense
}

fn assert_ad_jacobian_parity_cases() {
    let values = [1.2, 2.3, 7.0, 4.6];

    for input_shape in [
        MatrixShapeKind::Column,
        MatrixShapeKind::Row,
        MatrixShapeKind::Matrix,
    ] {
        for output_shape in [
            MatrixShapeKind::Column,
            MatrixShapeKind::Row,
            MatrixShapeKind::Matrix,
        ] {
            for input_density in [MatrixDensityKind::Dense, MatrixDensityKind::Sparse] {
                for output_density in [MatrixDensityKind::Dense, MatrixDensityKind::Sparse] {
                    let input = ad_input_case(input_shape, input_density);
                    let output = ad_output_case(output_shape, output_density, &input);
                    let vars = input
                        .nonzeros()
                        .iter()
                        .copied()
                        .zip(values)
                        .map(|(symbol, value)| (symbol.id(), value))
                        .collect::<HashMap<_, _>>();
                    let jacobian = output.jacobian(&input).unwrap();
                    let actual = expand_slot_jacobian(&output, &input, &jacobian, &vars);
                    let expected = expected_ad_jacobian(input_density, output_density, 2.3, 7.0);

                    assert_eq!(actual.len(), expected.len());
                    assert_eq!(actual[0].len(), expected[0].len());
                    for row in 0..expected.len() {
                        for col in 0..expected[row].len() {
                            assert_abs_diff_eq!(
                                actual[row][col],
                                expected[row][col],
                                epsilon = 1e-9
                            );
                        }
                    }
                }
            }
        }
    }
}

fn assert_ad_jacobian_sparsity_parity_cases() {
    for input_shape in [
        MatrixShapeKind::Column,
        MatrixShapeKind::Row,
        MatrixShapeKind::Matrix,
    ] {
        for output_shape in [
            MatrixShapeKind::Column,
            MatrixShapeKind::Row,
            MatrixShapeKind::Matrix,
        ] {
            for input_density in [MatrixDensityKind::Dense, MatrixDensityKind::Sparse] {
                for output_density in [MatrixDensityKind::Dense, MatrixDensityKind::Sparse] {
                    let input = ad_input_case(input_shape, input_density);
                    let output = ad_output_case(output_shape, output_density, &input);
                    let jacobian_ccs = output.jacobian_ccs(&input).unwrap();
                    let actual = expand_slot_jacobian_sparsity(&output, &input, &jacobian_ccs);
                    let expected = expected_ad_jacobian(input_density, output_density, 2.3, 7.0)
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|value| u8::from(value != 0.0))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();

                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

fn assert_scalar_case(expr: SX, x: SX, point: f64, expected: f64, _label: &str) {
    let vars = HashMap::from([(x.id(), point)]);
    assert_abs_diff_eq!(eval(expr, &vars), expected, epsilon = 1e-10);
}

fn jacobian_scalar(expr: SX, wrt: SX) -> SX {
    let expr = SXMatrix::scalar(expr);
    let wrt = SXMatrix::scalar(wrt);
    expr.jacobian(&wrt).unwrap().get(0, 0)
}

fn unary_jacobian_cases(x: SX, point: f64) -> [(SX, f64); 17] {
    [
        (x.sqrt(), 1.0 / (2.0 * point.sqrt())),
        (x.sin(), point.cos()),
        (x.abs(), point.signum()),
        (x.sign(), 0.0),
        (x.cos(), -point.sin()),
        (x.tan(), 1.0 / point.cos().powi(2)),
        (x.atan(), 1.0 / (point.powi(2) + 1.0)),
        (x.asin(), 1.0 / (1.0 - point.powi(2)).sqrt()),
        (x.acos(), -1.0 / (1.0 - point.powi(2)).sqrt()),
        (x.exp(), point.exp()),
        (x.log(), 1.0 / point),
        (x.powi(0), 0.0),
        (x.powi(1), 1.0),
        (x.powi(-2), -2.0 / point.powi(3)),
        (x.powf(0.3), 0.3 / point.powf(0.7)),
        (x.log1p(), 1.0 / (1.0 + point)),
        (x.expm1(), point.exp()),
    ]
}

type UnaryExprBuilder = fn(SX) -> SX;
type UnaryDerivative = fn(f64) -> f64;

fn unary_matrix_jacobian_cases() -> [(UnaryExprBuilder, UnaryDerivative); 17] {
    [
        ((|value: SX| value.sqrt()) as UnaryExprBuilder, |point| {
            1.0 / (2.0 * point.sqrt())
        }),
        ((|value: SX| value.sin()) as UnaryExprBuilder, |point| {
            point.cos()
        }),
        ((|value: SX| value.abs()) as UnaryExprBuilder, |point| {
            point.signum()
        }),
        ((|value: SX| value.sign()) as UnaryExprBuilder, |_| 0.0),
        ((|value: SX| value.cos()) as UnaryExprBuilder, |point| {
            -point.sin()
        }),
        ((|value: SX| value.tan()) as UnaryExprBuilder, |point| {
            1.0 / point.cos().powi(2)
        }),
        ((|value: SX| value.atan()) as UnaryExprBuilder, |point| {
            1.0 / (point.powi(2) + 1.0)
        }),
        ((|value: SX| value.asin()) as UnaryExprBuilder, |point| {
            1.0 / (1.0 - point.powi(2)).sqrt()
        }),
        ((|value: SX| value.acos()) as UnaryExprBuilder, |point| {
            -1.0 / (1.0 - point.powi(2)).sqrt()
        }),
        ((|value: SX| value.exp()) as UnaryExprBuilder, |point| {
            point.exp()
        }),
        ((|value: SX| value.log()) as UnaryExprBuilder, |point| {
            1.0 / point
        }),
        ((|value: SX| value.powi(0)) as UnaryExprBuilder, |_| 0.0),
        ((|value: SX| value.powi(1)) as UnaryExprBuilder, |_| 1.0),
        ((|value: SX| value.powi(-2)) as UnaryExprBuilder, |point| {
            -2.0 / point.powi(3)
        }),
        ((|value: SX| value.powf(0.3)) as UnaryExprBuilder, |point| {
            0.3 / point.powf(0.7)
        }),
        ((|value: SX| value.log1p()) as UnaryExprBuilder, |point| {
            1.0 / (1.0 + point)
        }),
        ((|value: SX| value.expm1()) as UnaryExprBuilder, |point| {
            point.exp()
        }),
    ]
}

fn assert_elementwise_unary_jacobian(matrix: &SXMatrix, points: [f64; 3]) {
    let vars = matrix
        .nonzeros()
        .iter()
        .copied()
        .zip(points)
        .map(|(symbol, point)| (symbol.id(), point))
        .collect::<HashMap<_, _>>();
    let point_values = points;

    for (expr_builder, derivative) in unary_matrix_jacobian_cases() {
        let output = matrix.map_nonzeros(expr_builder);
        let jacobian = output.jacobian(matrix).unwrap();
        let dense = eval_matrix(&jacobian, &vars);
        assert_eq!(jacobian.shape(), (3, 3));
        for (row, dense_row) in dense.iter().enumerate().take(3) {
            for (col, value) in dense_row.iter().enumerate().take(3) {
                let expected = if row == col {
                    derivative(point_values[row])
                } else {
                    0.0
                };
                assert_abs_diff_eq!(*value, expected, epsilon = 1e-9);
            }
        }
    }
}

type UnaryNumeric = fn(f64) -> f64;

fn unary_matrix_value_cases() -> [(UnaryExprBuilder, UnaryNumeric); 19] {
    [
        ((|value: SX| value.sqrt()) as UnaryExprBuilder, |point| {
            point.sqrt()
        }),
        ((|value: SX| value.sin()) as UnaryExprBuilder, |point| {
            point.sin()
        }),
        ((|value: SX| value.cos()) as UnaryExprBuilder, |point| {
            point.cos()
        }),
        ((|value: SX| value.tan()) as UnaryExprBuilder, |point| {
            point.tan()
        }),
        ((|value: SX| value.abs()) as UnaryExprBuilder, |point| {
            point.abs()
        }),
        ((|value: SX| value.sign()) as UnaryExprBuilder, |point| {
            if point > 0.0 {
                1.0
            } else if point < 0.0 {
                -1.0
            } else {
                0.0
            }
        }),
        ((|value: SX| value.atan()) as UnaryExprBuilder, |point| {
            point.atan()
        }),
        ((|value: SX| value.asin()) as UnaryExprBuilder, |point| {
            point.asin()
        }),
        ((|value: SX| value.acos()) as UnaryExprBuilder, |point| {
            point.acos()
        }),
        ((|value: SX| value.exp()) as UnaryExprBuilder, |point| {
            point.exp()
        }),
        ((|value: SX| value.log()) as UnaryExprBuilder, |point| {
            point.ln()
        }),
        ((|value: SX| value.powi(0)) as UnaryExprBuilder, |point| {
            point.powi(0)
        }),
        ((|value: SX| value.powi(1)) as UnaryExprBuilder, |point| {
            point.powi(1)
        }),
        ((|value: SX| value.powi(-2)) as UnaryExprBuilder, |point| {
            point.powi(-2)
        }),
        ((|value: SX| value.powf(0.3)) as UnaryExprBuilder, |point| {
            point.powf(0.3)
        }),
        ((|value: SX| value.floor()) as UnaryExprBuilder, |point| {
            point.floor()
        }),
        ((|value: SX| value.ceil()) as UnaryExprBuilder, |point| {
            point.ceil()
        }),
        ((|value: SX| value.log1p()) as UnaryExprBuilder, |point| {
            point.ln_1p()
        }),
        ((|value: SX| value.expm1()) as UnaryExprBuilder, |point| {
            point.exp_m1()
        }),
    ]
}

fn dense_matrix_unary_adaptor(matrix: &SXMatrix, op: UnaryExprBuilder) -> SXMatrix {
    let (nrow, ncol) = matrix.shape();
    let mut values = Vec::with_capacity(nrow * ncol);
    for col in 0..ncol {
        for row in 0..nrow {
            values.push(op(matrix.get(row, col)));
        }
    }
    SXMatrix::dense(nrow, ncol, values).unwrap()
}

fn sparse_matrix_unary_adaptor(matrix: &SXMatrix, op: UnaryExprBuilder) -> SXMatrix {
    matrix.map_nonzeros(op)
}

fn numeric_dense_unary(values: &[Vec<f64>], op: UnaryNumeric) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|row| row.iter().copied().map(op).collect::<Vec<_>>())
        .collect()
}

fn numeric_sparse_unary(ccs: &CCS, nonzeros: &[f64], op: UnaryNumeric) -> DenseValue {
    let mut dense = DenseValue::zeros(ccs.nrow(), ccs.ncol());
    for ((row, col), value) in ccs.positions().into_iter().zip(nonzeros.iter().copied()) {
        dense.values[row + col * ccs.nrow()] = op(value);
    }
    dense
}

fn det3_adaptor(matrix: &SXMatrix) -> SX {
    assert_eq!(matrix.shape(), (3, 3));
    let a = matrix.get(0, 0);
    let b = matrix.get(0, 1);
    let c = matrix.get(0, 2);
    let d = matrix.get(1, 0);
    let e = matrix.get(1, 1);
    let f = matrix.get(1, 2);
    let g = matrix.get(2, 0);
    let h = matrix.get(2, 1);
    let i = matrix.get(2, 2);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

fn det3_numeric(matrix: &DenseValue) -> f64 {
    assert_eq!(matrix.shape(), (3, 3));
    let a = matrix.values[0];
    let d = matrix.values[1];
    let g = matrix.values[2];
    let b = matrix.values[3];
    let e = matrix.values[4];
    let h = matrix.values[5];
    let c = matrix.values[6];
    let f = matrix.values[7];
    let i = matrix.values[8];
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

fn inv3_adaptor(matrix: &SXMatrix) -> SXMatrix {
    assert_eq!(matrix.shape(), (3, 3));
    let a = matrix.get(0, 0);
    let b = matrix.get(0, 1);
    let c = matrix.get(0, 2);
    let d = matrix.get(1, 0);
    let e = matrix.get(1, 1);
    let f = matrix.get(1, 2);
    let g = matrix.get(2, 0);
    let h = matrix.get(2, 1);
    let i = matrix.get(2, 2);
    let det = det3_adaptor(matrix);
    SXMatrix::dense(
        3,
        3,
        vec![
            (e * i - f * h) / det,
            (f * g - d * i) / det,
            (d * h - e * g) / det,
            (c * h - b * i) / det,
            (a * i - c * g) / det,
            (b * g - a * h) / det,
            (b * f - c * e) / det,
            (c * d - a * f) / det,
            (a * e - b * d) / det,
        ],
    )
    .unwrap()
}

fn inv3_numeric(matrix: &DenseValue) -> DenseValue {
    let det = det3_numeric(matrix);
    let a = matrix.values[0];
    let d = matrix.values[1];
    let g = matrix.values[2];
    let b = matrix.values[3];
    let e = matrix.values[4];
    let h = matrix.values[5];
    let c = matrix.values[6];
    let f = matrix.values[7];
    let i = matrix.values[8];
    DenseValue {
        nrow: 3,
        ncol: 3,
        values: vec![
            (e * i - f * h) / det,
            (f * g - d * i) / det,
            (d * h - e * g) / det,
            (c * h - b * i) / det,
            (a * i - c * g) / det,
            (b * g - a * h) / det,
            (b * f - c * e) / det,
            (c * d - a * f) / det,
            (a * e - b * d) / det,
        ],
    }
}

#[rstest]
fn sx_scalar_sx_matches_casadi_test_scalar_sx() {
    let x = SX::sym("x");
    let point: f64 = 0.738;

    assert_scalar_case(x.sqrt(), x, point, point.sqrt(), "sqrt");
    assert_scalar_case(x.sin(), x, point, point.sin(), "sin");
    assert_scalar_case(x.cos(), x, point, point.cos(), "cos");
    assert_scalar_case(x.tan(), x, point, point.tan(), "tan");
    assert_scalar_case(x.abs(), x, point, point.abs(), "fabs");
    assert_scalar_case(x.sign(), x, point, point.signum(), "sign");
    assert_scalar_case(x.atan(), x, point, point.atan(), "arctan");
    assert_scalar_case(x.asin(), x, point, point.asin(), "arcsin");
    assert_scalar_case(x.acos(), x, point, point.acos(), "arccos");
    assert_scalar_case(x.exp(), x, point, point.exp(), "exp");
    assert_scalar_case(x.log(), x, point, point.ln(), "log");
    assert_scalar_case(x.powi(0), x, point, point.powi(0), "x^0");
    assert_scalar_case(x.powi(1), x, point, point.powi(1), "^1");
    assert_scalar_case(x.powi(-2), x, point, point.powi(-2), "^-2");
    assert_scalar_case(x.powf(0.3), x, point, point.powf(0.3), "^0.3");
    assert_scalar_case(x.floor(), x, point, point.floor(), "floor");
    assert_scalar_case(x.ceil(), x, point, point.ceil(), "ceil");
    assert_scalar_case(x.log1p(), x, point, point.ln_1p(), "log1p");
    assert_scalar_case(x.expm1(), x, point, point.exp_m1(), "expm1");
}

#[rstest]
fn sx_gradient_matches_casadi_test_gradient() {
    let x = SX::sym("x");
    let point = 1.0;
    let order = 3;
    let mut expr = x.powi(order);
    let vars = HashMap::from([(x.id(), point)]);

    let first = jacobian_scalar(expr, x);
    assert_abs_diff_eq!(eval(first, &vars), order as f64, epsilon = 1e-10);

    let mut expected = 1.0;
    for i in 0..order {
        expr = jacobian_scalar(expr, x);
        expected *= f64::from(order - i);
    }
    assert_abs_diff_eq!(eval(expr, &vars), expected, epsilon = 1e-10);
}

#[rstest]
fn sx_gradient2_matches_casadi_test_gradient2() {
    let x = SX::sym("x");
    let p = SX::sym("p");
    let x0 = 1.0;
    let p0 = 3.0;
    let mut expr = x.pow(p);
    let vars = HashMap::from([(x.id(), x0), (p.id(), p0)]);

    let first = jacobian_scalar(expr, x);
    assert_abs_diff_eq!(eval(first, &vars), p0, epsilon = 1e-10);

    let mut expected = 1.0;
    for i in 0..3 {
        expr = jacobian_scalar(expr, x);
        expected *= 3.0 - f64::from(i);
    }
    assert_abs_diff_eq!(eval(expr, &vars), expected, epsilon = 1e-10);
}

#[rstest]
fn sx_unary_jacobian_matches_casadi_test_sx_jacobian() {
    let x = SX::sym("x");
    let point: f64 = 0.738;
    let vars = HashMap::from([(x.id(), point)]);

    for (expr, expected) in unary_jacobian_cases(x, point) {
        let jac = jacobian_scalar(expr, x);
        assert_abs_diff_eq!(eval(jac, &vars), expected, epsilon = 1e-9);
    }
}

#[rstest]
fn sx_unary_jac_matches_casadi_test_sx_jac() {
    let x = SX::sym("x");
    let point: f64 = 0.738;
    let vars = HashMap::from([(x.id(), point)]);

    for (expr, expected) in unary_jacobian_cases(x, point) {
        let jac = jacobian_scalar(expr, x);
        assert_abs_diff_eq!(eval(jac, &vars), expected, epsilon = 1e-9);
    }
}

#[rstest]
fn sx_unary_column_jacobian_matches_casadi_test_sx_jacobians() {
    let matrix = SXMatrix::sym_dense("x", 3, 1).unwrap();
    assert_elementwise_unary_jacobian(&matrix, [0.738, 0.9, 0.3]);
}

#[rstest]
fn sx_unary_row_jacobian_matches_casadi_test_sx_jacobians2() {
    let matrix = SXMatrix::sym_dense("x", 1, 3).unwrap();
    assert_elementwise_unary_jacobian(&matrix, [0.738, 0.9, 0.3]);
}

#[rstest]
fn sx_dense_unary_matches_casadi_test_sx() {
    let matrix = SXMatrix::sym_dense("x", 3, 2).unwrap();
    let values = vec![vec![0.738, 0.2], vec![0.1, 0.39], vec![0.99, 0.999_999]];
    let vars = dense_matrix_bindings(&matrix, &values);

    for (builder, numeric) in unary_matrix_value_cases() {
        let actual = eval_matrix(&dense_matrix_unary_adaptor(&matrix, builder), &vars);
        let expected = numeric_dense_unary(&values, numeric);
        assert_dense_matrix_close(&actual, &expected);
    }

    let invertible = DenseValue::from_rows(&[
        &[0.738, 0.2, 0.3],
        &[0.1, 0.39, -6.0],
        &[0.99, 0.999_999, -12.0],
    ]);
    let invertible_matrix = SXMatrix::sym_dense("a", 3, 3).unwrap();
    let invertible_vars = dense_matrix_bindings(&invertible_matrix, &invertible.to_rows());
    let det_actual = eval(det3_adaptor(&invertible_matrix), &invertible_vars);
    let det_expected = det3_numeric(&invertible);
    assert_abs_diff_eq!(det_actual, det_expected, epsilon = 1e-9);

    let inv_actual = eval_matrix(&inv3_adaptor(&invertible_matrix), &invertible_vars);
    let inv_expected = inv3_numeric(&invertible).to_rows();
    assert_dense_matrix_close_with_epsilon(&inv_actual, &inv_expected, 1e-8);
}

#[rstest]
fn sx_sparse_unary_matches_casadi_test_sxsparse() {
    let ccs = CCS::new(4, 3, vec![0, 2, 2, 3], vec![1, 2, 1]).unwrap();
    let matrix =
        SXMatrix::new(ccs.clone(), vec![SX::sym("x"), SX::sym("y"), SX::sym("z")]).unwrap();
    let dense_values = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.738, 0.0, 0.99],
        vec![0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let nonzeros = vec![0.738, 0.1, 0.99];
    let vars = dense_matrix_bindings(&matrix, &dense_values);

    for (builder, numeric) in unary_matrix_value_cases() {
        let actual = eval_matrix(&sparse_matrix_unary_adaptor(&matrix, builder), &vars);
        let expected = numeric_sparse_unary(&ccs, &nonzeros, numeric).to_rows();
        assert_dense_matrix_close(&actual, &expected);
    }
}

#[rstest]
fn sx_slicing_matches_casadi_test_sxslicing_via_adaptor() {
    let dense = SXMatrix::sym_dense("x", 3, 2).unwrap();
    let dense_values = vec![vec![0.738, 0.2], vec![0.1, 0.39], vec![0.99, 0.999_999]];
    let dense_vars = dense_matrix_bindings(&dense, &dense_values);

    assert_abs_diff_eq!(eval(dense.get(0, 0), &dense_vars), 0.738, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(dense.get(1, 0), &dense_vars), 0.1, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(dense.get(0, 1), &dense_vars), 0.2, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(dense.get(0, 1), &dense_vars), 0.2, epsilon = 1e-12);

    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1, 2], &[0]), &dense_vars),
        &[vec![0.738], vec![0.1], vec![0.99]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1, 2], &[1]), &dense_vars),
        &[vec![0.2], vec![0.39], vec![0.999_999]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[1], &[0, 1]), &dense_vars),
        &[vec![0.1, 0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0], &[0, 1]), &dense_vars),
        &[vec![0.738, 0.2]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[-1], &[0, 1]), &dense_vars),
        &[vec![0.99, 0.999_999]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1, 2], &[0]), &dense_vars),
        &[vec![0.738], vec![0.1], vec![0.99]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0], &[0]), &dense_vars),
        &[vec![0.738]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1], &[0]), &dense_vars),
        &[vec![0.738], vec![0.1]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1], &[0, 1]), &dense_vars),
        &[vec![0.738, 0.2], vec![0.1, 0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1], &[0, 1]), &dense_vars),
        &[vec![0.738, 0.2], vec![0.1, 0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1], &[0, 1]), &dense_vars),
        &[vec![0.738, 0.2], vec![0.1, 0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&dense, &[0, 1], &[0, 1]), &dense_vars),
        &[vec![0.738, 0.2], vec![0.1, 0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&dense, &[0, 2, 3]), &dense_vars),
        &[vec![0.738], vec![0.99], vec![0.2]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&dense, &[0, 1]), &dense_vars),
        &[vec![0.738], vec![0.1]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&dense, &[1]), &dense_vars),
        &[vec![0.1]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&dense, &[-1]), &dense_vars),
        &[vec![0.999_999]],
    );

    let sparse_ccs = CCS::new(4, 3, vec![0, 2, 2, 3], vec![1, 2, 1]).unwrap();
    let sparse = SXMatrix::new(sparse_ccs, vec![SX::sym("x"), SX::sym("y"), SX::sym("z")]).unwrap();
    let sparse_values = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.738, 0.0, 0.99],
        vec![0.39, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let sparse_vars = dense_matrix_bindings(&sparse, &sparse_values);

    assert_abs_diff_eq!(eval(sparse.get(0, 0), &sparse_vars), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(sparse.get(1, 0), &sparse_vars), 0.738, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(sparse.get(0, 2), &sparse_vars), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(sparse.get(0, 2), &sparse_vars), 0.0, epsilon = 1e-12);

    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[0, 1, 2, 3], &[0]),
            &sparse_vars,
        ),
        &[vec![0.0], vec![0.738], vec![0.39], vec![0.0]],
    );
    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[0, 1, 2, 3], &[1]),
            &sparse_vars,
        ),
        &[vec![0.0], vec![0.0], vec![0.0], vec![0.0]],
    );
    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[1], &[0, 1, 2]),
            &sparse_vars,
        ),
        &[vec![0.738, 0.0, 0.99]],
    );
    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[0], &[0, 1, 2]),
            &sparse_vars,
        ),
        &[vec![0.0, 0.0, 0.0]],
    );
    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[-1], &[0, 1, 2]),
            &sparse_vars,
        ),
        &[vec![0.0, 0.0, 0.0]],
    );
    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[0, 1, 2], &[1]),
            &sparse_vars,
        ),
        &[vec![0.0], vec![0.0], vec![0.0]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&dense_select_adaptor(&sparse, &[0, 1], &[0]), &sparse_vars),
        &[vec![0.0], vec![0.738]],
    );
    assert_dense_matrix_close(
        &eval_matrix(
            &dense_select_adaptor(&sparse, &[0, 1], &[0, 1]),
            &sparse_vars,
        ),
        &[vec![0.0, 0.0], vec![0.738, 0.0]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&sparse, &[2, 1]), &sparse_vars),
        &[vec![0.99], vec![0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&sparse, &[0, 1]), &sparse_vars),
        &[vec![0.738], vec![0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&sparse, &[1]), &sparse_vars),
        &[vec![0.39]],
    );
    assert_dense_matrix_close(
        &eval_matrix(&nz_select_adaptor(&sparse, &[-1]), &sparse_vars),
        &[vec![0.99]],
    );
}

#[rstest]
fn sx_equivalence_matches_casadi_test_equivalence() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let vars = HashMap::from([(x.id(), 1.3), (y.id(), 1.7)]);

    let cases = [
        (x.log1p(), (SX::one() + x).log()),
        (x.expm1(), x.exp() - SX::one()),
        (x.hypot(y), (x.sqr() + y.sqr()).sqrt()),
    ];

    for (lhs, rhs) in cases {
        assert_abs_diff_eq!(eval(lhs, &vars), eval(rhs, &vars), epsilon = 1e-10);
    }
}

#[rstest]
fn sx_binary_dense_matches_casadi_test_sxbinary_via_dense_adaptor() {
    let lhs = SXMatrix::sym_dense("x", 3, 2).unwrap();
    let rhs = SXMatrix::sym_dense("y", 3, 2).unwrap();
    let lhs_values = vec![vec![0.738, 0.2], vec![0.1, 0.39], vec![0.99, 0.999_999]];
    let rhs_values = vec![vec![1.738, 0.6], vec![0.7, 12.0], vec![0.0, -6.0]];
    let mut vars = dense_matrix_bindings(&lhs, &lhs_values);
    vars.extend(dense_matrix_bindings(&rhs, &rhs_values));

    for op in [
        MatrixBinaryOp::Add,
        MatrixBinaryOp::Sub,
        MatrixBinaryOp::Mul,
        MatrixBinaryOp::Max,
        MatrixBinaryOp::Min,
        MatrixBinaryOp::Hypot,
        MatrixBinaryOp::MtimesRhsTranspose,
    ] {
        let actual = eval_matrix(&dense_matrix_binary_adaptor(&lhs, &rhs, op), &vars);
        let expected = numeric_dense_matrix_binary(&lhs_values, &rhs_values, op);
        assert_dense_matrix_close(&actual, &expected);
    }

    assert!(dense_matrix_multiply_adaptor(&lhs, &rhs).is_err());
}

#[rstest]
fn sx_binary_sparse_matches_casadi_test_sxbinary_sparse_via_dense_adaptor() {
    let lhs = SXMatrix::new(
        CCS::new(4, 3, vec![0, 2, 2, 3], vec![1, 2, 1]).unwrap(),
        vec![SX::sym("x"), SX::sym("y"), SX::sym("z")],
    )
    .unwrap();
    let rhs = SXMatrix::new(
        CCS::new(4, 3, vec![0, 2, 2, 3], vec![0, 2, 3]).unwrap(),
        vec![SX::sym("x2"), SX::sym("z2"), SX::sym("y2")],
    )
    .unwrap();
    let lhs_values = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.738, 0.0, 0.99],
        vec![0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let rhs_values = vec![
        vec![1.738, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.7, 0.0, 0.0],
        vec![0.0, 0.0, -6.0],
    ];
    let mut vars = dense_matrix_bindings(&lhs, &lhs_values);
    vars.extend(dense_matrix_bindings(&rhs, &rhs_values));

    for op in [
        MatrixBinaryOp::Add,
        MatrixBinaryOp::Sub,
        MatrixBinaryOp::Mul,
        MatrixBinaryOp::Max,
        MatrixBinaryOp::Min,
        MatrixBinaryOp::Hypot,
        MatrixBinaryOp::MtimesRhsTranspose,
    ] {
        let actual = eval_matrix(&dense_matrix_binary_adaptor(&lhs, &rhs, op), &vars);
        let expected = numeric_dense_matrix_binary(&lhs_values, &rhs_values, op);
        assert_dense_matrix_close(&actual, &expected);
    }

    assert!(dense_matrix_multiply_adaptor(&lhs, &rhs).is_err());
}

#[rstest]
fn sx_binary_diff_matches_casadi_test_sxbinary_diff_via_dense_adaptor() {
    let lhs = SXMatrix::sym_dense("x", 4, 2).unwrap();
    let rhs = SXMatrix::sym_dense("y", 4, 2).unwrap();
    let lhs_values = vec![
        vec![0.738, 0.2],
        vec![0.1, 0.39],
        vec![0.99, 0.999_999],
        vec![1.0, 2.0],
    ];
    let rhs_values = vec![
        vec![1.738, 0.6],
        vec![0.7, 12.0],
        vec![0.0, -6.0],
        vec![1.0, 2.0],
    ];
    let dx_values = vec![
        vec![0.3, -0.2],
        vec![0.4, 0.1],
        vec![-0.5, 0.6],
        vec![0.2, -0.3],
    ];
    let dy_values = vec![
        vec![-0.1, 0.5],
        vec![0.2, -0.4],
        vec![0.7, -0.3],
        vec![0.1, 0.2],
    ];
    let mut vars = dense_matrix_bindings(&lhs, &lhs_values);
    vars.extend(dense_matrix_bindings(&rhs, &rhs_values));
    let wrt = SXMatrix::dense_column(
        lhs.nonzeros()
            .iter()
            .chain(rhs.nonzeros().iter())
            .copied()
            .collect(),
    )
    .unwrap();
    let epsilon = 1e-7;

    for op in [
        MatrixBinaryOp::Add,
        MatrixBinaryOp::Sub,
        MatrixBinaryOp::Mul,
        MatrixBinaryOp::Max,
        MatrixBinaryOp::Min,
        MatrixBinaryOp::Hypot,
        MatrixBinaryOp::MtimesRhsTranspose,
    ] {
        let direction_rhs = match op {
            MatrixBinaryOp::Max | MatrixBinaryOp::Min => &dx_values,
            _ => &dy_values,
        };
        let seed = SXMatrix::dense_column(
            dense_matrix_column_major(&dx_values)
                .into_iter()
                .chain(dense_matrix_column_major(direction_rhs))
                .map(SX::from)
                .collect(),
        )
        .unwrap();
        let output = dense_matrix_binary_adaptor(&lhs, &rhs, op);
        let actual = eval_matrix(&output.forward(&wrt, &seed).unwrap(), &vars);
        let expected = numeric_dense_matrix_binary(
            &add_scaled(&lhs_values, &dx_values, epsilon),
            &add_scaled(&rhs_values, direction_rhs, epsilon),
            op,
        )
        .into_iter()
        .zip(numeric_dense_matrix_binary(&lhs_values, &rhs_values, op))
        .map(|(lhs_row, rhs_row)| {
            lhs_row
                .into_iter()
                .zip(rhs_row)
                .map(|(lhs_value, rhs_value)| (lhs_value - rhs_value) / epsilon)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
        assert_dense_matrix_close_with_epsilon(&actual, &expected, 1e-4);
    }
}

#[rstest]
fn dm_binary_matches_casadi_test_dmbinary_via_dense_adaptor() {
    let lhs_values = vec![vec![0.738, 0.2], vec![0.1, 0.39], vec![0.99, 0.999_999]];
    let rhs_values = vec![vec![1.738, 0.6], vec![0.7, 12.0], vec![0.0, -6.0]];
    let lhs = SXMatrix::dense(
        3,
        2,
        dense_matrix_column_major(&lhs_values)
            .into_iter()
            .map(SX::from)
            .collect(),
    )
    .unwrap();
    let rhs = SXMatrix::dense(
        3,
        2,
        dense_matrix_column_major(&rhs_values)
            .into_iter()
            .map(SX::from)
            .collect(),
    )
    .unwrap();

    for op in [
        MatrixBinaryOp::Add,
        MatrixBinaryOp::Sub,
        MatrixBinaryOp::Mul,
        MatrixBinaryOp::Max,
        MatrixBinaryOp::Min,
        MatrixBinaryOp::Hypot,
        MatrixBinaryOp::MtimesRhsTranspose,
    ] {
        let actual = eval_matrix(
            &dense_matrix_binary_adaptor(&lhs, &rhs, op),
            &HashMap::new(),
        );
        let expected = numeric_dense_matrix_binary(&lhs_values, &rhs_values, op);
        assert_dense_matrix_close(&actual, &expected);
    }
}

fn assert_simplifies_to_identity(
    build: &(dyn Fn(SX, SX) -> SX + Send + Sync),
    x: SX,
    same_named_y: SX,
    bindings: &HashMap<u32, f64>,
) {
    let expr = build(x, same_named_y);
    assert_abs_diff_eq!(eval(expr, bindings), 0.3, epsilon = 1e-10);
    assert_eq!(expr.to_string(), "x");

    let neg_expr = build(-x, same_named_y);
    assert_abs_diff_eq!(eval(neg_expr, bindings), -0.3, epsilon = 1e-10);
    assert_eq!(neg_expr.to_string(), "(-x)");
}

#[rstest]
fn sx_const_folding_on_the_fly_matches_casadi() {
    let x = SX::sym("x");
    for a in [2.0, 7.0] {
        for b in [2.0, 7.0] {
            let reference = (a * b) * x;
            assert_eq!(reference.to_string(), (a * (b * x)).to_string());
            assert_eq!(reference.to_string(), ((b * x) * a).to_string());
        }
    }
}

#[rstest]
fn sx_simplifications_match_casadi_test_sxsimplifications() {
    let x = SX::sym("x");
    let same_named_y = SX::sym("x");
    let bindings = HashMap::from([(x.id(), 0.3), (same_named_y.id(), 0.8)]);
    let builders: Vec<Box<dyn Fn(SX, SX) -> SX + Send + Sync>> = vec![
        Box::new(|x, _| {
            let y = 0.5 * x;
            y + y
        }),
        Box::new(|x, _| {
            let y = x / 2.0;
            y + y
        }),
        Box::new(|x, _| {
            let y = x * 0.5;
            y + y
        }),
        Box::new(|x, _| {
            let y = x * x;
            ((-y) / y) * (-x)
        }),
        Box::new(|x, _| ((-(x * x)) / (x * x)) * (-x)),
        Box::new(|x, _| {
            let y = x * x;
            (y / (-y)) * (-x)
        }),
        Box::new(|x, _| {
            let y = x * x;
            ((-y) / (-y)) * x
        }),
        Box::new(|x, _| (x - x) + x),
        Box::new(|x, _| ((x * x) - (x * x)) + x),
        Box::new(|x, _| 4.0 * (0.25 * x)),
        Box::new(|x, _| 4.0 * (x * 0.25)),
        Box::new(|x, _| (0.25 * x) * 4.0),
        Box::new(|x, _| (x * 0.25) * 4.0),
        Box::new(|x, _| (4.0 * x) / 4.0),
        Box::new(|x, _| 4.0 * (x / 4.0)),
        Box::new(|x, _| (x / 4.0) / 0.25),
        Box::new(|x, _| x * (((4.0 / x) * x) / 4.0)),
        Box::new(|x, _| x * ((x * (2.0 / x)) / 2.0)),
        Box::new(|x, _| x * (((2.0 * x) / x) / 2.0)),
        Box::new(|x, _| x * ((x / (2.0 * x)) * 2.0)),
        Box::new(|x, _| x + 0.0),
        Box::new(|x, _| 0.0 + x),
        Box::new(|x, _| x - 0.0),
        Box::new(|x, _| 0.0 - (-x)),
        Box::new(|x, _| x * 1.0),
        Box::new(|x, _| 1.0 * x),
        Box::new(|x, _| 1.0 * (x * 1.0)),
        Box::new(|x, _| (1.0 * x) * 1.0),
        Box::new(|x, _| (0.5 * x) + (0.5 * x)),
        Box::new(|x, _| (x / 2.0) + (x / 2.0)),
        Box::new(|x, _| (x * 0.5) + (0.5 * x)),
        Box::new(|x, _| (SX::from(4.0) - SX::from(4.0)) + x),
        Box::new(|x, y| ((x + y) - (y + x)) + x),
        Box::new(|x, y| ((x * y) - (y * x)) + x),
        Box::new(|x, _| ((-x) - (-x)) + x),
    ];

    for build in &builders {
        assert_simplifies_to_identity(build.as_ref(), x, same_named_y, &bindings);
    }
}

#[rstest]
fn sx_copysign_matches_casadi_test_copysign_via_adaptor() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let expr = copysign_adaptor(x, y);

    for ((x_value, y_value), expected) in [
        ((2.0, 0.5), 2.0),
        ((2.0, -0.5), -2.0),
        ((-2.0, 0.5), 2.0),
        ((-2.0, -0.5), -2.0),
        ((2.0, 0.0), 2.0),
    ] {
        let vars = HashMap::from([(x.id(), x_value), (y.id(), y_value)]);
        assert_abs_diff_eq!(eval(expr, &vars), expected, epsilon = 1e-10);
    }

    let jac_x = jacobian_scalar(expr, x);
    for ((x_value, y_value), expected) in [
        ((2.0, 0.5), 1.0),
        ((2.0, -0.5), -1.0),
        ((-2.0, 0.5), -1.0),
        ((-2.0, -0.5), 1.0),
        ((2.0, 0.0), 1.0),
    ] {
        let vars = HashMap::from([(x.id(), x_value), (y.id(), y_value)]);
        assert_abs_diff_eq!(eval(jac_x, &vars), expected, epsilon = 1e-10);
    }

    let jac_y = jacobian_scalar(expr, y);
    for (x_value, y_value) in [
        (2.0, 0.5),
        (2.0, -0.5),
        (-2.0, 0.5),
        (-2.0, -0.5),
        (2.0, 0.0),
    ] {
        let vars = HashMap::from([(x.id(), x_value), (y.id(), y_value)]);
        assert_abs_diff_eq!(eval(jac_y, &vars), 0.0, epsilon = 1e-10);
    }
}

#[rstest]
fn sx_primitivefunctions_match_casadi_test_primitivefunctions_via_adaptors() {
    let x = SX::sym("x");
    let nums = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0];
    let cases = [
        (
            "sign",
            x.sign(),
            [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            "heaviside",
            heaviside_adaptor(x),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
        (
            "ramp",
            ramp_adaptor(x),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        ),
        (
            "rectangle",
            rectangle_adaptor(x),
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        ),
        (
            "triangle",
            triangle_adaptor(x),
            [0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 0.75, 0.5, 0.0, 0.0, 0.0],
        ),
    ];

    for (name, expr, reference) in cases {
        for (value, expected) in nums.iter().copied().zip(reference) {
            let vars = HashMap::from([(x.id(), value)]);
            let actual = eval(expr, &vars);
            assert!(
                (actual - expected).abs() <= 1e-10,
                "primitive function sample mismatch: name={name}, actual={actual}, expected={expected}, value={value}"
            );
        }
    }
}

#[rstest]
fn sx_issue107_add_assign_preserves_original_symbolicity() {
    let x = SX::sym("x");
    let y = SX::sym("y");

    let mut z = x;
    z += y;

    assert!(x.is_symbolic());
    assert!(!z.is_symbolic());

    let x = SX::sym("x");
    let y = SX::sym("y");

    let mut z = x;
    z += y;

    assert!(x.is_symbolic());
    assert!(!z.is_symbolic());
}

#[rstest]
fn ad_hessian_matches_casadi_test_hessian() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let z = SX::sym("z");
    let wrt = SXMatrix::dense_column(vec![x, y, z]).unwrap();
    let expr = SXMatrix::scalar(x + 2.0 * y.powi(3) + 3.0 * z.powi(4));

    let gradient = expr.gradient(&wrt).unwrap();
    let hessian = gradient.jacobian(&wrt).unwrap();
    let vars = HashMap::from([(x.id(), 1.2), (y.id(), 2.3), (z.id(), 7.0)]);

    let dense = eval_matrix(&hessian, &vars);
    assert_eq!(hessian.shape(), (3, 3));
    assert_eq!(hessian.ccs().positions(), vec![(1, 1), (2, 2)]);
    assert_abs_diff_eq!(dense[1][1], 27.6, epsilon = 1e-10);
    assert_abs_diff_eq!(dense[2][2], 1764.0, epsilon = 1e-10);
}

#[rstest]
fn ad_jacobian_sx_matches_casadi_test_jacobian_sx_via_dense_adaptor() {
    assert_ad_jacobian_parity_cases();
}

#[rstest]
fn ad_jacobian_matches_casadi_test_jacobian_via_dense_adaptor() {
    for _mode in ["forward", "reverse"] {
        assert_ad_jacobian_parity_cases();
    }
}

#[rstest]
fn ad_jacobian_sparsity_matches_casadi_test_jacsparsity_via_dense_adaptor() {
    assert_ad_jacobian_sparsity_parity_cases();
}

#[rstest]
fn ad_bugglibc_reconstructing_jacobian_after_evaluation_matches_casadi_regression() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let input = SXMatrix::new(
        CCS::from_positions(5, 1, &[(0, 0), (3, 0)]).unwrap(),
        vec![x, y],
    )
    .unwrap();
    let output = SXMatrix::dense_column(vec![x + y, x, y]).unwrap();
    let vars = HashMap::from([(x.id(), 2.0), (y.id(), 7.0)]);

    let first = output.jacobian(&input).unwrap();
    let first_dense = eval_matrix(&first, &vars);
    let second = output.jacobian(&input).unwrap();
    let second_dense = eval_matrix(&second, &vars);

    assert_eq!(first.shape(), (3, 2));
    assert_eq!(second.shape(), (3, 2));
    assert_eq!(first.ccs(), second.ccs());
    assert_eq!(first_dense, second_dense);
}

#[rstest]
fn sx_depends_on_matches_casadi_test_depends_on_via_adaptor() {
    let a = SX::sym("a");
    let b = SX::sym("b");
    let ab = SXMatrix::dense_column(vec![a, b]).unwrap();

    assert!(depends_on_adaptor(
        &SXMatrix::scalar(a.powi(2)),
        &SXMatrix::scalar(a)
    ));
    assert!(depends_on_adaptor(
        &SXMatrix::scalar(a),
        &SXMatrix::scalar(a)
    ));
    assert!(!depends_on_adaptor(
        &SXMatrix::scalar(0.0),
        &SXMatrix::scalar(a)
    ));
    assert!(depends_on_adaptor(&SXMatrix::scalar(a.powi(2)), &ab));
    assert!(depends_on_adaptor(&SXMatrix::scalar(a), &ab));
    assert!(!depends_on_adaptor(&SXMatrix::scalar(0.0), &ab));
    assert!(depends_on_adaptor(&SXMatrix::scalar(b.powi(2)), &ab));
    assert!(depends_on_adaptor(&SXMatrix::scalar(b), &ab));
    assert!(depends_on_adaptor(
        &SXMatrix::scalar(a.powi(2) + b.powi(2)),
        &ab
    ));
    assert!(depends_on_adaptor(&SXMatrix::scalar(a + b), &ab));
    assert!(depends_on_adaptor(
        &SXMatrix::dense_column(vec![0.0.into(), a]).unwrap(),
        &SXMatrix::scalar(a),
    ));
    assert!(depends_on_adaptor(
        &SXMatrix::dense_column(vec![a, 0.0.into()]).unwrap(),
        &SXMatrix::scalar(a),
    ));
    assert!(depends_on_adaptor(
        &SXMatrix::dense_column(vec![a.powi(2), b.powi(2)]).unwrap(),
        &ab,
    ));
    assert!(depends_on_adaptor(
        &SXMatrix::dense_column(vec![a, 0.0.into()]).unwrap(),
        &ab,
    ));
    assert!(depends_on_adaptor(
        &SXMatrix::dense_column(vec![0.0.into(), b]).unwrap(),
        &ab,
    ));
    assert!(depends_on_adaptor(
        &SXMatrix::dense_column(vec![b, 0.0.into()]).unwrap(),
        &ab,
    ));
    assert!(!depends_on_adaptor(
        &SXMatrix::dense_column(vec![0.0.into(), 0.0.into()]).unwrap(),
        &ab,
    ));
}

#[rstest]
fn sx_symvar_matches_casadi_test_symvar_via_adaptor() {
    let a = SX::sym("a");
    let b = SX::sym("b");
    let c = SX::sym("c");
    let expr = (a * b).cos() + c;

    let vars = symvar_adaptor(expr);
    assert_eq!(vars.len(), 3);
    assert_eq!(vars, vec![a, b, c]);
}

#[rstest]
fn sx_contains_matches_casadi_test_contains_via_adaptor() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let z = SX::sym("z");
    let e = y * z;

    let xyz = [
        SXMatrix::scalar(x),
        SXMatrix::scalar(y),
        SXMatrix::scalar(z),
    ];
    let xy = [SXMatrix::scalar(x), SXMatrix::scalar(y)];
    let yz = [SXMatrix::scalar(y), SXMatrix::scalar(z)];
    let ex = [SXMatrix::scalar(e), SXMatrix::scalar(x)];

    assert!(contains_adaptor(&xyz, x).unwrap());
    assert!(!contains_adaptor(&xy, z).unwrap());
    assert!(contains_any_adaptor(&xy, &yz).unwrap());
    assert!(!contains_all_adaptor(&xy, &yz).unwrap());
    assert!(contains_any_adaptor(&xy, &xy).unwrap());
    assert!(contains_all_adaptor(&xy, &xy).unwrap());
    assert!(contains_adaptor(&ex, e).unwrap());

    let err = contains_adaptor(&[SXMatrix::dense_column(vec![x, y]).unwrap()], x).unwrap_err();
    assert!(
        err.to_string()
            .contains("Can only convert 1-by-1 matrices to scalars")
    );
}

#[rstest]
fn sx_pow_matches_casadi_test_pow_via_dense_adaptor() {
    let x = SX::sym("x");
    let zero_pow_nnz = |values: Vec<SX>| {
        SXMatrix::dense_column(values)
            .unwrap()
            .map_nonzeros(|entry| entry.powi(0))
            .nnz()
    };

    assert_eq!(
        zero_pow_nnz(vec![0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()]),
        4
    );
    assert_eq!(zero_pow_nnz(vec![0.0.into(), x, 0.0.into(), 0.0.into()]), 4);
    assert_eq!(
        zero_pow_nnz(vec![0.0.into(), 0.0.into(), 0.0.into(), 0.0.into()]),
        4
    );
    assert_eq!(
        zero_pow_nnz(vec![0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()]),
        4
    );
}

#[rstest]
fn sx_is_regular_matches_casadi_test_is_regular_via_adaptor() {
    let x = SX::sym("x");

    assert!(is_regular_adaptor(&SXMatrix::scalar(0.0)).unwrap());
    assert!(!is_regular_adaptor(&SXMatrix::scalar(f64::INFINITY)).unwrap());
    assert!(
        is_regular_adaptor(&SXMatrix::dense_column(vec![0.0.into(), 1.0.into()]).unwrap()).unwrap()
    );
    assert!(
        !is_regular_adaptor(
            &SXMatrix::dense_column(vec![0.0.into(), f64::INFINITY.into()]).unwrap()
        )
        .unwrap()
    );
    assert!(
        !is_regular_adaptor(&SXMatrix::dense_column(vec![x, f64::INFINITY.into()]).unwrap())
            .unwrap()
    );
    assert!(is_regular_adaptor(&SXMatrix::dense_column(vec![x, x]).unwrap()).is_err());
}

#[rstest]
fn ccs_nz_matches_casadi_test_nz() {
    let positions = vec![(0, 0), (0, 1), (2, 0), (2, 3), (2, 4), (3, 1)];
    let expected = CCS::from_positions(4, 5, &positions).unwrap();
    let rows = positions.iter().map(|(row, _)| *row).collect::<Vec<_>>();
    let cols = positions.iter().map(|(_, col)| *col).collect::<Vec<_>>();
    let actual = CCS::triplet(4, 5, &rows, &cols).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
fn ccs_rowcol_matches_casadi_test_rowcol() {
    let rows = [0, 1, 3];
    let cols = [1, 4];
    let expected_positions = vec![(0, 1), (1, 1), (3, 1), (0, 4), (1, 4), (3, 4)];
    let expected = CCS::from_positions(4, 5, &expected_positions).unwrap();
    let actual = CCS::rowcol(&rows, &cols, 4, 5).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
fn ccs_union_matches_casadi_test_union() {
    let lhs_positions = vec![(0, 0), (0, 1), (2, 0), (3, 1)];
    let rhs_positions = vec![(0, 2), (0, 0), (2, 2)];
    let lhs = CCS::from_positions(4, 5, &lhs_positions).unwrap();
    let rhs = CCS::from_positions(4, 5, &rhs_positions).unwrap();
    let actual = ccs_union_adaptor(&lhs, &rhs);
    let expected =
        CCS::from_positions(4, 5, &[(0, 0), (0, 1), (2, 0), (3, 1), (0, 2), (2, 2)]).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
fn ccs_intersection_matches_casadi_test_intersection() {
    let lhs_positions = vec![(0, 0), (0, 1), (2, 0), (3, 1), (2, 3)];
    let rhs_positions = vec![(0, 2), (0, 0), (2, 2), (2, 3)];
    let lhs = CCS::from_positions(4, 5, &lhs_positions).unwrap();
    let rhs = CCS::from_positions(4, 5, &rhs_positions).unwrap();
    let actual = ccs_intersection_adaptor(&lhs, &rhs);
    let expected = CCS::from_positions(4, 5, &[(0, 0), (2, 3)]).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
fn ccs_get_nz_dense_matches_casadi_test_get_nz_dense() {
    let ccs = CCS::from_positions(4, 5, &[(0, 0), (0, 1), (2, 0), (3, 1)]).unwrap();
    let mut dense_linear = vec![0.0; ccs.nrow() * ccs.ncol()];
    for linear in ccs.find() {
        dense_linear[linear] = 1.0;
    }

    for linear in ccs.find() {
        assert_abs_diff_eq!(dense_linear[linear], 1.0, epsilon = 1e-12);
    }
}

#[rstest]
fn ccs_get_lower_matches_casadi_test_splower() {
    let ccs = CCS::new(4, 3, vec![0, 2, 2, 3], vec![1, 2, 1]).unwrap();
    let lower = ccs.get_lower().unwrap();
    let expected = CCS::from_positions(4, 3, &[(1, 0), (2, 0)]).unwrap();
    assert_eq!(lower, expected);
}

#[rstest]
fn ccs_pattern_inverse_matches_casadi_test_inverse() {
    for ccs in [
        CCS::from_positions(4, 4, &[(0, 0), (1, 0), (0, 1), (2, 1), (1, 2), (3, 3)]).unwrap(),
        CCS::dense(4, 4).unwrap(),
        CCS::empty(4, 4),
        CCS::lower_triangular(4),
        CCS::lower_triangular(4).transpose(),
    ] {
        let inverse = ccs.pattern_inverse().unwrap();
        let expected = CCS::from_positions(
            ccs.nrow(),
            ccs.ncol(),
            &(0..ccs.ncol())
                .flat_map(|col| (0..ccs.nrow()).map(move |row| (row, col)))
                .filter(|&(row, col)| ccs.nz_index(row, col).is_none())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(inverse, expected);
    }
}

#[rstest]
fn ccs_kron_matches_casadi_test_kron() {
    let lhs = CCS::from_positions(2, 3, &[(0, 0), (1, 0), (1, 1), (0, 2)]).unwrap();
    let rhs = CCS::from_positions(
        4,
        3,
        &[
            (0, 0),
            (1, 0),
            (3, 0),
            (1, 1),
            (3, 1),
            (1, 2),
            (2, 2),
            (3, 2),
        ],
    )
    .unwrap();
    let actual = lhs.kron(&rhs).unwrap();
    let rhs_positions = rhs.positions();
    let rhs_nrow = rhs.nrow();
    let rhs_ncol = rhs.ncol();
    let expected_positions = lhs
        .positions()
        .into_iter()
        .flat_map(|(lhs_row, lhs_col)| {
            rhs_positions
                .iter()
                .copied()
                .map(move |(rhs_row, rhs_col)| {
                    (lhs_row * rhs_nrow + rhs_row, lhs_col * rhs_ncol + rhs_col)
                })
        })
        .collect::<Vec<_>>();
    let expected = CCS::from_positions(8, 9, &expected_positions).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.nrow(), lhs.nrow() * rhs_nrow);
    assert_eq!(actual.ncol(), lhs.ncol() * rhs_ncol);
    assert_eq!(actual.nnz(), lhs.nnz() * rhs.nnz());
}

#[rstest]
fn ccs_get_nz_matches_casadi_test_nz_method() {
    let ccs = CCS::from_positions(4, 5, &[(0, 0), (2, 0), (1, 2), (3, 4)]).unwrap();
    let queries = vec![0, 2, 5, 9, 13, 19];
    let actual = ccs.get_nz(&queries).unwrap();
    let expected = queries
        .iter()
        .copied()
        .map(|linear| {
            let row = linear % ccs.nrow();
            let col = linear / ccs.nrow();
            ccs.nz_index(row, col).map_or(-1, |idx| idx as isize)
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[rstest]
fn ccs_serialize_matches_casadi_test_serialize() {
    for ccs in [
        CCS::empty(0, 0),
        CCS::dense(4, 5).unwrap(),
        CCS::lower_triangular(5),
    ] {
        let serialized = ccs.serialize();
        let deserialized = CCS::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, ccs);
    }
}

#[rstest]
fn ccs_is_subset_matches_casadi_test_is_subset() {
    let cases = [
        (CCS::lower_triangular(3), CCS::dense(3, 3).unwrap()),
        (CCS::diag(3), CCS::dense(3, 3).unwrap()),
        (CCS::diag(3), CCS::lower_triangular(3)),
        (CCS::empty(3, 3), CCS::lower_triangular(3)),
    ];

    for (lhs, rhs) in cases {
        assert!(ccs_is_subset_adaptor(&lhs, &rhs));
        assert!(!ccs_is_subset_adaptor(&rhs, &lhs));
    }
}

#[rstest]
fn ccs_get_ccs_matches_casadi_test_get_ccs() {
    let positions = vec![(0, 0), (0, 1), (2, 0), (2, 3), (3, 3), (2, 4), (3, 1)];
    let ccs = CCS::from_positions(4, 5, &positions).unwrap();
    let (col_ptrs, row_indices) = ccs.get_ccs();
    assert_eq!(col_ptrs, vec![0, 2, 4, 4, 6, 7]);
    assert_eq!(row_indices, vec![0, 2, 0, 3, 2, 3, 2]);

    let (row_ptrs, col_indices) = ccs.transpose().get_crs();
    assert_eq!(row_ptrs, col_ptrs);
    assert_eq!(col_indices, row_indices);
}

#[rstest]
fn ccs_find_nonzero_matches_casadi_test_find_nonzero() {
    let positions = vec![(0, 0), (2, 0), (4, 1), (1, 3), (3, 7), (19, 9)];
    let ccs = CCS::from_positions(20, 10, &positions).unwrap();
    let rebuilt = CCS::nonzeros(20, 10, &ccs.find()).unwrap();
    assert_eq!(rebuilt, ccs);
}

#[rstest]
fn ccs_reshape_matches_casadi_test_reshape() {
    let positions = vec![(0, 0), (0, 1), (2, 0), (2, 3), (2, 4), (3, 1)];
    let original = CCS::from_positions(4, 5, &positions).unwrap();
    let reshaped = original.reshape(2, 10).unwrap();
    let expected =
        CCS::from_positions(2, 10, &[(0, 0), (0, 1), (0, 2), (1, 3), (0, 7), (0, 9)]).unwrap();
    assert_eq!(reshaped, expected);
}

#[rstest]
fn ccs_diag_matches_casadi_test_diag() {
    let matrix = CCS::from_positions(5, 5, &[(1, 1), (2, 4), (3, 3)]).unwrap();
    let (diag_column, mapping) = matrix.get_diag().unwrap();
    let expected_column = CCS::from_positions(5, 1, &[(1, 0), (3, 0)]).unwrap();
    assert_eq!(diag_column, expected_column);
    assert_eq!(mapping, vec![0, 1]);

    let column = CCS::from_positions(5, 1, &[(1, 0), (2, 0), (4, 0)]).unwrap();
    let (diag_matrix, mapping) = column.get_diag().unwrap();
    let expected_diag = CCS::from_positions(5, 5, &[(1, 1), (2, 2), (4, 4)]).unwrap();
    assert_eq!(diag_matrix, expected_diag);
    assert_eq!(mapping, vec![0, 1, 2]);

    let row = CCS::from_positions(1, 5, &[(0, 1), (0, 2), (0, 4)]).unwrap();
    let (diag_matrix, mapping) = row.get_diag().unwrap();
    assert_eq!(diag_matrix, expected_diag);
    assert_eq!(mapping, vec![0, 1, 2]);
}

#[rstest]
fn ccs_enlarge_matches_casadi_test_enlarge() {
    let dense = CCS::dense(3, 4).unwrap();
    let dense_enlarged = dense.enlarge(7, 8, &[1, 2, 4], &[0, 3, 4, 6]).unwrap();
    let dense_expected = CCS::from_positions(
        7,
        8,
        &[
            (1, 0),
            (2, 0),
            (4, 0),
            (1, 3),
            (2, 3),
            (4, 3),
            (1, 4),
            (2, 4),
            (4, 4),
            (1, 6),
            (2, 6),
            (4, 6),
        ],
    )
    .unwrap();
    assert_eq!(dense_enlarged, dense_expected);

    let sparse = CCS::new(4, 3, vec![0, 2, 2, 3], vec![1, 2, 1])
        .unwrap()
        .transpose();
    let sparse_enlarged = sparse.enlarge(7, 8, &[1, 2, 4], &[0, 3, 4, 6]).unwrap();
    let sparse_expected = CCS::from_positions(7, 8, &[(1, 3), (1, 4), (4, 3)]).unwrap();
    assert_eq!(sparse_enlarged, sparse_expected);
}

#[rstest]
fn sx_function_constructors_match_casadi_test_sx_func() {
    let y = SXMatrix::sym_dense("y", 2, 3).unwrap();
    let function = SXFunction::new(
        "f",
        vec![named_matrix("y", y.clone())],
        vec![named_matrix("out", y)],
    )
    .unwrap();
    assert_eq!(function.n_in(), 1);
    assert_eq!(function.n_out(), 1);
    assert_eq!(function.size_in(0), (2, 3));
    assert_eq!(function.size_out(0), (2, 3));
}

#[rstest]
fn sx_conversion_matches_casadi_test_sxconversion() {
    let y = SX::sym("y");
    let matrix = SXMatrix::sym_dense("x", 3, 3).unwrap();
    let constant = SX::from(2.3);
    assert!(y.is_symbolic());
    assert_eq!(matrix.shape(), (3, 3));
    assert_eq!(matrix.nnz(), 9);
    assert_abs_diff_eq!(eval(constant, &HashMap::new()), 2.3, epsilon = 1e-12);
}

#[rstest]
fn sx_function_typemap_constructors_match_casadi_test_sx_func2() {
    let numeric_scalar = SXMatrix::scalar(2.3);
    let symbolic_scalar = SXMatrix::scalar(SX::sym("x"));
    for matrix in [numeric_scalar, symbolic_scalar] {
        let roundtrip = matrix.transpose().transpose();
        assert_eq!(roundtrip.shape(), (1, 1));
        assert_eq!(roundtrip.nnz(), 1);
    }
}

#[rstest]
fn sx_function_vector_constructors_match_casadi_test_sx_func3() {
    let x = SXMatrix::dense_column(vec![SX::sym("x0"), SX::sym("x1"), SX::sym("x2")]).unwrap();
    let y = SXMatrix::scalar(SX::sym("y"));
    let empty = SXMatrix::new(CCS::empty(0, 1), Vec::new()).unwrap();
    assert_eq!(
        dense_vcat_adaptor(&[x.clone(), x.clone()]).unwrap().shape(),
        (6, 1)
    );
    assert_eq!(dense_vcat_adaptor(&[y.clone(), y]).unwrap().shape(), (2, 1));
    assert_eq!(dense_vcat_adaptor(&[x, empty]).unwrap().shape(), (3, 1));
}

#[rstest]
fn sx_function_evaluation_matches_casadi_test_sx1() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let input = SXMatrix::dense_column(vec![x, y]).unwrap();
    let output = SXMatrix::dense_column(vec![x + y, x * y, x.powi(2) + y.powi(3)]).unwrap();
    let function = SXFunction::new(
        "f",
        vec![named_matrix("xy", input.clone())],
        vec![named_matrix("out", output.clone())],
    )
    .unwrap();
    let outputs =
        jit_eval_function(&function, &[DenseValue::from_rows(&[&[2.0], &[3.0]])]).unwrap();
    assert_dense_value_close(
        &outputs[0],
        &DenseValue::from_rows(&[&[5.0], &[6.0], &[31.0]]),
    );

    let jacobian = output.jacobian(&input).unwrap();
    let jacobian_fn = SXFunction::new(
        "jacobian",
        vec![named_matrix("xy", input)],
        vec![named_matrix("jac", jacobian)],
    )
    .unwrap();
    let jacobian_outputs =
        jit_eval_function(&jacobian_fn, &[DenseValue::from_rows(&[&[2.0], &[3.0]])]).unwrap();
    assert_dense_value_close(
        &jacobian_outputs[0],
        &DenseValue::from_rows(&[&[1.0, 1.0], &[3.0, 2.0], &[4.0, 27.0]]),
    );
}

#[rstest]
fn sx_function_evaluation_matches_casadi_test_sx2() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let input = SXMatrix::dense_column(vec![x, y]).unwrap();
    let output = SXMatrix::dense_column(vec![3.0 - (x * x).sin() - y, y.sqrt() * x]).unwrap();
    let function = SXFunction::new(
        "fcn",
        vec![named_matrix("xy", input)],
        vec![named_matrix("out", output)],
    )
    .unwrap();
    let outputs =
        jit_eval_function(&function, &[DenseValue::from_rows(&[&[2.0], &[3.0]])]).unwrap();
    assert_dense_value_close(
        &outputs[0],
        &DenseValue::from_rows(&[&[3.0 - 4.0_f64.sin() - 3.0], &[3.0_f64.sqrt() * 2.0]]),
    );
}

#[rstest]
fn sx_function_eval_matches_casadi_test_eval() {
    let x = SXMatrix::sym_dense("x", 2, 2).unwrap();
    let y = SXMatrix::sym_dense("y", 2, 2).unwrap();
    let output = dense_matrix_binary_adaptor(&x, &y, MatrixBinaryOp::Mul);
    let function = SXFunction::new(
        "eval",
        vec![named_matrix("x", x), named_matrix("y", y)],
        vec![named_matrix("out", output)],
    )
    .unwrap();
    let x0 = DenseValue::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let y0 = DenseValue::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
    let outputs = jit_eval_function(&function, &[x0, y0]).unwrap();
    assert_dense_value_close(
        &outputs[0],
        &DenseValue::from_rows(&[&[5.0, 12.0], &[21.0, 32.0]]),
    );
}

#[rstest]
fn sx_symbolcheck_matches_casadi_test_symbolcheck() {
    let err = SXFunction::new(
        "f",
        vec![named_matrix("x", SXMatrix::scalar(0.0))],
        vec![named_matrix("out", SXMatrix::scalar(SX::sym("x")))],
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("must contain only symbolic primitives")
    );
}

#[rstest]
fn sx_sparseconstr_matches_casadi_test_sparseconstr() {
    assert_dense_value_close(
        &dense_pattern_from_ccs(&CCS::lower_triangular(3)),
        &DenseValue::from_rows(&[&[1.0, 0.0, 0.0], &[1.0, 1.0, 0.0], &[1.0, 1.0, 1.0]]),
    );
    assert_dense_value_close(
        &dense_pattern_from_ccs(&CCS::diag(3)),
        &DenseValue::from_rows(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]),
    );
}

#[rstest]
fn sx_null_matches_casadi_test_null() {
    let x = SX::sym("x");
    let empty = SXMatrix::new(CCS::empty(0, 0), Vec::new()).unwrap();

    let unary = SXFunction::new(
        "null_out",
        vec![named_matrix("x", SXMatrix::scalar(x))],
        vec![
            named_matrix("square", SXMatrix::scalar(x.powi(2))),
            named_matrix("empty", empty.clone()),
        ],
    )
    .unwrap();
    let unary_outputs = jit_eval_function(&unary, &[DenseValue::scalar(0.0)]).unwrap();
    assert_eq!(unary_outputs[1], DenseValue::empty(0, 0));

    let binary = SXFunction::new(
        "null_io",
        vec![
            named_matrix("x", SXMatrix::scalar(x)),
            named_matrix("empty_in", empty.clone()),
        ],
        vec![
            named_matrix("square", SXMatrix::scalar(x.powi(2))),
            named_matrix("empty_out", empty),
        ],
    )
    .unwrap();
    for empty_input in [
        DenseValue::empty(0, 0),
        DenseValue::empty(0, 1),
        DenseValue::empty(1, 0),
    ] {
        let outputs = jit_eval_function(&binary, &[DenseValue::scalar(0.0), empty_input]).unwrap();
        assert_eq!(outputs[1], DenseValue::empty(0, 0));
    }
}

#[rstest]
fn sx_evalchecking_matches_casadi_test_evalchecking() {
    let x = SXMatrix::sym_dense("x", 1, 5).unwrap();
    let output = x.map_nonzeros(SX::sqr);
    let function = SXFunction::new(
        "evalchecking",
        vec![named_matrix("x", x)],
        vec![named_matrix("out", output)],
    )
    .unwrap();
    let wrong_short =
        jit_eval_function(&function, &[DenseValue::from_rows(&[&[1.0, 2.0, 3.0]])]).unwrap_err();
    assert!(wrong_short.to_string().contains("input shape mismatch"));

    let wrong_long = jit_eval_function(
        &function,
        &[DenseValue::from_rows(&[&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])],
    )
    .unwrap_err();
    assert!(wrong_long.to_string().contains("input shape mismatch"));

    let outputs = jit_eval_function(
        &function,
        &[DenseValue::from_rows(&[
            &[1.0],
            &[2.0],
            &[3.0],
            &[4.0],
            &[5.0],
        ])],
    )
    .unwrap();
    assert_dense_value_close(
        &outputs[0],
        &DenseValue::from_rows(&[&[1.0, 4.0, 9.0, 16.0, 25.0]]),
    );
}

#[rstest]
fn ccs_refcount_matches_casadi_test_refcount() {
    let ccs = CCS::lower_triangular(4);
    let symbols = (0..ccs.nnz())
        .map(|index| SX::sym(format!("x_{index}")))
        .collect::<Vec<_>>();
    let x = SXMatrix::new(ccs, symbols).unwrap();
    let product = dense_matrix_multiply_adaptor(&x, &x).unwrap();
    assert_eq!(product.shape(), (4, 4));
    assert_eq!(product.shape().0 * product.shape().1, 16);
}
