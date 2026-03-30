use anyhow::{Result, bail};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{NonnegativeConeT, ZeroConeT};
use clarabel::solver::implementations::default::DefaultSettingsBuilder;
use clarabel::solver::{DefaultSolver, IPSolver, SolverStatus};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::io::{self, IsTerminal};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(feature = "ipopt")]
mod ipopt_backend;

#[cfg(feature = "ipopt")]
pub use ipopt_backend::{
    IpoptMuStrategy, IpoptOptions, IpoptSolveError, IpoptSummary, solve_nlp_ipopt,
};

pub type Index = usize;
const NLP_INF: f64 = 1e20;
const BOX_LABEL_WIDTH: usize = 13;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendTimingMetadata {
    pub function_creation_time: Option<Duration>,
    pub jit_time: Option<Duration>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalTimingStat {
    pub calls: Index,
    pub total_time: Duration,
}

impl EvalTimingStat {
    fn record(&mut self, elapsed: Duration) {
        self.calls += 1;
        self.total_time += elapsed;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClarabelSqpProfiling {
    pub objective_value: EvalTimingStat,
    pub objective_gradient: EvalTimingStat,
    pub equality_values: EvalTimingStat,
    pub inequality_values: EvalTimingStat,
    pub equality_jacobian_values: EvalTimingStat,
    pub inequality_jacobian_values: EvalTimingStat,
    pub lagrangian_hessian_values: EvalTimingStat,
    pub qp_setups: Index,
    pub qp_setup_time: Duration,
    pub qp_solves: Index,
    pub qp_solve_time: Duration,
    pub preprocessing_time: Duration,
    pub total_time: Duration,
    pub unaccounted_time: Duration,
    pub backend_timing: BackendTimingMetadata,
}

impl ClarabelSqpProfiling {
    fn total_callback_time(&self) -> Duration {
        self.objective_value.total_time
            + self.objective_gradient.total_time
            + self.equality_values.total_time
            + self.inequality_values.total_time
            + self.equality_jacobian_values.total_time
            + self.inequality_jacobian_values.total_time
            + self.lagrangian_hessian_values.total_time
    }

    fn total_callback_calls(&self) -> Index {
        self.objective_value.calls
            + self.objective_gradient.calls
            + self.equality_values.calls
            + self.inequality_values.calls
            + self.equality_jacobian_values.calls
            + self.inequality_jacobian_values.calls
            + self.lagrangian_hessian_values.calls
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CCS {
    pub nrow: Index,
    pub ncol: Index,
    pub col_ptrs: Vec<Index>,
    pub row_indices: Vec<Index>,
}

impl CCS {
    pub fn new(nrow: Index, ncol: Index, col_ptrs: Vec<Index>, row_indices: Vec<Index>) -> Self {
        Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        }
    }

    pub fn empty(nrow: Index, ncol: Index) -> Self {
        Self {
            nrow,
            ncol,
            col_ptrs: vec![0; ncol + 1],
            row_indices: Vec::new(),
        }
    }

    pub fn dense(nrow: Index, ncol: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(ncol + 1);
        let mut row_indices = Vec::with_capacity(nrow * ncol);
        col_ptrs.push(0);
        for _ in 0..ncol {
            row_indices.extend(0..nrow);
            col_ptrs.push(row_indices.len());
        }
        Self {
            nrow,
            ncol,
            col_ptrs,
            row_indices,
        }
    }

    pub fn nnz(&self) -> Index {
        self.row_indices.len()
    }

    pub fn lower_triangular_dense(size: Index) -> Self {
        let mut col_ptrs = Vec::with_capacity(size + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col in 0..size {
            row_indices.extend(col..size);
            col_ptrs.push(row_indices.len());
        }
        Self {
            nrow: size,
            ncol: size,
            col_ptrs,
            row_indices,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParameterMatrix<'a> {
    pub ccs: &'a CCS,
    pub values: &'a [f64],
}

pub trait CompiledNlpProblem {
    fn dimension(&self) -> Index;
    fn parameter_count(&self) -> Index;
    fn parameter_ccs(&self, parameter_index: Index) -> &CCS;
    fn variable_bounds(&self, lower: &mut [f64], upper: &mut [f64]) -> bool {
        lower.fill(-NLP_INF);
        upper.fill(NLP_INF);
        true
    }
    fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        BackendTimingMetadata::default()
    }
    fn equality_count(&self) -> Index;
    fn inequality_count(&self) -> Index;
    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64;
    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]);
    fn equality_jacobian_ccs(&self) -> &CCS;
    fn equality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]);
    fn equality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    );
    fn inequality_jacobian_ccs(&self) -> &CCS;
    fn inequality_values(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]);
    fn inequality_jacobian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    );
    fn lagrangian_hessian_ccs(&self) -> &CCS;
    fn lagrangian_hessian_values(
        &self,
        x: &[f64],
        parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        out: &mut [f64],
    );
}

#[derive(Clone, Debug)]
pub struct ClarabelSqpOptions {
    pub max_iters: Index,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub complementarity_tol: f64,
    pub merit_penalty: f64,
    pub regularization: f64,
    pub armijo_c1: f64,
    pub line_search_beta: f64,
    pub min_step: f64,
    pub penalty_increase_factor: f64,
    pub max_penalty_updates: Index,
    pub verbose: bool,
}

impl Default for ClarabelSqpOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            dual_tol: 1e-6,
            constraint_tol: 1e-6,
            complementarity_tol: 1e-6,
            merit_penalty: 10.0,
            regularization: 1e-6,
            armijo_c1: 1e-4,
            line_search_beta: 0.5,
            min_step: 1e-8,
            penalty_increase_factor: 10.0,
            max_penalty_updates: 8,
            verbose: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClarabelSqpSummary {
    pub x: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub equality_inf_norm: f64,
    pub inequality_inf_norm: f64,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: f64,
    pub profiling: ClarabelSqpProfiling,
}

#[derive(Debug, Error)]
pub enum ClarabelSqpError {
    #[error("invalid SQP input: {0}")]
    InvalidInput(String),
    #[error("clarabel SQP failed to converge in {iterations} iterations")]
    MaxIterations { iterations: Index },
    #[error("clarabel solver setup failed: {0}")]
    Setup(String),
    #[error("clarabel returned status {status:?}")]
    QpSolve { status: SolverStatus },
    #[error("unconstrained SQP subproblem solve failed")]
    UnconstrainedStepSolve,
    #[error(
        "armijo line search failed to find sufficient decrease (directional derivative {directional_derivative}, step inf-norm {step_inf_norm}, penalty {penalty})"
    )]
    LineSearchFailed {
        directional_derivative: f64,
        step_inf_norm: f64,
        penalty: f64,
    },
    #[error(
        "sqp stalled with step inf-norm {step_inf_norm}, primal inf-norm {primal_inf_norm}, dual inf-norm {dual_inf_norm}, complementarity inf-norm {complementarity_inf_norm}"
    )]
    Stalled {
        step_inf_norm: f64,
        primal_inf_norm: f64,
        dual_inf_norm: f64,
        complementarity_inf_norm: f64,
    },
}

fn ccs_to_dense(sp: &CCS, values: &[f64]) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(sp.nrow, sp.ncol);
    for col in 0..sp.ncol {
        let start = sp.col_ptrs[col];
        let end = sp.col_ptrs[col + 1];
        for (offset, &row) in sp.row_indices[start..end].iter().enumerate() {
            dense[(row, col)] = values[start + offset];
        }
    }
    dense
}

fn lower_triangle_to_symmetric_dense(sp: &CCS, values: &[f64]) -> DMatrix<f64> {
    let mut dense = DMatrix::<f64>::zeros(sp.nrow, sp.ncol);
    for col in 0..sp.ncol {
        let start = sp.col_ptrs[col];
        let end = sp.col_ptrs[col + 1];
        for (offset, &row) in sp.row_indices[start..end].iter().enumerate() {
            let value = values[start + offset];
            dense[(row, col)] = value;
            dense[(col, row)] = value;
        }
    }
    dense
}

fn dense_to_csc_upper(matrix: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = matrix.ncols();
    let mut col_ptrs = Vec::with_capacity(n + 1);
    let mut row_vals = Vec::new();
    let mut nz_vals = Vec::new();
    col_ptrs.push(0);
    for col in 0..n {
        for row in 0..=col {
            let value = matrix[(row, col)];
            if value.abs() > 1e-12 || row == col {
                row_vals.push(row);
                nz_vals.push(value);
            }
        }
        col_ptrs.push(row_vals.len());
    }
    CscMatrix::new(n, n, col_ptrs, row_vals, nz_vals)
}

fn dense_to_csc(matrix: &DMatrix<f64>) -> CscMatrix<f64> {
    let ncol = matrix.ncols();
    let nrow = matrix.nrows();
    let mut col_ptrs = Vec::with_capacity(ncol + 1);
    let mut row_vals = Vec::new();
    let mut nz_vals = Vec::new();
    col_ptrs.push(0);
    for col in 0..ncol {
        for row in 0..nrow {
            let value = matrix[(row, col)];
            if value.abs() > 1e-12 {
                row_vals.push(row);
                nz_vals.push(value);
            }
        }
        col_ptrs.push(row_vals.len());
    }
    CscMatrix::new(nrow, ncol, col_ptrs, row_vals, nz_vals)
}

fn stack_jacobians(
    equality_jacobian: &DMatrix<f64>,
    inequality_jacobian: &DMatrix<f64>,
) -> DMatrix<f64> {
    let ncol = equality_jacobian.ncols().max(inequality_jacobian.ncols());
    let total_rows = equality_jacobian.nrows() + inequality_jacobian.nrows();
    let mut stacked = DMatrix::<f64>::zeros(total_rows, ncol);
    for row in 0..equality_jacobian.nrows() {
        for col in 0..equality_jacobian.ncols() {
            stacked[(row, col)] = equality_jacobian[(row, col)];
        }
    }
    let row_offset = equality_jacobian.nrows();
    for row in 0..inequality_jacobian.nrows() {
        for col in 0..inequality_jacobian.ncols() {
            stacked[(row_offset + row, col)] = inequality_jacobian[(row, col)];
        }
    }
    stacked
}

#[derive(Clone, Debug, Default)]
struct BoundConstraints {
    lower_indices: Vec<Index>,
    lower_values: Vec<f64>,
    upper_indices: Vec<Index>,
    upper_values: Vec<f64>,
}

impl BoundConstraints {
    fn total_count(&self) -> Index {
        self.lower_indices.len() + self.upper_indices.len()
    }
}

fn collect_bound_constraints<P>(
    problem: &P,
) -> std::result::Result<BoundConstraints, ClarabelSqpError>
where
    P: CompiledNlpProblem,
{
    let dimension = problem.dimension();
    let mut lower = vec![0.0; dimension];
    let mut upper = vec![0.0; dimension];
    if !problem.variable_bounds(&mut lower, &mut upper) {
        return Ok(BoundConstraints::default());
    }

    let mut bounds = BoundConstraints::default();
    for idx in 0..dimension {
        if lower[idx] > upper[idx] {
            return Err(ClarabelSqpError::InvalidInput(format!(
                "variable bound interval is empty at index {idx}: lower={} > upper={}",
                lower[idx], upper[idx]
            )));
        }
        if lower[idx] > -NLP_INF {
            bounds.lower_indices.push(idx);
            bounds.lower_values.push(lower[idx]);
        }
        if upper[idx] < NLP_INF {
            bounds.upper_indices.push(idx);
            bounds.upper_values.push(upper[idx]);
        }
    }
    Ok(bounds)
}

fn build_bound_jacobian(bounds: &BoundConstraints, dimension: Index) -> DMatrix<f64> {
    let mut jacobian = DMatrix::<f64>::zeros(bounds.total_count(), dimension);
    for (row, &idx) in bounds.lower_indices.iter().enumerate() {
        jacobian[(row, idx)] = -1.0;
    }
    let row_offset = bounds.lower_indices.len();
    for (row, &idx) in bounds.upper_indices.iter().enumerate() {
        jacobian[(row_offset + row, idx)] = 1.0;
    }
    jacobian
}

fn augment_inequality_values(
    nonlinear_values: &[f64],
    x: &[f64],
    bounds: &BoundConstraints,
    out: &mut [f64],
) {
    debug_assert_eq!(
        out.len(),
        nonlinear_values.len() + bounds.lower_indices.len() + bounds.upper_indices.len()
    );
    out[..nonlinear_values.len()].copy_from_slice(nonlinear_values);
    let mut cursor = nonlinear_values.len();
    for (&idx, &bound) in bounds.lower_indices.iter().zip(bounds.lower_values.iter()) {
        out[cursor] = bound - x[idx];
        cursor += 1;
    }
    for (&idx, &bound) in bounds.upper_indices.iter().zip(bounds.upper_values.iter()) {
        out[cursor] = x[idx] - bound;
        cursor += 1;
    }
}

fn split_augmented_inequality_multipliers(
    multipliers: &[f64],
    nonlinear_count: Index,
    lower_bound_count: Index,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let nonlinear = multipliers[..nonlinear_count].to_vec();
    let lower = multipliers[nonlinear_count..nonlinear_count + lower_bound_count].to_vec();
    let upper = multipliers[nonlinear_count + lower_bound_count..].to_vec();
    (nonlinear, lower, upper)
}

fn inf_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn positive_part(value: f64) -> f64 {
    value.max(0.0)
}

fn positive_part_inf_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0, |acc, value| acc.max(positive_part(*value)))
}

fn complementarity_inf_norm(inequality_values: &[f64], inequality_multipliers: &[f64]) -> f64 {
    inequality_values
        .iter()
        .zip(inequality_multipliers.iter())
        .fold(0.0, |acc, (value, multiplier)| {
            acc.max((value * multiplier).abs())
        })
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| lhs_value * rhs_value)
        .sum()
}

fn mat_vec(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; matrix.nrows()];
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            product[row] += matrix[(row, col)] * vector[col];
        }
    }
    product
}

fn lagrangian_gradient(
    gradient: &[f64],
    equality_jacobian: &DMatrix<f64>,
    equality_multipliers: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    inequality_multipliers: &[f64],
) -> Vec<f64> {
    let mut residual = gradient.to_vec();
    for row in 0..equality_jacobian.nrows() {
        let lambda = equality_multipliers[row];
        for col in 0..equality_jacobian.ncols() {
            residual[col] += equality_jacobian[(row, col)] * lambda;
        }
    }
    for row in 0..inequality_jacobian.nrows() {
        let mu = inequality_multipliers[row];
        for col in 0..inequality_jacobian.ncols() {
            residual[col] += inequality_jacobian[(row, col)] * mu;
        }
    }
    residual
}

fn exact_merit_value(
    objective_value: f64,
    equality_values: &[f64],
    inequality_values: &[f64],
    penalty: f64,
) -> f64 {
    objective_value
        + penalty
            * (equality_values.iter().map(|value| value.abs()).sum::<f64>()
                + inequality_values
                    .iter()
                    .map(|value| positive_part(*value))
                    .sum::<f64>())
}

fn exact_merit_directional_derivative(
    gradient: &[f64],
    equality_values: &[f64],
    equality_jacobian: &DMatrix<f64>,
    inequality_values: &[f64],
    inequality_jacobian: &DMatrix<f64>,
    step: &[f64],
    penalty: f64,
) -> f64 {
    let equality_directional = mat_vec(equality_jacobian, step);
    let inequality_directional = mat_vec(inequality_jacobian, step);
    let equality_penalty_rate = equality_values
        .iter()
        .zip(equality_directional.iter())
        .map(|(value, directional)| {
            if *value > 0.0 {
                *directional
            } else if *value < 0.0 {
                -*directional
            } else {
                directional.abs()
            }
        })
        .sum::<f64>();
    let inequality_penalty_rate = inequality_values
        .iter()
        .zip(inequality_directional.iter())
        .map(|(value, directional)| {
            if *value > 0.0 {
                *directional
            } else if *value < 0.0 {
                0.0
            } else {
                positive_part(*directional)
            }
        })
        .sum::<f64>();
    dot(gradient, step) + penalty * (equality_penalty_rate + inequality_penalty_rate)
}

fn update_merit_penalty(
    current_penalty: f64,
    equality_multipliers: &[f64],
    inequality_multipliers: &[f64],
) -> f64 {
    let equality_multiplier_inf = inf_norm(equality_multipliers);
    let inequality_multiplier_inf = inequality_multipliers
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(positive_part(*value)));
    current_penalty.max(equality_multiplier_inf.max(inequality_multiplier_inf) + 1.0)
}

fn regularize_hessian(hessian: &mut DMatrix<f64>, regularization: f64) {
    let eigen = SymmetricEigen::new(hessian.clone());
    let min_eig = eigen
        .eigenvalues
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value));
    let shift = if min_eig < regularization {
        regularization - min_eig
    } else {
        regularization
    };
    for idx in 0..hessian.nrows() {
        hessian[(idx, idx)] += shift;
    }
}

fn split_multipliers(multipliers: &[f64], equality_count: Index) -> (Vec<f64>, Vec<f64>) {
    let equality = multipliers[..equality_count].to_vec();
    let inequality = multipliers[equality_count..]
        .iter()
        .map(|value| positive_part(*value))
        .collect::<Vec<_>>();
    (equality, inequality)
}

fn solve_unconstrained_quadratic_step(
    hessian: &DMatrix<f64>,
    gradient: &[f64],
) -> std::result::Result<Vec<f64>, ClarabelSqpError> {
    let rhs = DVector::<f64>::from_iterator(gradient.len(), gradient.iter().map(|value| -value));
    let lu = hessian.clone().lu();
    let Some(step) = lu.solve(&rhs) else {
        return Err(ClarabelSqpError::UnconstrainedStepSolve);
    };
    Ok(step.iter().copied().collect())
}

fn trial_merit<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    buffers: (&mut [f64], &mut [f64], &mut [f64]),
    bounds: &BoundConstraints,
    penalty: f64,
    timing: (&mut ClarabelSqpProfiling, &mut Duration),
) -> f64
where
    P: CompiledNlpProblem,
{
    let (equality_values, inequality_values, augmented_inequality_values) = buffers;
    let (profiling, iteration_callback_time) = timing;
    time_callback(
        &mut profiling.equality_values,
        iteration_callback_time,
        || problem.equality_values(x, parameters, equality_values),
    );
    time_callback(
        &mut profiling.inequality_values,
        iteration_callback_time,
        || problem.inequality_values(x, parameters, inequality_values),
    );
    augment_inequality_values(inequality_values, x, bounds, augmented_inequality_values);
    exact_merit_value(
        time_callback(
            &mut profiling.objective_value,
            iteration_callback_time,
            || problem.objective_value(x, parameters),
        ),
        equality_values,
        augmented_inequality_values,
        penalty,
    )
}

struct SqpIterationLog {
    iteration: Index,
    flags: u8,
    objective_value: f64,
    equality_inf: f64,
    inequality_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    step_inf_norm: Option<f64>,
    merit_penalty: f64,
    alpha: Option<f64>,
    line_search_iterations: Option<Index>,
    qp_status: Option<SolverStatus>,
    qp_iterations: Option<u32>,
    qp_solve_time_secs: Option<f64>,
    constraint_tol: f64,
    dual_tol: f64,
    complementarity_tol: f64,
}

#[derive(Default)]
struct SqpEventLegendState {
    seen: u8,
}

const SQP_LOG_ITERATION_LIMIT_REACHED: u8 = 1 << 0;
const SQP_LOG_HAS_EQUALITIES: u8 = 1 << 1;
const SQP_LOG_HAS_INEQUALITIES: u8 = 1 << 2;
const SQP_LOG_PENALTY_UPDATED: u8 = 1 << 3;
const SQP_EVENT_SEEN_PENALTY: u8 = 1 << 0;
const SQP_EVENT_SEEN_LINE_SEARCH: u8 = 1 << 1;
const SQP_EVENT_SEEN_QP: u8 = 1 << 2;
const SQP_EVENT_SEEN_MAX_ITER: u8 = 1 << 3;

impl SqpIterationLog {
    fn flag(&self, bit: u8) -> bool {
        self.flags & bit != 0
    }
}

impl SqpEventLegendState {
    fn mark_if_new(&mut self, bit: u8) -> bool {
        let is_new = self.seen & bit == 0;
        self.seen |= bit;
        is_new
    }
}

fn ansi_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none())
}

fn style(text: &str, code: &str) -> String {
    if ansi_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn style_bold(text: &str) -> String {
    style(text, "1")
}

fn style_cyan_bold(text: &str) -> String {
    style(text, "1;36")
}

fn style_green_bold(text: &str) -> String {
    style(text, "1;32")
}

fn style_yellow_bold(text: &str) -> String {
    style(text, "1;33")
}

fn style_red_bold(text: &str) -> String {
    style(text, "1;31")
}

fn sci_text(value: f64) -> String {
    let raw = format!("{value:.2e}");
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return raw;
    };
    let Ok(exponent_value) = exponent.parse::<i32>() else {
        return raw;
    };
    format!("{mantissa}e{exponent_value:+03}")
}

fn fmt_sci(value: f64) -> String {
    format!("{:>9}", sci_text(value))
}

fn fmt_optional_sci(value: Option<f64>) -> String {
    match value {
        Some(value) => fmt_sci(value),
        None => format!("{:>9}", "--"),
    }
}

fn fmt_qp_iterations(iterations: Option<u32>) -> String {
    match iterations {
        Some(iterations) => format!("{iterations:>5}"),
        None => format!("{:>5}", "--"),
    }
}

fn compact_duration_text(seconds: f64) -> String {
    let units = [
        (1e-9_f64, "ns"),
        (1e-6_f64, "us"),
        (1e-3_f64, "ms"),
        (1.0_f64, "s"),
    ];
    let mut value = seconds * 1e9;
    let mut unit = "ns";
    for &(scale, candidate_unit) in &units {
        let candidate_value = seconds / scale;
        if (0.1..100.0).contains(&candidate_value) {
            value = candidate_value;
            unit = candidate_unit;
            break;
        }
        if candidate_value >= 100.0 {
            value = candidate_value;
            unit = candidate_unit;
        }
    }
    if value < 9.95 {
        format!("{value:.1}{unit}")
    } else {
        format!("{value:.0}{unit}")
    }
}

#[derive(Clone, Copy)]
enum SummaryDurationUnit {
    Nanoseconds,
    Microseconds,
    Milliseconds,
    Seconds,
}

impl SummaryDurationUnit {
    fn suffix(self) -> &'static str {
        match self {
            Self::Nanoseconds => "ns",
            Self::Microseconds => "us",
            Self::Milliseconds => "ms",
            Self::Seconds => "s",
        }
    }

    fn scale_seconds(self) -> f64 {
        match self {
            Self::Nanoseconds => 1e-9,
            Self::Microseconds => 1e-6,
            Self::Milliseconds => 1e-3,
            Self::Seconds => 1.0,
        }
    }
}

fn choose_summary_duration_unit(durations: &[Duration]) -> SummaryDurationUnit {
    let max_seconds = durations
        .iter()
        .map(Duration::as_secs_f64)
        .fold(0.0_f64, f64::max);
    if max_seconds >= 0.1 {
        SummaryDurationUnit::Seconds
    } else if max_seconds >= 1e-4 {
        SummaryDurationUnit::Milliseconds
    } else if max_seconds >= 1e-7 {
        SummaryDurationUnit::Microseconds
    } else {
        SummaryDurationUnit::Nanoseconds
    }
}

fn fmt_duration_in_unit(duration: Duration, unit: SummaryDurationUnit) -> String {
    let value = duration.as_secs_f64() / unit.scale_seconds();
    format!("{value:>6.1}{}", unit.suffix())
}

fn fmt_optional_duration_in_unit(duration: Option<Duration>, unit: SummaryDurationUnit) -> String {
    match duration {
        Some(duration) => fmt_duration_in_unit(duration, unit),
        None => format!("{:>8}", "--"),
    }
}

fn fmt_qp_time(seconds: Option<f64>) -> String {
    match seconds {
        Some(seconds) => format!("{:>7}", compact_duration_text(seconds)),
        None => format!("{:>7}", "--"),
    }
}

fn fmt_alpha(alpha: Option<f64>) -> String {
    match alpha {
        Some(alpha) => fmt_sci(alpha),
        None => format!("{:>9}", "--"),
    }
}

fn fmt_line_search_iterations(iterations: Option<Index>) -> String {
    match iterations {
        Some(iterations) => format!("{iterations:>5}"),
        None => format!("{:>5}", "--"),
    }
}

fn fmt_iteration(iteration: Index) -> String {
    format!("{iteration:>4}")
}

fn style_iteration_cell(iteration: Index, iteration_limit_reached: bool) -> String {
    let cell = fmt_iteration(iteration);
    if iteration_limit_reached {
        style_red_bold(&cell)
    } else {
        cell
    }
}

fn time_callback<R>(
    stat: &mut EvalTimingStat,
    iteration_callback_time: &mut Duration,
    f: impl FnOnce() -> R,
) -> R {
    let started = Instant::now();
    let result = f();
    let elapsed = started.elapsed();
    stat.record(elapsed);
    *iteration_callback_time += elapsed;
    result
}

fn finalize_profiling(profiling: &mut ClarabelSqpProfiling, solve_started: Instant) {
    profiling.total_time = solve_started.elapsed();
    profiling.unaccounted_time = profiling.total_time.saturating_sub(
        profiling.total_callback_time()
            + profiling.qp_setup_time
            + profiling.qp_solve_time
            + profiling.preprocessing_time,
    );
}

fn dense_fill_percent(nnz: Index, nrow: Index, ncol: Index) -> f64 {
    let denominator = nrow.saturating_mul(ncol);
    if denominator == 0 {
        0.0
    } else {
        100.0 * nnz as f64 / denominator as f64
    }
}

fn lower_tri_fill_percent(nnz: Index, size: Index) -> f64 {
    let denominator = size.saturating_mul(size + 1) / 2;
    if denominator == 0 {
        0.0
    } else {
        100.0 * nnz as f64 / denominator as f64
    }
}

fn declared_box_constraint_count<P>(problem: &P) -> Index
where
    P: CompiledNlpProblem,
{
    let mut lower = vec![0.0; problem.dimension()];
    let mut upper = vec![0.0; problem.dimension()];
    if !problem.variable_bounds(&mut lower, &mut upper) {
        return 0;
    }
    lower.iter().filter(|&&bound| bound > -NLP_INF).count()
        + upper.iter().filter(|&&bound| bound < NLP_INF).count()
}

fn visible_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut idx = 0;
    let mut len = 0;
    while idx < bytes.len() {
        if bytes[idx] == b'\x1b' {
            idx += 1;
            if idx < bytes.len() && bytes[idx] == b'[' {
                idx += 1;
                while idx < bytes.len() {
                    let byte = bytes[idx];
                    idx += 1;
                    if ('@'..='~').contains(&(byte as char)) {
                        break;
                    }
                }
            }
        } else {
            len += 1;
            idx += 1;
        }
    }
    len
}

fn boxed_line(label: &str, detail: impl Into<String>) -> String {
    format!("{label:<BOX_LABEL_WIDTH$}  {}", detail.into())
}

fn log_boxed_section(title: &str, lines: &[String], title_style: fn(&str) -> String) {
    let width = lines
        .iter()
        .map(|line| visible_len(line))
        .chain(std::iter::once(title.len()))
        .max()
        .unwrap_or(title.len());
    let border = format!("+{}+", "-".repeat(width + 2));
    eprintln!();
    eprintln!("{border}");
    let title_padding = " ".repeat(width.saturating_sub(title.len()));
    eprintln!("| {}{} |", title_style(title), title_padding);
    eprintln!("+{}+", "-".repeat(width + 2));
    for line in lines {
        let padding = " ".repeat(width.saturating_sub(visible_len(line)));
        eprintln!("| {line}{padding} |");
    }
    eprintln!("{border}");
}

fn log_sqp_status_summary(summary: &ClarabelSqpSummary, options: &ClarabelSqpOptions) {
    let has_inequality_like_constraints = !summary.inequality_multipliers.is_empty()
        || !summary.lower_bound_multipliers.is_empty()
        || !summary.upper_bound_multipliers.is_empty();
    let eq_text = if summary.equality_multipliers.is_empty() {
        "--".to_string()
    } else {
        style_residual_text(summary.equality_inf_norm, options.constraint_tol)
    };
    let ineq_text = if has_inequality_like_constraints {
        style_residual_text(summary.inequality_inf_norm, options.constraint_tol)
    } else {
        "--".to_string()
    };
    let comp_text = if has_inequality_like_constraints {
        style_residual_text(
            summary.complementarity_inf_norm,
            options.complementarity_tol,
        )
    } else {
        "--".to_string()
    };
    let callback_total_time = summary.profiling.total_callback_time();
    let callback_rows = [
        ("objective", &summary.profiling.objective_value),
        ("gradient", &summary.profiling.objective_gradient),
        ("eq values", &summary.profiling.equality_values),
        ("ineq values", &summary.profiling.inequality_values),
        ("eq jac", &summary.profiling.equality_jacobian_values),
        ("ineq jac", &summary.profiling.inequality_jacobian_values),
        ("hessian", &summary.profiling.lagrangian_hessian_values),
    ];
    let callback_unit = choose_summary_duration_unit(&[
        callback_total_time,
        summary.profiling.objective_value.total_time,
        summary.profiling.objective_gradient.total_time,
        summary.profiling.equality_values.total_time,
        summary.profiling.inequality_values.total_time,
        summary.profiling.equality_jacobian_values.total_time,
        summary.profiling.inequality_jacobian_values.total_time,
        summary.profiling.lagrangian_hessian_values.total_time,
    ]);
    let timing_unit = choose_summary_duration_unit(&[
        summary.profiling.qp_setup_time,
        summary.profiling.qp_solve_time,
        summary.profiling.preprocessing_time,
        summary.profiling.unaccounted_time,
        summary.profiling.total_time,
        summary
            .profiling
            .backend_timing
            .function_creation_time
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .backend_timing
            .jit_time
            .unwrap_or(Duration::ZERO),
    ]);
    let callback_row = |name: &str, calls: Index, duration: Duration| {
        format!(
            "{name:<12}  calls={calls:>4}  time={}",
            fmt_duration_in_unit(duration, callback_unit)
        )
    };
    let timing_row = |name: &str, count: Option<Index>, duration: Duration| {
        let count_cell = match count {
            Some(count) => format!("{count:>4}"),
            None => format!("{:>4}", "--"),
        };
        format!(
            "{name:<12}  count={count_cell}  time={}",
            fmt_duration_in_unit(duration, timing_unit)
        )
    };
    let mut lines = vec![
        boxed_line(
            "result",
            format!(
                "objective={}  primal_inf={}  dual_inf={}",
                sci_text(summary.objective),
                style_residual_text(summary.primal_inf_norm, options.constraint_tol),
                style_residual_text(summary.dual_inf_norm, options.dual_tol),
            ),
        ),
        boxed_line(
            "",
            format!(
                "eq_inf={}  ineq_inf={}  comp_inf={}  iterations={}",
                eq_text, ineq_text, comp_text, summary.iterations,
            ),
        ),
        String::new(),
        boxed_line(
            "callbacks",
            callback_row(
                "total",
                summary.profiling.total_callback_calls(),
                callback_total_time,
            ),
        ),
    ];
    for (name, stat) in callback_rows {
        lines.push(boxed_line(
            "",
            callback_row(name, stat.calls, stat.total_time),
        ));
    }
    lines.push(String::new());
    lines.push(boxed_line(
        "timing",
        timing_row(
            "qp setup",
            Some(summary.profiling.qp_setups),
            summary.profiling.qp_setup_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row(
            "qp solve",
            Some(summary.profiling.qp_solves),
            summary.profiling.qp_solve_time,
        ),
    ));
    lines.push(boxed_line(
        "",
        timing_row("preprocess", None, summary.profiling.preprocessing_time),
    ));
    lines.push(boxed_line(
        "",
        timing_row("unaccounted", None, summary.profiling.unaccounted_time),
    ));
    lines.push(boxed_line(
        "",
        timing_row("total", None, summary.profiling.total_time),
    ));
    lines.push(String::new());
    lines.push(boxed_line(
        "backend",
        format!(
            "create={}  jit={}",
            fmt_optional_duration_in_unit(
                summary.profiling.backend_timing.function_creation_time,
                timing_unit,
            ),
            fmt_optional_duration_in_unit(summary.profiling.backend_timing.jit_time, timing_unit),
        ),
    ));
    log_boxed_section("SQP converged", &lines, style_green_bold);
}

fn log_sqp_problem_header<P>(
    problem: &P,
    parameters: &[ParameterMatrix<'_>],
    options: &ClarabelSqpOptions,
) where
    P: CompiledNlpProblem,
{
    let n = problem.dimension();
    let equality_count = problem.equality_count();
    let inequality_count = problem.inequality_count();
    let declared_box_constraints = declared_box_constraint_count(problem);
    let degrees_of_freedom = n as i128 - equality_count as i128;
    let total_jacobian_rows = equality_count + inequality_count;
    let total_jacobian_nnz =
        problem.equality_jacobian_ccs().nnz() + problem.inequality_jacobian_ccs().nnz();
    let total_jacobian_fill = dense_fill_percent(total_jacobian_nnz, total_jacobian_rows, n);
    let hessian_nnz = problem.lagrangian_hessian_ccs().nnz();
    let hessian_fill = lower_tri_fill_percent(hessian_nnz, n);
    let parameter_nnz = parameters
        .iter()
        .map(|parameter| parameter.values.len())
        .sum::<usize>();
    let lines = vec![
        boxed_line(
            "dimensions",
            format!(
                "vars={n}  eq={equality_count}  ineq={inequality_count}  box={declared_box_constraints}  dof={degrees_of_freedom}"
            ),
        ),
        boxed_line(
            "sparsity",
            format!(
                "jac={total_jacobian_nnz} nnz ({total_jacobian_fill:.2}%)  hess={hessian_nnz} nnz ({hessian_fill:.2}% lower)"
            ),
        ),
        boxed_line(
            "parameters",
            format!("matrices={}  nnz={parameter_nnz}", parameters.len()),
        ),
        String::new(),
        boxed_line(
            "tolerances",
            format!(
                "dual={}  constraint={}  complementarity={}",
                sci_text(options.dual_tol),
                sci_text(options.constraint_tol),
                sci_text(options.complementarity_tol),
            ),
        ),
        boxed_line(
            "line search",
            format!(
                "armijo_c1={}  beta={}  min_step={}",
                sci_text(options.armijo_c1),
                sci_text(options.line_search_beta),
                sci_text(options.min_step),
            ),
        ),
        boxed_line(
            "globalize",
            format!(
                "penalty0={}  regularization={}",
                sci_text(options.merit_penalty),
                sci_text(options.regularization),
            ),
        ),
        boxed_line(
            "",
            format!(
                "factor={}  max_penalty_updates={}",
                sci_text(options.penalty_increase_factor),
                options.max_penalty_updates,
            ),
        ),
        boxed_line(
            "iteration",
            format!(
                "max_iters={}  verbose={}",
                options.max_iters, options.verbose
            ),
        ),
    ];
    log_boxed_section("SQP problem / settings", &lines, style_cyan_bold);
}

fn style_residual_text(value: f64, tolerance: f64) -> String {
    let text = sci_text(value);
    if value <= tolerance {
        style_green_bold(&text)
    } else {
        text
    }
}

fn style_residual_cell(value: f64, tolerance: f64, is_applicable: bool) -> String {
    if !is_applicable {
        return format!("{:>9}", "--");
    }
    let cell = fmt_sci(value);
    if value <= tolerance {
        style_green_bold(&cell)
    } else {
        cell
    }
}

fn style_line_search_iterations_cell(iterations: Option<Index>) -> String {
    let cell = fmt_line_search_iterations(iterations);
    match iterations {
        Some(iterations) if iterations >= 10 => style_red_bold(&cell),
        Some(iterations) if iterations >= 4 => style_yellow_bold(&cell),
        _ => cell,
    }
}

fn qp_reduced_accuracy(status: Option<SolverStatus>) -> bool {
    matches!(status, Some(SolverStatus::AlmostSolved))
}

fn fmt_event_codes(log: &SqpIterationLog) -> String {
    let long_linesearch = matches!(log.line_search_iterations, Some(iterations) if iterations >= 4);
    let qp_reduced = qp_reduced_accuracy(log.qp_status);
    let mut codes = String::new();
    if log.flag(SQP_LOG_PENALTY_UPDATED) {
        codes.push('P');
    }
    if long_linesearch {
        codes.push('L');
    }
    if qp_reduced {
        codes.push('R');
    }
    if log.flag(SQP_LOG_ITERATION_LIMIT_REACHED) {
        codes.push('M');
    }
    codes
}

fn style_event_cell(log: &SqpIterationLog) -> String {
    let codes = fmt_event_codes(log);
    let cell = format!("{:>4}", codes);
    if codes.is_empty() {
        return cell;
    }
    let long_linesearch = matches!(log.line_search_iterations, Some(iterations) if iterations >= 4);
    let qp_reduced = qp_reduced_accuracy(log.qp_status);
    if log.flag(SQP_LOG_ITERATION_LIMIT_REACHED) {
        style_red_bold(&cell)
    } else if log.flag(SQP_LOG_PENALTY_UPDATED) || long_linesearch || qp_reduced {
        style_yellow_bold(&cell)
    } else {
        cell
    }
}

fn event_legend_suffix(log: &SqpIterationLog, state: &mut SqpEventLegendState) -> String {
    let mut parts = Vec::new();
    let long_linesearch = matches!(log.line_search_iterations, Some(iterations) if iterations >= 4);
    let qp_reduced = qp_reduced_accuracy(log.qp_status);

    if log.flag(SQP_LOG_PENALTY_UPDATED) && state.mark_if_new(SQP_EVENT_SEEN_PENALTY) {
        parts.push("P=merit penalty increased");
    }
    if long_linesearch && state.mark_if_new(SQP_EVENT_SEEN_LINE_SEARCH) {
        parts.push("L=line search backtracked >=4 times");
    }
    if qp_reduced && state.mark_if_new(SQP_EVENT_SEEN_QP) {
        parts.push("R=QP solved to reduced accuracy");
    }
    if log.flag(SQP_LOG_ITERATION_LIMIT_REACHED) && state.mark_if_new(SQP_EVENT_SEEN_MAX_ITER) {
        parts.push("M=maximum SQP iterations reached");
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!("  {}", parts.join("  "))
    }
}

fn log_sqp_iteration(log: &SqpIterationLog, event_state: &mut SqpEventLegendState) {
    if log.iteration.is_multiple_of(20) {
        eprintln!();
        let header = [
            format!("{:>4}", "iter"),
            format!("{:>9}", "f"),
            format!("{:>9}", "eq_inf"),
            format!("{:>9}", "ineq_inf"),
            format!("{:>9}", "dual_inf"),
            format!("{:>9}", "comp_inf"),
            format!("{:>9}", "step_inf"),
            format!("{:>9}", "penalty"),
            format!("{:>9}", "alpha"),
            format!("{:>5}", "ls_it"),
            format!("{:>4}", "evt"),
            format!("{:>5}", "qp_it"),
            format!("{:>7}", "qp_time"),
        ];
        eprintln!("{}", style_bold(&header.join("  ")));
    }
    let row = [
        style_iteration_cell(log.iteration, log.flag(SQP_LOG_ITERATION_LIMIT_REACHED)),
        fmt_sci(log.objective_value),
        style_residual_cell(
            log.equality_inf,
            log.constraint_tol,
            log.flag(SQP_LOG_HAS_EQUALITIES),
        ),
        style_residual_cell(
            log.inequality_inf,
            log.constraint_tol,
            log.flag(SQP_LOG_HAS_INEQUALITIES),
        ),
        style_residual_cell(log.dual_inf, log.dual_tol, true),
        style_residual_cell(
            log.complementarity_inf,
            log.complementarity_tol,
            log.flag(SQP_LOG_HAS_INEQUALITIES),
        ),
        fmt_optional_sci(log.step_inf_norm),
        fmt_sci(log.merit_penalty),
        fmt_alpha(log.alpha),
        style_line_search_iterations_cell(log.line_search_iterations),
        style_event_cell(log),
        fmt_qp_iterations(log.qp_iterations),
        fmt_qp_time(log.qp_solve_time_secs),
    ];
    let mut rendered = row.join("  ");
    rendered.push_str(&event_legend_suffix(log, event_state));
    eprintln!("{rendered}");
}

pub fn solve_nlp_sqp<P>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &ClarabelSqpOptions,
) -> std::result::Result<ClarabelSqpSummary, ClarabelSqpError>
where
    P: CompiledNlpProblem,
{
    let solve_started = Instant::now();
    let mut profiling = ClarabelSqpProfiling {
        backend_timing: problem.backend_timing_metadata(),
        ..ClarabelSqpProfiling::default()
    };
    let validation_started = Instant::now();
    validate_nlp_problem_shapes(problem)
        .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
    validate_parameter_inputs(problem, parameters)
        .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
    profiling.preprocessing_time += validation_started.elapsed();
    let n = problem.dimension();
    if x0.len() != n {
        return Err(ClarabelSqpError::InvalidInput(format!(
            "x0 has length {}, expected {n}",
            x0.len()
        )));
    }

    let equality_count = problem.equality_count();
    let inequality_count = problem.inequality_count();
    let bounds = collect_bound_constraints(problem)?;
    let lower_bound_count = bounds.lower_indices.len();
    let augmented_inequality_count = inequality_count + bounds.total_count();
    let total_constraint_count = equality_count + augmented_inequality_count;
    let bound_jacobian = build_bound_jacobian(&bounds, n);
    let mut x = x0.to_vec();
    let mut gradient = vec![0.0; n];
    let mut equality_values = vec![0.0; equality_count];
    let mut inequality_values = vec![0.0; inequality_count];
    let mut augmented_inequality_values = vec![0.0; augmented_inequality_count];
    let mut hessian_values = vec![0.0; problem.lagrangian_hessian_ccs().nnz()];
    let mut equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    let mut trial_equality_values = vec![0.0; equality_count];
    let mut trial_inequality_values = vec![0.0; inequality_count];
    let mut trial_augmented_inequality_values = vec![0.0; augmented_inequality_count];
    let mut equality_multipliers = vec![0.0; equality_count];
    let mut inequality_multipliers = vec![0.0; inequality_count];
    let mut merit_penalty = options.merit_penalty;
    let mut event_state = SqpEventLegendState::default();

    if options.verbose {
        log_sqp_problem_header(problem, parameters, options);
    }

    for iteration in 0..options.max_iters {
        let iteration_started = Instant::now();
        let mut iteration_callback_time = Duration::ZERO;
        let mut iteration_qp_setup_time = Duration::ZERO;
        let mut iteration_qp_solve_time = Duration::ZERO;
        let objective_value = time_callback(
            &mut profiling.objective_value,
            &mut iteration_callback_time,
            || problem.objective_value(&x, parameters),
        );
        time_callback(
            &mut profiling.objective_gradient,
            &mut iteration_callback_time,
            || problem.objective_gradient(&x, parameters, &mut gradient),
        );
        time_callback(
            &mut profiling.equality_values,
            &mut iteration_callback_time,
            || problem.equality_values(&x, parameters, &mut equality_values),
        );
        time_callback(
            &mut profiling.inequality_values,
            &mut iteration_callback_time,
            || problem.inequality_values(&x, parameters, &mut inequality_values),
        );
        augment_inequality_values(
            &inequality_values,
            &x,
            &bounds,
            &mut augmented_inequality_values,
        );
        time_callback(
            &mut profiling.equality_jacobian_values,
            &mut iteration_callback_time,
            || problem.equality_jacobian_values(&x, parameters, &mut equality_jacobian_values),
        );
        time_callback(
            &mut profiling.inequality_jacobian_values,
            &mut iteration_callback_time,
            || problem.inequality_jacobian_values(&x, parameters, &mut inequality_jacobian_values),
        );
        let equality_jacobian =
            ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
        let nonlinear_inequality_jacobian = ccs_to_dense(
            problem.inequality_jacobian_ccs(),
            &inequality_jacobian_values,
        );
        let inequality_jacobian = stack_jacobians(&nonlinear_inequality_jacobian, &bound_jacobian);
        let equality_inf = inf_norm(&equality_values);
        let inequality_inf = positive_part_inf_norm(&augmented_inequality_values);
        let primal_inf = equality_inf.max(inequality_inf);

        time_callback(
            &mut profiling.lagrangian_hessian_values,
            &mut iteration_callback_time,
            || {
                problem.lagrangian_hessian_values(
                    &x,
                    parameters,
                    &equality_multipliers,
                    &inequality_multipliers,
                    &mut hessian_values,
                );
            },
        );
        let mut hessian =
            lower_triangle_to_symmetric_dense(problem.lagrangian_hessian_ccs(), &hessian_values);
        regularize_hessian(&mut hessian, options.regularization);

        let (
            step,
            candidate_equality_multipliers,
            candidate_inequality_multipliers,
            candidate_lower_bound_multipliers,
            candidate_upper_bound_multipliers,
            qp_status,
            qp_iterations,
            qp_solve_time_secs,
        ) = if total_constraint_count == 0 {
            (
                solve_unconstrained_quadratic_step(&hessian, &gradient)?,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
                None,
                None,
            )
        } else {
            let qp_setup_started = Instant::now();
            let p = dense_to_csc_upper(&hessian);
            let stacked_jacobian = stack_jacobians(&equality_jacobian, &inequality_jacobian);
            let a = dense_to_csc(&stacked_jacobian);
            let mut b = equality_values
                .iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            b.extend(augmented_inequality_values.iter().map(|value| -value));
            let mut cones = Vec::with_capacity(2);
            if equality_count > 0 {
                cones.push(ZeroConeT(equality_count));
            }
            if augmented_inequality_count > 0 {
                cones.push(NonnegativeConeT(augmented_inequality_count));
            }
            let settings = DefaultSettingsBuilder::default()
                .verbose(false)
                .build()
                .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
            let mut solver = DefaultSolver::new(&p, &gradient, &a, &b, &cones, settings)
                .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
            let qp_setup_elapsed = qp_setup_started.elapsed();
            profiling.qp_setups += 1;
            profiling.qp_setup_time += qp_setup_elapsed;
            iteration_qp_setup_time += qp_setup_elapsed;
            let qp_solve_started = Instant::now();
            solver.solve();
            let qp_solve_elapsed = qp_solve_started.elapsed();
            profiling.qp_solves += 1;
            profiling.qp_solve_time += qp_solve_elapsed;
            iteration_qp_solve_time += qp_solve_elapsed;
            let status = solver.solution.status;
            if !matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
                return Err(ClarabelSqpError::QpSolve { status });
            }
            let (candidate_equality_multipliers, candidate_augmented_inequality_multipliers) =
                split_multipliers(&solver.solution.z, equality_count);
            let (
                candidate_inequality_multipliers,
                candidate_lower_bound_multipliers,
                candidate_upper_bound_multipliers,
            ) = split_augmented_inequality_multipliers(
                &candidate_augmented_inequality_multipliers,
                inequality_count,
                lower_bound_count,
            );
            (
                solver.solution.x.clone(),
                candidate_equality_multipliers,
                candidate_inequality_multipliers,
                candidate_lower_bound_multipliers,
                candidate_upper_bound_multipliers,
                Some(solver.solution.status),
                Some(solver.solution.iterations),
                Some(qp_solve_elapsed.as_secs_f64()),
            )
        };

        let all_inequality_multipliers = [
            candidate_inequality_multipliers.as_slice(),
            candidate_lower_bound_multipliers.as_slice(),
            candidate_upper_bound_multipliers.as_slice(),
        ]
        .concat();
        let dual_residual = lagrangian_gradient(
            &gradient,
            &equality_jacobian,
            &candidate_equality_multipliers,
            &inequality_jacobian,
            &all_inequality_multipliers,
        );
        let dual_inf = inf_norm(&dual_residual);
        let complementarity_inf =
            complementarity_inf_norm(&augmented_inequality_values, &all_inequality_multipliers);
        if primal_inf <= options.constraint_tol
            && dual_inf <= options.dual_tol
            && complementarity_inf <= options.complementarity_tol
        {
            profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
                iteration_callback_time + iteration_qp_setup_time + iteration_qp_solve_time,
            );
            finalize_profiling(&mut profiling, solve_started);
            let summary = ClarabelSqpSummary {
                x,
                equality_multipliers: candidate_equality_multipliers,
                inequality_multipliers: candidate_inequality_multipliers,
                lower_bound_multipliers: candidate_lower_bound_multipliers,
                upper_bound_multipliers: candidate_upper_bound_multipliers,
                objective: objective_value,
                iterations: iteration,
                equality_inf_norm: equality_inf,
                inequality_inf_norm: inequality_inf,
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: complementarity_inf,
                profiling,
            };
            if options.verbose {
                let flags = (if equality_count > 0 {
                    SQP_LOG_HAS_EQUALITIES
                } else {
                    0
                }) | (if augmented_inequality_count > 0 {
                    SQP_LOG_HAS_INEQUALITIES
                } else {
                    0
                });
                log_sqp_iteration(
                    &SqpIterationLog {
                        iteration,
                        flags,
                        objective_value,
                        equality_inf,
                        inequality_inf,
                        dual_inf,
                        complementarity_inf,
                        step_inf_norm: None,
                        merit_penalty,
                        alpha: None,
                        line_search_iterations: None,
                        qp_status,
                        qp_iterations,
                        qp_solve_time_secs,
                        constraint_tol: options.constraint_tol,
                        dual_tol: options.dual_tol,
                        complementarity_tol: options.complementarity_tol,
                    },
                    &mut event_state,
                );
                log_sqp_status_summary(&summary, options);
            }
            return Ok(summary);
        }

        let step_inf_norm = inf_norm(&step);
        if step_inf_norm <= options.min_step {
            return Err(ClarabelSqpError::Stalled {
                step_inf_norm,
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: complementarity_inf,
            });
        }

        let penalty_before_updates = merit_penalty;
        merit_penalty = update_merit_penalty(
            merit_penalty,
            &candidate_equality_multipliers,
            &all_inequality_multipliers,
        );
        let current_merit = exact_merit_value(
            objective_value,
            &equality_values,
            &augmented_inequality_values,
            merit_penalty,
        );
        let mut directional_derivative = exact_merit_directional_derivative(
            &gradient,
            &equality_values,
            &equality_jacobian,
            &augmented_inequality_values,
            &inequality_jacobian,
            &step,
            merit_penalty,
        );
        for _ in 0..options.max_penalty_updates {
            if directional_derivative < -1e-12 {
                break;
            }
            merit_penalty *= options.penalty_increase_factor;
            directional_derivative = exact_merit_directional_derivative(
                &gradient,
                &equality_values,
                &equality_jacobian,
                &augmented_inequality_values,
                &inequality_jacobian,
                &step,
                merit_penalty,
            );
        }
        if directional_derivative >= 0.0 {
            return Err(ClarabelSqpError::LineSearchFailed {
                directional_derivative,
                step_inf_norm,
                penalty: merit_penalty,
            });
        }
        let penalty_updated = merit_penalty != penalty_before_updates;

        let mut alpha = 1.0;
        let mut accepted_trial = None;
        let mut line_search_iterations = 0;
        while alpha * step_inf_norm >= options.min_step {
            let trial = x
                .iter()
                .zip(step.iter())
                .map(|(xi, di)| xi + alpha * di)
                .collect::<Vec<_>>();
            let trial_merit_value = trial_merit(
                problem,
                &trial,
                parameters,
                (
                    &mut trial_equality_values,
                    &mut trial_inequality_values,
                    &mut trial_augmented_inequality_values,
                ),
                &bounds,
                merit_penalty,
                (&mut profiling, &mut iteration_callback_time),
            );
            if trial_merit_value
                <= current_merit + options.armijo_c1 * alpha * directional_derivative
            {
                accepted_trial = Some(trial);
                break;
            }
            alpha *= options.line_search_beta;
            line_search_iterations += 1;
        }
        let Some(trial) = accepted_trial else {
            return Err(ClarabelSqpError::LineSearchFailed {
                directional_derivative,
                step_inf_norm,
                penalty: merit_penalty,
            });
        };

        if options.verbose {
            let flags = (if iteration + 1 == options.max_iters {
                SQP_LOG_ITERATION_LIMIT_REACHED
            } else {
                0
            }) | (if equality_count > 0 {
                SQP_LOG_HAS_EQUALITIES
            } else {
                0
            }) | (if augmented_inequality_count > 0 {
                SQP_LOG_HAS_INEQUALITIES
            } else {
                0
            }) | (if penalty_updated {
                SQP_LOG_PENALTY_UPDATED
            } else {
                0
            });
            log_sqp_iteration(
                &SqpIterationLog {
                    iteration,
                    flags,
                    objective_value,
                    equality_inf,
                    inequality_inf,
                    dual_inf,
                    complementarity_inf,
                    step_inf_norm: Some(step_inf_norm),
                    merit_penalty,
                    alpha: Some(alpha),
                    line_search_iterations: Some(line_search_iterations),
                    qp_status,
                    qp_iterations,
                    qp_solve_time_secs,
                    constraint_tol: options.constraint_tol,
                    dual_tol: options.dual_tol,
                    complementarity_tol: options.complementarity_tol,
                },
                &mut event_state,
            );
        }

        profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
            iteration_callback_time + iteration_qp_setup_time + iteration_qp_solve_time,
        );
        x = trial;
        equality_multipliers = candidate_equality_multipliers;
        inequality_multipliers = candidate_inequality_multipliers;
    }

    Err(ClarabelSqpError::MaxIterations {
        iterations: options.max_iters,
    })
}

pub fn validate_nlp_problem_shapes<P>(problem: &P) -> Result<()>
where
    P: CompiledNlpProblem,
{
    let dimension = problem.dimension();
    if problem.lagrangian_hessian_ccs().nrow != dimension
        || problem.lagrangian_hessian_ccs().ncol != dimension
    {
        bail!("Lagrangian Hessian CCS must be square with dimension {dimension}");
    }
    if problem.equality_jacobian_ccs().nrow != problem.equality_count()
        || problem.equality_jacobian_ccs().ncol != dimension
    {
        bail!("equality Jacobian CCS does not match declared dimensions");
    }
    if problem.inequality_jacobian_ccs().nrow != problem.inequality_count()
        || problem.inequality_jacobian_ccs().ncol != dimension
    {
        bail!("inequality Jacobian CCS does not match declared dimensions");
    }
    Ok(())
}

pub fn validate_parameter_inputs<P>(problem: &P, parameters: &[ParameterMatrix<'_>]) -> Result<()>
where
    P: CompiledNlpProblem,
{
    if parameters.len() != problem.parameter_count() {
        bail!(
            "parameter count mismatch: got {}, expected {}",
            parameters.len(),
            problem.parameter_count()
        );
    }
    for (index, parameter) in parameters.iter().enumerate() {
        let expected_ccs = problem.parameter_ccs(index);
        if parameter.ccs != expected_ccs {
            bail!("parameter {index} CCS does not match declared dimensions/pattern");
        }
        if parameter.values.len() != expected_ccs.nnz() {
            bail!(
                "parameter {index} value length mismatch: got {}, expected {}",
                parameter.values.len(),
                expected_ccs.nnz()
            );
        }
    }
    Ok(())
}
