use anyhow::{Result, bail};
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{NonnegativeConeT, ZeroConeT};
use clarabel::solver::implementations::default::DefaultSettingsBuilder;
use clarabel::solver::{DefaultSolver, IPSolver, SolverStatus};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::io::{self, IsTerminal};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
pub use sx_codegen_llvm::LlvmOptimizationLevel;
use thiserror::Error;

mod interior_point;
#[cfg(feature = "ipopt")]
mod ipopt_backend;
mod symbolic;
mod vectorize;

pub use interior_point::{
    InteriorPointIterationEvent, InteriorPointIterationPhase, InteriorPointIterationSnapshot,
    InteriorPointLinearSolver, InteriorPointOptions, InteriorPointProfiling,
    InteriorPointSolveError, InteriorPointSummary, solve_nlp_interior_point,
    solve_nlp_interior_point_with_callback,
};
#[cfg(feature = "ipopt")]
pub use ipopt::SolveStatus as IpoptSolveStatus;
#[cfg(feature = "ipopt")]
pub use ipopt_backend::{
    IpoptIterationPhase, IpoptIterationSnapshot, IpoptMuStrategy, IpoptOptions, IpoptSolveError,
    IpoptSummary, solve_nlp_ipopt,
};
pub use optimization_derive::Vectorize;
pub use symbolic::{
    ConstraintBounds, RuntimeBoundedJitNlp, RuntimeNlpBounds, SymbolicNlpBuildError,
    SymbolicNlpCompileError, SymbolicNlpOutputs, TypedCompiledJitNlp, TypedRuntimeNlpBounds,
    TypedSymbolicNlp, symbolic_nlp,
};
pub use vectorize::{ScalarLeaf, Vectorize, flatten_value, symbolic_column, symbolic_value};

pub type Index = usize;
pub(crate) const NLP_INF: f64 = 1e20;
const BOX_LABEL_WIDTH: usize = 13;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendTimingMetadata {
    pub function_creation_time: Option<Duration>,
    pub derivative_generation_time: Option<Duration>,
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
    pub elastic_recovery_activations: Index,
    pub elastic_recovery_qp_solves: Index,
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
    pub elastic_mode: bool,
    pub elastic_weight: f64,
    pub elastic_primal_regularization: f64,
    pub elastic_slack_regularization: f64,
    pub elastic_restore_reduction_factor: f64,
    pub elastic_restore_abs_tol: f64,
    pub elastic_restore_max_iters: Index,
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
            elastic_mode: true,
            elastic_weight: 100.0,
            elastic_primal_regularization: 1.0,
            elastic_slack_regularization: 1e-8,
            elastic_restore_reduction_factor: 1e-2,
            elastic_restore_abs_tol: 1e-4,
            elastic_restore_max_iters: 5,
            verbose: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpIterationPhase {
    Initial,
    AcceptedStep,
    PostConvergence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpIterationEvent {
    PenaltyUpdated,
    LongLineSearch,
    QpReducedAccuracy,
    ElasticRecoveryUsed,
    MaxIterationsReached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpQpStatus {
    Solved,
    ReducedAccuracy,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpTermination {
    Converged,
    MaxIterations,
    QpSolve,
    LineSearchFailed,
    Stalled,
    NonFiniteInput,
    NonFiniteCallbackOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SqpFinalStateKind {
    InitialPoint,
    AcceptedIterate,
    TrialPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFiniteInputStage {
    InitialGuess,
    ParameterValues { parameter_index: Index },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFiniteCallbackStage {
    ObjectiveValue,
    ObjectiveGradient,
    EqualityValues,
    InequalityValues,
    EqualityJacobianValues,
    InequalityJacobianValues,
    LagrangianHessianValues,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SqpIterationTiming {
    pub objective_value: Duration,
    pub objective_gradient: Duration,
    pub equality_values: Duration,
    pub inequality_values: Duration,
    pub equality_jacobian_values: Duration,
    pub inequality_jacobian_values: Duration,
    pub lagrangian_hessian_values: Duration,
    pub qp_setup: Duration,
    pub qp_solve: Duration,
    pub preprocess: Duration,
    pub total: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqpLineSearchInfo {
    pub accepted_alpha: f64,
    pub last_tried_alpha: f64,
    pub backtrack_count: Index,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SqpQpInfo {
    pub status: SqpQpStatus,
    pub raw_status: SolverStatus,
    pub setup_time: Duration,
    pub solve_time: Duration,
    pub iteration_count: Index,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SqpIterationSnapshot {
    pub iteration: Index,
    pub phase: SqpIterationPhase,
    pub x: Vec<f64>,
    pub objective: f64,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub dual_inf: f64,
    pub comp_inf: Option<f64>,
    pub step_inf: Option<f64>,
    pub penalty: f64,
    pub line_search: Option<SqpLineSearchInfo>,
    pub qp: Option<SqpQpInfo>,
    pub timing: SqpIterationTiming,
    pub events: Vec<SqpIterationEvent>,
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
    pub equality_inf_norm: Option<f64>,
    pub inequality_inf_norm: Option<f64>,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: Option<f64>,
    pub termination: SqpTermination,
    pub final_state: SqpIterationSnapshot,
    pub final_state_kind: SqpFinalStateKind,
    pub last_accepted_state: Option<SqpIterationSnapshot>,
    pub profiling: ClarabelSqpProfiling,
}

#[derive(Clone, Debug)]
pub struct SqpFailureContext {
    pub termination: SqpTermination,
    pub final_state: Option<SqpIterationSnapshot>,
    pub final_state_kind: Option<SqpFinalStateKind>,
    pub last_accepted_state: Option<SqpIterationSnapshot>,
    pub profiling: ClarabelSqpProfiling,
}

#[derive(Debug, Error)]
pub enum ClarabelSqpError {
    #[error("invalid SQP input: {0}")]
    InvalidInput(String),
    #[error("non-finite SQP input at {stage:?}")]
    NonFiniteInput { stage: NonFiniteInputStage },
    #[error("clarabel SQP failed to converge in {iterations} iterations")]
    MaxIterations {
        iterations: Index,
        context: Box<SqpFailureContext>,
    },
    #[error("clarabel solver setup failed: {0}")]
    Setup(String),
    #[error("clarabel returned status {status:?}")]
    QpSolve {
        status: SolverStatus,
        context: Box<SqpFailureContext>,
    },
    #[error("unconstrained SQP subproblem solve failed")]
    UnconstrainedStepSolve { context: Box<SqpFailureContext> },
    #[error(
        "armijo line search failed to find sufficient decrease (directional derivative {directional_derivative}, step inf-norm {step_inf_norm}, penalty {penalty})"
    )]
    LineSearchFailed {
        directional_derivative: f64,
        step_inf_norm: f64,
        penalty: f64,
        context: Box<SqpFailureContext>,
    },
    #[error(
        "sqp stalled with step inf-norm {step_inf_norm}, primal inf-norm {primal_inf_norm}, dual inf-norm {dual_inf_norm}, complementarity inf-norm {complementarity_inf_norm}"
    )]
    Stalled {
        step_inf_norm: f64,
        primal_inf_norm: f64,
        dual_inf_norm: f64,
        complementarity_inf_norm: f64,
        context: Box<SqpFailureContext>,
    },
    #[error("non-finite SQP callback output at {stage:?}")]
    NonFiniteCallbackOutput {
        stage: NonFiniteCallbackStage,
        context: Box<SqpFailureContext>,
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
) -> std::result::Result<Vec<f64>, ()> {
    let rhs = DVector::<f64>::from_iterator(gradient.len(), gradient.iter().map(|value| -value));
    let lu = hessian.clone().lu();
    let Some(step) = lu.solve(&rhs) else {
        return Err(());
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
) -> std::result::Result<f64, NonFiniteCallbackStage>
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
    if equality_values.iter().any(|value| !value.is_finite()) {
        return Err(NonFiniteCallbackStage::EqualityValues);
    }
    time_callback(
        &mut profiling.inequality_values,
        iteration_callback_time,
        || problem.inequality_values(x, parameters, inequality_values),
    );
    if inequality_values.iter().any(|value| !value.is_finite()) {
        return Err(NonFiniteCallbackStage::InequalityValues);
    }
    augment_inequality_values(inequality_values, x, bounds, augmented_inequality_values);
    let objective_value = time_callback(
        &mut profiling.objective_value,
        iteration_callback_time,
        || problem.objective_value(x, parameters),
    );
    if !objective_value.is_finite() {
        return Err(NonFiniteCallbackStage::ObjectiveValue);
    }
    Ok(exact_merit_value(
        objective_value,
        equality_values,
        augmented_inequality_values,
        penalty,
    ))
}

fn snapshot_events(
    penalty_updated: bool,
    line_search_backtracks: Option<Index>,
    qp_info: Option<&SqpQpInfo>,
    elastic_recovery_used: bool,
    max_iterations_reached: bool,
) -> Vec<SqpIterationEvent> {
    let mut events = Vec::new();
    if penalty_updated {
        events.push(SqpIterationEvent::PenaltyUpdated);
    }
    if matches!(line_search_backtracks, Some(iterations) if iterations >= 4) {
        events.push(SqpIterationEvent::LongLineSearch);
    }
    if matches!(
        qp_info.map(|info| info.status),
        Some(SqpQpStatus::ReducedAccuracy)
    ) {
        events.push(SqpIterationEvent::QpReducedAccuracy);
    }
    if elastic_recovery_used {
        events.push(SqpIterationEvent::ElasticRecoveryUsed);
    }
    if max_iterations_reached {
        events.push(SqpIterationEvent::MaxIterationsReached);
    }
    events
}

fn final_state_kind(snapshot: &SqpIterationSnapshot) -> SqpFinalStateKind {
    match snapshot.phase {
        SqpIterationPhase::Initial => SqpFinalStateKind::InitialPoint,
        SqpIterationPhase::AcceptedStep | SqpIterationPhase::PostConvergence => {
            SqpFinalStateKind::AcceptedIterate
        }
    }
}

fn failure_context(
    termination: SqpTermination,
    final_state: Option<SqpIterationSnapshot>,
    last_accepted_state: Option<SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> Box<SqpFailureContext> {
    let final_state_kind = final_state.as_ref().map(final_state_kind);
    Box::new(SqpFailureContext {
        termination,
        final_state,
        final_state_kind,
        last_accepted_state,
        profiling: profiling.clone(),
    })
}

#[derive(Default)]
struct SqpEventLegendState {
    seen: u8,
}

const SQP_EVENT_SEEN_PENALTY: u8 = 1 << 0;
const SQP_EVENT_SEEN_LINE_SEARCH: u8 = 1 << 1;
const SQP_EVENT_SEEN_QP: u8 = 1 << 2;
const SQP_EVENT_SEEN_MAX_ITER: u8 = 1 << 3;
const SQP_EVENT_SEEN_ELASTIC: u8 = 1 << 4;
pub(crate) const SQP_LOG_ITERATION_LIMIT_REACHED: u8 = 1 << 0;
pub(crate) const SQP_LOG_HAS_EQUALITIES: u8 = 1 << 1;
pub(crate) const SQP_LOG_HAS_INEQUALITIES: u8 = 1 << 2;
pub(crate) const SQP_LOG_PENALTY_UPDATED: u8 = 1 << 3;

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
    let eq_text = summary.equality_inf_norm.map_or_else(
        || "--".to_string(),
        |value| style_residual_text(value, options.constraint_tol),
    );
    let ineq_text = summary.inequality_inf_norm.map_or_else(
        || "--".to_string(),
        |value| style_residual_text(value, options.constraint_tol),
    );
    let comp_text = summary.complementarity_inf_norm.map_or_else(
        || "--".to_string(),
        |value| style_residual_text(value, options.complementarity_tol),
    );
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
            .derivative_generation_time
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
        "elastic",
        format!(
            "activations={:>4}  recovery_qps={:>4}",
            summary.profiling.elastic_recovery_activations,
            summary.profiling.elastic_recovery_qp_solves,
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
            "create={}  derive={}  jit={}",
            fmt_optional_duration_in_unit(
                summary.profiling.backend_timing.function_creation_time,
                timing_unit,
            ),
            fmt_optional_duration_in_unit(
                summary.profiling.backend_timing.derivative_generation_time,
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
                "penalty0={}  regularization={}  elastic={}",
                sci_text(options.merit_penalty),
                sci_text(options.regularization),
                if options.elastic_mode { "on" } else { "off" },
            ),
        ),
        boxed_line(
            "",
            format!(
                "elastic_weight={}  elastic_primal_reg={}  elastic_slack_reg={}",
                sci_text(options.elastic_weight),
                sci_text(options.elastic_primal_regularization),
                sci_text(options.elastic_slack_regularization),
            ),
        ),
        boxed_line(
            "",
            format!(
                "elastic_restore_reduction={}  elastic_restore_abs_tol={}  elastic_restore_max_iters={}",
                sci_text(options.elastic_restore_reduction_factor),
                sci_text(options.elastic_restore_abs_tol),
                options.elastic_restore_max_iters,
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

fn has_event(snapshot: &SqpIterationSnapshot, event: SqpIterationEvent) -> bool {
    snapshot.events.contains(&event)
}

fn fmt_event_codes(snapshot: &SqpIterationSnapshot) -> String {
    let mut codes = String::new();
    if has_event(snapshot, SqpIterationEvent::PenaltyUpdated) {
        codes.push('P');
    }
    if has_event(snapshot, SqpIterationEvent::LongLineSearch) {
        codes.push('L');
    }
    if has_event(snapshot, SqpIterationEvent::QpReducedAccuracy) {
        codes.push('R');
    }
    if has_event(snapshot, SqpIterationEvent::ElasticRecoveryUsed) {
        codes.push('E');
    }
    if has_event(snapshot, SqpIterationEvent::MaxIterationsReached) {
        codes.push('M');
    }
    codes
}

fn style_event_cell(snapshot: &SqpIterationSnapshot) -> String {
    let codes = fmt_event_codes(snapshot);
    let cell = format!("{:>4}", codes);
    if codes.is_empty() {
        return cell;
    }
    if has_event(snapshot, SqpIterationEvent::MaxIterationsReached) {
        style_red_bold(&cell)
    } else if has_event(snapshot, SqpIterationEvent::PenaltyUpdated)
        || has_event(snapshot, SqpIterationEvent::LongLineSearch)
        || has_event(snapshot, SqpIterationEvent::QpReducedAccuracy)
        || has_event(snapshot, SqpIterationEvent::ElasticRecoveryUsed)
    {
        style_yellow_bold(&cell)
    } else {
        cell
    }
}

fn event_legend_suffix(snapshot: &SqpIterationSnapshot, state: &mut SqpEventLegendState) -> String {
    let mut parts = Vec::new();

    if has_event(snapshot, SqpIterationEvent::PenaltyUpdated)
        && state.mark_if_new(SQP_EVENT_SEEN_PENALTY)
    {
        parts.push("P=merit penalty increased");
    }
    if has_event(snapshot, SqpIterationEvent::LongLineSearch)
        && state.mark_if_new(SQP_EVENT_SEEN_LINE_SEARCH)
    {
        parts.push("L=line search backtracked >=4 times");
    }
    if has_event(snapshot, SqpIterationEvent::QpReducedAccuracy)
        && state.mark_if_new(SQP_EVENT_SEEN_QP)
    {
        parts.push("R=QP solved to reduced accuracy");
    }
    if has_event(snapshot, SqpIterationEvent::ElasticRecoveryUsed)
        && state.mark_if_new(SQP_EVENT_SEEN_ELASTIC)
    {
        parts.push("E=elastic recovery QP used after primal-infeasible linearization");
    }
    if has_event(snapshot, SqpIterationEvent::MaxIterationsReached)
        && state.mark_if_new(SQP_EVENT_SEEN_MAX_ITER)
    {
        parts.push("M=maximum SQP iterations reached");
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!("  {}", parts.join("  "))
    }
}

fn log_sqp_iteration(
    snapshot: &SqpIterationSnapshot,
    options: &ClarabelSqpOptions,
    event_state: &mut SqpEventLegendState,
) {
    if snapshot.iteration.is_multiple_of(10) {
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
    let line_search = snapshot.line_search.as_ref();
    let qp = snapshot.qp.as_ref();
    let row = [
        style_iteration_cell(
            snapshot.iteration,
            has_event(snapshot, SqpIterationEvent::MaxIterationsReached),
        ),
        fmt_sci(snapshot.objective),
        style_residual_cell(
            snapshot.eq_inf.unwrap_or(0.0),
            options.constraint_tol,
            snapshot.eq_inf.is_some(),
        ),
        style_residual_cell(
            snapshot.ineq_inf.unwrap_or(0.0),
            options.constraint_tol,
            snapshot.ineq_inf.is_some(),
        ),
        style_residual_cell(snapshot.dual_inf, options.dual_tol, true),
        style_residual_cell(
            snapshot.comp_inf.unwrap_or(0.0),
            options.complementarity_tol,
            snapshot.comp_inf.is_some(),
        ),
        fmt_optional_sci(snapshot.step_inf),
        fmt_sci(snapshot.penalty),
        fmt_alpha(line_search.map(|info| info.accepted_alpha)),
        style_line_search_iterations_cell(line_search.map(|info| info.backtrack_count)),
        style_event_cell(snapshot),
        fmt_qp_iterations(qp.and_then(|info| u32::try_from(info.iteration_count).ok())),
        fmt_qp_time(qp.map(|info| info.solve_time.as_secs_f64())),
    ];
    let mut rendered = row.join("  ");
    rendered.push_str(&event_legend_suffix(snapshot, event_state));
    eprintln!("{rendered}");
}

fn validate_finite_inputs(
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
) -> std::result::Result<(), ClarabelSqpError> {
    if x0.iter().any(|value| !value.is_finite()) {
        return Err(ClarabelSqpError::NonFiniteInput {
            stage: NonFiniteInputStage::InitialGuess,
        });
    }
    for (parameter_index, parameter) in parameters.iter().enumerate() {
        if parameter.values.iter().any(|value| !value.is_finite()) {
            return Err(ClarabelSqpError::NonFiniteInput {
                stage: NonFiniteInputStage::ParameterValues { parameter_index },
            });
        }
    }
    Ok(())
}

fn validate_finite_scalar_output(
    value: f64,
    stage: NonFiniteCallbackStage,
    current_state: Option<&SqpIterationSnapshot>,
    last_accepted_state: Option<&SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> std::result::Result<f64, ClarabelSqpError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: Box::new(SqpFailureContext {
                termination: SqpTermination::NonFiniteCallbackOutput,
                final_state: current_state.cloned(),
                final_state_kind: current_state.map(|_| SqpFinalStateKind::AcceptedIterate),
                last_accepted_state: last_accepted_state.cloned(),
                profiling: profiling.clone(),
            }),
        })
    }
}

fn validate_finite_slice_output(
    values: &[f64],
    stage: NonFiniteCallbackStage,
    current_state: Option<&SqpIterationSnapshot>,
    last_accepted_state: Option<&SqpIterationSnapshot>,
    profiling: &ClarabelSqpProfiling,
) -> std::result::Result<(), ClarabelSqpError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ClarabelSqpError::NonFiniteCallbackOutput {
            stage,
            context: Box::new(SqpFailureContext {
                termination: SqpTermination::NonFiniteCallbackOutput,
                final_state: current_state.cloned(),
                final_state_kind: current_state.map(|_| SqpFinalStateKind::AcceptedIterate),
                last_accepted_state: last_accepted_state.cloned(),
                profiling: profiling.clone(),
            }),
        })
    }
}

fn qp_status_info(
    raw_status: SolverStatus,
    setup_time: Duration,
    solve_time: Duration,
    iteration_count: u32,
) -> SqpQpInfo {
    let status = match raw_status {
        SolverStatus::Solved => SqpQpStatus::Solved,
        SolverStatus::AlmostSolved => SqpQpStatus::ReducedAccuracy,
        _ => SqpQpStatus::Failed,
    };
    SqpQpInfo {
        status,
        raw_status,
        setup_time,
        solve_time,
        iteration_count: iteration_count as Index,
    }
}

#[derive(Clone, Debug)]
struct RawQpSolve {
    primal: Vec<f64>,
    dual: Vec<f64>,
    qp_info: SqpQpInfo,
}

#[derive(Clone, Debug)]
struct SqpSubproblemSolution {
    step: Vec<f64>,
    equality_multipliers: Vec<f64>,
    inequality_multipliers: Vec<f64>,
    lower_bound_multipliers: Vec<f64>,
    upper_bound_multipliers: Vec<f64>,
    qp_info: SqpQpInfo,
    elastic_recovery_used: bool,
}

struct QpSolveContext<'a> {
    profiling: &'a mut ClarabelSqpProfiling,
    iteration_qp_setup_time: &'a mut Duration,
    iteration_qp_solve_time: &'a mut Duration,
}

struct ElasticRecoveryModel<'a> {
    hessian: &'a DMatrix<f64>,
    gradient: &'a [f64],
    equality_values: &'a [f64],
    equality_jacobian: &'a DMatrix<f64>,
    nonlinear_inequality_values: &'a [f64],
    nonlinear_inequality_jacobian: &'a DMatrix<f64>,
    augmented_inequality_values: &'a [f64],
    bound_jacobian: &'a DMatrix<f64>,
}

fn solve_clarabel_qp_from_dense(
    hessian: &DMatrix<f64>,
    linear_objective: &[f64],
    constraint_matrix: &DMatrix<f64>,
    rhs: &[f64],
    cones: &[clarabel::solver::SupportedConeT<f64>],
    qp_ctx: &mut QpSolveContext<'_>,
) -> std::result::Result<RawQpSolve, ClarabelSqpError> {
    let qp_setup_started = Instant::now();
    let p = dense_to_csc_upper(hessian);
    let a = dense_to_csc(constraint_matrix);
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .build()
        .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
    let mut solver = DefaultSolver::new(&p, linear_objective, &a, rhs, cones, settings)
        .map_err(|err| ClarabelSqpError::Setup(err.to_string()))?;
    let qp_setup_elapsed = qp_setup_started.elapsed();
    qp_ctx.profiling.qp_setups += 1;
    qp_ctx.profiling.qp_setup_time += qp_setup_elapsed;
    *qp_ctx.iteration_qp_setup_time += qp_setup_elapsed;

    let qp_solve_started = Instant::now();
    solver.solve();
    let qp_solve_elapsed = qp_solve_started.elapsed();
    qp_ctx.profiling.qp_solves += 1;
    qp_ctx.profiling.qp_solve_time += qp_solve_elapsed;
    *qp_ctx.iteration_qp_solve_time += qp_solve_elapsed;

    Ok(RawQpSolve {
        primal: solver.solution.x.clone(),
        dual: solver.solution.z.clone(),
        qp_info: qp_status_info(
            solver.solution.status,
            qp_setup_elapsed,
            qp_solve_elapsed,
            solver.solution.iterations,
        ),
    })
}

fn should_try_elastic_recovery(
    status: SolverStatus,
    equality_count: Index,
    nonlinear_inequality_count: Index,
    options: &ClarabelSqpOptions,
) -> bool {
    options.elastic_mode
        && (equality_count > 0 || nonlinear_inequality_count > 0)
        && matches!(
            status,
            SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible
        )
}

fn decode_normal_qp_solution(
    raw: RawQpSolve,
    equality_count: Index,
    nonlinear_inequality_count: Index,
    lower_bound_count: Index,
) -> SqpSubproblemSolution {
    let (equality_multipliers, candidate_augmented_inequality_multipliers) =
        split_multipliers(&raw.dual, equality_count);
    let (inequality_multipliers, lower_bound_multipliers, upper_bound_multipliers) =
        split_augmented_inequality_multipliers(
            &candidate_augmented_inequality_multipliers,
            nonlinear_inequality_count,
            lower_bound_count,
        );
    SqpSubproblemSolution {
        step: raw.primal,
        equality_multipliers,
        inequality_multipliers,
        lower_bound_multipliers,
        upper_bound_multipliers,
        qp_info: raw.qp_info,
        elastic_recovery_used: false,
    }
}

fn decode_elastic_qp_solution(
    raw: RawQpSolve,
    step_dimension: Index,
    equality_count: Index,
    nonlinear_inequality_count: Index,
    lower_bound_count: Index,
) -> SqpSubproblemSolution {
    let mut cursor = 0;
    let equality_upper = &raw.dual[cursor..cursor + equality_count];
    cursor += equality_count;
    let equality_lower = &raw.dual[cursor..cursor + equality_count];
    cursor += equality_count;
    let inequality_multipliers = raw.dual[cursor..cursor + nonlinear_inequality_count].to_vec();
    cursor += nonlinear_inequality_count;
    let augmented_bound_multipliers = &raw.dual[cursor..];
    let (lower_bound_multipliers, upper_bound_multipliers) =
        augmented_bound_multipliers.split_at(lower_bound_count);
    let equality_multipliers = equality_upper
        .iter()
        .zip(equality_lower.iter())
        .map(|(upper, lower)| upper - lower)
        .collect();
    SqpSubproblemSolution {
        step: raw.primal[..step_dimension].to_vec(),
        equality_multipliers,
        inequality_multipliers,
        lower_bound_multipliers: lower_bound_multipliers.to_vec(),
        upper_bound_multipliers: upper_bound_multipliers.to_vec(),
        qp_info: raw.qp_info,
        elastic_recovery_used: true,
    }
}

fn solve_elastic_recovery_qp(
    model: &ElasticRecoveryModel<'_>,
    options: &ClarabelSqpOptions,
    qp_ctx: &mut QpSolveContext<'_>,
) -> std::result::Result<RawQpSolve, ClarabelSqpError> {
    // Follow SNOPT-style elastic mode precedent: add nonnegative elastic variables to the
    // linearized nonlinear constraints, penalize their L1 norm in the QP objective, keep simple
    // bounds hard, and return to the normal SQP model on the next major iteration.
    let step_dimension = model.gradient.len();
    let equality_count = model.equality_values.len();
    let nonlinear_inequality_count = model.nonlinear_inequality_values.len();
    let bound_count = model.augmented_inequality_values.len() - nonlinear_inequality_count;
    let total_elastic = equality_count + nonlinear_inequality_count;
    let variable_count = step_dimension + total_elastic;
    let equality_elastic_offset = step_dimension;
    let inequality_elastic_offset = step_dimension + equality_count;

    let mut elastic_hessian = DMatrix::<f64>::zeros(variable_count, variable_count);
    for row in 0..step_dimension {
        for col in 0..step_dimension {
            elastic_hessian[(row, col)] = model.hessian[(row, col)];
        }
        elastic_hessian[(row, row)] += options.elastic_primal_regularization;
    }
    for idx in step_dimension..variable_count {
        elastic_hessian[(idx, idx)] = options.elastic_slack_regularization;
    }

    let mut linear_objective = model.gradient.to_vec();
    linear_objective.extend(std::iter::repeat_n(options.elastic_weight, total_elastic));

    let total_rows = 2 * equality_count
        + nonlinear_inequality_count
        + bound_count
        + equality_count
        + nonlinear_inequality_count;
    let mut constraint_matrix = DMatrix::<f64>::zeros(total_rows, variable_count);
    let mut rhs = vec![0.0; total_rows];
    let mut cones = Vec::new();
    let mut row = 0;

    if equality_count > 0 {
        for eq_row in 0..equality_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + eq_row, col)] = model.equality_jacobian[(eq_row, col)];
            }
            constraint_matrix[(row + eq_row, equality_elastic_offset + eq_row)] = -1.0;
            rhs[row + eq_row] = -model.equality_values[eq_row];
        }
        cones.push(NonnegativeConeT(equality_count));
        row += equality_count;

        for eq_row in 0..equality_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + eq_row, col)] = -model.equality_jacobian[(eq_row, col)];
            }
            constraint_matrix[(row + eq_row, equality_elastic_offset + eq_row)] = -1.0;
            rhs[row + eq_row] = model.equality_values[eq_row];
        }
        cones.push(NonnegativeConeT(equality_count));
        row += equality_count;
    }

    if nonlinear_inequality_count > 0 {
        for ineq_row in 0..nonlinear_inequality_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + ineq_row, col)] =
                    model.nonlinear_inequality_jacobian[(ineq_row, col)];
            }
            constraint_matrix[(row + ineq_row, inequality_elastic_offset + ineq_row)] = -1.0;
            rhs[row + ineq_row] = -model.nonlinear_inequality_values[ineq_row];
        }
        cones.push(NonnegativeConeT(nonlinear_inequality_count));
        row += nonlinear_inequality_count;
    }

    if bound_count > 0 {
        for bound_row in 0..bound_count {
            for col in 0..step_dimension {
                constraint_matrix[(row + bound_row, col)] = model.bound_jacobian[(bound_row, col)];
            }
            rhs[row + bound_row] =
                -model.augmented_inequality_values[nonlinear_inequality_count + bound_row];
        }
        cones.push(NonnegativeConeT(bound_count));
        row += bound_count;
    }

    if equality_count > 0 {
        for eq_row in 0..equality_count {
            constraint_matrix[(row + eq_row, equality_elastic_offset + eq_row)] = -1.0;
        }
        cones.push(NonnegativeConeT(equality_count));
        row += equality_count;
    }

    if nonlinear_inequality_count > 0 {
        for ineq_row in 0..nonlinear_inequality_count {
            constraint_matrix[(row + ineq_row, inequality_elastic_offset + ineq_row)] = -1.0;
        }
        cones.push(NonnegativeConeT(nonlinear_inequality_count));
    }

    let solve = solve_clarabel_qp_from_dense(
        &elastic_hessian,
        &linear_objective,
        &constraint_matrix,
        &rhs,
        &cones,
        qp_ctx,
    )?;
    qp_ctx.profiling.elastic_recovery_qp_solves += 1;
    Ok(solve)
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
    solve_nlp_sqp_with_callback(problem, x0, parameters, options, |_| {})
}

pub fn solve_nlp_sqp_with_callback<P, C>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &ClarabelSqpOptions,
    mut callback: C,
) -> std::result::Result<ClarabelSqpSummary, ClarabelSqpError>
where
    P: CompiledNlpProblem,
    C: FnMut(&SqpIterationSnapshot),
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
    validate_finite_inputs(x0, parameters)?;
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
    let mut lower_bound_multipliers = vec![0.0; lower_bound_count];
    let mut upper_bound_multipliers = vec![0.0; bounds.upper_indices.len()];
    let mut merit_penalty = options.merit_penalty;
    let mut event_state = SqpEventLegendState::default();
    let mut previous_step_inf = None;
    let mut previous_line_search = None;
    let mut previous_qp = None;
    let mut previous_elastic_recovery_used = false;
    let mut previous_events = Vec::new();
    let mut last_accepted_state = None;
    let mut elastic_mode_active = false;
    let mut elastic_mode_entry_primal_inf = 0.0;
    let mut elastic_mode_iters = 0;

    if options.verbose {
        log_sqp_problem_header(problem, parameters, options);
    }

    for iteration in 0..options.max_iters {
        let iteration_started = Instant::now();
        let mut iteration_callback_time = Duration::ZERO;
        let mut iteration_qp_setup_time = Duration::ZERO;
        let mut iteration_qp_solve_time = Duration::ZERO;
        let objective_value = validate_finite_scalar_output(
            time_callback(
                &mut profiling.objective_value,
                &mut iteration_callback_time,
                || problem.objective_value(&x, parameters),
            ),
            NonFiniteCallbackStage::ObjectiveValue,
            None,
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.objective_gradient,
            &mut iteration_callback_time,
            || problem.objective_gradient(&x, parameters, &mut gradient),
        );
        validate_finite_slice_output(
            &gradient,
            NonFiniteCallbackStage::ObjectiveGradient,
            None,
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.equality_values,
            &mut iteration_callback_time,
            || problem.equality_values(&x, parameters, &mut equality_values),
        );
        validate_finite_slice_output(
            &equality_values,
            NonFiniteCallbackStage::EqualityValues,
            None,
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.inequality_values,
            &mut iteration_callback_time,
            || problem.inequality_values(&x, parameters, &mut inequality_values),
        );
        validate_finite_slice_output(
            &inequality_values,
            NonFiniteCallbackStage::InequalityValues,
            None,
            last_accepted_state.as_ref(),
            &profiling,
        )?;
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
        validate_finite_slice_output(
            &equality_jacobian_values,
            NonFiniteCallbackStage::EqualityJacobianValues,
            None,
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        time_callback(
            &mut profiling.inequality_jacobian_values,
            &mut iteration_callback_time,
            || problem.inequality_jacobian_values(&x, parameters, &mut inequality_jacobian_values),
        );
        validate_finite_slice_output(
            &inequality_jacobian_values,
            NonFiniteCallbackStage::InequalityJacobianValues,
            None,
            last_accepted_state.as_ref(),
            &profiling,
        )?;
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
        let all_inequality_multipliers = [
            inequality_multipliers.as_slice(),
            lower_bound_multipliers.as_slice(),
            upper_bound_multipliers.as_slice(),
        ]
        .concat();
        let dual_residual = lagrangian_gradient(
            &gradient,
            &equality_jacobian,
            &equality_multipliers,
            &inequality_jacobian,
            &all_inequality_multipliers,
        );
        let dual_inf = inf_norm(&dual_residual);
        let complementarity_inf =
            complementarity_inf_norm(&augmented_inequality_values, &all_inequality_multipliers);
        let preprocess_duration = iteration_started
            .elapsed()
            .saturating_sub(iteration_callback_time);
        let phase = if iteration == 0 {
            SqpIterationPhase::Initial
        } else if primal_inf <= options.constraint_tol
            && dual_inf <= options.dual_tol
            && complementarity_inf <= options.complementarity_tol
        {
            SqpIterationPhase::PostConvergence
        } else {
            SqpIterationPhase::AcceptedStep
        };
        let current_snapshot = SqpIterationSnapshot {
            iteration,
            phase,
            x: x.clone(),
            objective: objective_value,
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
            step_inf: previous_step_inf,
            penalty: merit_penalty,
            line_search: previous_line_search.clone(),
            qp: previous_qp.clone(),
            timing: SqpIterationTiming {
                objective_value: profiling.objective_value.total_time,
                objective_gradient: profiling.objective_gradient.total_time,
                equality_values: profiling.equality_values.total_time,
                inequality_values: profiling.inequality_values.total_time,
                equality_jacobian_values: profiling.equality_jacobian_values.total_time,
                inequality_jacobian_values: profiling.inequality_jacobian_values.total_time,
                lagrangian_hessian_values: Duration::ZERO,
                qp_setup: Duration::ZERO,
                qp_solve: Duration::ZERO,
                preprocess: preprocess_duration,
                total: iteration_started.elapsed(),
            },
            events: previous_events.clone(),
        };
        callback(&current_snapshot);
        if options.verbose {
            log_sqp_iteration(&current_snapshot, options, &mut event_state);
        }
        if primal_inf <= options.constraint_tol
            && dual_inf <= options.dual_tol
            && complementarity_inf <= options.complementarity_tol
        {
            profiling.preprocessing_time += preprocess_duration;
            finalize_profiling(&mut profiling, solve_started);
            let summary = ClarabelSqpSummary {
                x: x.clone(),
                equality_multipliers,
                inequality_multipliers,
                lower_bound_multipliers,
                upper_bound_multipliers,
                objective: objective_value,
                iterations: iteration,
                equality_inf_norm: (equality_count > 0).then_some(equality_inf),
                inequality_inf_norm: (augmented_inequality_count > 0).then_some(inequality_inf),
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: (augmented_inequality_count > 0)
                    .then_some(complementarity_inf),
                termination: SqpTermination::Converged,
                final_state: current_snapshot.clone(),
                final_state_kind: final_state_kind(&current_snapshot),
                last_accepted_state: None,
                profiling,
            };
            if options.verbose {
                log_sqp_status_summary(&summary, options);
            }
            return Ok(summary);
        }

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
        validate_finite_slice_output(
            &hessian_values,
            NonFiniteCallbackStage::LagrangianHessianValues,
            Some(&current_snapshot),
            last_accepted_state.as_ref(),
            &profiling,
        )?;
        let mut hessian =
            lower_triangle_to_symmetric_dense(problem.lagrangian_hessian_ccs(), &hessian_values);
        regularize_hessian(&mut hessian, options.regularization);

        let mut activated_elastic_mode = false;
        let SqpSubproblemSolution {
            step,
            equality_multipliers: candidate_equality_multipliers,
            inequality_multipliers: candidate_inequality_multipliers,
            lower_bound_multipliers: candidate_lower_bound_multipliers,
            upper_bound_multipliers: candidate_upper_bound_multipliers,
            qp_info: current_qp_info,
            elastic_recovery_used,
        } = if total_constraint_count == 0 {
            SqpSubproblemSolution {
                step: solve_unconstrained_quadratic_step(&hessian, &gradient).map_err(|()| {
                    ClarabelSqpError::UnconstrainedStepSolve {
                        context: failure_context(
                            SqpTermination::QpSolve,
                            Some(current_snapshot.clone()),
                            last_accepted_state.clone(),
                            &profiling,
                        ),
                    }
                })?,
                equality_multipliers: Vec::new(),
                inequality_multipliers: Vec::new(),
                lower_bound_multipliers: Vec::new(),
                upper_bound_multipliers: Vec::new(),
                qp_info: SqpQpInfo {
                    status: SqpQpStatus::Solved,
                    raw_status: SolverStatus::Solved,
                    setup_time: Duration::ZERO,
                    solve_time: Duration::ZERO,
                    iteration_count: 0,
                },
                elastic_recovery_used: false,
            }
        } else {
            let stacked_jacobian = stack_jacobians(&equality_jacobian, &inequality_jacobian);
            let mut rhs = equality_values
                .iter()
                .map(|value| -value)
                .collect::<Vec<_>>();
            rhs.extend(augmented_inequality_values.iter().map(|value| -value));
            let mut cones = Vec::with_capacity(2);
            if equality_count > 0 {
                cones.push(ZeroConeT(equality_count));
            }
            if augmented_inequality_count > 0 {
                cones.push(NonnegativeConeT(augmented_inequality_count));
            }
            let elastic_model = ElasticRecoveryModel {
                hessian: &hessian,
                gradient: &gradient,
                equality_values: &equality_values,
                equality_jacobian: &equality_jacobian,
                nonlinear_inequality_values: &inequality_values,
                nonlinear_inequality_jacobian: &nonlinear_inequality_jacobian,
                augmented_inequality_values: &augmented_inequality_values,
                bound_jacobian: &bound_jacobian,
            };
            let mut qp_ctx = QpSolveContext {
                profiling: &mut profiling,
                iteration_qp_setup_time: &mut iteration_qp_setup_time,
                iteration_qp_solve_time: &mut iteration_qp_solve_time,
            };
            if elastic_mode_active {
                let elastic_qp = solve_elastic_recovery_qp(&elastic_model, options, &mut qp_ctx)?;
                match elastic_qp.qp_info.raw_status {
                    SolverStatus::Solved | SolverStatus::AlmostSolved => {
                        decode_elastic_qp_solution(
                            elastic_qp,
                            n,
                            equality_count,
                            inequality_count,
                            lower_bound_count,
                        )
                    }
                    status => {
                        return Err(ClarabelSqpError::QpSolve {
                            status,
                            context: failure_context(
                                SqpTermination::QpSolve,
                                Some(current_snapshot.clone()),
                                last_accepted_state.clone(),
                                &profiling,
                            ),
                        });
                    }
                }
            } else {
                let normal_qp = solve_clarabel_qp_from_dense(
                    &hessian,
                    &gradient,
                    &stacked_jacobian,
                    &rhs,
                    &cones,
                    &mut qp_ctx,
                )?;
                match normal_qp.qp_info.raw_status {
                    SolverStatus::Solved | SolverStatus::AlmostSolved => decode_normal_qp_solution(
                        normal_qp,
                        equality_count,
                        inequality_count,
                        lower_bound_count,
                    ),
                    status
                        if should_try_elastic_recovery(
                            status,
                            equality_count,
                            inequality_count,
                            options,
                        ) =>
                    {
                        activated_elastic_mode = true;
                        qp_ctx.profiling.elastic_recovery_activations += 1;
                        let elastic_qp =
                            solve_elastic_recovery_qp(&elastic_model, options, &mut qp_ctx)?;
                        match elastic_qp.qp_info.raw_status {
                            SolverStatus::Solved | SolverStatus::AlmostSolved => {
                                decode_elastic_qp_solution(
                                    elastic_qp,
                                    n,
                                    equality_count,
                                    inequality_count,
                                    lower_bound_count,
                                )
                            }
                            status => {
                                return Err(ClarabelSqpError::QpSolve {
                                    status,
                                    context: failure_context(
                                        SqpTermination::QpSolve,
                                        Some(current_snapshot.clone()),
                                        last_accepted_state.clone(),
                                        &profiling,
                                    ),
                                });
                            }
                        }
                    }
                    status => {
                        return Err(ClarabelSqpError::QpSolve {
                            status,
                            context: failure_context(
                                SqpTermination::QpSolve,
                                Some(current_snapshot.clone()),
                                last_accepted_state.clone(),
                                &profiling,
                            ),
                        });
                    }
                }
            }
        };
        let current_qp = (total_constraint_count > 0).then_some(current_qp_info);

        let candidate_all_inequality_multipliers = [
            candidate_inequality_multipliers.as_slice(),
            candidate_lower_bound_multipliers.as_slice(),
            candidate_upper_bound_multipliers.as_slice(),
        ]
        .concat();
        let candidate_dual_inf = inf_norm(&lagrangian_gradient(
            &gradient,
            &equality_jacobian,
            &candidate_equality_multipliers,
            &inequality_jacobian,
            &candidate_all_inequality_multipliers,
        ));
        let candidate_complementarity_inf = complementarity_inf_norm(
            &augmented_inequality_values,
            &candidate_all_inequality_multipliers,
        );
        if primal_inf <= options.constraint_tol
            && candidate_dual_inf <= options.dual_tol
            && candidate_complementarity_inf <= options.complementarity_tol
        {
            let post_convergence_state = SqpIterationSnapshot {
                iteration,
                phase: SqpIterationPhase::PostConvergence,
                x: x.clone(),
                objective: objective_value,
                eq_inf: (equality_count > 0).then_some(equality_inf),
                ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
                dual_inf: candidate_dual_inf,
                comp_inf: (augmented_inequality_count > 0).then_some(candidate_complementarity_inf),
                step_inf: None,
                penalty: merit_penalty,
                line_search: None,
                qp: current_qp.clone(),
                timing: SqpIterationTiming {
                    objective_value: current_snapshot.timing.objective_value,
                    objective_gradient: current_snapshot.timing.objective_gradient,
                    equality_values: current_snapshot.timing.equality_values,
                    inequality_values: current_snapshot.timing.inequality_values,
                    equality_jacobian_values: current_snapshot.timing.equality_jacobian_values,
                    inequality_jacobian_values: current_snapshot.timing.inequality_jacobian_values,
                    lagrangian_hessian_values: profiling.lagrangian_hessian_values.total_time,
                    qp_setup: iteration_qp_setup_time,
                    qp_solve: iteration_qp_solve_time,
                    preprocess: Duration::ZERO,
                    total: iteration_started.elapsed(),
                },
                events: snapshot_events(
                    false,
                    None,
                    current_qp.as_ref(),
                    elastic_recovery_used,
                    false,
                ),
            };
            callback(&post_convergence_state);
            if options.verbose {
                log_sqp_iteration(&post_convergence_state, options, &mut event_state);
            }
            profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
                iteration_callback_time + iteration_qp_setup_time + iteration_qp_solve_time,
            );
            finalize_profiling(&mut profiling, solve_started);
            let summary = ClarabelSqpSummary {
                x: x.clone(),
                equality_multipliers: candidate_equality_multipliers,
                inequality_multipliers: candidate_inequality_multipliers,
                lower_bound_multipliers: candidate_lower_bound_multipliers,
                upper_bound_multipliers: candidate_upper_bound_multipliers,
                objective: objective_value,
                iterations: iteration,
                equality_inf_norm: (equality_count > 0).then_some(equality_inf),
                inequality_inf_norm: (augmented_inequality_count > 0).then_some(inequality_inf),
                primal_inf_norm: primal_inf,
                dual_inf_norm: candidate_dual_inf,
                complementarity_inf_norm: (augmented_inequality_count > 0)
                    .then_some(candidate_complementarity_inf),
                termination: SqpTermination::Converged,
                final_state: post_convergence_state.clone(),
                final_state_kind: final_state_kind(&post_convergence_state),
                last_accepted_state: Some(current_snapshot.clone()),
                profiling,
            };
            if options.verbose {
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
                context: failure_context(
                    SqpTermination::Stalled,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    &profiling,
                ),
            });
        }

        let penalty_before_updates = merit_penalty;
        if !elastic_recovery_used {
            merit_penalty = update_merit_penalty(
                merit_penalty,
                &candidate_equality_multipliers,
                &[
                    candidate_inequality_multipliers.as_slice(),
                    candidate_lower_bound_multipliers.as_slice(),
                    candidate_upper_bound_multipliers.as_slice(),
                ]
                .concat(),
            );
        }
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
                context: failure_context(
                    SqpTermination::LineSearchFailed,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    &profiling,
                ),
            });
        }
        let penalty_updated = merit_penalty != penalty_before_updates;

        let mut alpha = 1.0;
        let mut accepted_trial = None;
        let mut line_search_iterations = 0;
        let mut last_tried_alpha = alpha;
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
            )
            .map_err(|stage| ClarabelSqpError::NonFiniteCallbackOutput {
                stage,
                context: failure_context(
                    SqpTermination::NonFiniteCallbackOutput,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    &profiling,
                ),
            })?;
            if trial_merit_value
                <= current_merit + options.armijo_c1 * alpha * directional_derivative
            {
                accepted_trial = Some(trial);
                last_tried_alpha = alpha;
                break;
            }
            last_tried_alpha = alpha;
            alpha *= options.line_search_beta;
            line_search_iterations += 1;
        }
        let Some(trial) = accepted_trial else {
            return Err(ClarabelSqpError::LineSearchFailed {
                directional_derivative,
                step_inf_norm,
                penalty: merit_penalty,
                context: failure_context(
                    SqpTermination::LineSearchFailed,
                    Some(current_snapshot.clone()),
                    last_accepted_state.clone(),
                    &profiling,
                ),
            });
        };
        let accepted_equality_inf = inf_norm(&trial_equality_values);
        let accepted_inequality_inf = positive_part_inf_norm(&trial_augmented_inequality_values);
        let accepted_primal_inf = accepted_equality_inf.max(accepted_inequality_inf);

        profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
            iteration_callback_time + iteration_qp_setup_time + iteration_qp_solve_time,
        );
        x = trial;
        equality_multipliers = candidate_equality_multipliers;
        inequality_multipliers = candidate_inequality_multipliers;
        lower_bound_multipliers = candidate_lower_bound_multipliers;
        upper_bound_multipliers = candidate_upper_bound_multipliers;
        previous_step_inf = Some(step_inf_norm);
        previous_line_search = Some(SqpLineSearchInfo {
            accepted_alpha: alpha,
            last_tried_alpha,
            backtrack_count: line_search_iterations,
        });
        previous_qp = current_qp;
        previous_elastic_recovery_used = elastic_recovery_used;
        if elastic_recovery_used {
            if activated_elastic_mode {
                elastic_mode_entry_primal_inf = primal_inf;
                elastic_mode_iters = 0;
            }
            elastic_mode_iters += 1;
            let elastic_exit_target = (elastic_mode_entry_primal_inf
                * options.elastic_restore_reduction_factor)
                .max(options.elastic_restore_abs_tol);
            elastic_mode_active = elastic_mode_iters < options.elastic_restore_max_iters
                && accepted_primal_inf > elastic_exit_target;
            if !elastic_mode_active {
                elastic_mode_entry_primal_inf = 0.0;
                elastic_mode_iters = 0;
            }
        } else {
            elastic_mode_active = false;
            elastic_mode_entry_primal_inf = 0.0;
            elastic_mode_iters = 0;
        }
        previous_events = snapshot_events(
            penalty_updated,
            Some(line_search_iterations),
            previous_qp.as_ref(),
            previous_elastic_recovery_used,
            false,
        );
        last_accepted_state = Some(current_snapshot);
    }

    let max_iteration = options.max_iters;
    let iteration_started = Instant::now();
    let mut iteration_callback_time = Duration::ZERO;
    time_callback(
        &mut profiling.objective_gradient,
        &mut iteration_callback_time,
        || problem.objective_gradient(&x, parameters, &mut gradient),
    );
    validate_finite_slice_output(
        &gradient,
        NonFiniteCallbackStage::ObjectiveGradient,
        last_accepted_state.as_ref(),
        last_accepted_state.as_ref(),
        &profiling,
    )?;
    time_callback(
        &mut profiling.equality_values,
        &mut iteration_callback_time,
        || problem.equality_values(&x, parameters, &mut equality_values),
    );
    validate_finite_slice_output(
        &equality_values,
        NonFiniteCallbackStage::EqualityValues,
        last_accepted_state.as_ref(),
        last_accepted_state.as_ref(),
        &profiling,
    )?;
    time_callback(
        &mut profiling.inequality_values,
        &mut iteration_callback_time,
        || problem.inequality_values(&x, parameters, &mut inequality_values),
    );
    validate_finite_slice_output(
        &inequality_values,
        NonFiniteCallbackStage::InequalityValues,
        last_accepted_state.as_ref(),
        last_accepted_state.as_ref(),
        &profiling,
    )?;
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
    validate_finite_slice_output(
        &equality_jacobian_values,
        NonFiniteCallbackStage::EqualityJacobianValues,
        last_accepted_state.as_ref(),
        last_accepted_state.as_ref(),
        &profiling,
    )?;
    time_callback(
        &mut profiling.inequality_jacobian_values,
        &mut iteration_callback_time,
        || problem.inequality_jacobian_values(&x, parameters, &mut inequality_jacobian_values),
    );
    validate_finite_slice_output(
        &inequality_jacobian_values,
        NonFiniteCallbackStage::InequalityJacobianValues,
        last_accepted_state.as_ref(),
        last_accepted_state.as_ref(),
        &profiling,
    )?;
    let equality_jacobian =
        ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
    let nonlinear_inequality_jacobian = ccs_to_dense(
        problem.inequality_jacobian_ccs(),
        &inequality_jacobian_values,
    );
    let inequality_jacobian = stack_jacobians(&nonlinear_inequality_jacobian, &bound_jacobian);
    let objective_value = validate_finite_scalar_output(
        time_callback(
            &mut profiling.objective_value,
            &mut iteration_callback_time,
            || problem.objective_value(&x, parameters),
        ),
        NonFiniteCallbackStage::ObjectiveValue,
        last_accepted_state.as_ref(),
        last_accepted_state.as_ref(),
        &profiling,
    )?;
    let equality_inf = inf_norm(&equality_values);
    let inequality_inf = positive_part_inf_norm(&augmented_inequality_values);
    let _primal_inf = equality_inf.max(inequality_inf);
    let all_inequality_multipliers = [
        inequality_multipliers.as_slice(),
        lower_bound_multipliers.as_slice(),
        upper_bound_multipliers.as_slice(),
    ]
    .concat();
    let dual_inf = inf_norm(&lagrangian_gradient(
        &gradient,
        &equality_jacobian,
        &equality_multipliers,
        &inequality_jacobian,
        &all_inequality_multipliers,
    ));
    let complementarity_inf =
        complementarity_inf_norm(&augmented_inequality_values, &all_inequality_multipliers);
    let final_snapshot = SqpIterationSnapshot {
        iteration: max_iteration,
        phase: SqpIterationPhase::AcceptedStep,
        x: x.clone(),
        objective: objective_value,
        eq_inf: (equality_count > 0).then_some(equality_inf),
        ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
        dual_inf,
        comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
        step_inf: previous_step_inf,
        penalty: merit_penalty,
        line_search: previous_line_search.clone(),
        qp: previous_qp.clone(),
        timing: SqpIterationTiming {
            objective_value: profiling.objective_value.total_time,
            objective_gradient: profiling.objective_gradient.total_time,
            equality_values: profiling.equality_values.total_time,
            inequality_values: profiling.inequality_values.total_time,
            equality_jacobian_values: profiling.equality_jacobian_values.total_time,
            inequality_jacobian_values: profiling.inequality_jacobian_values.total_time,
            lagrangian_hessian_values: profiling.lagrangian_hessian_values.total_time,
            qp_setup: Duration::ZERO,
            qp_solve: Duration::ZERO,
            preprocess: iteration_started
                .elapsed()
                .saturating_sub(iteration_callback_time),
            total: iteration_started.elapsed(),
        },
        events: snapshot_events(
            false,
            previous_line_search
                .as_ref()
                .map(|info| info.backtrack_count),
            previous_qp.as_ref(),
            previous_elastic_recovery_used,
            true,
        ),
    };
    callback(&final_snapshot);
    if options.verbose {
        log_sqp_iteration(&final_snapshot, options, &mut event_state);
    }
    profiling.preprocessing_time += iteration_started
        .elapsed()
        .saturating_sub(iteration_callback_time);
    finalize_profiling(&mut profiling, solve_started);
    Err(ClarabelSqpError::MaxIterations {
        iterations: options.max_iters,
        context: failure_context(
            SqpTermination::MaxIterations,
            Some(final_snapshot),
            last_accepted_state,
            &profiling,
        ),
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
