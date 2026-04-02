use clarabel::algebra::CscMatrix;
use clarabel::qdldl::{QDLDLFactorisation, QDLDLSettings};
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

use super::{
    BackendTimingMetadata, BoundConstraints, CompiledNlpProblem, EvalTimingStat, Index,
    ParameterMatrix, SQP_EVENT_SEEN_LINE_SEARCH, SQP_EVENT_SEEN_MAX_ITER, SQP_EVENT_SEEN_PENALTY,
    SQP_LOG_HAS_EQUALITIES, SQP_LOG_HAS_INEQUALITIES, SQP_LOG_ITERATION_LIMIT_REACHED,
    SQP_LOG_PENALTY_UPDATED, SolverAdapterTiming, SqpEventLegendState, augment_inequality_values,
    boxed_line, build_bound_jacobian, ccs_to_dense, choose_summary_duration_unit,
    collect_bound_constraints, compact_duration_text, complementarity_inf_norm,
    declared_box_constraint_count, dense_fill_percent, fmt_duration_in_unit,
    fmt_optional_duration_in_unit, inf_norm, lagrangian_gradient, log_boxed_section,
    lower_tri_fill_percent, lower_triangle_to_symmetric_dense, positive_part_inf_norm,
    regularize_hessian, sci_text, split_augmented_inequality_multipliers, style_bold,
    style_cyan_bold, style_green_bold, style_red_bold, style_yellow_bold, time_callback,
    validate_nlp_problem_shapes, validate_parameter_inputs,
};

const IP_LOG_LINEAR_FALLBACK: u8 = 1 << 4;
const IP_EVENT_SEEN_LINEAR_FALLBACK: u8 = 1 << 4;
const AUTO_SPARSE_QDLDL_MIN_DIM: usize = 10;
const LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM: f64 = 1e12;
const LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL: f64 = 1e-7;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointLinearSolver {
    Auto,
    SparseQdldl,
    DenseRegularizedLdl,
    DenseLu,
}

impl InteriorPointLinearSolver {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::SparseQdldl => "sparse_qdldl",
            Self::DenseRegularizedLdl => "dense_ldl",
            Self::DenseLu => "dense_lu",
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InteriorPointOptions {
    pub max_iters: Index,
    pub dual_tol: f64,
    pub constraint_tol: f64,
    pub complementarity_tol: f64,
    pub fraction_to_boundary: f64,
    pub line_search_beta: f64,
    pub line_search_c1: f64,
    pub min_step: f64,
    pub regularization: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub mu_min: f64,
    pub linear_solver: InteriorPointLinearSolver,
    pub verbose: bool,
}

impl Default for InteriorPointOptions {
    fn default() -> Self {
        Self {
            max_iters: 80,
            dual_tol: 2e-6,
            constraint_tol: 1e-6,
            complementarity_tol: 1e-6,
            fraction_to_boundary: 0.995,
            line_search_beta: 0.5,
            line_search_c1: 1e-4,
            min_step: 1e-8,
            regularization: 1e-6,
            sigma_min: 1e-4,
            sigma_max: 1.0,
            mu_min: 1e-12,
            linear_solver: InteriorPointLinearSolver::Auto,
            verbose: true,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteriorPointProfiling {
    pub objective_value: EvalTimingStat,
    pub objective_gradient: EvalTimingStat,
    pub equality_values: EvalTimingStat,
    pub inequality_values: EvalTimingStat,
    pub equality_jacobian_values: EvalTimingStat,
    pub inequality_jacobian_values: EvalTimingStat,
    pub lagrangian_hessian_values: EvalTimingStat,
    pub kkt_assemblies: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub kkt_assembly_time: Duration,
    pub linear_solves: Index,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub linear_solve_time: Duration,
    pub adapter_timing: Option<SolverAdapterTiming>,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub preprocessing_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub total_time: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub unaccounted_time: Duration,
    pub backend_timing: BackendTimingMetadata,
}

impl InteriorPointProfiling {
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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct InteriorPointSummary {
    pub x: Vec<f64>,
    pub equality_multipliers: Vec<f64>,
    pub inequality_multipliers: Vec<f64>,
    pub lower_bound_multipliers: Vec<f64>,
    pub upper_bound_multipliers: Vec<f64>,
    pub slack: Vec<f64>,
    pub objective: f64,
    pub iterations: Index,
    pub equality_inf_norm: f64,
    pub inequality_inf_norm: f64,
    pub primal_inf_norm: f64,
    pub dual_inf_norm: f64,
    pub complementarity_inf_norm: f64,
    pub barrier_parameter: f64,
    pub profiling: InteriorPointProfiling,
    pub linear_solver: InteriorPointLinearSolver,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointIterationPhase {
    AcceptedStep,
    Converged,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteriorPointIterationEvent {
    SigmaAdjusted,
    LongLineSearch,
    LinearSolverFallback,
    MaxIterationsReached,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InteriorPointIterationTiming {
    pub adapter_timing: Option<SolverAdapterTiming>,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub callback: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub kkt_assembly: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub linear_solve: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub preprocess: Duration,
    #[cfg_attr(feature = "serde", serde(with = "crate::duration_seconds_serde"))]
    pub total: Duration,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct InteriorPointIterationSnapshot {
    pub iteration: Index,
    pub phase: InteriorPointIterationPhase,
    pub x: Vec<f64>,
    pub objective: f64,
    pub eq_inf: Option<f64>,
    pub ineq_inf: Option<f64>,
    pub dual_inf: f64,
    pub comp_inf: Option<f64>,
    pub barrier_parameter: Option<f64>,
    pub alpha: Option<f64>,
    pub line_search_iterations: Option<Index>,
    pub linear_solver: InteriorPointLinearSolver,
    #[cfg_attr(
        feature = "serde",
        serde(with = "crate::option_duration_seconds_serde")
    )]
    pub linear_solve_time: Option<Duration>,
    pub timing: InteriorPointIterationTiming,
    pub events: Vec<InteriorPointIterationEvent>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Error)]
pub enum InteriorPointSolveError {
    #[error("invalid interior-point input: {0}")]
    InvalidInput(String),
    #[error("interior-point linear solve failed using {solver:?}")]
    LinearSolve { solver: InteriorPointLinearSolver },
    #[error(
        "interior-point line search failed (residual merit {merit}, mu {mu}, step inf-norm {step_inf_norm})"
    )]
    LineSearchFailed {
        merit: f64,
        mu: f64,
        step_inf_norm: f64,
    },
    #[error("interior-point failed to converge in {iterations} iterations")]
    MaxIterations { iterations: Index },
}

struct NewtonDirection {
    dx: Vec<f64>,
    d_lambda: Vec<f64>,
    ds: Vec<f64>,
    dz: Vec<f64>,
    solver_used: InteriorPointLinearSolver,
}

struct ActiveSetPolishDirection {
    dx: Vec<f64>,
    d_lambda: Vec<f64>,
    d_z: Vec<f64>,
    active_indices: Vec<Index>,
}

struct ActiveSetPolishSystem<'a> {
    hessian: &'a DMatrix<f64>,
    equality_jacobian: &'a DMatrix<f64>,
    inequality_jacobian: &'a DMatrix<f64>,
    equality_values: &'a [f64],
    augmented_inequality_values: &'a [f64],
    z: &'a [f64],
    dual_residual: &'a [f64],
    regularization: f64,
}

struct ReducedKktSystem<'a> {
    hessian: &'a DMatrix<f64>,
    equality_jacobian: &'a DMatrix<f64>,
    inequality_jacobian: &'a DMatrix<f64>,
    slack: &'a [f64],
    multipliers: &'a [f64],
    r_dual: &'a [f64],
    r_eq: &'a [f64],
    r_ineq: &'a [f64],
    r_cent: &'a [f64],
    solver: InteriorPointLinearSolver,
    regularization: f64,
}

#[derive(Clone)]
struct EvalState {
    objective_value: f64,
    gradient: Vec<f64>,
    equality_values: Vec<f64>,
    augmented_inequality_values: Vec<f64>,
    equality_jacobian: DMatrix<f64>,
    inequality_jacobian: DMatrix<f64>,
}

fn merit_residual(
    equality_inf: f64,
    inequality_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    mu: f64,
) -> f64 {
    equality_inf
        .max(inequality_inf)
        .max(dual_inf)
        .max(complementarity_inf)
        .max(mu)
}

fn barrier_objective_value(objective_value: f64, slack: &[f64], barrier_parameter: f64) -> f64 {
    if slack.is_empty() || barrier_parameter <= 0.0 {
        return objective_value;
    }
    objective_value - barrier_parameter * slack.iter().map(|value| value.ln()).sum::<f64>()
}

fn step_inf_norm(step: &[f64]) -> f64 {
    step.iter().fold(0.0, |acc, value| acc.max(value.abs()))
}

fn fraction_to_boundary(current: &[f64], direction: &[f64], tau: f64) -> f64 {
    let mut alpha = 1.0_f64;
    for (&value, &delta) in current.iter().zip(direction.iter()) {
        if delta < 0.0 {
            alpha = alpha.min(-tau * value / delta);
        }
    }
    alpha.clamp(0.0, 1.0)
}

fn barrier_parameter(slack: &[f64], multipliers: &[f64]) -> f64 {
    if slack.is_empty() {
        0.0
    } else {
        slack
            .iter()
            .zip(multipliers.iter())
            .map(|(s, z)| s * z)
            .sum::<f64>()
            / slack.len() as f64
    }
}

fn adapter_timing_delta<P>(
    problem: &P,
    previous: &mut Option<SolverAdapterTiming>,
) -> Option<SolverAdapterTiming>
where
    P: CompiledNlpProblem,
{
    let current = problem.adapter_timing_snapshot();
    let delta = match (current, *previous) {
        (Some(current), Some(previous)) => Some(current.saturating_sub(previous)),
        (Some(current), None) => Some(current),
        (None, _) => None,
    };
    *previous = current;
    delta
}

fn project_initial_point_into_box_interior(x: &mut [f64], bounds: &BoundConstraints) {
    for (&idx, &lower) in bounds.lower_indices.iter().zip(bounds.lower_values.iter()) {
        let margin = (0.01 * lower.abs().max(1.0)).max(1e-4);
        x[idx] = x[idx].max(lower + margin);
    }
    for (&idx, &upper) in bounds.upper_indices.iter().zip(bounds.upper_values.iter()) {
        let margin = (0.01 * upper.abs().max(1.0)).max(1e-4);
        x[idx] = x[idx].min(upper - margin);
    }
    for ((&idx, &lower), upper) in bounds
        .lower_indices
        .iter()
        .zip(bounds.lower_values.iter())
        .filter_map(|(idx, lower)| {
            bounds
                .upper_indices
                .iter()
                .position(|upper_idx| upper_idx == idx)
                .map(|position| ((idx, lower), bounds.upper_values[position]))
        })
    {
        if lower >= upper {
            continue;
        }
        let width = upper - lower;
        let margin = (0.1 * width).max(1e-4).min(0.5 * width);
        let interior_lower = lower + margin;
        let interior_upper = upper - margin;
        if interior_lower <= interior_upper {
            x[idx] = x[idx].clamp(interior_lower, interior_upper);
        } else {
            x[idx] = 0.5 * (lower + upper);
        }
    }
}

fn initialise_slack_and_multipliers(
    augmented_inequality_values: &[f64],
    slack: &mut [f64],
    multipliers: &mut [f64],
) {
    debug_assert_eq!(augmented_inequality_values.len(), slack.len());
    debug_assert_eq!(slack.len(), multipliers.len());
    for ((&g_i, s_i), z_i) in augmented_inequality_values
        .iter()
        .zip(slack.iter_mut())
        .zip(multipliers.iter_mut())
    {
        *s_i = if g_i < 0.0 { -g_i } else { g_i + 1.0 };
        *s_i = s_i.max(1e-4);
        *z_i = (1.0 / *s_i).max(1e-4);
    }
}

fn trial_state<P>(
    problem: &P,
    x: &[f64],
    parameters: &[ParameterMatrix<'_>],
    bounds: &BoundConstraints,
    profiling: &mut InteriorPointProfiling,
    callback_time: &mut Duration,
) -> EvalState
where
    P: CompiledNlpProblem,
{
    let mut gradient = vec![0.0; problem.dimension()];
    let mut equality_values = vec![0.0; problem.equality_count()];
    let mut nonlinear_inequality_values = vec![0.0; problem.inequality_count()];
    let mut augmented_inequality_values =
        vec![0.0; problem.inequality_count() + bounds.total_count()];
    let mut equality_jacobian_values = vec![0.0; problem.equality_jacobian_ccs().nnz()];
    let mut inequality_jacobian_values = vec![0.0; problem.inequality_jacobian_ccs().nnz()];
    let objective_value = time_callback(&mut profiling.objective_value, callback_time, || {
        problem.objective_value(x, parameters)
    });
    time_callback(&mut profiling.objective_gradient, callback_time, || {
        problem.objective_gradient(x, parameters, &mut gradient);
    });
    time_callback(&mut profiling.equality_values, callback_time, || {
        problem.equality_values(x, parameters, &mut equality_values);
    });
    time_callback(&mut profiling.inequality_values, callback_time, || {
        problem.inequality_values(x, parameters, &mut nonlinear_inequality_values);
    });
    augment_inequality_values(
        &nonlinear_inequality_values,
        x,
        bounds,
        &mut augmented_inequality_values,
    );
    time_callback(
        &mut profiling.equality_jacobian_values,
        callback_time,
        || problem.equality_jacobian_values(x, parameters, &mut equality_jacobian_values),
    );
    time_callback(
        &mut profiling.inequality_jacobian_values,
        callback_time,
        || problem.inequality_jacobian_values(x, parameters, &mut inequality_jacobian_values),
    );
    let equality_jacobian =
        ccs_to_dense(problem.equality_jacobian_ccs(), &equality_jacobian_values);
    let nonlinear_inequality_jacobian = ccs_to_dense(
        problem.inequality_jacobian_ccs(),
        &inequality_jacobian_values,
    );
    let bound_jacobian = build_bound_jacobian(bounds, problem.dimension());
    let inequality_jacobian =
        super::stack_jacobians(&nonlinear_inequality_jacobian, &bound_jacobian);
    EvalState {
        objective_value,
        gradient,
        equality_values,
        augmented_inequality_values,
        equality_jacobian,
        inequality_jacobian,
    }
}

fn factor_solve_dense_ldl(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    regularization: f64,
) -> Option<DVector<f64>> {
    let n = matrix.nrows();
    if matrix.ncols() != n || rhs.len() != n {
        return None;
    }
    let mut lower = DMatrix::<f64>::identity(n, n);
    let mut diag = vec![0.0; n];
    for k in 0..n {
        let mut d = matrix[(k, k)];
        for j in 0..k {
            d -= lower[(k, j)] * lower[(k, j)] * diag[j];
        }
        if d.abs() < regularization {
            d = if d.is_sign_negative() {
                -regularization
            } else {
                regularization
            };
        }
        if !d.is_finite() || d.abs() < f64::EPSILON {
            return None;
        }
        diag[k] = d;
        for i in (k + 1)..n {
            let mut value = matrix[(i, k)];
            for j in 0..k {
                value -= lower[(i, j)] * lower[(k, j)] * diag[j];
            }
            lower[(i, k)] = value / diag[k];
        }
    }

    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[(i, j)] * y[j];
        }
        y[i] = value;
    }

    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = y[i] / diag[i];
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut value = z[i];
        for j in (i + 1)..n {
            value -= lower[(j, i)] * x[j];
        }
        x[i] = value;
    }

    if x.iter().all(|value| value.is_finite()) {
        Some(DVector::from_vec(x))
    } else {
        None
    }
}

fn dense_symmetric_to_triu_csc(matrix: &DMatrix<f64>) -> Option<CscMatrix<f64>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return None;
    }

    let upper_bound_nnz = n.checked_mul(n + 1)?.checked_div(2)?;
    let mut rows = Vec::with_capacity(upper_bound_nnz);
    let mut cols = Vec::with_capacity(upper_bound_nnz);
    let mut values = Vec::with_capacity(upper_bound_nnz);

    for col in 0..n {
        for row in 0..=col {
            let value = matrix[(row, col)];
            if row == col || value != 0.0 {
                rows.push(row);
                cols.push(col);
                values.push(value);
            }
        }
    }

    Some(CscMatrix::new_from_triplets(n, n, rows, cols, values))
}

fn linear_solution_looks_reasonable(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    solution: &DVector<f64>,
) -> bool {
    if !solution.iter().all(|value| value.is_finite()) {
        return false;
    }

    let rhs_inf = rhs.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let solution_inf = solution
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if solution_inf > LINEAR_SOLUTION_MAX_RELATIVE_INF_NORM * (1.0 + rhs_inf) {
        return false;
    }

    let residual = matrix * solution - rhs;
    let residual_inf = residual
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    residual_inf <= LINEAR_SOLUTION_MAX_RELATIVE_RESIDUAL * (1.0 + rhs_inf)
}

fn default_dsigns(dimension: usize) -> Vec<i8> {
    vec![1_i8; dimension]
}

fn quasidefinite_dsigns(primal_dimension: usize, dual_dimension: usize) -> Vec<i8> {
    let mut dsigns = default_dsigns(primal_dimension + dual_dimension);
    dsigns[primal_dimension..].fill(-1);
    dsigns
}

fn factor_solve_sparse_qdldl(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    regularization: f64,
    dsigns: Option<&[i8]>,
) -> Option<DVector<f64>> {
    let n = matrix.nrows();
    if matrix.ncols() != n || rhs.len() != n {
        return None;
    }
    let csc = dense_symmetric_to_triu_csc(matrix)?;
    let dsigns = dsigns.map_or_else(|| default_dsigns(n), ToOwned::to_owned);
    if dsigns.len() != n {
        return None;
    }
    let settings = QDLDLSettings {
        amd_dense_scale: 1.5,
        Dsigns: Some(dsigns),
        regularize_enable: true,
        regularize_eps: regularization.max(1e-12),
        regularize_delta: regularization.max(1e-9),
        ..Default::default()
    };
    let mut factor = QDLDLFactorisation::new(&csc, Some(settings)).ok()?;
    let mut solution = rhs.iter().copied().collect::<Vec<_>>();
    factor.solve(&mut solution);
    let solution = DVector::from_vec(solution);
    if linear_solution_looks_reasonable(matrix, rhs, &solution) {
        Some(solution)
    } else {
        None
    }
}

fn auto_prefers_sparse_qdldl(matrix: &DMatrix<f64>) -> bool {
    matrix.nrows() >= AUTO_SPARSE_QDLDL_MIN_DIM
}

fn solve_symmetric_system(
    solver: InteriorPointLinearSolver,
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    regularization: f64,
    dsigns: Option<&[i8]>,
) -> std::result::Result<(DVector<f64>, InteriorPointLinearSolver), InteriorPointSolveError> {
    let try_sparse_qdldl = |matrix: &DMatrix<f64>, rhs: &DVector<f64>| {
        factor_solve_sparse_qdldl(matrix, rhs, regularization, dsigns)
            .map(|solution| (solution, InteriorPointLinearSolver::SparseQdldl))
    };
    let try_ldl = |matrix: &DMatrix<f64>, rhs: &DVector<f64>| {
        factor_solve_dense_ldl(matrix, rhs, regularization)
            .map(|solution| (solution, InteriorPointLinearSolver::DenseRegularizedLdl))
    };
    let try_lu = |matrix: &DMatrix<f64>, rhs: &DVector<f64>| {
        matrix
            .clone()
            .lu()
            .solve(rhs)
            .map(|solution| (solution, InteriorPointLinearSolver::DenseLu))
    };

    let result = match solver {
        InteriorPointLinearSolver::SparseQdldl => try_sparse_qdldl(matrix, rhs),
        InteriorPointLinearSolver::DenseRegularizedLdl => try_ldl(matrix, rhs),
        InteriorPointLinearSolver::DenseLu => try_lu(matrix, rhs),
        InteriorPointLinearSolver::Auto => {
            if auto_prefers_sparse_qdldl(matrix) {
                try_sparse_qdldl(matrix, rhs)
                    .or_else(|| try_ldl(matrix, rhs))
                    .or_else(|| try_lu(matrix, rhs))
            } else {
                try_ldl(matrix, rhs)
                    .or_else(|| try_lu(matrix, rhs))
                    .or_else(|| try_sparse_qdldl(matrix, rhs))
            }
        }
    };
    result.ok_or(InteriorPointSolveError::LinearSolve { solver })
}

fn refine_multipliers_on_active_set(
    gradient: &[f64],
    equality_jacobian: &DMatrix<f64>,
    inequality_jacobian: &DMatrix<f64>,
    augmented_inequality_values: &[f64],
    inequality_multipliers: &[f64],
    regularization: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let active_tolerance = 1e-4;
    let multiplier_tolerance = 1e-5;
    let active_indices = augmented_inequality_values
        .iter()
        .zip(inequality_multipliers.iter())
        .enumerate()
        .filter_map(|(idx, (&g_i, &z_i))| {
            if g_i >= -active_tolerance || z_i >= multiplier_tolerance {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let meq = equality_jacobian.nrows();
    let mactive = active_indices.len();
    if meq + mactive == 0 {
        return None;
    }

    let n = gradient.len();
    let mut stacked = DMatrix::<f64>::zeros(meq + mactive, n);
    for row in 0..meq {
        for col in 0..n {
            stacked[(row, col)] = equality_jacobian[(row, col)];
        }
    }
    for (offset, &active_row) in active_indices.iter().enumerate() {
        for col in 0..n {
            stacked[(meq + offset, col)] = inequality_jacobian[(active_row, col)];
        }
    }

    let mut normal_matrix = &stacked * stacked.transpose();
    for diag in 0..normal_matrix.nrows() {
        normal_matrix[(diag, diag)] += regularization;
    }
    let rhs = -(&stacked * DVector::from_column_slice(gradient));
    let (solution, _) = solve_symmetric_system(
        InteriorPointLinearSolver::Auto,
        &normal_matrix,
        &rhs,
        regularization,
        None,
    )
    .ok()?;

    let mut lambda_eq = vec![0.0; meq];
    for row in 0..meq {
        lambda_eq[row] = solution[row];
    }
    let mut z = vec![0.0; augmented_inequality_values.len()];
    for (offset, &active_row) in active_indices.iter().enumerate() {
        z[active_row] = solution[meq + offset].max(0.0);
    }
    Some((lambda_eq, z))
}

fn solve_active_set_polish_direction(
    system: &ActiveSetPolishSystem<'_>,
) -> Option<ActiveSetPolishDirection> {
    let active_tolerance = 1e-4;
    let multiplier_tolerance = 1e-5;
    let active_indices = system
        .augmented_inequality_values
        .iter()
        .zip(system.z.iter())
        .enumerate()
        .filter_map(|(idx, (&g_i, &z_i))| {
            if g_i >= -active_tolerance || z_i >= multiplier_tolerance {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let n = system.hessian.nrows();
    let meq = system.equality_jacobian.nrows();
    let mactive = active_indices.len();
    if meq + mactive == 0 {
        return None;
    }
    let total_rows = meq + mactive;
    let mut jacobian = DMatrix::<f64>::zeros(total_rows, n);
    for row in 0..meq {
        for col in 0..n {
            jacobian[(row, col)] = system.equality_jacobian[(row, col)];
        }
    }
    for (offset, &active_row) in active_indices.iter().enumerate() {
        for col in 0..n {
            jacobian[(meq + offset, col)] = system.inequality_jacobian[(active_row, col)];
        }
    }
    let mut kkt = DMatrix::<f64>::zeros(n + total_rows, n + total_rows);
    for row in 0..n {
        for col in 0..n {
            kkt[(row, col)] = system.hessian[(row, col)];
        }
    }
    for row in 0..total_rows {
        for col in 0..n {
            kkt[(col, n + row)] = jacobian[(row, col)];
            kkt[(n + row, col)] = jacobian[(row, col)];
        }
    }
    let mut rhs = DVector::<f64>::zeros(n + total_rows);
    for row in 0..n {
        rhs[row] = -system.dual_residual[row];
    }
    for row in 0..meq {
        rhs[n + row] = -system.equality_values[row];
    }
    for (offset, &active_row) in active_indices.iter().enumerate() {
        rhs[n + meq + offset] = -system.augmented_inequality_values[active_row];
    }
    let dsigns = quasidefinite_dsigns(n, total_rows);
    let (solution, _) = solve_symmetric_system(
        InteriorPointLinearSolver::Auto,
        &kkt,
        &rhs,
        system.regularization,
        Some(&dsigns),
    )
    .ok()?;
    let dx = solution.rows(0, n).iter().copied().collect::<Vec<_>>();
    let mut d_lambda = vec![0.0; meq];
    for row in 0..meq {
        d_lambda[row] = solution[n + row];
    }
    let mut d_z = vec![0.0; active_indices.len()];
    for (offset, &active_row) in active_indices.iter().enumerate() {
        let current = system.z[active_row];
        d_z[offset] = solution[n + meq + offset] - current;
    }
    Some(ActiveSetPolishDirection {
        dx,
        d_lambda,
        d_z,
        active_indices,
    })
}

fn solve_reduced_kkt(
    system: &ReducedKktSystem<'_>,
) -> std::result::Result<NewtonDirection, InteriorPointSolveError> {
    let n = system.hessian.nrows();
    let meq = system.equality_jacobian.nrows();
    let mineq = system.inequality_jacobian.nrows();

    let mut hbar = system.hessian.clone();
    let mut rhs_top = DVector::<f64>::from_iterator(n, system.r_dual.iter().map(|value| -value));

    if mineq > 0 {
        let diagonal = system
            .slack
            .iter()
            .zip(system.multipliers.iter())
            .map(|(s, z)| z / s)
            .collect::<Vec<_>>();
        for (row, scale) in diagonal.iter().copied().enumerate().take(mineq) {
            for col_i in 0..n {
                let a_i = system.inequality_jacobian[(row, col_i)];
                if a_i == 0.0 {
                    continue;
                }
                for col_j in 0..n {
                    hbar[(col_i, col_j)] += scale * a_i * system.inequality_jacobian[(row, col_j)];
                }
            }
        }
        let sz_term = system
            .r_cent
            .iter()
            .zip(system.multipliers.iter())
            .zip(system.r_ineq.iter())
            .zip(system.slack.iter())
            .map(|(((r_cent_i, z_i), r_ineq_i), s_i)| (r_cent_i - z_i * r_ineq_i) / s_i)
            .collect::<Vec<_>>();
        let correction = system.inequality_jacobian.transpose() * DVector::from_vec(sz_term);
        rhs_top += correction;
    }

    let (solution, solver_used) = if meq == 0 {
        solve_symmetric_system(system.solver, &hbar, &rhs_top, system.regularization, None)?
    } else {
        let mut kkt = DMatrix::<f64>::zeros(n + meq, n + meq);
        for row in 0..n {
            for col in 0..n {
                kkt[(row, col)] = hbar[(row, col)];
            }
        }
        for row in 0..meq {
            for col in 0..n {
                kkt[(col, n + row)] = system.equality_jacobian[(row, col)];
                kkt[(n + row, col)] = system.equality_jacobian[(row, col)];
            }
        }
        let mut rhs = DVector::<f64>::zeros(n + meq);
        for row in 0..n {
            rhs[row] = rhs_top[row];
        }
        for row in 0..meq {
            rhs[n + row] = -system.r_eq[row];
        }
        let dsigns = quasidefinite_dsigns(n, meq);
        solve_symmetric_system(
            system.solver,
            &kkt,
            &rhs,
            system.regularization,
            Some(&dsigns),
        )?
    };

    let dx = solution.rows(0, n).iter().copied().collect::<Vec<_>>();
    let d_lambda = if meq == 0 {
        Vec::new()
    } else {
        solution.rows(n, meq).iter().copied().collect::<Vec<_>>()
    };

    let (ds, dz) = if mineq == 0 {
        (Vec::new(), Vec::new())
    } else {
        let dx_vec = DVector::from_column_slice(&dx);
        let a_dx = system.inequality_jacobian * &dx_vec;
        let ds = system
            .r_ineq
            .iter()
            .zip(a_dx.iter())
            .map(|(r_ineq_i, a_dx_i)| -r_ineq_i - a_dx_i)
            .collect::<Vec<_>>();
        let dz = system
            .r_cent
            .iter()
            .zip(system.multipliers.iter())
            .zip(ds.iter())
            .zip(system.slack.iter())
            .map(|(((r_cent_i, z_i), ds_i), s_i)| (-r_cent_i - z_i * ds_i) / s_i)
            .collect::<Vec<_>>();
        (ds, dz)
    };

    Ok(NewtonDirection {
        dx,
        d_lambda,
        ds,
        dz,
        solver_used,
    })
}

fn style_ip_residual_text(value: f64, tolerance: f64, applicable: bool) -> String {
    if !applicable {
        return "--".to_string();
    }
    let text = sci_text(value);
    if value <= tolerance {
        style_green_bold(&text)
    } else {
        text
    }
}

fn style_ip_residual_cell(value: f64, tolerance: f64, applicable: bool) -> String {
    if !applicable {
        return format!("{:>9}", "--");
    }
    let cell = format!("{:>9}", sci_text(value));
    if value <= tolerance {
        style_green_bold(&cell)
    } else {
        cell
    }
}

fn fmt_optional_ip_sci(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{:>9}", sci_text(value)),
        None => format!("{:>9}", "--"),
    }
}

fn fmt_optional_index(value: Option<Index>) -> String {
    match value {
        Some(value) => format!("{value:>5}"),
        None => format!("{:>5}", "--"),
    }
}

fn style_ip_line_search_cell(iterations: Option<Index>) -> String {
    let cell = fmt_optional_index(iterations);
    match iterations {
        Some(iterations) if iterations >= 10 => style_red_bold(&cell),
        Some(iterations) if iterations >= 4 => style_yellow_bold(&cell),
        _ => cell,
    }
}

struct InteriorPointIterationLog {
    iteration: Index,
    flags: u8,
    objective_value: f64,
    equality_inf: f64,
    inequality_inf: f64,
    dual_inf: f64,
    complementarity_inf: f64,
    barrier_parameter: f64,
    alpha: Option<f64>,
    line_search_iterations: Option<Index>,
    linear_time_secs: Option<f64>,
    constraint_tol: f64,
    dual_tol: f64,
    complementarity_tol: f64,
}

fn fmt_ip_event_codes(log: &InteriorPointIterationLog) -> String {
    let mut codes = String::new();
    if log.flags & SQP_LOG_PENALTY_UPDATED != 0 {
        codes.push('P');
    }
    if matches!(log.line_search_iterations, Some(iterations) if iterations >= 4) {
        codes.push('L');
    }
    if log.flags & IP_LOG_LINEAR_FALLBACK != 0 {
        codes.push('F');
    }
    if log.flags & SQP_LOG_ITERATION_LIMIT_REACHED != 0 {
        codes.push('M');
    }
    codes
}

fn style_ip_event_cell(log: &InteriorPointIterationLog) -> String {
    let codes = fmt_ip_event_codes(log);
    let cell = format!("{:>4}", codes);
    if codes.is_empty() {
        cell
    } else if log.flags & SQP_LOG_ITERATION_LIMIT_REACHED != 0 {
        style_red_bold(&cell)
    } else {
        style_yellow_bold(&cell)
    }
}

fn ip_event_suffix(log: &InteriorPointIterationLog, state: &mut SqpEventLegendState) -> String {
    let mut parts = Vec::new();
    if log.flags & SQP_LOG_PENALTY_UPDATED != 0 && state.mark_if_new(SQP_EVENT_SEEN_PENALTY) {
        parts.push("P=sigma clipped or barrier safeguard engaged");
    }
    if matches!(log.line_search_iterations, Some(iterations) if iterations >= 4)
        && state.mark_if_new(SQP_EVENT_SEEN_LINE_SEARCH)
    {
        parts.push("L=line search backtracked >=4 times");
    }
    if log.flags & IP_LOG_LINEAR_FALLBACK != 0 && state.mark_if_new(IP_EVENT_SEEN_LINEAR_FALLBACK) {
        parts.push("F=linear solver fell back to LU");
    }
    if log.flags & SQP_LOG_ITERATION_LIMIT_REACHED != 0
        && state.mark_if_new(SQP_EVENT_SEEN_MAX_ITER)
    {
        parts.push("M=maximum interior-point iterations reached");
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("  {}", parts.join("  "))
    }
}

fn log_interior_point_iteration(
    log: &InteriorPointIterationLog,
    event_state: &mut SqpEventLegendState,
) {
    if log.iteration.is_multiple_of(10) {
        eprintln!();
        let header = [
            format!("{:>4}", "iter"),
            format!("{:>9}", "f"),
            format!("{:>9}", "eq_inf"),
            format!("{:>9}", "ineq_inf"),
            format!("{:>9}", "dual_inf"),
            format!("{:>9}", "comp_inf"),
            format!("{:>9}", "mu"),
            format!("{:>9}", "alpha"),
            format!("{:>5}", "ls_it"),
            format!("{:>4}", "evt"),
            format!("{:>7}", "lin_t"),
        ];
        eprintln!("{}", style_bold(&header.join("  ")));
    }
    let row = [
        if log.flags & SQP_LOG_ITERATION_LIMIT_REACHED != 0 {
            style_red_bold(&format!("{:>4}", log.iteration))
        } else {
            format!("{:>4}", log.iteration)
        },
        format!("{:>9}", sci_text(log.objective_value)),
        style_ip_residual_cell(
            log.equality_inf,
            log.constraint_tol,
            log.flags & SQP_LOG_HAS_EQUALITIES != 0,
        ),
        style_ip_residual_cell(
            log.inequality_inf,
            log.constraint_tol,
            log.flags & SQP_LOG_HAS_INEQUALITIES != 0,
        ),
        style_ip_residual_cell(log.dual_inf, log.dual_tol, true),
        style_ip_residual_cell(
            log.complementarity_inf,
            log.complementarity_tol,
            log.flags & SQP_LOG_HAS_INEQUALITIES != 0,
        ),
        format!("{:>9}", sci_text(log.barrier_parameter)),
        fmt_optional_ip_sci(log.alpha),
        style_ip_line_search_cell(log.line_search_iterations),
        style_ip_event_cell(log),
        match log.linear_time_secs {
            Some(seconds) => format!("{:>7}", compact_duration_text(seconds)),
            None => format!("{:>7}", "--"),
        },
    ];
    let mut rendered = row.join("  ");
    rendered.push_str(&ip_event_suffix(log, event_state));
    eprintln!("{rendered}");
}

fn log_interior_point_problem_header<P>(
    problem: &P,
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
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
                "beta={}  c1={}  min_step={}  fraction_to_boundary={}",
                sci_text(options.line_search_beta),
                sci_text(options.line_search_c1),
                sci_text(options.min_step),
                sci_text(options.fraction_to_boundary),
            ),
        ),
        boxed_line(
            "barrier",
            format!(
                "sigma_min={}  sigma_max={}  mu_min={}",
                sci_text(options.sigma_min),
                sci_text(options.sigma_max),
                sci_text(options.mu_min),
            ),
        ),
        boxed_line(
            "linear solve",
            format!(
                "solver={}  regularization={}",
                options.linear_solver.label(),
                sci_text(options.regularization),
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
    log_boxed_section("Interior-point problem / settings", &lines, style_cyan_bold);
}

fn log_interior_point_status_summary(
    summary: &InteriorPointSummary,
    options: &InteriorPointOptions,
) {
    let has_inequality_like_constraints = !summary.inequality_multipliers.is_empty()
        || !summary.lower_bound_multipliers.is_empty()
        || !summary.upper_bound_multipliers.is_empty();
    let eq_text = if summary.equality_multipliers.is_empty() {
        "--".to_string()
    } else {
        style_ip_residual_text(summary.equality_inf_norm, options.constraint_tol, true)
    };
    let ineq_text = if has_inequality_like_constraints {
        style_ip_residual_text(summary.inequality_inf_norm, options.constraint_tol, true)
    } else {
        "--".to_string()
    };
    let comp_text = if has_inequality_like_constraints {
        style_ip_residual_text(
            summary.complementarity_inf_norm,
            options.complementarity_tol,
            true,
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
        summary.profiling.kkt_assembly_time,
        summary.profiling.linear_solve_time,
        summary.profiling.preprocessing_time,
        summary.profiling.unaccounted_time,
        summary.profiling.total_time,
        summary
            .profiling
            .adapter_timing
            .map(|timing| timing.callback_evaluation)
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .adapter_timing
            .map(|timing| timing.output_marshalling)
            .unwrap_or(Duration::ZERO),
        summary
            .profiling
            .adapter_timing
            .map(|timing| timing.layout_projection)
            .unwrap_or(Duration::ZERO),
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
    let optional_timing_row = |name: &str, count: Option<Index>, duration: Option<Duration>| {
        let count_cell = match count {
            Some(count) => format!("{count:>4}"),
            None => format!("{:>4}", "--"),
        };
        let time_cell = duration
            .map(|duration| fmt_duration_in_unit(duration, timing_unit))
            .unwrap_or_else(|| format!("{:>7}", "--"));
        format!("{name:<12}  count={count_cell}  time={time_cell}")
    };
    let lines = vec![
        boxed_line(
            "result",
            format!(
                "objective={}  primal_inf={}  dual_inf={}  mu={}",
                sci_text(summary.objective),
                style_ip_residual_text(summary.primal_inf_norm, options.constraint_tol, true),
                style_ip_residual_text(summary.dual_inf_norm, options.dual_tol, true),
                sci_text(summary.barrier_parameter),
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
        boxed_line(
            "",
            callback_row(
                callback_rows[0].0,
                callback_rows[0].1.calls,
                callback_rows[0].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[1].0,
                callback_rows[1].1.calls,
                callback_rows[1].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[2].0,
                callback_rows[2].1.calls,
                callback_rows[2].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[3].0,
                callback_rows[3].1.calls,
                callback_rows[3].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[4].0,
                callback_rows[4].1.calls,
                callback_rows[4].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[5].0,
                callback_rows[5].1.calls,
                callback_rows[5].1.total_time,
            ),
        ),
        boxed_line(
            "",
            callback_row(
                callback_rows[6].0,
                callback_rows[6].1.calls,
                callback_rows[6].1.total_time,
            ),
        ),
        String::new(),
        boxed_line(
            "timing",
            timing_row(
                "kkt assembly",
                Some(summary.profiling.kkt_assemblies),
                summary.profiling.kkt_assembly_time,
            ),
        ),
        boxed_line(
            "",
            timing_row(
                "linear solve",
                Some(summary.profiling.linear_solves),
                summary.profiling.linear_solve_time,
            ),
        ),
        boxed_line(
            "",
            optional_timing_row(
                "adapter cb",
                None,
                summary
                    .profiling
                    .adapter_timing
                    .map(|timing| timing.callback_evaluation),
            ),
        ),
        boxed_line(
            "",
            optional_timing_row(
                "adapter io",
                None,
                summary
                    .profiling
                    .adapter_timing
                    .map(|timing| timing.output_marshalling),
            ),
        ),
        boxed_line(
            "",
            optional_timing_row(
                "layout",
                None,
                summary
                    .profiling
                    .adapter_timing
                    .map(|timing| timing.layout_projection),
            ),
        ),
        boxed_line(
            "",
            timing_row("preprocess", None, summary.profiling.preprocessing_time),
        ),
        boxed_line(
            "",
            timing_row("unaccounted", None, summary.profiling.unaccounted_time),
        ),
        boxed_line("", timing_row("total", None, summary.profiling.total_time)),
        String::new(),
        boxed_line(
            "backend",
            format!(
                "create={}  jit={}  linear_solver={}",
                fmt_optional_duration_in_unit(
                    summary.profiling.backend_timing.function_creation_time,
                    timing_unit,
                ),
                fmt_optional_duration_in_unit(
                    summary.profiling.backend_timing.jit_time,
                    timing_unit
                ),
                summary.linear_solver.label(),
            ),
        ),
    ];
    log_boxed_section("Interior-point converged", &lines, style_green_bold);
}

fn finalise_interior_point_profiling(
    profiling: &mut InteriorPointProfiling,
    solve_started: Instant,
) {
    profiling.total_time = solve_started.elapsed();
    profiling.unaccounted_time = profiling.total_time.saturating_sub(
        profiling.total_callback_time()
            + profiling.kkt_assembly_time
            + profiling.linear_solve_time
            + profiling.preprocessing_time,
    );
}

pub fn solve_nlp_interior_point<P>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
) -> std::result::Result<InteriorPointSummary, InteriorPointSolveError>
where
    P: CompiledNlpProblem,
{
    solve_nlp_interior_point_with_callback(problem, x0, parameters, options, |_| {})
}

pub fn solve_nlp_interior_point_with_callback<P, C>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: &InteriorPointOptions,
    mut callback: C,
) -> std::result::Result<InteriorPointSummary, InteriorPointSolveError>
where
    P: CompiledNlpProblem,
    C: FnMut(&InteriorPointIterationSnapshot),
{
    let solve_started = Instant::now();
    let mut profiling = InteriorPointProfiling {
        backend_timing: problem.backend_timing_metadata(),
        ..InteriorPointProfiling::default()
    };
    let validation_started = Instant::now();
    validate_nlp_problem_shapes(problem)
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
    validate_parameter_inputs(problem, parameters)
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
    profiling.preprocessing_time += validation_started.elapsed();

    let n = problem.dimension();
    if x0.len() != n {
        return Err(InteriorPointSolveError::InvalidInput(format!(
            "x0 has length {}, expected {n}",
            x0.len()
        )));
    }

    let equality_count = problem.equality_count();
    let inequality_count = problem.inequality_count();
    let bounds = collect_bound_constraints(problem)
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
    let augmented_inequality_count = inequality_count + bounds.total_count();
    let lower_bound_count = bounds.lower_indices.len();
    let mut x = x0.to_vec();
    project_initial_point_into_box_interior(&mut x, &bounds);
    let mut lambda_eq = vec![0.0; equality_count];
    let mut z = vec![1.0; augmented_inequality_count];
    let mut slack = vec![1.0; augmented_inequality_count];
    let mut event_state = SqpEventLegendState::default();
    let mut last_adapter_timing = problem.adapter_timing_snapshot();
    profiling.adapter_timing = last_adapter_timing;

    let setup_started = Instant::now();
    let mut setup_callback_time = Duration::ZERO;
    let initial_state = trial_state(
        problem,
        &x,
        parameters,
        &bounds,
        &mut profiling,
        &mut setup_callback_time,
    );
    profiling.preprocessing_time += setup_started.elapsed().saturating_sub(setup_callback_time);
    initialise_slack_and_multipliers(
        &initial_state.augmented_inequality_values,
        &mut slack,
        &mut z,
    );

    if options.verbose {
        log_interior_point_problem_header(problem, parameters, options);
    }

    let mut nonlinear_inequality_multipliers = vec![0.0; inequality_count];
    let mut last_linear_solver = options.linear_solver;

    'iterations: for iteration in 0..options.max_iters {
        let iteration_started = Instant::now();
        let mut iteration_callback_time = Duration::ZERO;
        let mut iteration_kkt_assembly_time = Duration::ZERO;
        let mut iteration_linear_solve_time = Duration::ZERO;
        let state = trial_state(
            problem,
            &x,
            parameters,
            &bounds,
            &mut profiling,
            &mut iteration_callback_time,
        );
        let equality_inf = inf_norm(&state.equality_values);
        let inequality_inf = positive_part_inf_norm(&state.augmented_inequality_values);
        let primal_inf = equality_inf.max(inequality_inf);
        let mut dual_residual = lagrangian_gradient(
            &state.gradient,
            &state.equality_jacobian,
            &lambda_eq,
            &state.inequality_jacobian,
            &z,
        );
        let mut dual_inf = inf_norm(&dual_residual);
        let complementarity_inf = if augmented_inequality_count > 0 {
            complementarity_inf_norm(&slack, &z)
        } else {
            0.0
        };
        if augmented_inequality_count > 0
            && primal_inf <= (100.0 * options.constraint_tol).max(1e-6)
            && complementarity_inf <= (100.0 * options.complementarity_tol).max(1e-8)
            && dual_inf > options.dual_tol
            && let Some((refined_lambda_eq, refined_z)) = refine_multipliers_on_active_set(
                &state.gradient,
                &state.equality_jacobian,
                &state.inequality_jacobian,
                &state.augmented_inequality_values,
                &z,
                options.regularization,
            )
        {
            let refined_dual_residual = lagrangian_gradient(
                &state.gradient,
                &state.equality_jacobian,
                &refined_lambda_eq,
                &state.inequality_jacobian,
                &refined_z,
            );
            let refined_dual_inf = inf_norm(&refined_dual_residual);
            if refined_dual_inf < dual_inf {
                lambda_eq = refined_lambda_eq;
                z = refined_z;
                dual_residual = refined_dual_residual;
                dual_inf = refined_dual_inf;
            }
        }
        let mu = barrier_parameter(&slack, &z);

        if primal_inf <= options.constraint_tol
            && dual_inf <= options.dual_tol
            && complementarity_inf <= options.complementarity_tol
        {
            let adapter_timing = adapter_timing_delta(problem, &mut last_adapter_timing);
            profiling.adapter_timing = last_adapter_timing;
            let iteration_total = iteration_started.elapsed();
            let iteration_preprocess = iteration_total.saturating_sub(
                iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
            );
            let snapshot = InteriorPointIterationSnapshot {
                iteration,
                phase: InteriorPointIterationPhase::Converged,
                x: x.clone(),
                objective: state.objective_value,
                eq_inf: (equality_count > 0).then_some(equality_inf),
                ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
                dual_inf,
                comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
                barrier_parameter: (augmented_inequality_count > 0).then_some(mu),
                alpha: None,
                line_search_iterations: None,
                linear_solver: last_linear_solver,
                linear_solve_time: None,
                timing: InteriorPointIterationTiming {
                    adapter_timing,
                    callback: iteration_callback_time,
                    kkt_assembly: iteration_kkt_assembly_time,
                    linear_solve: iteration_linear_solve_time,
                    preprocess: iteration_preprocess,
                    total: iteration_total,
                },
                events: Vec::new(),
            };
            callback(&snapshot);
            let (nonlinear_ineq, lower_bounds, upper_bounds) =
                split_augmented_inequality_multipliers(&z, inequality_count, lower_bound_count);
            let summary = InteriorPointSummary {
                x,
                equality_multipliers: lambda_eq,
                inequality_multipliers: nonlinear_ineq,
                lower_bound_multipliers: lower_bounds,
                upper_bound_multipliers: upper_bounds,
                slack,
                objective: state.objective_value,
                iterations: iteration,
                equality_inf_norm: equality_inf,
                inequality_inf_norm: inequality_inf,
                primal_inf_norm: primal_inf,
                dual_inf_norm: dual_inf,
                complementarity_inf_norm: complementarity_inf,
                barrier_parameter: mu,
                profiling,
                linear_solver: last_linear_solver,
            };
            let mut summary = summary;
            finalise_interior_point_profiling(&mut summary.profiling, solve_started);
            if options.verbose {
                log_interior_point_iteration(
                    &InteriorPointIterationLog {
                        iteration,
                        flags: (if equality_count > 0 {
                            SQP_LOG_HAS_EQUALITIES
                        } else {
                            0
                        }) | (if augmented_inequality_count > 0 {
                            SQP_LOG_HAS_INEQUALITIES
                        } else {
                            0
                        }),
                        objective_value: state.objective_value,
                        equality_inf,
                        inequality_inf,
                        dual_inf,
                        complementarity_inf,
                        barrier_parameter: mu,
                        alpha: None,
                        line_search_iterations: None,
                        linear_time_secs: None,
                        constraint_tol: options.constraint_tol,
                        dual_tol: options.dual_tol,
                        complementarity_tol: options.complementarity_tol,
                    },
                    &mut event_state,
                );
                log_interior_point_status_summary(&summary, options);
            }
            return Ok(summary);
        }

        let hessian_started = Instant::now();
        let mut hessian_values = vec![0.0; problem.lagrangian_hessian_ccs().nnz()];
        time_callback(
            &mut profiling.lagrangian_hessian_values,
            &mut iteration_callback_time,
            || {
                problem.lagrangian_hessian_values(
                    &x,
                    parameters,
                    &lambda_eq,
                    &nonlinear_inequality_multipliers,
                    &mut hessian_values,
                );
            },
        );
        let mut hessian =
            lower_triangle_to_symmetric_dense(problem.lagrangian_hessian_ccs(), &hessian_values);
        regularize_hessian(&mut hessian, options.regularization);
        let hessian_elapsed = hessian_started.elapsed();
        profiling.kkt_assemblies += 1;
        profiling.kkt_assembly_time += hessian_elapsed;
        iteration_kkt_assembly_time += hessian_elapsed;

        if augmented_inequality_count > 0
            && primal_inf <= (100.0 * options.constraint_tol).max(1e-6)
            && complementarity_inf <= (100.0 * options.complementarity_tol).max(1e-8)
            && dual_inf > options.dual_tol
            && let Some(polish_direction) =
                solve_active_set_polish_direction(&ActiveSetPolishSystem {
                    hessian: &hessian,
                    equality_jacobian: &state.equality_jacobian,
                    inequality_jacobian: &state.inequality_jacobian,
                    equality_values: &state.equality_values,
                    augmented_inequality_values: &state.augmented_inequality_values,
                    z: &z,
                    dual_residual: &dual_residual,
                    regularization: options.regularization,
                })
        {
            let active_z = polish_direction
                .active_indices
                .iter()
                .map(|&idx| z[idx])
                .collect::<Vec<_>>();
            let mut alpha = fraction_to_boundary(
                &active_z,
                &polish_direction.d_z,
                options.fraction_to_boundary,
            );
            alpha = alpha.clamp(0.0, 1.0);
            while alpha >= options.min_step {
                let trial_x = x
                    .iter()
                    .zip(polish_direction.dx.iter())
                    .map(|(value, delta)| value + alpha * delta)
                    .collect::<Vec<_>>();
                let trial_lambda = lambda_eq
                    .iter()
                    .zip(polish_direction.d_lambda.iter())
                    .map(|(value, delta)| value + alpha * delta)
                    .collect::<Vec<_>>();
                let mut trial_z = vec![0.0; augmented_inequality_count];
                for (offset, &active_row) in polish_direction.active_indices.iter().enumerate() {
                    trial_z[active_row] =
                        (z[active_row] + alpha * polish_direction.d_z[offset]).max(0.0);
                }
                let mut polish_callback_time = Duration::ZERO;
                let trial_state = trial_state(
                    problem,
                    &trial_x,
                    parameters,
                    &bounds,
                    &mut profiling,
                    &mut polish_callback_time,
                );
                let trial_eq_inf = inf_norm(&trial_state.equality_values);
                let trial_ineq_inf =
                    positive_part_inf_norm(&trial_state.augmented_inequality_values);
                if trial_eq_inf > primal_inf.max(options.constraint_tol)
                    || trial_ineq_inf > primal_inf.max(options.constraint_tol)
                {
                    alpha *= options.line_search_beta;
                    continue;
                }
                let trial_dual_residual = lagrangian_gradient(
                    &trial_state.gradient,
                    &trial_state.equality_jacobian,
                    &trial_lambda,
                    &trial_state.inequality_jacobian,
                    &trial_z,
                );
                let trial_dual_inf = inf_norm(&trial_dual_residual);
                if trial_dual_inf < dual_inf {
                    x = trial_x;
                    lambda_eq = trial_lambda;
                    z = trial_z;
                    for (slack_i, &g_i) in slack
                        .iter_mut()
                        .zip(trial_state.augmented_inequality_values.iter())
                    {
                        *slack_i = (-g_i).max(1e-8);
                    }
                    let (nonlinear, _, _) = split_augmented_inequality_multipliers(
                        &z,
                        inequality_count,
                        lower_bound_count,
                    );
                    nonlinear_inequality_multipliers = nonlinear;
                    profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
                        iteration_callback_time
                            + iteration_kkt_assembly_time
                            + iteration_linear_solve_time,
                    );
                    continue 'iterations;
                }
                alpha *= options.line_search_beta;
            }
        }

        let r_cent_aff = slack
            .iter()
            .zip(z.iter())
            .map(|(s, z_i)| s * z_i)
            .collect::<Vec<_>>();
        let linear_started = Instant::now();
        let affine_direction = solve_reduced_kkt(&ReducedKktSystem {
            hessian: &hessian,
            equality_jacobian: &state.equality_jacobian,
            inequality_jacobian: &state.inequality_jacobian,
            slack: &slack,
            multipliers: &z,
            r_dual: &dual_residual,
            r_eq: &state.equality_values,
            r_ineq: &state.augmented_inequality_values,
            r_cent: &r_cent_aff,
            solver: options.linear_solver,
            regularization: options.regularization,
        })?;
        let linear_elapsed = linear_started.elapsed();
        profiling.linear_solves += 1;
        profiling.linear_solve_time += linear_elapsed;
        iteration_linear_solve_time += linear_elapsed;
        let alpha_pr_aff = fraction_to_boundary(
            &slack,
            &affine_direction.ds,
            1.0_f64.min(options.fraction_to_boundary),
        );
        let alpha_du_aff = fraction_to_boundary(
            &z,
            &affine_direction.dz,
            1.0_f64.min(options.fraction_to_boundary),
        );
        let mu_aff = if augmented_inequality_count > 0 {
            let slack_aff = slack
                .iter()
                .zip(affine_direction.ds.iter())
                .map(|(s, ds)| s + alpha_pr_aff * ds)
                .collect::<Vec<_>>();
            let z_aff = z
                .iter()
                .zip(affine_direction.dz.iter())
                .map(|(zi, dz)| zi + alpha_du_aff * dz)
                .collect::<Vec<_>>();
            barrier_parameter(&slack_aff, &z_aff)
        } else {
            0.0
        };
        let mut sigma = if augmented_inequality_count > 0 && mu > 0.0 {
            (mu_aff / mu).powi(3)
        } else {
            0.0
        };
        let sigma_clipped = sigma.clamp(options.sigma_min, options.sigma_max);
        let sigma_adjusted = augmented_inequality_count > 0 && (sigma_clipped - sigma).abs() > 0.0;
        sigma = sigma_clipped;
        let r_cent = if augmented_inequality_count > 0 {
            slack
                .iter()
                .zip(z.iter())
                .zip(affine_direction.ds.iter())
                .zip(affine_direction.dz.iter())
                .map(|(((s, z_i), ds), dz)| s * z_i + ds * dz - sigma * mu)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let linear_started = Instant::now();
        let direction = solve_reduced_kkt(&ReducedKktSystem {
            hessian: &hessian,
            equality_jacobian: &state.equality_jacobian,
            inequality_jacobian: &state.inequality_jacobian,
            slack: &slack,
            multipliers: &z,
            r_dual: &dual_residual,
            r_eq: &state.equality_values,
            r_ineq: &state.augmented_inequality_values,
            r_cent: &r_cent,
            solver: options.linear_solver,
            regularization: options.regularization,
        })?;
        last_linear_solver = direction.solver_used;
        let linear_elapsed = linear_started.elapsed();
        profiling.linear_solves += 1;
        profiling.linear_solve_time += linear_elapsed;
        iteration_linear_solve_time += linear_elapsed;

        let alpha_pr = if augmented_inequality_count > 0 {
            fraction_to_boundary(&slack, &direction.ds, options.fraction_to_boundary)
        } else {
            1.0
        };
        let alpha_du = if augmented_inequality_count > 0 {
            fraction_to_boundary(&z, &direction.dz, options.fraction_to_boundary)
        } else {
            1.0
        };
        let mut alpha = alpha_pr.min(alpha_du).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return Err(InteriorPointSolveError::LineSearchFailed {
                merit: merit_residual(
                    equality_inf,
                    inequality_inf,
                    dual_inf,
                    complementarity_inf,
                    mu,
                ),
                mu,
                step_inf_norm: step_inf_norm(&direction.dx),
            });
        }

        let current_merit = merit_residual(
            equality_inf,
            inequality_inf,
            dual_inf,
            complementarity_inf,
            mu,
        );
        let current_primal_inf = primal_inf;
        let current_barrier_objective =
            barrier_objective_value(state.objective_value, &slack, mu.max(options.mu_min));
        let mut line_search_iterations = 0;
        let mut accepted = None;
        let mut best_feasible = None;
        let mut best_feasible_merit = f64::INFINITY;
        while alpha >= options.min_step {
            let trial_x = x
                .iter()
                .zip(direction.dx.iter())
                .map(|(value, delta)| value + alpha * delta)
                .collect::<Vec<_>>();
            let trial_lambda = lambda_eq
                .iter()
                .zip(direction.d_lambda.iter())
                .map(|(value, delta)| value + alpha * delta)
                .collect::<Vec<_>>();
            let trial_slack = slack
                .iter()
                .zip(direction.ds.iter())
                .map(|(value, delta)| value + alpha * delta)
                .collect::<Vec<_>>();
            let trial_z = z
                .iter()
                .zip(direction.dz.iter())
                .map(|(value, delta)| value + alpha * delta)
                .collect::<Vec<_>>();
            if trial_slack.iter().any(|value| *value <= 0.0)
                || trial_z.iter().any(|value| *value <= 0.0)
            {
                alpha *= options.line_search_beta;
                line_search_iterations += 1;
                continue;
            }
            let mut trial_callback_time = Duration::ZERO;
            let trial_state = trial_state(
                problem,
                &trial_x,
                parameters,
                &bounds,
                &mut profiling,
                &mut trial_callback_time,
            );
            let trial_eq_inf = inf_norm(&trial_state.equality_values);
            let trial_ineq_inf = positive_part_inf_norm(&trial_state.augmented_inequality_values);
            let trial_dual_residual = lagrangian_gradient(
                &trial_state.gradient,
                &trial_state.equality_jacobian,
                &trial_lambda,
                &trial_state.inequality_jacobian,
                &trial_z,
            );
            let trial_dual_inf = inf_norm(&trial_dual_residual);
            let trial_comp_inf = if augmented_inequality_count > 0 {
                complementarity_inf_norm(&trial_slack, &trial_z)
            } else {
                0.0
            };
            let trial_mu = barrier_parameter(&trial_slack, &trial_z);
            let trial_merit = merit_residual(
                trial_eq_inf,
                trial_ineq_inf,
                trial_dual_inf,
                trial_comp_inf,
                trial_mu,
            );
            let trial_primal_inf = trial_eq_inf.max(trial_ineq_inf);
            let trial_barrier_objective = barrier_objective_value(
                trial_state.objective_value,
                &trial_slack,
                mu.max(options.mu_min),
            );
            if trial_merit < best_feasible_merit {
                best_feasible_merit = trial_merit;
                best_feasible = Some((
                    trial_x.clone(),
                    trial_lambda.clone(),
                    trial_slack.clone(),
                    trial_z.clone(),
                ));
            }
            let residual_accept = trial_merit
                <= (1.0 - options.line_search_c1 * alpha) * current_merit
                || trial_merit < current_merit;
            let filter_accept = trial_primal_inf
                <= (1.0 - options.line_search_c1 * alpha) * current_primal_inf
                || trial_barrier_objective
                    <= current_barrier_objective
                        - options.line_search_c1 * alpha * current_primal_inf.max(1.0);
            if residual_accept || filter_accept {
                accepted = Some((trial_x, trial_lambda, trial_slack, trial_z));
                break;
            }
            alpha *= options.line_search_beta;
            line_search_iterations += 1;
        }
        let accepted = accepted.or(best_feasible);
        let Some((trial_x, trial_lambda, trial_slack, trial_z)) = accepted else {
            return Err(InteriorPointSolveError::LineSearchFailed {
                merit: current_merit,
                mu,
                step_inf_norm: step_inf_norm(&direction.dx),
            });
        };

        if options.verbose {
            let solver_fell_back = options.linear_solver != InteriorPointLinearSolver::Auto
                && direction.solver_used != options.linear_solver;
            let flags = (if equality_count > 0 {
                SQP_LOG_HAS_EQUALITIES
            } else {
                0
            }) | (if augmented_inequality_count > 0 {
                SQP_LOG_HAS_INEQUALITIES
            } else {
                0
            }) | (if sigma_adjusted {
                SQP_LOG_PENALTY_UPDATED
            } else {
                0
            }) | (if solver_fell_back {
                IP_LOG_LINEAR_FALLBACK
            } else {
                0
            }) | (if iteration + 1 == options.max_iters {
                SQP_LOG_ITERATION_LIMIT_REACHED
            } else {
                0
            });
            log_interior_point_iteration(
                &InteriorPointIterationLog {
                    iteration,
                    flags,
                    objective_value: state.objective_value,
                    equality_inf,
                    inequality_inf,
                    dual_inf,
                    complementarity_inf,
                    barrier_parameter: if augmented_inequality_count > 0 {
                        mu.max(options.mu_min)
                    } else {
                        0.0
                    },
                    alpha: Some(alpha),
                    line_search_iterations: Some(line_search_iterations),
                    linear_time_secs: Some(
                        profiling.linear_solve_time.as_secs_f64()
                            / profiling.linear_solves.max(1) as f64,
                    ),
                    constraint_tol: options.constraint_tol,
                    dual_tol: options.dual_tol,
                    complementarity_tol: options.complementarity_tol,
                },
                &mut event_state,
            );
        }

        let solver_fell_back = options.linear_solver != InteriorPointLinearSolver::Auto
            && direction.solver_used != options.linear_solver;
        let mut events = Vec::new();
        if sigma_adjusted {
            events.push(InteriorPointIterationEvent::SigmaAdjusted);
        }
        if line_search_iterations >= 4 {
            events.push(InteriorPointIterationEvent::LongLineSearch);
        }
        if solver_fell_back {
            events.push(InteriorPointIterationEvent::LinearSolverFallback);
        }
        if iteration + 1 == options.max_iters {
            events.push(InteriorPointIterationEvent::MaxIterationsReached);
        }
        let adapter_timing = adapter_timing_delta(problem, &mut last_adapter_timing);
        profiling.adapter_timing = last_adapter_timing;
        let iteration_total = iteration_started.elapsed();
        let iteration_preprocess = iteration_total.saturating_sub(
            iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
        );
        callback(&InteriorPointIterationSnapshot {
            iteration,
            phase: InteriorPointIterationPhase::AcceptedStep,
            x: x.clone(),
            objective: state.objective_value,
            eq_inf: (equality_count > 0).then_some(equality_inf),
            ineq_inf: (augmented_inequality_count > 0).then_some(inequality_inf),
            dual_inf,
            comp_inf: (augmented_inequality_count > 0).then_some(complementarity_inf),
            barrier_parameter: (augmented_inequality_count > 0).then_some(mu.max(options.mu_min)),
            alpha: Some(alpha),
            line_search_iterations: Some(line_search_iterations),
            linear_solver: direction.solver_used,
            linear_solve_time: Some(iteration_linear_solve_time),
            timing: InteriorPointIterationTiming {
                adapter_timing,
                callback: iteration_callback_time,
                kkt_assembly: iteration_kkt_assembly_time,
                linear_solve: iteration_linear_solve_time,
                preprocess: iteration_preprocess,
                total: iteration_total,
            },
            events,
        });

        profiling.preprocessing_time += iteration_started.elapsed().saturating_sub(
            iteration_callback_time + iteration_kkt_assembly_time + iteration_linear_solve_time,
        );
        x = trial_x;
        lambda_eq = trial_lambda;
        slack = trial_slack;
        z = trial_z;
        let (nonlinear, _, _) =
            split_augmented_inequality_multipliers(&z, inequality_count, lower_bound_count);
        nonlinear_inequality_multipliers = nonlinear;
    }

    Err(InteriorPointSolveError::MaxIterations {
        iterations: options.max_iters,
    })
}
