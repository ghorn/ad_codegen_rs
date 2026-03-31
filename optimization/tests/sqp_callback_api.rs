use approx::assert_abs_diff_eq;
use optimization::{
    CCS, ClarabelSqpError, ClarabelSqpOptions, CompiledNlpProblem, NonFiniteCallbackStage,
    NonFiniteInputStage, ParameterMatrix, SqpFinalStateKind, SqpIterationEvent, SqpIterationPhase,
    SqpTermination, SymbolicNlpOutputs, solve_nlp_sqp, solve_nlp_sqp_with_callback, symbolic_nlp,
};
use rstest::rstest;
use std::sync::OnceLock;
use sx_core::SX;

#[derive(Clone, optimization::Vectorize)]
struct Pair<T> {
    x: T,
    y: T,
}

fn quiet_options() -> ClarabelSqpOptions {
    ClarabelSqpOptions {
        verbose: false,
        ..ClarabelSqpOptions::default()
    }
}

fn scalar_ccs() -> &'static CCS {
    static CCS_1X1: OnceLock<CCS> = OnceLock::new();
    CCS_1X1.get_or_init(|| CCS::new(1, 1, vec![0, 1], vec![0]))
}

fn empty_ccs_1d() -> &'static CCS {
    static EMPTY: OnceLock<CCS> = OnceLock::new();
    EMPTY.get_or_init(|| CCS::empty(0, 1))
}

fn unconstrained_rosenbrock_problem() -> optimization::TypedCompiledJitNlp<Pair<SX>, (), ()> {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), _>("telemetry_rosenbrock", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            constraints: (),
        })
        .expect("symbolic NLP should build");
    symbolic.compile_jit().expect("JIT compile should succeed")
}

struct OneDimQuadraticProblem;

impl CompiledNlpProblem for OneDimQuadraticProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("one-dimensional quadratic has no parameters")
    }

    fn equality_count(&self) -> usize {
        0
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] - 2.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = 2.0 * (x[0] - 2.0);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn equality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {}

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn inequality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = 2.0;
    }
}

struct EqualityQuadraticProblem;

impl CompiledNlpProblem for EqualityQuadraticProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("equality quadratic has no parameters")
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] - 1.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = 2.0 * (x[0] - 1.0);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0] - 3.0;
    }

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = 1.0;
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn inequality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = 2.0;
    }
}

struct InfeasibleLinearizedEqualityProblem;

impl CompiledNlpProblem for InfeasibleLinearizedEqualityProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("infeasible equality problem has no parameters")
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        x[0] * x[0]
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = 2.0 * x[0];
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn equality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = 1.0;
    }

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = 0.0;
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn inequality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = 2.0;
    }
}

struct RecoverableElasticEqualityProblem;

impl CompiledNlpProblem for RecoverableElasticEqualityProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("recoverable elastic equality problem has no parameters")
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        (x[0] - 1.0).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = 2.0 * (x[0] - 1.0);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = x[0] * x[0] - 1.0;
    }

    fn equality_jacobian_values(
        &self,
        x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = 2.0 * x[0];
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn inequality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        let lambda = equality_multipliers.first().copied().unwrap_or(0.0);
        out[0] = 2.0 + 2.0 * lambda;
    }
}

struct ScalarParameterizedProblem;

impl CompiledNlpProblem for ScalarParameterizedProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        1
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        scalar_ccs()
    }

    fn equality_count(&self) -> usize {
        0
    }

    fn inequality_count(&self) -> usize {
        0
    }

    fn objective_value(&self, x: &[f64], parameters: &[ParameterMatrix<'_>]) -> f64 {
        let shift = parameters[0].values[0];
        (x[0] - shift).powi(2)
    }

    fn objective_gradient(&self, x: &[f64], parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        let shift = parameters[0].values[0];
        out[0] = 2.0 * (x[0] - shift);
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn equality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {}

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        empty_ccs_1d()
    }

    fn inequality_values(&self, _x: &[f64], _parameters: &[ParameterMatrix<'_>], _out: &mut [f64]) {
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _out: &mut [f64],
    ) {
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = 2.0;
    }
}

#[derive(Clone, Copy, Debug)]
struct NonFiniteProblem {
    stage: NonFiniteCallbackStage,
}

impl CompiledNlpProblem for NonFiniteProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameter_ccs(&self, _parameter_index: usize) -> &CCS {
        unreachable!("non-finite problem has no parameters")
    }

    fn equality_count(&self) -> usize {
        1
    }

    fn inequality_count(&self) -> usize {
        1
    }

    fn objective_value(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>]) -> f64 {
        if self.stage == NonFiniteCallbackStage::ObjectiveValue {
            f64::NAN
        } else {
            (x[0] - 1.0).powi(2)
        }
    }

    fn objective_gradient(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = if self.stage == NonFiniteCallbackStage::ObjectiveGradient {
            f64::INFINITY
        } else {
            2.0 * (x[0] - 1.0)
        };
    }

    fn equality_jacobian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn equality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = if self.stage == NonFiniteCallbackStage::EqualityValues {
            f64::NAN
        } else {
            x[0] - 1.0
        };
    }

    fn equality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = if self.stage == NonFiniteCallbackStage::EqualityJacobianValues {
            f64::INFINITY
        } else {
            1.0
        };
    }

    fn inequality_jacobian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn inequality_values(&self, x: &[f64], _parameters: &[ParameterMatrix<'_>], out: &mut [f64]) {
        out[0] = if self.stage == NonFiniteCallbackStage::InequalityValues {
            f64::NAN
        } else {
            -x[0]
        };
    }

    fn inequality_jacobian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        out: &mut [f64],
    ) {
        out[0] = if self.stage == NonFiniteCallbackStage::InequalityJacobianValues {
            f64::NEG_INFINITY
        } else {
            -1.0
        };
    }

    fn lagrangian_hessian_ccs(&self) -> &CCS {
        scalar_ccs()
    }

    fn lagrangian_hessian_values(
        &self,
        _x: &[f64],
        _parameters: &[ParameterMatrix<'_>],
        _equality_multipliers: &[f64],
        _inequality_multipliers: &[f64],
        out: &mut [f64],
    ) {
        out[0] = if self.stage == NonFiniteCallbackStage::LagrangianHessianValues {
            f64::NAN
        } else {
            2.0
        };
    }
}

#[test]
fn sqp_callback_initial_snapshot_is_iteration_zero_without_step_or_qp() {
    let problem = OneDimQuadraticProblem;
    let mut snapshots = Vec::new();
    let summary =
        solve_nlp_sqp_with_callback(&problem, &[0.0], &[], &quiet_options(), |snapshot| {
            snapshots.push(snapshot.clone());
        })
        .expect("solve should succeed");

    assert!(!snapshots.is_empty());
    assert_eq!(snapshots[0].iteration, 0);
    assert_eq!(snapshots[0].phase, SqpIterationPhase::Initial);
    assert_eq!(snapshots[0].eq_inf, None);
    assert_eq!(snapshots[0].ineq_inf, None);
    assert_eq!(snapshots[0].comp_inf, None);
    assert_eq!(snapshots[0].step_inf, None);
    assert_eq!(snapshots[0].line_search, None);
    assert_eq!(snapshots[0].qp, None);
    assert_eq!(
        summary.final_state.iteration,
        snapshots.last().expect("snapshot").iteration
    );
}

#[test]
fn sqp_callback_one_step_exact_solve_on_unconstrained_1d_quadratic() {
    let problem = OneDimQuadraticProblem;
    let mut snapshots = Vec::new();
    let summary =
        solve_nlp_sqp_with_callback(&problem, &[0.0], &[], &quiet_options(), |snapshot| {
            snapshots.push(snapshot.clone());
        })
        .expect("solve should succeed");

    assert_abs_diff_eq!(summary.x[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-12);
    assert_eq!(
        summary.final_state.phase,
        SqpIterationPhase::PostConvergence
    );
    assert_eq!(summary.final_state_kind, SqpFinalStateKind::AcceptedIterate);
    assert_eq!(summary.equality_inf_norm, None);
    assert_eq!(summary.inequality_inf_norm, None);
    assert_eq!(summary.complementarity_inf_norm, None);
    assert!(snapshots.len() >= 2);
    let final_snapshot = snapshots.last().expect("final snapshot should exist");
    assert_eq!(final_snapshot.phase, SqpIterationPhase::PostConvergence);
    let line_search = final_snapshot
        .line_search
        .as_ref()
        .expect("post-convergence snapshot should carry previous line-search info");
    assert_abs_diff_eq!(line_search.accepted_alpha, 1.0, epsilon = 1e-15);
    assert_eq!(line_search.backtrack_count, 0);
}

#[test]
fn sqp_callback_constrained_1d_quadratic_reports_only_available_metrics() {
    let problem = EqualityQuadraticProblem;
    let mut snapshots = Vec::new();
    let summary =
        solve_nlp_sqp_with_callback(&problem, &[0.0], &[], &quiet_options(), |snapshot| {
            snapshots.push(snapshot.clone());
        })
        .expect("solve should succeed");

    assert_abs_diff_eq!(summary.x[0], 3.0, epsilon = 1e-12);
    assert!(
        summary
            .equality_inf_norm
            .is_some_and(|value| value <= 1e-12)
    );
    assert_eq!(summary.inequality_inf_norm, None);
    assert_eq!(summary.complementarity_inf_norm, None);
    assert!(snapshots.iter().all(|snapshot| snapshot.eq_inf.is_some()));
    assert!(snapshots.iter().all(|snapshot| snapshot.ineq_inf.is_none()));
    assert!(snapshots.iter().all(|snapshot| snapshot.comp_inf.is_none()));
}

#[test]
fn sqp_callback_rosenbrock_reports_line_search_telemetry() {
    let compiled = unconstrained_rosenbrock_problem();
    let problem = compiled
        .bind_runtime_bounds(&optimization::TypedRuntimeNlpBounds::default())
        .expect("runtime bounds should validate");
    let mut snapshots = Vec::new();
    let options = ClarabelSqpOptions {
        verbose: false,
        max_iters: 80,
        dual_tol: 1e-7,
        ..ClarabelSqpOptions::default()
    };
    let summary = solve_nlp_sqp_with_callback(&problem, &[-1.2, 1.0], &[], &options, |snapshot| {
        snapshots.push(snapshot.clone());
    })
    .expect("solve should succeed");

    assert!(summary.objective <= 1e-10);
    assert!(snapshots.iter().any(|snapshot| {
        snapshot
            .line_search
            .as_ref()
            .is_some_and(|line_search| line_search.backtrack_count > 0)
    }));
    for snapshot in &snapshots {
        if snapshot
            .line_search
            .as_ref()
            .is_some_and(|line_search| line_search.backtrack_count >= 4)
        {
            assert!(snapshot.events.contains(&SqpIterationEvent::LongLineSearch));
        }
    }
}

#[test]
fn sqp_callback_exposes_post_convergence_snapshot() {
    let compiled = unconstrained_rosenbrock_problem();
    let problem = compiled
        .bind_runtime_bounds(&optimization::TypedRuntimeNlpBounds::default())
        .expect("runtime bounds should validate");
    let mut snapshots = Vec::new();
    let options = ClarabelSqpOptions {
        verbose: false,
        max_iters: 80,
        dual_tol: 1e-7,
        ..ClarabelSqpOptions::default()
    };
    let summary = solve_nlp_sqp_with_callback(&problem, &[-1.2, 1.0], &[], &options, |snapshot| {
        snapshots.push(snapshot.clone());
    })
    .expect("solve should succeed");

    let final_snapshot = snapshots.last().expect("final snapshot should exist");
    assert_eq!(final_snapshot.phase, SqpIterationPhase::PostConvergence);
    assert_eq!(
        summary.final_state.phase,
        SqpIterationPhase::PostConvergence
    );
    assert_eq!(summary.final_state.iteration, final_snapshot.iteration);
}

#[test]
fn sqp_qp_failure_diagnostics_surface_on_infeasible_case() {
    let problem = InfeasibleLinearizedEqualityProblem;
    let mut snapshots = Vec::new();
    let options = ClarabelSqpOptions {
        elastic_mode: false,
        ..quiet_options()
    };
    let error = solve_nlp_sqp_with_callback(&problem, &[0.0], &[], &options, |snapshot| {
        snapshots.push(snapshot.clone());
    })
    .expect_err("infeasible linearized equality should fail the QP");

    match error {
        ClarabelSqpError::QpSolve { context, .. } => {
            assert_eq!(context.termination, SqpTermination::QpSolve);
            assert!(context.final_state.is_some());
            assert_eq!(
                context.final_state_kind,
                Some(SqpFinalStateKind::InitialPoint)
            );
            assert!(context.last_accepted_state.is_none());
            let final_state = context.final_state.expect("final state should be captured");
            assert_eq!(final_state.phase, SqpIterationPhase::Initial);
            assert_eq!(final_state.eq_inf, Some(1.0));
            assert_eq!(final_state.ineq_inf, None);
            assert_eq!(final_state.comp_inf, None);
        }
        other => panic!("expected QpSolve error, got {other:?}"),
    }
    assert_eq!(snapshots.len(), 1);
}

#[test]
fn sqp_elastic_recovery_handles_primal_infeasible_linearization() {
    let problem = RecoverableElasticEqualityProblem;
    let mut snapshots = Vec::new();
    let summary = solve_nlp_sqp_with_callback(
        &problem,
        &[0.0],
        &[],
        &ClarabelSqpOptions {
            verbose: false,
            max_iters: 20,
            elastic_mode: true,
            elastic_weight: 100.0,
            ..ClarabelSqpOptions::default()
        },
        |snapshot| {
            snapshots.push(snapshot.clone());
        },
    )
    .expect("elastic recovery should rescue the infeasible linearization");

    assert_abs_diff_eq!(summary.x[0].abs(), 1.0, epsilon = 1e-6);
    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-6));
    assert!(summary.dual_inf_norm <= 1e-6);
    assert_eq!(summary.profiling.elastic_recovery_activations, 1);
    assert!(summary.profiling.elastic_recovery_qp_solves >= 1);
    assert!(snapshots.iter().any(|snapshot| {
        snapshot
            .events
            .contains(&SqpIterationEvent::ElasticRecoveryUsed)
    }));
}

#[test]
fn sqp_rejects_non_finite_initial_guess() {
    let error = solve_nlp_sqp(&OneDimQuadraticProblem, &[f64::NAN], &[], &quiet_options())
        .expect_err("NaN initial guess should be rejected");
    match error {
        ClarabelSqpError::NonFiniteInput { stage } => {
            assert_eq!(stage, NonFiniteInputStage::InitialGuess);
        }
        other => panic!("expected NonFiniteInput error, got {other:?}"),
    }
}

#[test]
fn sqp_rejects_non_finite_parameters() {
    let parameter_values = [f64::INFINITY];
    let parameters = [ParameterMatrix {
        ccs: scalar_ccs(),
        values: &parameter_values,
    }];
    let error = solve_nlp_sqp(
        &ScalarParameterizedProblem,
        &[0.0],
        &parameters,
        &quiet_options(),
    )
    .expect_err("non-finite parameters should be rejected");
    match error {
        ClarabelSqpError::NonFiniteInput { stage } => {
            assert_eq!(
                stage,
                NonFiniteInputStage::ParameterValues { parameter_index: 0 }
            );
        }
        other => panic!("expected NonFiniteInput error, got {other:?}"),
    }
}

#[rstest]
#[case(NonFiniteCallbackStage::ObjectiveValue)]
#[case(NonFiniteCallbackStage::ObjectiveGradient)]
#[case(NonFiniteCallbackStage::EqualityValues)]
#[case(NonFiniteCallbackStage::InequalityValues)]
#[case(NonFiniteCallbackStage::EqualityJacobianValues)]
#[case(NonFiniteCallbackStage::InequalityJacobianValues)]
#[case(NonFiniteCallbackStage::LagrangianHessianValues)]
fn sqp_rejects_non_finite_callback_outputs(#[case] stage: NonFiniteCallbackStage) {
    let problem = NonFiniteProblem { stage };
    let error = solve_nlp_sqp(&problem, &[0.0], &[], &quiet_options())
        .expect_err("non-finite callback output should be rejected");

    match error {
        ClarabelSqpError::NonFiniteCallbackOutput {
            stage: actual,
            context,
        } => {
            assert_eq!(actual, stage);
            assert_eq!(context.termination, SqpTermination::NonFiniteCallbackOutput);
            if matches!(stage, NonFiniteCallbackStage::LagrangianHessianValues) {
                assert!(context.final_state.is_some());
                assert!(context.last_accepted_state.is_none());
            }
        }
        other => panic!("expected NonFiniteCallbackOutput error, got {other:?}"),
    }
}
