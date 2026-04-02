#![cfg(feature = "ipopt")]

use approx::assert_abs_diff_eq;
use optimization::{IpoptOptions, ParameterMatrix, solve_nlp_ipopt, solve_nlp_ipopt_with_callback};
use rstest::rstest;

#[path = "support/generated_problem.rs"]
mod generated_problem;

use generated_problem::{
    CallbackBackend, CallbackNlpProblem, casadi_rosenbrock_nlp_problem,
    constrained_rosenbrock_problem, hanging_chain_initial_guess, hanging_chain_problem,
    hs021_problem, hs035_problem, hs071_problem, invalid_shape_problem,
    parameterized_quadratic_parameter_ccs, parameterized_quadratic_problem, simple_nlp_problem,
};

fn build_problem_ok(
    result: anyhow::Result<CallbackNlpProblem>,
    backend: CallbackBackend,
) -> CallbackNlpProblem {
    assert!(
        result.is_ok(),
        "problem construction failed for backend {}: {:?}",
        backend.label(),
        result.as_ref().err()
    );
    match result {
        Ok(problem) => problem,
        Err(err) => unreachable!("asserted success for backend {}: {err}", backend.label()),
    }
}

fn solve_ok<P: optimization::CompiledNlpProblem>(
    problem: &P,
    x0: &[f64],
    parameters: &[ParameterMatrix<'_>],
    options: IpoptOptions,
) -> optimization::IpoptSummary {
    let solve_result = solve_nlp_ipopt(problem, x0, parameters, &options);
    assert!(solve_result.is_ok(), "Ipopt solve failed: {solve_result:?}");
    match solve_result {
        Ok(summary) => summary,
        Err(err) => unreachable!("asserted success: {err}"),
    }
}

#[rstest]
fn ipopt_callback_exposes_snapshots(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let mut snapshots = Vec::new();
    let summary = solve_nlp_ipopt_with_callback(
        &problem,
        &[0.0, 0.0],
        &[],
        &IpoptOptions::default(),
        |snapshot| snapshots.push(snapshot.clone()),
    )
    .expect("Ipopt solve should succeed");

    assert!(!snapshots.is_empty());
    assert_eq!(
        snapshots.last().map(|snapshot| snapshot.iteration),
        Some(summary.iterations)
    );
}

#[cfg(feature = "serde")]
#[rstest]
fn ipopt_summary_serializes_with_duration_seconds(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.0, 0.0], &[], IpoptOptions::default());

    let json = serde_json::to_value(&summary).expect("summary should serialize");
    assert!(json["profiling"]["objective_value"]["total_time"].is_number());
    assert!(json["profiling"]["total_time"].is_number());

    let roundtrip: optimization::IpoptSummary =
        serde_json::from_value(json).expect("summary should deserialize");
    assert_eq!(roundtrip.status, summary.status);
}

#[test]
fn ipopt_rejects_invalid_shape_problem() {
    let solve_result = solve_nlp_ipopt(
        &invalid_shape_problem(),
        &[0.0, 0.0],
        &[],
        &IpoptOptions::default(),
    );
    assert!(solve_result.is_err());
}

#[rstest]
fn ipopt_solves_equality_constrained_rosenbrock(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(constrained_rosenbrock_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.5, 0.5], &[], IpoptOptions::default());

    assert_abs_diff_eq!(summary.x[0], 0.61879562, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], 0.38120438, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.objective, 0.14560702, epsilon = 1e-6);
    assert!(summary.equality_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
}

#[rstest]
fn ipopt_solves_casadi_rosenbrock_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(casadi_rosenbrock_nlp_problem(backend), backend);
    let summary = solve_ok(&problem, &[2.5, 3.0, 0.75], &[], IpoptOptions::default());

    assert_abs_diff_eq!(summary.x[0], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.x[2], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-7);
    assert!(summary.equality_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
}

#[rstest]
fn ipopt_solves_casadi_simple_nlp_example(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(simple_nlp_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.0, 0.0], &[], IpoptOptions::default());

    assert_abs_diff_eq!(summary.x[0], 5.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.x[1], 5.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.objective, 50.0, epsilon = 1e-7);
    assert!(summary.equality_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
}

#[rstest]
fn ipopt_solves_hs021(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs021_problem(backend), backend);
    let summary = solve_ok(&problem, &[2.0, 2.0], &[], IpoptOptions::default());

    assert_abs_diff_eq!(summary.x[0], 2.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.x[1], 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.objective, -99.96, epsilon = 1e-7);
    assert!(summary.primal_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
    assert!(summary.complementarity_inf_norm <= 1e-6);
}

#[rstest]
fn ipopt_solves_hs035(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs035_problem(backend), backend);
    let summary = solve_ok(&problem, &[0.5, 0.5, 0.5], &[], IpoptOptions::default());

    assert_abs_diff_eq!(summary.x[0], 4.0 / 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 7.0 / 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[2], 4.0 / 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 1.0 / 9.0, epsilon = 1e-7);
    assert!(summary.primal_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
    assert!(summary.complementarity_inf_norm <= 1e-6);
}

#[rstest]
fn ipopt_solves_hs071(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hs071_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &[1.0, 5.0, 5.0, 1.0],
        &[],
        IpoptOptions {
            max_iters: 120,
            ..IpoptOptions::default()
        },
    );

    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[1], 4.74299964, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[2], 3.82114998, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.x[3], 1.37940829, epsilon = 1e-4);
    assert_abs_diff_eq!(summary.objective, 17.0140173, epsilon = 1e-5);
    assert!(summary.primal_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.complementarity_inf_norm <= 1e-5);
}

#[rstest]
fn ipopt_solves_parameterized_problem(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(parameterized_quadratic_problem(backend), backend);
    let parameter_ccs = parameterized_quadratic_parameter_ccs();
    let parameter_values = [0.25, 0.75];
    let parameters = [ParameterMatrix {
        ccs: &parameter_ccs,
        values: &parameter_values,
    }];
    let summary = solve_ok(&problem, &[0.9, 0.1], &parameters, IpoptOptions::default());

    assert_abs_diff_eq!(summary.x[0], 0.25, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.x[1], 0.75, epsilon = 1e-7);
    assert_abs_diff_eq!(summary.objective, 0.0, epsilon = 1e-9);
}

#[rstest]
fn ipopt_solves_hanging_chain(
    #[values(CallbackBackend::Aot, CallbackBackend::Jit)] backend: CallbackBackend,
) {
    let problem = build_problem_ok(hanging_chain_problem(backend), backend);
    let summary = solve_ok(
        &problem,
        &hanging_chain_initial_guess(),
        &[],
        IpoptOptions {
            max_iters: 200,
            tol: 1e-9,
            ..IpoptOptions::default()
        },
    );

    assert!(summary.equality_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-5);
    assert!(summary.objective < -1.35);
    assert_abs_diff_eq!(summary.x[0] + summary.x[6], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[2] + summary.x[4], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], summary.x[7], epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[3], summary.x[5], epsilon = 1e-5);
}
