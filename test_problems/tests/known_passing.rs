use test_problems::{
    KnownStatus, ProblemSpeed, RunRequest, SolverKind, render_markdown_report, run_cases,
};

fn assert_known_passing_solver(solver: SolverKind, speed: Option<ProblemSpeed>) {
    let results_result = run_cases(&RunRequest {
        solvers: vec![solver],
        jobs: std::thread::available_parallelism().ok().map(usize::from),
        problem_set: speed,
        ..RunRequest::default()
    });
    assert!(
        results_result.is_ok(),
        "runner should succeed for {}: {results_result:?}",
        solver.label()
    );
    let results = match results_result {
        Ok(results) => results,
        Err(err) => unreachable!("asserted success for {}: {err}", solver.label()),
    };
    let failing = results
        .records
        .iter()
        .filter(|record| record.expected == KnownStatus::KnownPassing && !record.status.accepted())
        .collect::<Vec<_>>();
    assert!(
        failing.is_empty(),
        "known-passing regressions for {}:\n{}",
        solver.label(),
        render_markdown_report(&results),
    );
}

#[test]
fn known_passing_sqp_fast_cases_stay_green() {
    assert_known_passing_solver(SolverKind::Sqp, Some(ProblemSpeed::Fast));
}

#[test]
fn known_passing_interior_point_fast_cases_stay_green() {
    assert_known_passing_solver(SolverKind::InteriorPoint, Some(ProblemSpeed::Fast));
}

#[test]
#[ignore = "slow full-suite regression"]
fn known_passing_sqp_full_suite_stays_green() {
    assert_known_passing_solver(SolverKind::Sqp, None);
}

#[test]
#[ignore = "slow full-suite regression"]
fn known_passing_interior_point_full_suite_stays_green() {
    assert_known_passing_solver(SolverKind::InteriorPoint, None);
}
