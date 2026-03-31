use anyhow::{Result, anyhow, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;

use crate::manifest::{KnownStatus, ProblemSpeed, manifest_entry_by_id};
use crate::model::{
    JitOptLevel, ProblemCase, ProblemDescriptor, ProblemRunOptions, ProblemRunRecord, RunStatus,
    SolverKind, ValidationOutcome, ValidationTier,
};
use crate::registry::registry;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunRequest {
    pub problem_ids: Option<Vec<String>>,
    pub solvers: Vec<SolverKind>,
    pub run_options: Vec<ProblemRunOptions>,
    pub jobs: Option<usize>,
    pub include_skipped: bool,
    pub problem_set: Option<ProblemSpeed>,
    pub progress: bool,
}

impl Default for RunRequest {
    fn default() -> Self {
        Self {
            problem_ids: None,
            solvers: vec![SolverKind::Sqp, SolverKind::InteriorPoint],
            run_options: vec![ProblemRunOptions {
                jit_opt_level: JitOptLevel::O3,
            }],
            jobs: None,
            include_skipped: false,
            problem_set: None,
            progress: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunResults {
    pub records: Vec<ProblemRunRecord>,
}

impl RunResults {
    pub fn total(&self) -> usize {
        self.records.len()
    }
}

pub fn run_cases(request: &RunRequest) -> Result<RunResults> {
    if request.solvers.is_empty() {
        bail!("at least one solver must be selected");
    }
    if request.run_options.is_empty() {
        bail!("at least one run option must be selected");
    }

    let cases = registry()?;
    let selected_cases = select_cases(&cases, request.problem_ids.as_ref())?;
    let tasks = selected_cases
        .iter()
        .flat_map(|case| {
            request.solvers.iter().flat_map(move |solver| {
                request.run_options.iter().map(move |run_options| {
                    let manifest = manifest_entry_by_id(case.id)
                        .ok_or_else(|| anyhow!("missing manifest entry for {}", case.id))?;
                    let expected = match solver {
                        SolverKind::Sqp => manifest.sqp,
                        SolverKind::InteriorPoint => manifest.interior_point,
                        #[cfg(feature = "ipopt")]
                        SolverKind::Ipopt => manifest.ipopt,
                    };
                    if request.problem_set.is_some_and(|set| manifest.speed != set) {
                        return Ok(None);
                    }
                    let max_iters_limit = match solver {
                        SolverKind::Sqp => manifest.max_iters.sqp,
                        SolverKind::InteriorPoint => manifest.max_iters.interior_point,
                        #[cfg(feature = "ipopt")]
                        SolverKind::Ipopt => manifest.max_iters.ipopt,
                    };
                    Ok(Some((
                        case,
                        *solver,
                        *run_options,
                        expected,
                        max_iters_limit,
                    )))
                })
            })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let progress = request.progress.then(|| make_progress_bar(tasks.len()));
    let execute = || {
        tasks
            .par_iter()
            .filter_map(|(case, solver, run_options, expected, max_iters_limit)| {
                if matches!(expected, KnownStatus::Skipped) && !request.include_skipped {
                    return None;
                }
                let record = match expected {
                    KnownStatus::Skipped => {
                        skipped_record(case, *solver, *run_options, *expected, *max_iters_limit)
                    }
                    _ => case.run(*solver, *run_options, *max_iters_limit, *expected),
                };
                if let Some(progress) = &progress {
                    progress.inc(1);
                    if record.status.failed() {
                        progress.println(format!(
                            "fail  {:<28}  {:<3}  {:<2}  {}",
                            record.id,
                            solver.label(),
                            run_options.label(),
                            failure_brief(&record),
                        ));
                    }
                }
                Some(record)
            })
            .collect::<Vec<_>>()
    };

    let mut records = if let Some(jobs) = request.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()?
            .install(execute)
    } else {
        execute()
    };
    records.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.solver.label().cmp(right.solver.label()))
            .then_with(|| left.options.label().cmp(right.options.label()))
    });

    if let Some(progress) = progress {
        progress.finish_with_message(format!(
            "{} runs complete ({} passed, {} not passed)",
            records.len(),
            records
                .iter()
                .filter(|record| record.status.accepted())
                .count(),
            records
                .iter()
                .filter(|record| !record.status.accepted())
                .count(),
        ));
    }

    Ok(RunResults { records })
}

fn select_cases<'a>(
    cases: &'a [ProblemCase],
    requested_ids: Option<&Vec<String>>,
) -> Result<Vec<&'a ProblemCase>> {
    let Some(requested_ids) = requested_ids else {
        return Ok(cases.iter().collect());
    };
    let mut selected = Vec::with_capacity(requested_ids.len());
    for requested_id in requested_ids {
        let case = cases
            .iter()
            .find(|case| case.id == requested_id)
            .ok_or_else(|| anyhow!("unknown problem id: {requested_id}"))?;
        selected.push(case);
    }
    Ok(selected)
}

fn skipped_record(
    case: &ProblemCase,
    solver: SolverKind,
    options: ProblemRunOptions,
    expected: KnownStatus,
    max_iters_limit: usize,
) -> ProblemRunRecord {
    ProblemRunRecord {
        id: case.id.to_string(),
        solver,
        options,
        expected,
        max_iters_limit,
        status: RunStatus::Skipped,
        descriptor: ProblemDescriptor {
            id: case.id.to_string(),
            family: case.family.to_string(),
            variant: case.variant.to_string(),
            source: case.source.to_string(),
            description: case.description.to_string(),
            parameterized: case.parameterized,
            num_vars: 0,
            num_eq: 0,
            num_ineq: 0,
            num_box: 0,
            dof: 0,
            constrained: false,
        },
        solution: None,
        metrics: crate::model::SolverMetrics::default(),
        timing: crate::model::SolverTimingBreakdown::default(),
        validation: ValidationOutcome {
            tier: ValidationTier::Passed,
            tolerance: "skipped".to_string(),
            detail: "skipped by manifest".to_string(),
        },
        solver_thresholds: None,
        error: None,
        console_output: None,
        console_output_path: None,
    }
}

fn make_progress_bar(total: usize) -> ProgressBar {
    let progress = ProgressBar::new(total as u64);
    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>3}/{len:<3} {msg}",
    )
    .expect("valid progress template")
    .progress_chars("=> ");
    progress.set_style(style);
    progress.set_message("running");
    progress
}

fn failure_brief(record: &ProblemRunRecord) -> String {
    if let Some(error) = &record.error {
        if error.contains("failed to converge") {
            "max_iters".to_string()
        } else if error.contains("line search failed") {
            "line_search".to_string()
        } else if error.contains("PrimalInfeasible") {
            "primal_infeasible".to_string()
        } else {
            error.clone()
        }
    } else {
        record.validation.detail.clone()
    }
}
