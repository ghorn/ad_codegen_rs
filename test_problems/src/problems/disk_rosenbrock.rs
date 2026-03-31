use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use crate::model::{ProblemRunRecord, ValidationOutcome, ValidationTier};

use super::{CaseMetadata, Pair, ProblemCase, make_typed_case, symbolic_compile};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, _, _>(
        CaseMetadata::new(
            "disk_rosenbrock",
            "disk_rosenbrock",
            "disk<=1.5, y<=2",
            "manual",
            "Rosenbrock with runtime nonlinear disk constraint and variable upper bound",
            false,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, _>(
                "disk_rosenbrock",
                |x, ()| SymbolicNlpOutputs {
                    objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
                    constraints: x.x.sqr() + x.y.sqr(),
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: Pair { x: -1.2, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: Some(Pair {
                        x: f64::INFINITY,
                        y: 2.0,
                    }),
                    constraint_lower: Some(-f64::INFINITY),
                    constraint_upper: Some(1.5),
                },
            })
        },
        validate_disk_rosenbrock,
    )
}

fn validate_disk_rosenbrock(record: &ProblemRunRecord) -> ValidationOutcome {
    let Some(solution) = record.solution.as_ref() else {
        return ValidationOutcome {
            tier: ValidationTier::Failed,
            tolerance:
                "objective<=2e-1, primal<=1e-6, dual<=1e-5, comp<=1e-5; reduced: primal<=1e-6, dual<=1e-6, comp<=1e-6"
                    .to_string(),
            detail: "missing solution".to_string(),
        };
    };
    let x = solution[0];
    let y = solution[1];
    let objective = record.metrics.objective.unwrap_or(f64::INFINITY);
    let primal = record.metrics.primal_inf.unwrap_or(f64::INFINITY);
    let dual = record.metrics.dual_inf.unwrap_or(f64::INFINITY);
    let comp = record.metrics.complementarity_inf.unwrap_or(f64::INFINITY);
    let passed = objective <= 2e-1
        && primal <= 1e-6
        && dual <= 1e-5
        && comp <= 1e-5
        && x * x + y * y <= 1.5 + 1e-6;
    let tier = if passed {
        ValidationTier::Passed
    } else if record.error.is_none() && primal <= 1e-6 && dual <= 1e-6 && comp <= 1e-6 {
        ValidationTier::ReducedAccuracy
    } else {
        ValidationTier::Failed
    };
    ValidationOutcome {
        tier,
        tolerance:
            "objective<=2e-1, primal<=1e-6, dual<=1e-5, comp<=1e-5; reduced: primal<=1e-6, dual<=1e-6, comp<=1e-6"
                .to_string(),
        detail: format!(
            "objective={objective:.6e}, primal={primal:.3e}, dual={dual:.3e}, comp={comp:.3e}, disk={:.6e}",
            x * x + y * y
        ),
    }
}
