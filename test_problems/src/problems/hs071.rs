use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use crate::model::{ProblemRunRecord, ValidationOutcome, ValidationTier};

use super::{CaseMetadata, ProblemCase, VecN, make_typed_case, symbolic_compile};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), SX, SX, _, _>(
        CaseMetadata::new(
            "hs071",
            "hock_schittkowski",
            "hs071",
            "manual",
            "HS071 with runtime box bounds, one equality, and one nonlinear inequality",
            false,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), SX, SX, _>(
                "hs071",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: x0 * x3 * (x0 + x1 + x2) + x2,
                        equalities: x0.sqr() + x1.sqr() + x2.sqr() + x3.sqr() - 40.0,
                        inequalities: 25.0 - x0 * x1 * x2 * x3,
                    }
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: VecN {
                    values: [1.0, 5.0, 5.0, 1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(VecN {
                        values: [1.0, 1.0, 1.0, 1.0],
                    }),
                    variable_upper: Some(VecN {
                        values: [5.0, 5.0, 5.0, 5.0],
                    }),
                    inequality_lower: Some(-f64::INFINITY),
                    inequality_upper: Some(0.0),
                },
            })
        },
        validate_hs071,
    )
}

fn validate_hs071(record: &ProblemRunRecord) -> ValidationOutcome {
    let Some(solution) = record.solution.as_ref() else {
        return ValidationOutcome {
            tier: ValidationTier::Failed,
            tolerance:
                "x<=5e-3, obj<=5e-3, primal<=1e-5, dual<=1e-4, comp<=1e-5; reduced: primal<=1e-6, dual<=1e-6, comp<=1e-6"
                    .to_string(),
            detail: "missing solution".to_string(),
        };
    };
    let expected = [1.0, 4.742_999_64, 3.821_149_98, 1.379_408_29];
    let objective = record.metrics.objective.unwrap_or(f64::INFINITY);
    let primal = record.metrics.primal_inf.unwrap_or(f64::INFINITY);
    let dual = record.metrics.dual_inf.unwrap_or(f64::INFINITY);
    let comp = record.metrics.complementarity_inf.unwrap_or(f64::INFINITY);
    let x_ok = solution
        .iter()
        .zip(expected.iter())
        .all(|(actual, expected)| (actual - expected).abs() <= 5e-3);
    let passed = x_ok
        && (objective - 17.014_017_3).abs() <= 5e-3
        && primal <= 1e-5
        && dual <= 1e-4
        && comp <= 1e-5;
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
            "x<=5e-3, obj<=5e-3, primal<=1e-5, dual<=1e-4, comp<=1e-5; reduced: primal<=1e-6, dual<=1e-6, comp<=1e-6"
                .to_string(),
        detail: format!(
            "objective={objective:.6e}, primal={primal:.3e}, dual={dual:.3e}, comp={comp:.3e}"
        ),
    }
}
