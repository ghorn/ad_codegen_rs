use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, ProblemCase, VecN, make_typed_case, objective_validation, symbolic_compile,
};

pub(crate) fn cases() -> Vec<ProblemCase> {
    vec![
        case_for::<4>("trigonometric_4", "n=4"),
        case_for::<8>("trigonometric_8", "n=8"),
        case_for::<16>("trigonometric_16", "n=16"),
    ]
}

fn case_for<const N: usize>(id: &'static str, variant: &'static str) -> ProblemCase {
    make_typed_case::<VecN<SX, N>, (), (), (), _, _>(
        CaseMetadata::new(
            id,
            "trigonometric",
            variant,
            "manual",
            "Trigonometric least-squares objective",
            true,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), (), (), _>(
                id,
                |x, ()| {
                    let cos_sum = x
                        .values
                        .iter()
                        .fold(SX::zero(), |acc, value| acc + value.cos());
                    let residuals: [SX; N] = std::array::from_fn(|idx| {
                        let scale = (idx + 1) as f64;
                        N as f64 - cos_sum + scale * (1.0 - x.values[idx].cos())
                            - x.values[idx].sin()
                    });
                    let objective = residuals
                        .into_iter()
                        .fold(SX::zero(), |acc, residual| acc + residual.sqr());
                    SymbolicNlpOutputs {
                        objective,
                        equalities: (),
                        inequalities: (),
                    }
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: VecN {
                    values: std::array::from_fn(|_| 1.0 / N as f64),
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(0.0, 1e-8, 1e-6, 1e-5, None),
    )
}
