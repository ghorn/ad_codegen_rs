use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, ProblemCase, VecN, make_typed_case, objective_validation, symbolic_compile,
};

pub(crate) fn cases() -> Vec<ProblemCase> {
    vec![
        case_for::<4>("brown_almost_linear_4", "n=4"),
        case_for::<8>("brown_almost_linear_8", "n=8"),
        case_for::<16>("brown_almost_linear_16", "n=16"),
    ]
}

fn case_for<const N: usize>(id: &'static str, variant: &'static str) -> ProblemCase {
    make_typed_case::<VecN<SX, N>, (), (), (), _, _>(
        CaseMetadata::new(
            id,
            "brown_almost_linear",
            variant,
            "manual",
            "Brown almost linear least-squares objective",
            true,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), (), (), _>(
                id,
                |x, ()| {
                    let sum = x.values.iter().fold(SX::zero(), |acc, value| acc + *value);
                    let product = x
                        .values
                        .iter()
                        .fold(SX::from(1.0), |acc, value| acc * *value);
                    let residuals: [SX; N] = std::array::from_fn(|idx| {
                        if idx + 1 == N {
                            product - 1.0
                        } else {
                            x.values[idx] + sum - (N + 1) as f64
                        }
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
                    values: std::array::from_fn(|_| 0.5),
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(0.0, 1e-8, 1e-6, 1e-5, None),
    )
}
