use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, ProblemCase, VecN, exact_solution_validation, make_typed_case, symbolic_compile,
};

pub(crate) fn cases() -> Vec<ProblemCase> {
    vec![
        rosenbrock_case::<2>("rosenbrock_2", "n=2"),
        rosenbrock_case::<4>("generalized_rosenbrock_4", "n=4"),
        rosenbrock_case::<8>("generalized_rosenbrock_8", "n=8"),
        rosenbrock_case::<16>("generalized_rosenbrock_16", "n=16"),
    ]
}

fn rosenbrock_case<const N: usize>(id: &'static str, variant: &'static str) -> ProblemCase {
    make_typed_case::<VecN<SX, N>, (), (), (), _, _>(
        CaseMetadata::new(
            id,
            "generalized_rosenbrock",
            variant,
            "manual",
            "Generalized Rosenbrock objective with alternating standard initial guess",
            true,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, N>, (), (), (), _>(
                id,
                |x, ()| {
                    let objective = x.values.windows(2).fold(SX::zero(), |acc, pair| {
                        acc + (1.0 - pair[0]).sqr() + 100.0 * (pair[1] - pair[0].sqr()).sqr()
                    });
                    SymbolicNlpOutputs {
                        objective,
                        equalities: (),
                        inequalities: (),
                    }
                },
                jit_opt_level,
            )?;
            let x0 = VecN {
                values: std::array::from_fn(|idx| if idx % 2 == 0 { -1.2 } else { 1.0 }),
            };
            Ok(super::TypedProblemData {
                compiled,
                x0,
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(&vec![1.0; N], 1e-4, 0.0, 1e-9, 1e-9, 1e-9, None),
    )
}
