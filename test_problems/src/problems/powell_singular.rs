use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, ProblemCase, VecN, make_typed_case, objective_validation, symbolic_compile,
};

pub(crate) fn case() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), (), _, _>(
        CaseMetadata::new(
            "powell_singular_4",
            "powell_singular",
            "n=4",
            "manual",
            "Powell singular least-squares objective",
            false,
        ),
        |jit_opt_level| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), (), _>(
                "powell_singular_4",
                |x, ()| {
                    let f1 = x.values[0] + 10.0 * x.values[1];
                    let f2 = 5.0_f64.sqrt() * (x.values[2] - x.values[3]);
                    let f3 = (x.values[1] - 2.0 * x.values[2]).sqr();
                    let f4 = 10.0_f64.sqrt() * (x.values[0] - x.values[3]).sqr();
                    SymbolicNlpOutputs {
                        objective: f1.sqr() + f2.sqr() + f3.sqr() + f4.sqr(),
                        constraints: (),
                    }
                },
                jit_opt_level,
            )?;
            Ok(super::TypedProblemData {
                compiled,
                x0: VecN {
                    values: [3.0, -1.0, 0.0, 1.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(0.0, 1e-8, 1e-6, 1e-5, None),
    )
}
