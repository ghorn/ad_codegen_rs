use approx::assert_abs_diff_eq;
use optimization::{
    ClarabelSqpOptions, InteriorPointOptions, SymbolicNlpOutputs, TypedRuntimeNlpBounds, flat_view,
    symbolic_nlp,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptOptions, IpoptRawStatus};
use sx_core::SX;

#[derive(Clone, optimization::Vectorize)]
struct Pair<T> {
    x: T,
    y: T,
}

#[derive(Clone, optimization::Vectorize)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Clone, optimization::Vectorize)]
struct Chain<T, const N: usize> {
    points: [Point<T>; N],
}

#[test]
fn typed_symbolic_rosenbrock_solves_end_to_end_with_jit() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let timing = compiled.backend_timing_metadata();
    let summary = compiled
        .solve_sqp(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &ClarabelSqpOptions {
                max_iters: 80,
                dual_tol: 1e-7,
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert!(timing.function_creation_time.is_some());
    assert!(timing.derivative_generation_time.is_some());
    assert!(timing.jit_time.is_some());
    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-6);
    assert!(summary.objective <= 1e-10);
    assert_eq!(summary.equality_inf_norm, None);
    assert_eq!(summary.inequality_inf_norm, None);
    assert_eq!(summary.complementarity_inf_norm, None);

    let final_view: PairView<'_, f64> =
        flat_view::<Pair<f64>, f64>(&summary.x).expect("solver output should project into a view");
    assert_abs_diff_eq!(*final_view.x, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(*final_view.y, 1.0, epsilon = 1e-6);
    let snapshot_view: PairView<'_, f64> = flat_view::<Pair<f64>, f64>(&summary.final_state.x)
        .expect("final snapshot should project into a view");
    assert_abs_diff_eq!(*snapshot_view.x, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(*snapshot_view.y, 1.0, epsilon = 1e-6);
}

#[test]
fn typed_symbolic_disk_constrained_rosenbrock_solves_with_runtime_constraint_bounds() {
    let symbolic = symbolic_nlp::<Pair<SX>, (), (), Pair<SX>, _>("disk_rosenbrock", |x, _| {
        SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: Pair {
                x: x.x.sqr() + x.y.sqr(),
                y: x.y,
            },
        }
    })
    .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: Some(Pair {
                    x: -f64::INFINITY,
                    y: -f64::INFINITY,
                }),
                inequality_upper: Some(Pair { x: 1.5, y: 2.0 }),
            },
            &ClarabelSqpOptions {
                max_iters: 80,
                merit_penalty: 25.0,
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert!(summary.primal_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-6);
    assert!(
        summary
            .complementarity_inf_norm
            .is_some_and(|value| value <= 1e-6)
    );
    assert!(summary.objective < 10.0);
    assert!(summary.x[0].powi(2) + summary.x[1].powi(2) <= 1.5 + 1e-5);
}

#[test]
fn typed_symbolic_hanging_chain_solves_end_to_end() {
    const N: usize = 4;
    let span = 3.0;
    let link_length = 0.75;
    let symbolic = symbolic_nlp::<Chain<SX, N>, (), [SX; N + 1], (), _>("hanging_chain", |q, _| {
        let objective = q.points.iter().fold(SX::zero(), |acc, point| acc + point.y);
        let mut constraints = std::array::from_fn(|_| SX::zero());
        let mut prev_x = SX::from(0.0);
        let mut prev_y = SX::from(0.0);
        let link_length_sq = link_length * link_length;
        for (index, point) in q.points.iter().enumerate() {
            constraints[index] =
                (point.x - prev_x).sqr() + (point.y - prev_y).sqr() - link_length_sq;
            prev_x = point.x;
            prev_y = point.y;
        }
        constraints[N] = (prev_x - span).sqr() + prev_y.sqr() - link_length_sq;
        SymbolicNlpOutputs {
            objective,
            equalities: constraints,
            inequalities: (),
        }
    })
    .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &Chain {
                points: [
                    Point { x: 0.75, y: 0.0 },
                    Point {
                        x: 1.125,
                        y: -0.649_519_052_8,
                    },
                    Point {
                        x: 1.875,
                        y: -0.649_519_052_8,
                    },
                    Point { x: 2.25, y: 0.0 },
                ],
            },
            &(),
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: None,
                inequality_upper: None,
            },
            &ClarabelSqpOptions {
                max_iters: 120,
                merit_penalty: 50.0,
                dual_tol: 1e-5,
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-6));
    assert_eq!(summary.inequality_inf_norm, None);
    assert_eq!(summary.complementarity_inf_norm, None);
    assert!(summary.objective < -1.35);
    assert_abs_diff_eq!(summary.x[0] + summary.x[6], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[2] + summary.x[4], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], summary.x[7], epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[3], summary.x[5], epsilon = 1e-5);
}

#[test]
fn typed_symbolic_parameterized_nlp_solves_end_to_end() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, Pair<SX>, SX, (), _>("parameterized_quadratic", |x, p| {
            SymbolicNlpOutputs {
                objective: (x.x - p.x).sqr() + (x.y - p.y).sqr(),
                equalities: x.x + x.y,
                inequalities: (),
            }
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &Pair { x: 0.9, y: 0.1 },
            &Pair { x: 0.25, y: 0.75 },
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: None,
                inequality_upper: None,
            },
            &ClarabelSqpOptions {
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert_abs_diff_eq!(summary.x[0], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 0.5, epsilon = 1e-9);
    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-8));
}

#[test]
fn typed_symbolic_compile_exposes_timing_metadata() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("timed_rosenbrock", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let timing = compiled.backend_timing_metadata();

    assert!(timing.function_creation_time.is_some());
    assert!(timing.derivative_generation_time.is_some());
    assert!(timing.jit_time.is_some());
}

#[cfg(feature = "ipopt")]
#[test]
fn typed_symbolic_rosenbrock_solves_with_ipopt_without_box_bounds() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock_ipopt", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_ipopt(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &IpoptOptions {
                max_iters: 120,
                tol: 1e-9,
                ..IpoptOptions::default()
            },
        )
        .expect("Ipopt solve should succeed");

    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-6);
    assert!(summary.objective <= 1e-10);
    assert!(summary.dual_inf_norm <= 1e-6);
}

#[cfg(feature = "ipopt")]
#[test]
fn typed_symbolic_inequality_only_problem_solves_with_ipopt_without_box_bounds() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), Pair<SX>, _>("disk_rosenbrock_ipopt", |x, _| {
            SymbolicNlpOutputs {
                objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
                equalities: (),
                inequalities: Pair {
                    x: x.x.sqr() + x.y.sqr(),
                    y: x.y,
                },
            }
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_ipopt(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: Some(Pair {
                    x: -f64::INFINITY,
                    y: -f64::INFINITY,
                }),
                inequality_upper: Some(Pair { x: 1.5, y: 2.0 }),
            },
            &IpoptOptions {
                max_iters: 200,
                tol: 1e-9,
                ..IpoptOptions::default()
            },
        )
        .expect("Ipopt solve should succeed");

    assert!(summary.primal_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
    assert!(summary.complementarity_inf_norm <= 1e-6);
    assert!(summary.objective < 10.0);
    assert!(summary.x[0].powi(2) + summary.x[1].powi(2) <= 1.5 + 1e-5);
}

#[test]
fn typed_symbolic_problem_reports_adapter_timing_with_interior_point() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock_ip", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let mut snapshots = Vec::new();
    let summary = compiled
        .solve_interior_point_with_callback(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &InteriorPointOptions {
                max_iters: 120,
                dual_tol: 1e-6,
                verbose: false,
                ..InteriorPointOptions::default()
            },
            |snapshot| snapshots.push(snapshot.clone()),
        )
        .expect("interior-point solve should succeed");

    assert!(summary.profiling.adapter_timing.is_some());
    assert!(
        snapshots
            .iter()
            .any(|snapshot| snapshot.timing.adapter_timing.is_some())
    );
}

#[cfg(feature = "ipopt")]
#[test]
fn typed_symbolic_problem_reports_adapter_timing_with_ipopt() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock_ipopt_cb", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let mut snapshots = Vec::new();
    let summary = compiled
        .solve_ipopt_with_callback(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &IpoptOptions {
                max_iters: 120,
                tol: 1e-9,
                ..IpoptOptions::default()
            },
            |snapshot| snapshots.push(snapshot.clone()),
        )
        .expect("Ipopt solve should succeed");

    assert!(matches!(
        summary.status,
        IpoptRawStatus::SolveSucceeded
            | IpoptRawStatus::SolvedToAcceptableLevel
            | IpoptRawStatus::FeasiblePointFound
    ));
    assert!(summary.profiling.adapter_timing.is_some());
    assert!(
        snapshots
            .iter()
            .any(|snapshot| snapshot.timing.adapter_timing.is_some())
    );
}
