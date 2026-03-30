use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use approx::assert_abs_diff_eq;
use proptest::prelude::*;
use rstest::rstest;
use serde::Deserialize;
use sx_core::{CCS, HessianStrategy, NamedMatrix, SX, SXFunction, SXMatrix};

#[path = "../../test_support/symbolic_eval.rs"]
mod symbolic_eval;

use symbolic_eval::eval;

fn assert_matrix_values_match(lhs: &SXMatrix, rhs: &SXMatrix, vars: &HashMap<u32, f64>) {
    assert_eq!(lhs.shape(), rhs.shape());
    assert_eq!(lhs.ccs(), rhs.ccs());
    let (nrow, ncol) = lhs.shape();
    for col in 0..ncol {
        for row in 0..nrow {
            assert_abs_diff_eq!(
                eval(lhs.get(row, col), vars),
                eval(rhs.get(row, col), vars),
                epsilon = 1e-9
            );
        }
    }
}

fn assert_unary_matches(
    expr: SX,
    x: SX,
    point: f64,
    reference: impl Fn(f64) -> f64,
    reference_derivative: impl Fn(f64) -> f64,
) {
    let x_vec = SXMatrix::dense_column(vec![x]).unwrap();
    let f = SXMatrix::scalar(expr);
    let gradient = f.gradient(&x_vec).unwrap();
    let vars = HashMap::from([(x.id(), point)]);
    assert_abs_diff_eq!(eval(f.nz(0), &vars), reference(point), epsilon = 1e-10);
    assert_abs_diff_eq!(
        eval(gradient.nz(0), &vars),
        reference_derivative(point),
        epsilon = 1e-9
    );
}

fn assert_binary_matches(
    expr: SX,
    x: SX,
    y: SX,
    point: (f64, f64),
    reference: impl Fn(f64, f64) -> (f64, f64, f64),
) {
    let vars_vec = SXMatrix::dense_column(vec![x, y]).unwrap();
    let f = SXMatrix::scalar(expr);
    let gradient = f.gradient(&vars_vec).unwrap();
    let (point_x, point_y) = point;
    let vars = HashMap::from([(x.id(), point_x), (y.id(), point_y)]);
    let (value, dx, dy) = reference(point_x, point_y);
    assert_abs_diff_eq!(eval(f.nz(0), &vars), value, epsilon = 1e-10);
    assert_abs_diff_eq!(eval(gradient.nz(0), &vars), dx, epsilon = 1e-8);
    assert_abs_diff_eq!(eval(gradient.nz(1), &vars), dy, epsilon = 1e-8);
}

#[rstest]
fn sx_cse_and_constant_folding() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    assert_eq!(x + SX::zero(), x);
    assert_eq!(x * SX::one(), x);
    assert_eq!(x + y, y + x);
    assert_eq!((x + y).id(), (y + x).id());
}

#[rstest]
fn ccs_transpose_roundtrip() {
    let ccs = CCS::from_positions(3, 2, &[(0, 0), (2, 0), (1, 1)]).unwrap();
    assert_eq!(ccs.transpose().transpose(), ccs);
}

#[rstest]
fn gradient_matches_manual_rosenbrock() {
    let x0 = SX::sym("x0");
    let x1 = SX::sym("x1");
    let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
    let f = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let grad = f.gradient(&x).unwrap();

    let values = HashMap::from([(x0.id(), 1.2), (x1.id(), 1.5)]);
    let g0 = eval(grad.nz(0), &values);
    let g1 = eval(grad.nz(1), &values);
    assert_abs_diff_eq!(g0, -28.4, epsilon = 1e-9);
    assert_abs_diff_eq!(g1, 12.0, epsilon = 1e-9);
}

#[rstest]
fn extended_unary_ops_match_numeric_reference() {
    let x = SX::sym("ops_x");
    assert_unary_matches(x.abs(), x, 1.3, |v| v.abs(), |v| v.signum());
    assert_unary_matches(x.sign(), x, 1.3, |v| v.signum(), |_| 0.0);
    assert_unary_matches(x.floor(), x, 1.3, |v| v.floor(), |_| 0.0);
    assert_unary_matches(x.ceil(), x, 1.3, |v| v.ceil(), |_| 0.0);
    assert_unary_matches(x.round(), x, 1.3, |v| v.round(), |_| 0.0);
    assert_unary_matches(x.trunc(), x, 1.3, |v| v.trunc(), |_| 0.0);
    assert_unary_matches(x.sqrt(), x, 1.7, |v| v.sqrt(), |v| 0.5 / v.sqrt());
    assert_unary_matches(x.exp(), x, 0.3, |v| v.exp(), |v| v.exp());
    assert_unary_matches(
        x.exp2(),
        x,
        0.3,
        |v| v.exp2(),
        |v| v.exp2() * std::f64::consts::LN_2,
    );
    assert_unary_matches(
        x.exp10(),
        x,
        0.3,
        |v| 10.0_f64.powf(v),
        |v| 10.0_f64.powf(v) * std::f64::consts::LN_10,
    );
    assert_unary_matches(x.log(), x, 1.7, |v| v.ln(), |v| 1.0 / v);
    assert_unary_matches(
        x.log2(),
        x,
        1.7,
        |v| v.log2(),
        |v| 1.0 / (v * std::f64::consts::LN_2),
    );
    assert_unary_matches(
        x.log10(),
        x,
        1.7,
        |v| v.log10(),
        |v| 1.0 / (v * std::f64::consts::LN_10),
    );
    assert_unary_matches(x.sin(), x, 0.3, |v| v.sin(), |v| v.cos());
    assert_unary_matches(x.cos(), x, 0.3, |v| v.cos(), |v| -v.sin());
    assert_unary_matches(x.tan(), x, 0.3, |v| v.tan(), |v| 1.0 / v.cos().powi(2));
    assert_unary_matches(
        x.asin(),
        x,
        0.3,
        |v| v.asin(),
        |v| 1.0 / (1.0 - v * v).sqrt(),
    );
    assert_unary_matches(
        x.acos(),
        x,
        0.3,
        |v| v.acos(),
        |v| -1.0 / (1.0 - v * v).sqrt(),
    );
    assert_unary_matches(x.atan(), x, 0.3, |v| v.atan(), |v| 1.0 / (1.0 + v * v));
    assert_unary_matches(x.sinh(), x, 0.3, |v| v.sinh(), |v| v.cosh());
    assert_unary_matches(x.cosh(), x, 0.3, |v| v.cosh(), |v| v.sinh());
    assert_unary_matches(x.tanh(), x, 0.3, |v| v.tanh(), |v| 1.0 / v.cosh().powi(2));
    assert_unary_matches(
        x.asinh(),
        x,
        0.3,
        |v| v.asinh(),
        |v| 1.0 / (v * v + 1.0).sqrt(),
    );
    assert_unary_matches(
        x.acosh(),
        x,
        1.7,
        |v| v.acosh(),
        |v| 1.0 / ((v - 1.0).sqrt() * (v + 1.0).sqrt()),
    );
    assert_unary_matches(x.atanh(), x, 0.3, |v| v.atanh(), |v| 1.0 / (1.0 - v * v));
    assert_unary_matches(
        x.log_base(3.0),
        x,
        1.7,
        |v| v.ln() / 3.0_f64.ln(),
        |v| 1.0 / (v * 3.0_f64.ln()),
    );
}

#[rstest]
fn extended_binary_ops_match_numeric_reference() {
    let x = SX::sym("ops_bx");
    let y = SX::sym("ops_by");
    assert_binary_matches(x.pow(y), x, y, (1.7, 0.8), |a, b| {
        (a.powf(b), b * a.powf(b - 1.0), a.powf(b) * a.ln())
    });
    assert_binary_matches(x.atan2(y), x, y, (1.7, 0.8), |a, b| {
        (a.atan2(b), b / (a * a + b * b), -a / (a * a + b * b))
    });
    assert_binary_matches(x.hypot(y), x, y, (1.7, 0.8), |a, b| {
        (a.hypot(b), a / a.hypot(b), b / a.hypot(b))
    });
    assert_binary_matches(x % y, x, y, (1.7, 0.8), |a, b| {
        (a % b, 1.0, -(a / b).trunc())
    });
    assert_binary_matches(x.max(y), x, y, (1.7, 0.8), |a, b| {
        (
            a.max(b),
            if a > b { 1.0 } else { 0.0 },
            if a > b { 0.0 } else { 1.0 },
        )
    });
    assert_binary_matches(x.min(y), x, y, (1.7, 0.8), |a, b| {
        (
            a.min(b),
            if a < b { 1.0 } else { 0.0 },
            if a < b { 0.0 } else { 1.0 },
        )
    });
    assert_unary_matches(x.powi(3), x, 1.7, |v| v.powi(3), |v| 3.0 * v.powi(2));
    assert_unary_matches(x.powf(1.5), x, 1.7, |v| v.powf(1.5), |v| 1.5 * v.powf(0.5));
}

#[rstest]
fn jacobian_and_hessian_shapes() {
    let x0 = SX::sym("x0");
    let x1 = SX::sym("x1");
    let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
    let f = SXMatrix::dense_column(vec![x0 + x1, x0 * x1]).unwrap();

    let jac = f.jacobian(&x).unwrap();
    assert_eq!(jac.ccs().nrow(), 2);
    assert_eq!(jac.ccs().ncol(), 2);

    let rosen = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let hess = rosen.hessian(&x).unwrap();
    assert_eq!(hess.ccs().nrow(), 2);
    assert_eq!(hess.ccs().ncol(), 2);
    assert!(hess.ccs().positions().iter().all(|(row, col)| row >= col));
}

#[rstest]
fn hessian_strategies_match_on_rosenbrock() {
    let x0 = SX::sym("hx0");
    let x1 = SX::sym("hx1");
    let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
    let rosen = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let values = HashMap::from([(x0.id(), 1.2), (x1.id(), 1.5)]);

    let reference = rosen
        .hessian_with_strategy(&x, HessianStrategy::LowerTriangleByColumn)
        .unwrap();
    for strategy in HessianStrategy::ALL {
        let candidate = rosen.hessian_with_strategy(&x, strategy).unwrap();
        assert_matrix_values_match(&reference, &candidate, &values);
    }

    let default_hessian = rosen.hessian(&x).unwrap();
    let selected_outputs_hessian = rosen
        .hessian_with_strategy(&x, HessianStrategy::LowerTriangleSelectedOutputs)
        .unwrap();
    assert_eq!(default_hessian, selected_outputs_hessian);
}

#[rstest]
fn sx_function_validates_free_symbols() {
    let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
    let y = SX::sym("y");
    let output = SXMatrix::scalar(y);
    let err = SXFunction::new(
        "bad",
        vec![NamedMatrix::new("x", x).unwrap()],
        vec![NamedMatrix::new("out", output).unwrap()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("free symbol"));
}

#[derive(Debug, Deserialize)]
struct ParityManifest {
    casadi_commit: String,
    entries: Vec<ParityEntry>,
}

#[derive(Debug, Deserialize)]
struct ParityEntry {
    casadi_case: String,
    rust_test: Option<String>,
    status: String,
}

#[rstest]
fn parity_manifest_is_pinned_and_supported_cases_have_tests() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../parity/casadi_v1.toml");
    let manifest: ParityManifest =
        toml::from_str(&fs::read_to_string(manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest.casadi_commit,
        "838d585b15f3de6f2a1e8d058c91e445a01d43e3"
    );
    assert!(
        manifest
            .entries
            .iter()
            .inspect(|entry| assert!(!entry.casadi_case.trim().is_empty()))
            .inspect(|entry| {
                assert!(matches!(
                    entry.status.as_str(),
                    "supported" | "unsupported" | "intentionally_descoped"
                ));
            })
            .filter(|entry| entry.status == "supported")
            .all(|entry| entry.rust_test.is_some())
    );
}

prop_compose! {
    fn expr_coeffs()(a in -3.0f64..3.0, b in -3.0f64..3.0, c in -3.0f64..3.0, d in 0.5f64..3.0) -> (f64, f64, f64, f64) {
        (a, b, c, d)
    }
}

prop_compose! {
    fn smooth_unary_samples()(
        trig in -0.8f64..0.8,
        positive in 0.2f64..3.0,
        acosh_point in 1.1f64..3.0,
    ) -> (f64, f64, f64) {
        (trig, positive, acosh_point)
    }
}

prop_compose! {
    fn smooth_binary_samples()(
        base in 0.3f64..3.0,
        exponent in -1.5f64..2.0,
        y in 0.3f64..2.5,
    ) -> (f64, f64, f64) {
        (base, exponent, y)
    }
}

prop_compose! {
    fn hessian_expr_coeffs()(
        a in -2.0f64..2.0,
        b in -2.0f64..2.0,
        c in -2.0f64..2.0,
        d in 0.5f64..3.0,
        e in -2.0f64..2.0,
        f in -2.0f64..2.0,
    ) -> (f64, f64, f64, f64, f64, f64) {
        (a, b, c, d, e, f)
    }
}

proptest! {
    #[test]
    fn forward_reverse_directionals_match_finite_difference((a, b, c, d) in expr_coeffs(), x0v in -1.5f64..1.5, x1v in -1.5f64..1.5, s0 in -1.0f64..1.0, s1 in -1.0f64..1.0) {
        let x0 = SX::sym("px0");
        let x1 = SX::sym("px1");
        let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
        let expr = SXMatrix::scalar(((a * x0 + b * x1) * (c * x0 - x1)) / d);
        let seed = SXMatrix::dense_column(vec![SX::from(s0), SX::from(s1)]).unwrap();

        let forward = expr.forward(&x, &seed).unwrap().scalar_expr().unwrap();
        let reverse = expr.reverse(&x, &SXMatrix::scalar(1.0)).unwrap();

        let vars = HashMap::from([(x0.id(), x0v), (x1.id(), x1v)]);
        let eps = 1e-6;
        let base = eval(expr.scalar_expr().unwrap(), &vars);
        let bumped = {
            let vars_eps = HashMap::from([(x0.id(), x0v + eps * s0), (x1.id(), x1v + eps * s1)]);
            eval(expr.scalar_expr().unwrap(), &vars_eps)
        };
        let fd = (bumped - base) / eps;
        prop_assert!((eval(forward, &vars) - fd).abs() < 1e-3);
        prop_assert_eq!(reverse.nnz(), 2);
    }

    #[test]
    fn hessian_strategies_match_on_random_nonlinear_scalar(
        (a, b, c, d, e, f) in hessian_expr_coeffs(),
        x0v in -1.5f64..1.5,
        x1v in -1.5f64..1.5,
        x2v in -1.5f64..1.5,
    ) {
        let x0 = SX::sym("phx0");
        let x1 = SX::sym("phx1");
        let x2 = SX::sym("phx2");
        let x = SXMatrix::dense_column(vec![x0, x1, x2]).unwrap();
        let expr = SXMatrix::scalar(
            ((a * x0 + b * x1) * (c * x1 - x2)) / d
                + e * x0 * x2
                + f * x1.sqr()
                + x0 * x1 * x2
                + x0.sqr() * x2,
        );
        let vars = HashMap::from([(x0.id(), x0v), (x1.id(), x1v), (x2.id(), x2v)]);

        let reference = expr
            .hessian_with_strategy(&x, HessianStrategy::LowerTriangleByColumn)
            .unwrap();
        for strategy in HessianStrategy::ALL {
            let candidate = expr.hessian_with_strategy(&x, strategy).unwrap();
            prop_assert_eq!(candidate.ccs(), reference.ccs());
            let (nrow, ncol) = candidate.shape();
            for col in 0..ncol {
                for row in 0..nrow {
                    prop_assert!(
                        (eval(candidate.get(row, col), &vars) - eval(reference.get(row, col), &vars)).abs() < 1e-9
                    );
                }
            }
        }
    }

    #[test]
    fn smooth_unary_operator_gradients_match_finite_difference(
        (trig, positive, acosh_point) in smooth_unary_samples(),
    ) {
        let x = SX::sym("fd_unary");
        let wrt = SXMatrix::dense_column(vec![x]).unwrap();
        let cases: [(&str, SX, f64); 16] = [
            ("sin", x.sin(), trig),
            ("cos", x.cos(), trig),
            ("tan", x.tan(), trig),
            ("exp", x.exp(), trig),
            ("log", x.log(), positive),
            ("sqrt", x.sqrt(), positive),
            ("asin", x.asin(), trig),
            ("acos", x.acos(), trig),
            ("atan", x.atan(), trig),
            ("sinh", x.sinh(), trig),
            ("cosh", x.cosh(), trig),
            ("tanh", x.tanh(), trig),
            ("asinh", x.asinh(), trig),
            ("acosh", x.acosh(), acosh_point),
            ("atanh", x.atanh(), trig),
            ("log_base", x.log_base(3.0), positive),
        ];

        let eps = 1e-6;
        for (label, expr, point) in cases {
            let f = SXMatrix::scalar(expr);
            let grad = f.gradient(&wrt).unwrap();
            let vars = HashMap::from([(x.id(), point)]);
            let vars_eps = HashMap::from([(x.id(), point + eps)]);
            let fd = (eval(expr, &vars_eps) - eval(expr, &vars)) / eps;
            let ad = eval(grad.nz(0), &vars);
            prop_assert!(
                (ad - fd).abs() < 1e-4,
                "{label} derivative mismatch: ad={ad}, fd={fd}, point={point}",
            );
        }
    }

    #[test]
    fn smooth_binary_operator_gradients_match_finite_difference(
        (base, exponent, y) in smooth_binary_samples(),
    ) {
        let x = SX::sym("fd_bin_x");
        let z = SX::sym("fd_bin_z");
        let wrt = SXMatrix::dense_column(vec![x, z]).unwrap();
        let cases: [(&str, SX, (f64, f64)); 3] = [
            ("pow", x.pow(z), (base, exponent)),
            ("atan2", x.atan2(z), (base, y)),
            ("hypot", x.hypot(z), (base, y)),
        ];

        let eps = 1e-6;
        for (label, expr, (xv, zv)) in cases {
            let f = SXMatrix::scalar(expr);
            let grad = f.gradient(&wrt).unwrap();
            let vars = HashMap::from([(x.id(), xv), (z.id(), zv)]);
            let vars_x = HashMap::from([(x.id(), xv + eps), (z.id(), zv)]);
            let vars_z = HashMap::from([(x.id(), xv), (z.id(), zv + eps)]);
            let fd_x = (eval(expr, &vars_x) - eval(expr, &vars)) / eps;
            let fd_z = (eval(expr, &vars_z) - eval(expr, &vars)) / eps;
            let ad_x = eval(grad.nz(0), &vars);
            let ad_z = eval(grad.nz(1), &vars);
            prop_assert!(
                (ad_x - fd_x).abs() < 1e-4,
                "{label} d/dx mismatch: ad={ad_x}, fd={fd_x}, point=({xv}, {zv})",
            );
            prop_assert!(
                (ad_z - fd_z).abs() < 1e-4,
                "{label} d/dz mismatch: ad={ad_z}, fd={fd_z}, point=({xv}, {zv})",
            );
        }
    }
}
