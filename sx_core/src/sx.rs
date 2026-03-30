use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::error::{Result, SxError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SX(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Trunc,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,
    Hypot,
    Mod,
    Copysign,
}

#[derive(Clone, Debug)]
enum NodeKind {
    Constant(f64),
    Symbol { serial: usize, name: String },
    Unary { op: UnaryOp, arg: SX },
    Binary { op: BinaryOp, lhs: SX, rhs: SX },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NodeKey {
    Constant(u64),
    Unary { op: UnaryOp, arg: SX },
    Binary { op: BinaryOp, lhs: SX, rhs: SX },
}

#[derive(Clone, Debug)]
struct Node {
    kind: NodeKind,
}

#[derive(Default)]
struct Interner {
    nodes: Vec<Node>,
    keyed: HashMap<NodeKey, SX>,
    next_symbol_serial: usize,
}

static INTERNER: OnceLock<Mutex<Interner>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq)]
pub enum NodeView {
    Constant(f64),
    Symbol { name: String, serial: usize },
    Unary { op: UnaryOp, arg: SX },
    Binary { op: BinaryOp, lhs: SX, rhs: SX },
}

impl UnaryOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Sign => "sign",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Trunc => "trunc",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
        }
    }

    fn apply_constant(self, arg: f64) -> f64 {
        match self {
            Self::Abs => arg.abs(),
            Self::Sign => {
                if arg > 0.0 {
                    1.0
                } else if arg < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => arg.floor(),
            Self::Ceil => arg.ceil(),
            Self::Round => arg.round(),
            Self::Trunc => arg.trunc(),
            Self::Sqrt => arg.sqrt(),
            Self::Exp => arg.exp(),
            Self::Log => arg.ln(),
            Self::Sin => arg.sin(),
            Self::Cos => arg.cos(),
            Self::Tan => arg.tan(),
            Self::Asin => arg.asin(),
            Self::Acos => arg.acos(),
            Self::Atan => arg.atan(),
            Self::Sinh => arg.sinh(),
            Self::Cosh => arg.cosh(),
            Self::Tanh => arg.tanh(),
            Self::Asinh => arg.asinh(),
            Self::Acosh => arg.acosh(),
            Self::Atanh => arg.atanh(),
        }
    }
}

impl BinaryOp {
    pub fn symbol(self) -> Option<&'static str> {
        match self {
            Self::Add => Some("+"),
            Self::Sub => Some("-"),
            Self::Mul => Some("*"),
            Self::Div => Some("/"),
            Self::Mod => Some("%"),
            Self::Pow | Self::Atan2 | Self::Hypot | Self::Copysign => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::Atan2 => "atan2",
            Self::Hypot => "hypot",
            Self::Mod => "mod",
            Self::Copysign => "copysign",
        }
    }

    fn apply_constant(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
            Self::Pow => lhs.powf(rhs),
            Self::Atan2 => lhs.atan2(rhs),
            Self::Hypot => lhs.hypot(rhs),
            Self::Mod => lhs % rhs,
            Self::Copysign => lhs.copysign(rhs),
        }
    }

    fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Hypot)
    }
}

fn with_interner<R>(f: impl FnOnce(&mut Interner) -> R) -> R {
    let mutex = INTERNER.get_or_init(|| Mutex::new(Interner::default()));
    let mut guard = lock_interner(mutex);
    f(&mut guard)
}

fn with_interner_ref<R>(f: impl FnOnce(&Interner) -> R) -> R {
    let mutex = INTERNER.get_or_init(|| Mutex::new(Interner::default()));
    let guard = lock_interner(mutex);
    f(&guard)
}

fn lock_interner(mutex: &Mutex<Interner>) -> MutexGuard<'_, Interner> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Interner {
    fn node(&self, sx: SX) -> &Node {
        &self.nodes[sx.0 as usize]
    }

    fn node_kind(&self, sx: SX) -> NodeKind {
        self.node(sx).kind.clone()
    }

    fn fresh_symbol(&mut self, name: impl Into<String>) -> SX {
        let serial = self.next_symbol_serial;
        self.next_symbol_serial += 1;
        let id = SX(self.nodes.len() as u32);
        self.nodes.push(Node {
            kind: NodeKind::Symbol {
                serial,
                name: name.into(),
            },
        });
        id
    }

    fn intern_keyed(&mut self, key: NodeKey, kind: NodeKind) -> SX {
        if let Some(existing) = self.keyed.get(&key) {
            return *existing;
        }
        let id = SX(self.nodes.len() as u32);
        self.nodes.push(Node { kind });
        self.keyed.insert(key, id);
        id
    }
}

fn node_kind(sx: SX) -> NodeKind {
    with_interner_ref(|interner| interner.node_kind(sx))
}

fn constant_value(sx: SX) -> Option<f64> {
    match node_kind(sx) {
        NodeKind::Constant(value) => Some(value),
        NodeKind::Symbol { .. } | NodeKind::Unary { .. } | NodeKind::Binary { .. } => None,
    }
}

fn mul_constant_factor(sx: SX) -> Option<(f64, SX)> {
    match node_kind(sx) {
        NodeKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } => {
            if let Some(value) = constant_value(lhs) {
                Some((value, rhs))
            } else {
                constant_value(rhs).map(|value| (value, lhs))
            }
        }
        NodeKind::Constant(_)
        | NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. } => None,
    }
}

#[derive(Clone, Debug)]
struct RationalFactors {
    coeff: f64,
    numerators: Vec<SX>,
    denominators: Vec<SX>,
}

fn intern_binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKey as K;
    use NodeKind as N;

    let (lhs, rhs) = if op.is_commutative() {
        canonical_pair(lhs, rhs)
    } else {
        (lhs, rhs)
    };
    with_interner(|interner| {
        interner.intern_keyed(K::Binary { op, lhs, rhs }, N::Binary { op, lhs, rhs })
    })
}

fn canonicalize_rational_factors(mut factors: RationalFactors) -> RationalFactors {
    if factors.coeff == 0.0 {
        factors.numerators.clear();
        factors.denominators.clear();
        return factors;
    }

    factors.numerators.sort_unstable();
    factors.denominators.sort_unstable();

    let mut numerators = Vec::with_capacity(factors.numerators.len());
    let mut denominators = Vec::with_capacity(factors.denominators.len());
    let mut numerator_idx = 0;
    let mut denominator_idx = 0;

    while numerator_idx < factors.numerators.len() && denominator_idx < factors.denominators.len() {
        let numerator = factors.numerators[numerator_idx];
        let denominator = factors.denominators[denominator_idx];
        match numerator.cmp(&denominator) {
            std::cmp::Ordering::Equal => {
                numerator_idx += 1;
                denominator_idx += 1;
            }
            std::cmp::Ordering::Less => {
                numerators.push(numerator);
                numerator_idx += 1;
            }
            std::cmp::Ordering::Greater => {
                denominators.push(denominator);
                denominator_idx += 1;
            }
        }
    }

    numerators.extend_from_slice(&factors.numerators[numerator_idx..]);
    denominators.extend_from_slice(&factors.denominators[denominator_idx..]);

    RationalFactors {
        coeff: factors.coeff,
        numerators,
        denominators,
    }
}

fn combine_rational_factors(lhs: RationalFactors, rhs: RationalFactors) -> RationalFactors {
    let mut numerators = lhs.numerators;
    numerators.extend(rhs.numerators);
    let mut denominators = lhs.denominators;
    denominators.extend(rhs.denominators);
    canonicalize_rational_factors(RationalFactors {
        coeff: lhs.coeff * rhs.coeff,
        numerators,
        denominators,
    })
}

fn divide_rational_factors(lhs: RationalFactors, rhs: RationalFactors) -> Option<RationalFactors> {
    if rhs.coeff == 0.0 {
        return None;
    }

    let mut numerators = lhs.numerators;
    numerators.extend(rhs.denominators);
    let mut denominators = lhs.denominators;
    denominators.extend(rhs.numerators);
    Some(canonicalize_rational_factors(RationalFactors {
        coeff: lhs.coeff / rhs.coeff,
        numerators,
        denominators,
    }))
}

fn rational_factors(expr: SX) -> RationalFactors {
    match node_kind(expr) {
        NodeKind::Constant(value) => canonicalize_rational_factors(RationalFactors {
            coeff: value,
            numerators: Vec::new(),
            denominators: Vec::new(),
        }),
        NodeKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } => combine_rational_factors(rational_factors(lhs), rational_factors(rhs)),
        NodeKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
        } => divide_rational_factors(rational_factors(lhs), rational_factors(rhs)).unwrap_or_else(
            || RationalFactors {
                coeff: 1.0,
                numerators: vec![expr],
                denominators: Vec::new(),
            },
        ),
        NodeKind::Symbol { .. } | NodeKind::Unary { .. } | NodeKind::Binary { .. } => {
            RationalFactors {
                coeff: 1.0,
                numerators: vec![expr],
                denominators: Vec::new(),
            }
        }
    }
}

fn rebuild_rational_factors(factors: RationalFactors) -> SX {
    if factors.coeff == 0.0 {
        return SX::zero();
    }

    let mut numerators = factors.numerators.into_iter();
    let mut expr = match numerators.next() {
        Some(first) if factors.coeff == 1.0 => first,
        Some(first) => intern_binary(BinaryOp::Mul, SX::from(factors.coeff), first),
        None => SX::from(factors.coeff),
    };

    for numerator in numerators {
        expr = intern_binary(BinaryOp::Mul, expr, numerator);
    }

    for denominator in factors.denominators {
        expr = intern_binary(BinaryOp::Div, expr, denominator);
    }

    expr
}

fn combine_like_terms(lhs: SX, rhs: SX, rhs_sign: f64) -> Option<SX> {
    let lhs_factors = rational_factors(lhs);
    let mut rhs_factors = rational_factors(rhs);
    rhs_factors.coeff *= rhs_sign;
    rhs_factors = canonicalize_rational_factors(rhs_factors);

    if lhs_factors.numerators != rhs_factors.numerators
        || lhs_factors.denominators != rhs_factors.denominators
    {
        return None;
    }

    Some(rebuild_rational_factors(canonicalize_rational_factors(
        RationalFactors {
            coeff: lhs_factors.coeff + rhs_factors.coeff,
            numerators: lhs_factors.numerators,
            denominators: lhs_factors.denominators,
        },
    )))
}

fn format_expression(expr: SX, interner: &Interner, memo: &mut HashMap<SX, String>) -> String {
    if let Some(existing) = memo.get(&expr) {
        return existing.clone();
    }
    let formatted = match interner.node(expr).kind.clone() {
        NodeKind::Constant(v) => {
            if v.fract() == 0.0 {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        NodeKind::Symbol { name, .. } => name,
        NodeKind::Unary { op, arg } => {
            format!("{}({})", op.name(), format_expression(arg, interner, memo))
        }
        NodeKind::Binary { op, lhs, rhs } => {
            if op == BinaryOp::Mul {
                if matches!(interner.node(lhs).kind, NodeKind::Constant(value) if value == -1.0) {
                    let rendered = format_expression(rhs, interner, memo);
                    let negated = format!("(-{rendered})");
                    memo.insert(expr, negated.clone());
                    return negated;
                }
                if matches!(interner.node(rhs).kind, NodeKind::Constant(value) if value == -1.0) {
                    let rendered = format_expression(lhs, interner, memo);
                    let negated = format!("(-{rendered})");
                    memo.insert(expr, negated.clone());
                    return negated;
                }
            }
            let lhs_rendered = format_expression(lhs, interner, memo);
            let rhs_rendered = format_expression(rhs, interner, memo);
            if let Some(symbol) = op.symbol() {
                format!("({lhs_rendered} {symbol} {rhs_rendered})")
            } else {
                format!("{}({lhs_rendered}, {rhs_rendered})", op.name())
            }
        }
    };
    memo.insert(expr, formatted.clone());
    formatted
}

fn canonical_pair(lhs: SX, rhs: SX) -> (SX, SX) {
    if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
}

fn unary(op: UnaryOp, arg: SX) -> SX {
    use NodeKey as K;
    use NodeKind as N;

    if let Some(value) = constant_value(arg) {
        return SX::from(op.apply_constant(value));
    }

    match op {
        UnaryOp::Abs => {
            if arg.is_zero() {
                return SX::zero();
            }
            if matches!(
                node_kind(arg),
                NodeKind::Unary {
                    op: UnaryOp::Abs,
                    ..
                }
            ) {
                return arg;
            }
        }
        UnaryOp::Sign => {
            if arg.is_zero() {
                return SX::zero();
            }
            if matches!(
                node_kind(arg),
                NodeKind::Unary {
                    op: UnaryOp::Sign,
                    ..
                }
            ) {
                return arg;
            }
        }
        UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Trunc
        | UnaryOp::Sin
        | UnaryOp::Tan
        | UnaryOp::Asin
        | UnaryOp::Atan
        | UnaryOp::Sinh
        | UnaryOp::Tanh
        | UnaryOp::Asinh => {
            if arg.is_zero() {
                return SX::zero();
            }
        }
        UnaryOp::Sqrt => {
            if arg.is_zero() {
                return SX::zero();
            }
            if arg.is_one() {
                return SX::one();
            }
        }
        UnaryOp::Exp | UnaryOp::Cos | UnaryOp::Cosh => {
            if arg.is_zero() {
                return SX::one();
            }
        }
        UnaryOp::Log => {
            if arg.is_one() {
                return SX::zero();
            }
        }
        UnaryOp::Acos | UnaryOp::Acosh | UnaryOp::Atanh => {}
    }

    with_interner(|interner| interner.intern_keyed(K::Unary { op, arg }, N::Unary { op, arg }))
}

fn binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKind as N;

    let lhs_kind = node_kind(lhs);
    let rhs_kind = node_kind(rhs);

    if let (N::Constant(a), N::Constant(b)) = (&lhs_kind, &rhs_kind) {
        return SX::from(op.apply_constant(*a, *b));
    }

    match op {
        BinaryOp::Add => {
            if lhs.is_zero() {
                return rhs;
            }
            if rhs.is_zero() {
                return lhs;
            }
            if let Some(combined) = combine_like_terms(lhs, rhs, 1.0) {
                return combined;
            }
        }
        BinaryOp::Sub => {
            if lhs.is_zero() {
                return -rhs;
            }
            if rhs.is_zero() {
                return lhs;
            }
            if lhs == rhs {
                return SX::zero();
            }
            if let Some(combined) = combine_like_terms(lhs, rhs, -1.0) {
                return combined;
            }
        }
        BinaryOp::Mul => {
            if lhs.is_zero() || rhs.is_zero() {
                return SX::zero();
            }
            if lhs.is_one() {
                return rhs;
            }
            if rhs.is_one() {
                return lhs;
            }
            if let Some(lhs_value) = constant_value(lhs)
                && let Some((rhs_value, factor)) = mul_constant_factor(rhs)
            {
                return SX::from(lhs_value * rhs_value) * factor;
            }
            if let Some(rhs_value) = constant_value(rhs)
                && let Some((lhs_value, factor)) = mul_constant_factor(lhs)
            {
                return SX::from(lhs_value * rhs_value) * factor;
            }
            return rebuild_rational_factors(combine_rational_factors(
                rational_factors(lhs),
                rational_factors(rhs),
            ));
        }
        BinaryOp::Div => {
            if lhs.is_zero() {
                return SX::zero();
            }
            if rhs.is_one() {
                return lhs;
            }
            if let Some(divided) =
                divide_rational_factors(rational_factors(lhs), rational_factors(rhs))
            {
                return rebuild_rational_factors(divided);
            }
        }
        BinaryOp::Pow => {
            if rhs.is_zero() {
                return SX::one();
            }
            if rhs.is_one() {
                return lhs;
            }
            if let Some(exponent) = constant_value(rhs) {
                if exponent == 2.0 {
                    return lhs.sqr();
                }
                if exponent == 0.5 {
                    return lhs.sqrt();
                }
                if lhs.is_zero() && exponent > 0.0 {
                    return SX::zero();
                }
            }
            if lhs.is_one() {
                return SX::one();
            }
        }
        BinaryOp::Atan2 => {
            if lhs.is_zero() && rhs.is_one() {
                return SX::zero();
            }
        }
        BinaryOp::Hypot => {
            if lhs.is_zero() {
                return rhs.abs();
            }
            if rhs.is_zero() {
                return lhs.abs();
            }
        }
        BinaryOp::Mod => {
            if lhs.is_zero() {
                return SX::zero();
            }
        }
        BinaryOp::Copysign => {
            if lhs.is_zero() {
                return SX::zero();
            }
            if rhs.is_zero() {
                return lhs.abs();
            }
        }
    }

    intern_binary(op, lhs, rhs)
}

fn topo_visit(node: SX, seen: &mut HashSet<SX>, order: &mut Vec<SX>) {
    if !seen.insert(node) {
        return;
    }
    match node_kind(node) {
        NodeKind::Unary { arg, .. } => {
            topo_visit(arg, seen, order);
            order.push(node);
        }
        NodeKind::Binary { lhs, rhs, .. } => {
            topo_visit(lhs, seen, order);
            topo_visit(rhs, seen, order);
            order.push(node);
        }
        NodeKind::Constant(_) | NodeKind::Symbol { .. } => {}
    }
}

fn unary_derivative(op: UnaryOp, arg: SX) -> SX {
    match op {
        UnaryOp::Abs => arg.sign(),
        UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc => {
            SX::zero()
        }
        UnaryOp::Sqrt => 0.5 / arg.sqrt(),
        UnaryOp::Exp => arg.exp(),
        UnaryOp::Log => SX::one() / arg,
        UnaryOp::Sin => arg.cos(),
        UnaryOp::Cos => -arg.sin(),
        UnaryOp::Tan => SX::one() / arg.cos().sqr(),
        UnaryOp::Asin => SX::one() / (SX::one() - arg.sqr()).sqrt(),
        UnaryOp::Acos => -SX::one() / (SX::one() - arg.sqr()).sqrt(),
        UnaryOp::Atan => SX::one() / (SX::one() + arg.sqr()),
        UnaryOp::Sinh => arg.cosh(),
        UnaryOp::Cosh => arg.sinh(),
        UnaryOp::Tanh => SX::one() / arg.cosh().sqr(),
        UnaryOp::Asinh => SX::one() / (arg.sqr() + 1.0).sqrt(),
        UnaryOp::Acosh => SX::one() / ((arg - 1.0).sqrt() * (arg + 1.0).sqrt()),
        UnaryOp::Atanh => SX::one() / (SX::one() - arg.sqr()),
    }
}

fn binary_partials(op: BinaryOp, lhs: SX, rhs: SX) -> (SX, SX) {
    match op {
        BinaryOp::Add => (SX::one(), SX::one()),
        BinaryOp::Sub => (SX::one(), -SX::one()),
        BinaryOp::Mul => (rhs, lhs),
        BinaryOp::Div => (SX::one() / rhs, -lhs / rhs.sqr()),
        BinaryOp::Pow => {
            let pow = lhs.pow(rhs);
            (rhs * lhs.pow(rhs - 1.0), pow * lhs.log())
        }
        BinaryOp::Atan2 => {
            let denom = lhs.sqr() + rhs.sqr();
            (rhs / denom, -lhs / denom)
        }
        BinaryOp::Hypot => {
            let hypot = lhs.hypot(rhs);
            (lhs / hypot, rhs / hypot)
        }
        BinaryOp::Mod => (SX::one(), -(lhs / rhs).trunc()),
        BinaryOp::Copysign => {
            let rhs_sign = rhs.sign() + (SX::one() - rhs.sign().abs());
            (lhs.sign() * rhs_sign, SX::zero())
        }
    }
}

fn directional_forward(expr: SX, seeds: &HashMap<SX, SX>, memo: &mut HashMap<SX, SX>) -> SX {
    if let Some(existing) = memo.get(&expr) {
        return *existing;
    }
    let derivative = match node_kind(expr) {
        NodeKind::Constant(_) => SX::zero(),
        NodeKind::Symbol { .. } => seeds.get(&expr).copied().unwrap_or_else(SX::zero),
        NodeKind::Unary { op, arg } => {
            directional_forward(arg, seeds, memo) * unary_derivative(op, arg)
        }
        NodeKind::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add => {
                directional_forward(lhs, seeds, memo) + directional_forward(rhs, seeds, memo)
            }
            BinaryOp::Sub => {
                directional_forward(lhs, seeds, memo) - directional_forward(rhs, seeds, memo)
            }
            BinaryOp::Mul => {
                directional_forward(lhs, seeds, memo) * rhs
                    + lhs * directional_forward(rhs, seeds, memo)
            }
            BinaryOp::Div => {
                let dl = directional_forward(lhs, seeds, memo);
                let dr = directional_forward(rhs, seeds, memo);
                (dl * rhs - lhs * dr) / rhs.sqr()
            }
            BinaryOp::Pow
            | BinaryOp::Atan2
            | BinaryOp::Hypot
            | BinaryOp::Mod
            | BinaryOp::Copysign => {
                let dl = directional_forward(lhs, seeds, memo);
                let dr = directional_forward(rhs, seeds, memo);
                let (d_lhs, d_rhs) = binary_partials(op, lhs, rhs);
                dl * d_lhs + dr * d_rhs
            }
        },
    };
    memo.insert(expr, derivative);
    derivative
}

pub(crate) fn forward_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if vars.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "forward seed length {} does not match variable length {}",
            seeds.len(),
            vars.len()
        )));
    }
    let seed_map = vars
        .iter()
        .copied()
        .zip(seeds.iter().copied())
        .collect::<HashMap<_, _>>();
    let mut memo = HashMap::new();
    Ok(outputs
        .iter()
        .copied()
        .map(|output| directional_forward(output, &seed_map, &mut memo))
        .collect())
}

pub(crate) fn reverse_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if outputs.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "reverse seed length {} does not match output length {}",
            seeds.len(),
            outputs.len()
        )));
    }
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for output in outputs.iter().copied() {
        topo_visit(output, &mut seen, &mut order);
    }

    let mut adjoints = HashMap::<SX, SX>::new();
    for (output, seed) in outputs.iter().copied().zip(seeds.iter().copied()) {
        adjoints
            .entry(output)
            .and_modify(|entry| *entry += seed)
            .or_insert(seed);
    }

    for node in order.into_iter().rev() {
        let Some(adj) = adjoints.get(&node).copied() else {
            continue;
        };
        match node_kind(node) {
            NodeKind::Unary { op, arg } => {
                let contrib = adj * unary_derivative(op, arg);
                adjoints
                    .entry(arg)
                    .and_modify(|entry| *entry += contrib)
                    .or_insert(contrib);
            }
            NodeKind::Binary { op, lhs, rhs } => match op {
                BinaryOp::Add => {
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry += adj)
                        .or_insert(adj);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry += adj)
                        .or_insert(adj);
                }
                BinaryOp::Sub => {
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry += adj)
                        .or_insert(adj);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry -= adj)
                        .or_insert(-adj);
                }
                BinaryOp::Mul => {
                    let lhs_contrib = adj * rhs;
                    let rhs_contrib = adj * lhs;
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry += lhs_contrib)
                        .or_insert(lhs_contrib);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry += rhs_contrib)
                        .or_insert(rhs_contrib);
                }
                BinaryOp::Div => {
                    let lhs_contrib = adj / rhs;
                    let rhs_contrib = -(adj * lhs) / rhs.sqr();
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry += lhs_contrib)
                        .or_insert(lhs_contrib);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry += rhs_contrib)
                        .or_insert(rhs_contrib);
                }
                BinaryOp::Pow
                | BinaryOp::Atan2
                | BinaryOp::Hypot
                | BinaryOp::Mod
                | BinaryOp::Copysign => {
                    let (d_lhs, d_rhs) = binary_partials(op, lhs, rhs);
                    let lhs_contrib = adj * d_lhs;
                    let rhs_contrib = adj * d_rhs;
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry += lhs_contrib)
                        .or_insert(lhs_contrib);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry += rhs_contrib)
                        .or_insert(rhs_contrib);
                }
            },
            NodeKind::Constant(_) | NodeKind::Symbol { .. } => {}
        }
    }

    Ok(vars
        .iter()
        .copied()
        .map(|var| adjoints.get(&var).copied().unwrap_or_else(SX::zero))
        .collect())
}

pub(crate) fn depends_on(expr: SX, target: SX, memo: &mut HashMap<SX, bool>) -> bool {
    if let Some(existing) = memo.get(&expr) {
        return *existing;
    }
    let answer = match node_kind(expr) {
        NodeKind::Constant(_) => false,
        NodeKind::Symbol { .. } => expr == target,
        NodeKind::Unary { arg, .. } => depends_on(arg, target, memo),
        NodeKind::Binary { lhs, rhs, .. } => {
            depends_on(lhs, target, memo) || depends_on(rhs, target, memo)
        }
    };
    memo.insert(expr, answer);
    answer
}

impl SX {
    pub fn sym(name: impl Into<String>) -> Self {
        with_interner(|interner| interner.fresh_symbol(name))
    }

    pub fn zero() -> Self {
        SX::from(0.0)
    }

    pub fn one() -> Self {
        SX::from(1.0)
    }

    pub fn sqr(self) -> Self {
        self * self
    }

    pub fn abs(self) -> Self {
        unary(UnaryOp::Abs, self)
    }

    pub fn sign(self) -> Self {
        unary(UnaryOp::Sign, self)
    }

    pub fn floor(self) -> Self {
        unary(UnaryOp::Floor, self)
    }

    pub fn ceil(self) -> Self {
        unary(UnaryOp::Ceil, self)
    }

    pub fn round(self) -> Self {
        unary(UnaryOp::Round, self)
    }

    pub fn trunc(self) -> Self {
        unary(UnaryOp::Trunc, self)
    }

    pub fn sqrt(self) -> Self {
        unary(UnaryOp::Sqrt, self)
    }

    pub fn exp(self) -> Self {
        unary(UnaryOp::Exp, self)
    }

    pub fn expm1(self) -> Self {
        self.exp() - SX::one()
    }

    pub fn exp2(self) -> Self {
        (std::f64::consts::LN_2 * self).exp()
    }

    pub fn exp10(self) -> Self {
        (std::f64::consts::LN_10 * self).exp()
    }

    pub fn log(self) -> Self {
        unary(UnaryOp::Log, self)
    }

    pub fn log1p(self) -> Self {
        (SX::one() + self).log()
    }

    pub fn log2(self) -> Self {
        self.log() / std::f64::consts::LN_2
    }

    pub fn log10(self) -> Self {
        self.log() / std::f64::consts::LN_10
    }

    pub fn log_base(self, base: impl Into<SX>) -> Self {
        self.log() / base.into().log()
    }

    pub fn sin(self) -> Self {
        unary(UnaryOp::Sin, self)
    }

    pub fn cos(self) -> Self {
        unary(UnaryOp::Cos, self)
    }

    pub fn tan(self) -> Self {
        unary(UnaryOp::Tan, self)
    }

    pub fn asin(self) -> Self {
        unary(UnaryOp::Asin, self)
    }

    pub fn acos(self) -> Self {
        unary(UnaryOp::Acos, self)
    }

    pub fn atan(self) -> Self {
        unary(UnaryOp::Atan, self)
    }

    pub fn sinh(self) -> Self {
        unary(UnaryOp::Sinh, self)
    }

    pub fn cosh(self) -> Self {
        unary(UnaryOp::Cosh, self)
    }

    pub fn tanh(self) -> Self {
        unary(UnaryOp::Tanh, self)
    }

    pub fn asinh(self) -> Self {
        unary(UnaryOp::Asinh, self)
    }

    pub fn acosh(self) -> Self {
        unary(UnaryOp::Acosh, self)
    }

    pub fn atanh(self) -> Self {
        unary(UnaryOp::Atanh, self)
    }

    pub fn pow(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Pow, self, rhs.into())
    }

    pub fn powf(self, rhs: f64) -> Self {
        self.pow(rhs)
    }

    pub fn powi(self, exponent: i32) -> Self {
        if exponent == 0 {
            return SX::one();
        }
        if exponent < 0 {
            return SX::one() / self.powi(-exponent);
        }
        let mut result = SX::one();
        let mut base = self;
        let mut remaining = exponent as u32;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result *= base;
            }
            remaining >>= 1;
            if remaining > 0 {
                base *= base;
            }
        }
        result
    }

    pub fn atan2(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Atan2, self, rhs.into())
    }

    pub fn hypot(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Hypot, self, rhs.into())
    }

    pub fn modulo(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Mod, self, rhs.into())
    }

    pub fn copysign(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Copysign, self, rhs.into())
    }

    pub fn min(self, rhs: impl Into<SX>) -> Self {
        let rhs = rhs.into();
        0.5 * (self + rhs - (self - rhs).abs())
    }

    pub fn max(self, rhs: impl Into<SX>) -> Self {
        let rhs = rhs.into();
        0.5 * (self + rhs + (self - rhs).abs())
    }

    pub fn inspect(self) -> NodeView {
        match node_kind(self) {
            NodeKind::Constant(v) => NodeView::Constant(v),
            NodeKind::Symbol { name, serial } => NodeView::Symbol { name, serial },
            NodeKind::Unary { op, arg } => NodeView::Unary { op, arg },
            NodeKind::Binary { op, lhs, rhs } => NodeView::Binary { op, lhs, rhs },
        }
    }

    pub fn is_symbolic(self) -> bool {
        matches!(node_kind(self), NodeKind::Symbol { .. })
    }

    pub fn is_zero(self) -> bool {
        matches!(node_kind(self), NodeKind::Constant(v) if v == 0.0)
    }

    pub fn is_one(self) -> bool {
        matches!(node_kind(self), NodeKind::Constant(v) if v == 1.0)
    }

    pub fn free_symbols(self) -> BTreeSet<SX> {
        let mut seen = HashSet::new();
        let mut stack = vec![self];
        let mut free = BTreeSet::new();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            match node_kind(node) {
                NodeKind::Symbol { .. } => {
                    free.insert(node);
                }
                NodeKind::Constant(_) => {}
                NodeKind::Unary { arg, .. } => {
                    stack.push(arg);
                }
                NodeKind::Binary { lhs, rhs, .. } => {
                    stack.push(lhs);
                    stack.push(rhs);
                }
            }
        }
        free
    }

    pub fn symbol_name(self) -> Option<String> {
        match node_kind(self) {
            NodeKind::Symbol { name, .. } => Some(name),
            NodeKind::Constant(_) | NodeKind::Unary { .. } | NodeKind::Binary { .. } => None,
        }
    }

    pub fn id(self) -> u32 {
        self.0
    }
}

impl From<f64> for SX {
    fn from(value: f64) -> Self {
        with_interner(|interner| {
            interner.intern_keyed(
                NodeKey::Constant(value.to_bits()),
                NodeKind::Constant(value),
            )
        })
    }
}

impl fmt::Display for SX {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_interner_ref(|interner| {
            let rendered = format_expression(*self, interner, &mut HashMap::new());
            write!(f, "{rendered}")
        })
    }
}

impl Neg for SX {
    type Output = SX;

    fn neg(self) -> Self::Output {
        SX::from(-1.0) * self
    }
}

impl Add for SX {
    type Output = SX;

    fn add(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Add, self, rhs)
    }
}

impl AddAssign for SX {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for SX {
    type Output = SX;

    fn sub(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Sub, self, rhs)
    }
}

impl SubAssign for SX {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for SX {
    type Output = SX;

    fn mul(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Mul, self, rhs)
    }
}

impl MulAssign for SX {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for SX {
    type Output = SX;

    fn div(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Div, self, rhs)
    }
}

impl DivAssign for SX {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for SX {
    type Output = SX;

    fn rem(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Mod, self, rhs)
    }
}

impl RemAssign for SX {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Add<f64> for SX {
    type Output = SX;

    fn add(self, rhs: f64) -> Self::Output {
        self + SX::from(rhs)
    }
}

impl Sub<f64> for SX {
    type Output = SX;

    fn sub(self, rhs: f64) -> Self::Output {
        self - SX::from(rhs)
    }
}

impl Mul<f64> for SX {
    type Output = SX;

    fn mul(self, rhs: f64) -> Self::Output {
        self * SX::from(rhs)
    }
}

impl Div<f64> for SX {
    type Output = SX;

    fn div(self, rhs: f64) -> Self::Output {
        self / SX::from(rhs)
    }
}

impl Rem<f64> for SX {
    type Output = SX;

    fn rem(self, rhs: f64) -> Self::Output {
        self % SX::from(rhs)
    }
}

impl Add<SX> for f64 {
    type Output = SX;

    fn add(self, rhs: SX) -> Self::Output {
        SX::from(self) + rhs
    }
}

impl Sub<SX> for f64 {
    type Output = SX;

    fn sub(self, rhs: SX) -> Self::Output {
        SX::from(self) - rhs
    }
}

impl Mul<SX> for f64 {
    type Output = SX;

    fn mul(self, rhs: SX) -> Self::Output {
        SX::from(self) * rhs
    }
}

impl Div<SX> for f64 {
    type Output = SX;

    fn div(self, rhs: SX) -> Self::Output {
        SX::from(self) / rhs
    }
}

impl Rem<SX> for f64 {
    type Output = SX;

    fn rem(self, rhs: SX) -> Self::Output {
        SX::from(self) % rhs
    }
}
