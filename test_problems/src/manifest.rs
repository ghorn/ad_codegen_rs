use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownStatus {
    KnownPassing,
    KnownFailing,
    Skipped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemSpeed {
    Fast,
    Slow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterationLimits {
    pub sqp: usize,
    pub interior_point: usize,
    pub ipopt: usize,
}

impl IterationLimits {
    pub const fn with_default(default: usize) -> Self {
        Self {
            sqp: default,
            interior_point: default,
            ipopt: default,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProblemManifestEntry {
    pub id: &'static str,
    pub sqp: KnownStatus,
    pub interior_point: KnownStatus,
    pub ipopt: KnownStatus,
    pub max_iters: IterationLimits,
    pub speed: ProblemSpeed,
}

pub const DEFAULT_MAX_ITERS: usize = 300;

pub const fn manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    interior_point: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        interior_point,
        ipopt: interior_point,
        max_iters,
        speed: ProblemSpeed::Fast,
    }
}

pub const fn manifest_entry_with_ipopt(
    id: &'static str,
    sqp: KnownStatus,
    interior_point: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        interior_point,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Fast,
    }
}

pub const fn slow_manifest_entry(
    id: &'static str,
    sqp: KnownStatus,
    interior_point: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        interior_point,
        ipopt: interior_point,
        max_iters,
        speed: ProblemSpeed::Slow,
    }
}

pub const fn slow_manifest_entry_with_ipopt(
    id: &'static str,
    sqp: KnownStatus,
    interior_point: KnownStatus,
    ipopt: KnownStatus,
    max_iters: IterationLimits,
) -> ProblemManifestEntry {
    ProblemManifestEntry {
        id,
        sqp,
        interior_point,
        ipopt,
        max_iters,
        speed: ProblemSpeed::Slow,
    }
}

const MANIFEST: &[ProblemManifestEntry] = &[
    manifest_entry_with_ipopt(
        "rosenbrock_2",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "generalized_rosenbrock_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "generalized_rosenbrock_8",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "generalized_rosenbrock_16",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 400,
            interior_point: 400,
            ipopt: 400,
        },
    ),
    manifest_entry_with_ipopt(
        "disk_rosenbrock",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "powell_singular_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "wood_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "brown_almost_linear_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "brown_almost_linear_8",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "brown_almost_linear_16",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 400,
            interior_point: 400,
            ipopt: 400,
        },
    ),
    manifest_entry_with_ipopt(
        "trigonometric_4",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry_with_ipopt(
        "trigonometric_8",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "trigonometric_16",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 400,
            interior_point: 400,
            ipopt: 400,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_eq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links06_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_eq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links12_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            interior_point: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            interior_point: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_eq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            interior_point: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            interior_point: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            interior_point: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links24_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 500,
            interior_point: 500,
            ipopt: 500,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_eq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            interior_point: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_eq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            interior_point: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_eq_quadratic",
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            interior_point: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_ineq_zigzag",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            interior_point: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_ineq_straight",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            interior_point: 800,
            ipopt: 800,
        },
    ),
    slow_manifest_entry_with_ipopt(
        "hanging_chain_links48_ineq_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits {
            sqp: 800,
            interior_point: 800,
            ipopt: 800,
        },
    ),
    manifest_entry_with_ipopt(
        "parameterized_quadratic",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry(
        "hs021",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry(
        "hs035",
        KnownStatus::KnownPassing,
        KnownStatus::KnownPassing,
        IterationLimits::with_default(DEFAULT_MAX_ITERS),
    ),
    manifest_entry(
        "hs071",
        KnownStatus::KnownPassing,
        KnownStatus::KnownFailing,
        IterationLimits {
            sqp: 400,
            interior_point: 400,
            ipopt: 400,
        },
    ),
];

pub fn manifest_entries() -> &'static [ProblemManifestEntry] {
    MANIFEST
}

pub fn manifest_entry_by_id(id: &str) -> Option<&'static ProblemManifestEntry> {
    MANIFEST.iter().find(|entry| entry.id == id)
}
