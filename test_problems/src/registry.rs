use std::collections::BTreeSet;

use anyhow::{Result, bail};

use crate::manifest::manifest_entries;
use crate::model::ProblemCase;
use crate::problems;

pub fn registry() -> Result<Vec<ProblemCase>> {
    let cases = problems::all_cases();
    let mut case_ids = BTreeSet::new();
    for case in &cases {
        if !case_ids.insert(case.id) {
            bail!("duplicate problem case id in registry: {}", case.id);
        }
    }

    let manifest_ids = manifest_entries()
        .iter()
        .map(|entry| entry.id)
        .collect::<BTreeSet<_>>();
    let extra_cases = case_ids
        .difference(&manifest_ids)
        .copied()
        .collect::<Vec<_>>();
    let missing_cases = manifest_ids
        .difference(&case_ids)
        .copied()
        .collect::<Vec<_>>();
    if !extra_cases.is_empty() || !missing_cases.is_empty() {
        bail!(
            "manifest/registry mismatch: extra_cases={extra_cases:?}, missing_cases={missing_cases:?}"
        );
    }

    Ok(cases)
}
