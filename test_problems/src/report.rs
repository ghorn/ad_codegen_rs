use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;

use crate::manifest::KnownStatus;
use crate::model::{ProblemRunRecord, RunStatus, SolverKind};
use crate::runner::RunResults;

pub fn render_markdown_report(results: &RunResults) -> String {
    let mut out = String::new();
    out.push_str("# Test Problems Report\n\n");
    out.push_str(
        "> Browser note: Chrome does not natively render `.md` files; it shows the source. Open `report.html` for the styled browser view, or open this file in a Markdown renderer.\n\n",
    );
    out.push_str("## Overview\n\n");
    out.push_str(&summary_table(results));
    out.push('\n');
    out.push_str(&family_summary(results));

    let regressions = collect_records(results, |record| {
        matches!(record.expected, KnownStatus::KnownPassing)
            && !record.status.accepted()
    });
    if !regressions.is_empty() {
        out.push_str("\n## Known-Passing Regressions\n\n");
        out.push_str(&record_table(
            &regressions,
            false,
            true,
            true,
            "These are the cases CI should care about most.",
        ));
    }

    let unexpected_passes = collect_records(results, |record| {
        matches!(record.expected, KnownStatus::KnownFailing)
            && record.status.accepted()
    });
    if !unexpected_passes.is_empty() {
        out.push_str("\n## Unexpected Passes\n\n");
        out.push_str(&record_table(
            &unexpected_passes,
            false,
            true,
            true,
            "These look good and are candidates to promote in the manifest.",
        ));
    }

    let failures = collect_records(results, |record| {
        record.status.failed()
    });
    if !failures.is_empty() {
        out.push_str("\n## Failure Details\n\n");
        for (family, family_records) in group_by_family(&failures) {
            let _ = writeln!(out, "### {family}\n");
            for record in family_records {
                let info = failure_info(record);
                let _ = writeln!(
                    out,
                    "- {} [{}/{}] `{}`: {}",
                    problem_markdown(record),
                    solver_display(record.solver),
                    record.options.label(),
                    info.code,
                    info.detail,
                );
            }
            out.push('\n');
        }
    }

    out.push_str("\n## Full Matrix\n\n");
    for (family, family_records) in group_by_family(&results.records.iter().collect::<Vec<_>>()) {
        let _ = writeln!(out, "### {family}\n");
        out.push_str(&record_table(&family_records, true, true, false, ""));
    }

    out
}

pub fn render_terminal_report(results: &RunResults) -> String {
    let mut out = String::new();
    out.push_str("Test Problems Summary\n\n");
    out.push_str(&summary_table(results));
    out.push('\n');
    out.push_str(&family_summary(results));

    let regressions = collect_records(results, |record| {
        matches!(record.expected, KnownStatus::KnownPassing)
            && !record.status.accepted()
    });
    if !regressions.is_empty() {
        out.push_str("\nKnown-Passing Regressions\n\n");
        out.push_str(&record_table(
            &regressions,
            true,
            true,
            false,
            "These are the cases CI should care about most.",
        ));
    }

    let failures = collect_records(results, |record| {
        record.status.failed()
    });
    if !failures.is_empty() {
        out.push_str("\nFailure Details\n\n");
        for (family, family_records) in group_by_family(&failures) {
            let _ = writeln!(out, "{family}");
            for record in family_records {
                let info = failure_info(record);
                let _ = writeln!(
                    out,
                    "  - {:<28}  {:<5}  {:<12}  {}",
                    record.id,
                    solver_display(record.solver),
                    info.code,
                    info.detail,
                );
            }
            out.push('\n');
        }
    }

    out
}

pub fn render_html_report(results: &RunResults) -> String {
    let mut body = String::new();
    body.push_str("<h1>Test Problems Report</h1>\n");
    body.push_str(
        "<p class=\"tp-note\"><strong>Tip:</strong> Chrome shows raw Markdown for <code>.md</code> files. Open <code>report.html</code> for the styled browser view, or use a Markdown renderer for <code>report.md</code>.</p>\n",
    );
    body.push_str("<h2>Overview</h2>\n");
    body.push_str(&summary_html_table(results));
    body.push_str("<h2>Family Summary</h2>\n");
    body.push_str(&family_summary_html(results));

    let regressions = collect_records(results, |record| {
        matches!(record.expected, KnownStatus::KnownPassing)
            && !record.status.accepted()
    });
    if !regressions.is_empty() {
        body.push_str("<h2>Known-Passing Regressions</h2>\n");
        body.push_str("<p class=\"tp-note\">These are the cases CI should care about most.</p>\n");
        body.push_str(&record_html_table(&regressions, false, true));
    }

    let unexpected_passes = collect_records(results, |record| {
        matches!(record.expected, KnownStatus::KnownFailing)
            && record.status.accepted()
    });
    if !unexpected_passes.is_empty() {
        body.push_str("<h2>Unexpected Passes</h2>\n");
        body.push_str(
            "<p class=\"tp-note\">These look good and are candidates to promote in the manifest.</p>\n",
        );
        body.push_str(&record_html_table(&unexpected_passes, false, true));
    }

    let failures = collect_records(results, |record| {
        record.status.failed()
    });
    if !failures.is_empty() {
        body.push_str("<h2>Failure Details</h2>\n");
        for (family, family_records) in group_by_family(&failures) {
            let _ = writeln!(body, "<h3>{family}</h3>");
            body.push_str("<ul class=\"tp-list\">\n");
            for record in family_records {
                let info = failure_info(record);
                let _ = writeln!(
                    body,
                    "<li>{} <span class=\"tp-inline-note\">[{}/{}]</span> <span class=\"{}\">{}</span></li>",
                    problem_html(record),
                    solver_display(record.solver),
                    record.options.label(),
                    info.text_class,
                    html_escape(&info.detail),
                );
            }
            body.push_str("</ul>\n");
        }
    }

    body.push_str("<h2>Full Matrix</h2>\n");
    for (family, family_records) in group_by_family(&results.records.iter().collect::<Vec<_>>()) {
        let _ = writeln!(body, "<h3>{family}</h3>");
        body.push_str(&record_html_table(&family_records, true, true));
    }

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ad_codegen_rs test problems report</title>
  {styles}
</head>
<body class="tp-page">
{body}
</body>
</html>"#,
        styles = report_styles(),
    )
}

pub fn write_json_report(results: &RunResults, path: &Path) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(results)?)?;
    Ok(())
}

pub fn write_html_report(results: &RunResults, path: &Path) -> Result<()> {
    fs::write(path, render_html_report(results))?;
    Ok(())
}

pub(crate) fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2}s")
    } else if secs >= 1e-3 {
        format!("{:.2}ms", secs * 1e3)
    } else if secs >= 1e-6 {
        format!("{:.2}us", secs * 1e6)
    } else {
        format!("{:.2}ns", secs * 1e9)
    }
}

fn summary_table(results: &RunResults) -> String {
    let mut grouped = BTreeMap::new();
    for record in &results.records {
        let entry = grouped
            .entry((solver_display(record.solver), record.options.label()))
            .or_insert((0usize, 0usize, 0usize, 0usize, Duration::ZERO));
        entry.0 += 1;
        if record.status.accepted() {
            entry.1 += 1;
        }
        if matches!(record.status, RunStatus::ReducedAccuracy) {
            entry.2 += 1;
        }
        if record.status.failed() || matches!(record.status, RunStatus::Skipped) {
            entry.3 += 1;
        }
        entry.4 += record.timing.total_wall_time;
    }
    let rows = grouped
        .into_iter()
        .map(|((solver, jit_opt), (cases, passed, reduced, not_passed, total_time))| {
            vec![
                solver.to_string(),
                jit_opt.to_string(),
                cases.to_string(),
                passed.to_string(),
                reduced.to_string(),
                not_passed.to_string(),
                format!("{:.1}%", percentage(passed, cases)),
                format_duration(total_time),
            ]
        })
        .collect::<Vec<_>>();
    render_text_table(
        ["Solver", "JIT", "Cases", "Pass", "Reduced", "Fail", "Pass %", "Total"],
        &[
            Align::Left,
            Align::Left,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
            Align::Right,
        ],
        &rows,
    )
}

fn summary_html_table(results: &RunResults) -> String {
    let mut grouped = BTreeMap::new();
    for record in &results.records {
        let entry = grouped
            .entry((solver_display(record.solver), record.options.label()))
            .or_insert((0usize, 0usize, 0usize, 0usize, Duration::ZERO));
        entry.0 += 1;
        if record.status.accepted() {
            entry.1 += 1;
        }
        if matches!(record.status, RunStatus::ReducedAccuracy) {
            entry.2 += 1;
        }
        if record.status.failed() || matches!(record.status, RunStatus::Skipped) {
            entry.3 += 1;
        }
        entry.4 += record.timing.total_wall_time;
    }

    let mut out = String::from("<div class=\"tp-overview-grid\">");
    for ((solver, jit_opt), (cases, passed, reduced, failed, total_time)) in grouped {
        let pass_class = if reduced > 0 || failed > 0 {
            "tp-warn"
        } else {
            "tp-ok"
        };
        let reduced_class = if reduced == 0 { "tp-neutral" } else { "tp-warn" };
        let fail_class = if failed == 0 { "tp-neutral" } else { "tp-fail" };
        let _ = writeln!(
            out,
            "<div class=\"tp-overview-card\"><div class=\"tp-overview-head\"><span class=\"tp-overview-solver\">{solver}</span><span class=\"tp-overview-jit\">{jit_opt}</span></div><div class=\"tp-overview-body\"><div><span class=\"tp-inline-note\">cases</span> <strong>{cases}</strong></div><div><span class=\"tp-inline-note\">pass</span> <span class=\"tp-badge {pass_class}\">{passed}</span></div><div><span class=\"tp-inline-note\">reduced</span> <span class=\"tp-badge {reduced_class}\">{reduced}</span></div><div><span class=\"tp-inline-note\">fail</span> <span class=\"tp-badge {fail_class}\">{failed}</span></div><div><span class=\"tp-inline-note\">pass %</span> <strong>{:.1}%</strong></div><div><span class=\"tp-inline-note\">total</span> <strong>{}</strong></div></div></div>",
            percentage(passed, cases),
            format_duration(total_time),
        );
    }
    out.push_str("</div>\n");
    out
}

fn family_summary(results: &RunResults) -> String {
    let (columns, rows_data) = family_summary_matrix(results);
    let mut rows = Vec::with_capacity(rows_data.len());
    for (family, by_column) in rows_data {
        let mut row = vec![family];
        let mut total = Duration::ZERO;
        for column in &columns {
            if let Some((cases, passed, reduced, total_time)) = by_column.get(column) {
                total += *total_time;
                if *reduced > 0 {
                    row.push(format!(
                        "{passed}/{cases} ({reduced} reduced, {:.0}%, {})",
                        percentage(*passed, *cases),
                        format_duration(*total_time)
                    ));
                } else {
                    row.push(format!(
                        "{passed}/{cases} ({:.0}%, {})",
                        percentage(*passed, *cases),
                        format_duration(*total_time)
                    ));
                }
            } else {
                row.push("--".to_string());
            }
        }
        row.push(format_duration(total));
        rows.push(row);
    }
    let mut out = String::from("## Family Summary\n\n");
    let mut headers = vec!["Family".to_string()];
    headers.extend(columns.iter().cloned());
    headers.push("Total".to_string());
    let mut align = vec![Align::Left];
    align.extend(std::iter::repeat_n(Align::Left, columns.len()));
    align.push(Align::Right);
    out.push_str(&render_text_table_dyn(&headers, &align, &rows));
    out.push('\n');
    out
}

fn family_summary_html(results: &RunResults) -> String {
    let (columns, rows_data) = family_summary_matrix(results);
    let mut out = String::from(
        "<div class=\"tp-table-wrap\"><table class=\"tp-table tp-table-compact\">\n<thead><tr><th>Family</th>",
    );
    for column in &columns {
        let _ = write!(out, "<th>{}</th>", html_escape(column));
    }
    out.push_str("<th>Total</th></tr></thead>\n<tbody>\n");
    for (family, by_column) in rows_data {
        let mut family_total = Duration::ZERO;
        let _ = write!(out, "<tr><td>{}</td>", html_escape(&family));
        for column in &columns {
            if let Some((cases, passed, reduced, total_time)) = by_column.get(column) {
                family_total += *total_time;
                let pass_class = if *passed == *cases && *reduced == 0 {
                    "tp-ok"
                } else if *passed == 0 {
                    "tp-fail"
                } else {
                    "tp-warn"
                };
                let cell_class = if *passed == *cases && *reduced == 0 {
                    "tp-family-cell tp-family-ok"
                } else if *passed == 0 {
                    "tp-family-cell tp-family-fail"
                } else {
                    "tp-family-cell tp-family-warn"
                };
                let reduced_meta = if *reduced > 0 {
                    format!(" · {reduced} reduced")
                } else {
                    String::new()
                };
                let _ = write!(
                    out,
                    "<td class=\"{cell_class}\"><span class=\"tp-badge {pass_class}\">{passed}/{cases}</span><div class=\"tp-family-meta\">{:.1}% · {}{reduced_meta}</div></td>",
                    percentage(*passed, *cases),
                    format_duration(*total_time),
                );
            } else {
                out.push_str("<td class=\"tp-family-cell tp-family-empty\">--</td>");
            }
        }
        let _ = writeln!(
            out,
            "<td class=\"tp-num\">{}</td></tr>",
            format_duration(family_total)
        );
    }
    out.push_str("</tbody></table></div>\n");
    out
}

type FamilySummaryCell = (usize, usize, usize, Duration);
type FamilySummaryRow = BTreeMap<String, FamilySummaryCell>;
type FamilySummaryRows = BTreeMap<String, FamilySummaryRow>;

fn family_summary_matrix(results: &RunResults) -> (Vec<String>, FamilySummaryRows) {
    let show_jit = results
        .records
        .iter()
        .map(|record| record.options.label())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1;

    let mut columns = std::collections::BTreeSet::new();
    let mut rows = FamilySummaryRows::new();
    for record in &results.records {
        let column = if show_jit {
            format!(
                "{}/{}",
                solver_display(record.solver),
                record.options.label()
            )
        } else {
            solver_display(record.solver).to_string()
        };
        columns.insert(column.clone());
        let entry = rows
            .entry(record.descriptor.family.clone())
            .or_default()
            .entry(column)
            .or_insert((0usize, 0usize, 0usize, Duration::ZERO));
        entry.0 += 1;
        if record.status.accepted() {
            entry.1 += 1;
        }
        if matches!(record.status, RunStatus::ReducedAccuracy) {
            entry.2 += 1;
        }
        entry.3 += record.timing.total_wall_time;
    }
    (columns.into_iter().collect(), rows)
}

fn format_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "--".to_string(), |value| value.to_string())
}

fn format_opt_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), |value| format!("{value:.3e}"))
}

fn format_expected_short(expected: KnownStatus) -> &'static str {
    match expected {
        KnownStatus::KnownPassing => "pass",
        KnownStatus::KnownFailing => "fail",
        KnownStatus::Skipped => "skip",
    }
}

fn format_status_short(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Passed => "ok",
        RunStatus::ReducedAccuracy => "reduced",
        RunStatus::FailedValidation => "bad",
        RunStatus::SolveError => "err",
        RunStatus::Skipped => "skip",
    }
}

fn solver_display(solver: SolverKind) -> &'static str {
    match solver {
        SolverKind::Sqp => "sqp",
        SolverKind::InteriorPoint => "ip",
        #[cfg(feature = "ipopt")]
        SolverKind::Ipopt => "ipopt",
    }
}

fn percentage(numer: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        (numer as f64) * 100.0 / (denom as f64)
    }
}

fn collect_records<F>(results: &RunResults, predicate: F) -> Vec<&ProblemRunRecord>
where
    F: Fn(&ProblemRunRecord) -> bool,
{
    results
        .records
        .iter()
        .filter(|record| predicate(record))
        .collect()
}

fn group_by_family<'a>(
    records: &'a [&'a ProblemRunRecord],
) -> BTreeMap<&'a str, Vec<&'a ProblemRunRecord>> {
    let mut grouped = BTreeMap::new();
    for record in records {
        grouped
            .entry(record.descriptor.family.as_str())
            .or_insert_with(Vec::new)
            .push(*record);
    }
    grouped
}

fn record_table(
    records: &[&ProblemRunRecord],
    include_expected: bool,
    include_reason: bool,
    include_detail: bool,
    preamble: &str,
) -> String {
    let mut rows = Vec::with_capacity(records.len());
    for record in records {
        let info = failure_info(record);
        let mut row = vec![
            problem_markdown(record),
            solver_display(record.solver).to_string(),
            record.options.label().to_string(),
        ];
        if include_expected {
            row.push(format_expected_short(record.expected).to_string());
        }
        row.extend([
            format_status_short(record.status).to_string(),
            record.descriptor.num_vars.to_string(),
            record.descriptor.dof.to_string(),
            constraint_summary(record),
            format_opt_usize(record.metrics.iterations),
            format_duration(record.timing.total_wall_time),
            format_opt_f64(record.metrics.objective),
            format_opt_f64(record.metrics.primal_inf),
            format_opt_f64(record.metrics.dual_inf),
        ]);
        if include_reason {
            row.push(info.code.into_owned());
        }
        if include_detail {
            row.push(ellipsize(&info.detail, 88));
        }
        rows.push(row);
    }

    let mut headers = vec!["Problem", "Solver", "JIT"];
    let mut align = vec![Align::Left, Align::Left, Align::Left];
    if include_expected {
        headers.push("Exp");
        align.push(Align::Left);
    }
    headers.extend([
        "Status", "Vars", "DOF", "Constr", "Iters", "Time", "Obj", "Primal", "Dual",
    ]);
    align.extend([
        Align::Left,
        Align::Right,
        Align::Right,
        Align::Left,
        Align::Right,
        Align::Right,
        Align::Right,
        Align::Right,
        Align::Right,
    ]);
    if include_reason {
        headers.push("Reason");
        align.push(Align::Left);
    }
    if include_detail {
        headers.push("Detail");
        align.push(Align::Left);
    }

    let mut out = String::new();
    if !preamble.is_empty() {
        let _ = writeln!(out, "{preamble}\n");
    }
    out.push_str(&render_text_table(headers, &align, &rows));
    out.push('\n');
    out
}

fn record_html_table(
    records: &[&ProblemRunRecord],
    include_expected: bool,
    include_detail: bool,
) -> String {
    let mut out = String::from("<table class=\"tp-table\">\n<thead><tr>");
    out.push_str("<th>Problem</th><th>Solver</th><th>JIT</th>");
    if include_expected {
        out.push_str("<th>Exp</th>");
    }
    out.push_str(
        "<th>Status</th><th>Reason</th><th>Vars</th><th>DOF</th><th>Constr</th><th>Iters</th><th>Time</th><th>Obj</th><th>Primal</th><th>Dual</th>",
    );
    if include_detail {
        out.push_str("<th>Detail</th>");
    }
    out.push_str("</tr></thead>\n<tbody>\n");

    for record in records {
        let info = failure_info(record);
        let status_badge = status_badge(record.status);
        let _ = write!(
            out,
            "<tr><td>{}</td><td>{}</td><td>{}</td>",
            problem_html(record),
            solver_display(record.solver),
            record.options.label(),
        );
        if include_expected {
            let _ = write!(
                out,
                "<td><span class=\"tp-badge {}\">{}</span></td>",
                expected_badge_class(record.expected),
                format_expected_short(record.expected),
            );
        }
        let _ = write!(
            out,
            "<td>{status_badge}</td><td class=\"{}\">{}</td><td class=\"tp-num\">{}</td><td class=\"tp-num\">{}</td><td>{}</td><td class=\"tp-num\">{}</td><td class=\"tp-num\">{}</td><td class=\"tp-num\">{}</td><td class=\"tp-num\">{}</td><td class=\"tp-num\">{}</td>",
            info.cell_class,
            reason_badge(&info),
            record.descriptor.num_vars,
            record.descriptor.dof,
            constraint_summary(record),
            format_opt_usize(record.metrics.iterations),
            format_duration(record.timing.total_wall_time),
            format_opt_f64(record.metrics.objective),
            format_opt_f64(record.metrics.primal_inf),
            format_opt_f64(record.metrics.dual_inf),
        );
        if include_detail {
            let _ = write!(
                out,
                "<td class=\"{}\"><code>{}</code></td>",
                info.cell_class,
                html_escape(&info.detail),
            );
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody></table>\n");
    out
}

fn constraint_summary(record: &ProblemRunRecord) -> String {
    match (
        record.descriptor.num_eq,
        record.descriptor.num_ineq,
        record.descriptor.num_box,
    ) {
        (0, 0, 0) => "none".to_string(),
        (eq, ineq, box_n) => format!("e{eq}/i{ineq}/b{box_n}"),
    }
}

fn classify_solve_error(error: &str) -> FailureInfo {
    if error.contains("failed to converge") {
        if let Some(start) = error.find("failed to converge in ") {
            let tail = &error[start + "failed to converge in ".len()..];
            if let Some(end) = tail.find(" iterations") {
                return FailureInfo::warn(
                    Cow::Borrowed("max_iters"),
                    format!("hit iteration limit after {} iterations", &tail[..end]),
                );
            }
        }
        FailureInfo::warn(
            Cow::Borrowed("max_iters"),
            "hit iteration limit".to_string(),
        )
    } else if error.contains("line search failed") {
        FailureInfo::fail(Cow::Borrowed("line_search"), squash_whitespace(error))
    } else if error.contains("PrimalInfeasible") {
        FailureInfo::fail(Cow::Borrowed("primal_infeasible"), squash_whitespace(error))
    } else if error.contains("NonFinite") || error.contains("NaN") || error.contains("Inf") {
        FailureInfo::fail(Cow::Borrowed("non_finite"), squash_whitespace(error))
    } else {
        FailureInfo::fail(Cow::Borrowed("solve_error"), squash_whitespace(error))
    }
}

fn classify_validation_failure(record: &ProblemRunRecord) -> FailureInfo {
    let detail = record.validation.detail.as_str();
    if detail.contains("solution_mismatch") {
        let objective = record
            .metrics
            .objective
            .map_or_else(|| "--".to_string(), |value| format!("{value:.2e}"));
        return FailureInfo::fail(
            Cow::Borrowed("wrong_solution"),
            format!("wrong solution basin (objective {objective})"),
        );
    }
    if let Some(objective) = record.metrics.objective
        && !objective.is_finite()
    {
        return FailureInfo::fail(
            Cow::Borrowed("non_finite"),
            "objective evaluated to a non-finite value".to_string(),
        );
    }
    if detail.contains("objective=") {
        let objective = record
            .metrics
            .objective
            .map_or_else(|| "--".to_string(), |value| format!("{value:.2e}"));
        let primal = record
            .metrics
            .primal_inf
            .map_or_else(|| "--".to_string(), |value| format!("{value:.2e}"));
        let dual = record
            .metrics
            .dual_inf
            .map_or_else(|| "--".to_string(), |value| format!("{value:.2e}"));
        return FailureInfo::warn(
            Cow::Borrowed("tol_miss"),
            format!("missed tolerance targets (obj {objective}, p {primal}, d {dual})"),
        );
    }
    FailureInfo::fail(Cow::Borrowed("validation"), squash_whitespace(detail))
}

fn status_badge(status: RunStatus) -> String {
    let (class, label) = match status {
        RunStatus::Passed => ("tp-ok", "PASS"),
        RunStatus::ReducedAccuracy => ("tp-warn", "REDUCED"),
        RunStatus::FailedValidation => ("tp-fail", "FAIL"),
        RunStatus::SolveError => ("tp-fail", "ERROR"),
        RunStatus::Skipped => ("tp-skip", "SKIP"),
    };
    format!("<span class=\"tp-badge {class}\">{label}</span>")
}

fn reason_badge(info: &FailureInfo) -> String {
    format!(
        "<span class=\"tp-badge {}\">{}</span>",
        info.badge_class,
        html_escape(info.code.as_ref()),
    )
}

fn expected_badge_class(expected: KnownStatus) -> &'static str {
    match expected {
        KnownStatus::KnownPassing => "tp-ok",
        KnownStatus::KnownFailing => "tp-warn",
        KnownStatus::Skipped => "tp-skip",
    }
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn problem_markdown(record: &ProblemRunRecord) -> String {
    if let Some(path) = &record.console_output_path {
        format!("[`{}`]({path})", record.id)
    } else {
        format!("`{}`", record.id)
    }
}

fn problem_html(record: &ProblemRunRecord) -> String {
    if let Some(path) = &record.console_output_path {
        format!(
            "<a href=\"{}\"><code>{}</code></a>",
            html_escape(path),
            html_escape(&record.id),
        )
    } else {
        format!("<code>{}</code>", html_escape(&record.id))
    }
}

fn squash_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn ellipsize(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    text.chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>()
        + "…"
}

fn report_styles() -> &'static str {
    r#"<style>
body.tp-page { font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; line-height: 1.45; }
h1, h2, h3 { margin: 0 0 12px; }
h2 { margin-top: 28px; }
h3 { margin-top: 20px; }
code { font-family: ui-monospace, SFMono-Regular, monospace; }
.tp-note { color: #cbd5e1; margin: 0 0 16px; }
.tp-inline-note { color: #94a3b8; font-size: 0.92em; }
.tp-table-wrap { overflow-x: auto; margin: 0 0 16px; }
.tp-table { border-collapse: collapse; width: max-content; min-width: 100%; font-size: 13px; margin: 0 0 16px; background: #111827; }
.tp-table-compact { min-width: 0; width: auto; }
.tp-table th, .tp-table td { border: 1px solid #334155; padding: 6px 8px; text-align: left; vertical-align: top; }
.tp-table th { background: #172033; color: #bfdbfe; }
.tp-num { text-align: right; font-variant-numeric: tabular-nums; }
.tp-badge { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 700; line-height: 1.4; white-space: nowrap; }
.tp-overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, max-content)); gap: 12px; margin: 0 0 16px; align-items: start; }
.tp-overview-card { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; }
.tp-overview-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.tp-overview-solver { font-weight: 700; color: #e2e8f0; text-transform: lowercase; }
.tp-overview-jit { color: #93c5fd; font-family: ui-monospace, SFMono-Regular, monospace; }
.tp-overview-body { display: grid; grid-template-columns: auto auto; gap: 6px 10px; align-items: center; }
.tp-ok { background: #dcfce7; color: #166534; }
.tp-fail { background: #fee2e2; color: #991b1b; }
.tp-warn { background: #fef3c7; color: #92400e; }
.tp-skip { background: #e5e7eb; color: #374151; }
.tp-neutral { background: #e2e8f0; color: #334155; }
.tp-family-cell { min-width: 110px; }
.tp-family-meta { margin-top: 4px; font-size: 11px; color: #cbd5e1; }
.tp-family-ok { background: #102417; }
.tp-family-warn { background: #2b2212; }
.tp-family-fail { background: #2a1417; }
.tp-family-empty { color: #94a3b8; }
.tp-cell-fail { background: #2a1417; }
.tp-cell-warn { background: #2b2212; }
.tp-cell-skip { background: #1f2937; }
.tp-text-fail { color: #fca5a5; }
.tp-text-warn { color: #fcd34d; }
.tp-list { margin: 0 0 16px 18px; padding: 0; }
.tp-list li { margin: 0 0 6px; }
</style>"#
}

#[derive(Clone, Debug)]
struct FailureInfo {
    code: Cow<'static, str>,
    detail: String,
    badge_class: &'static str,
    cell_class: &'static str,
    text_class: &'static str,
}

impl FailureInfo {
    fn neutral() -> Self {
        Self {
            code: Cow::Borrowed("--"),
            detail: "--".to_string(),
            badge_class: "tp-neutral",
            cell_class: "",
            text_class: "tp-inline-note",
        }
    }

    fn skip(detail: String) -> Self {
        Self {
            code: Cow::Borrowed("skip"),
            detail,
            badge_class: "tp-skip",
            cell_class: "tp-cell-skip",
            text_class: "tp-inline-note",
        }
    }

    fn warn(code: Cow<'static, str>, detail: String) -> Self {
        Self {
            code,
            detail,
            badge_class: "tp-warn",
            cell_class: "tp-cell-warn",
            text_class: "tp-text-warn",
        }
    }

    fn fail(code: Cow<'static, str>, detail: String) -> Self {
        Self {
            code,
            detail,
            badge_class: "tp-fail",
            cell_class: "tp-cell-fail",
            text_class: "tp-text-fail",
        }
    }
}

fn failure_info(record: &ProblemRunRecord) -> FailureInfo {
    match record.status {
        RunStatus::Passed => FailureInfo::neutral(),
        RunStatus::ReducedAccuracy => FailureInfo::warn(
            Cow::Borrowed("reduced_accuracy"),
            squash_whitespace(&record.validation.detail),
        ),
        RunStatus::Skipped => FailureInfo::skip(record.validation.detail.clone()),
        RunStatus::SolveError => {
            if let Some(error) = &record.error {
                classify_solve_error(error)
            } else {
                FailureInfo::fail(Cow::Borrowed("solve_error"), "solve error".to_string())
            }
        }
        RunStatus::FailedValidation => classify_validation_failure(record),
    }
}

#[derive(Clone, Copy)]
enum Align {
    Left,
    Right,
}

fn render_text_table(
    headers: impl AsRef<[&'static str]>,
    align: &[Align],
    rows: &[Vec<String>],
) -> String {
    let headers = headers.as_ref();
    let mut widths = headers
        .iter()
        .map(|header| header.len())
        .collect::<Vec<_>>();
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    let mut out = String::from("```text\n");
    push_aligned_row(&mut out, headers, &widths, align);
    push_separator(&mut out, &widths);
    for row in rows {
        let cells = row.iter().map(String::as_str).collect::<Vec<_>>();
        push_aligned_row_dyn(&mut out, &cells, &widths, align);
    }
    out.push_str("```\n");
    out
}

fn render_text_table_dyn(headers: &[String], align: &[Align], rows: &[Vec<String>]) -> String {
    let mut widths = headers.iter().map(String::len).collect::<Vec<_>>();
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    let mut out = String::from("```text\n");
    let header_cells = headers.iter().map(String::as_str).collect::<Vec<_>>();
    push_aligned_row_dyn(&mut out, &header_cells, &widths, align);
    push_separator(&mut out, &widths);
    for row in rows {
        let cells = row.iter().map(String::as_str).collect::<Vec<_>>();
        push_aligned_row_dyn(&mut out, &cells, &widths, align);
    }
    out.push_str("```\n");
    out
}

fn push_separator(out: &mut String, widths: &[usize]) {
    for (idx, width) in widths.iter().enumerate() {
        if idx > 0 {
            out.push_str("  ");
        }
        out.push_str(&"-".repeat(*width));
    }
    out.push('\n');
}

fn push_aligned_row(out: &mut String, row: &[&str], widths: &[usize], align: &[Align]) {
    for idx in 0..row.len() {
        if idx > 0 {
            out.push_str("  ");
        }
        match align[idx] {
            Align::Left => {
                let _ = write!(out, "{:<width$}", row[idx], width = widths[idx]);
            }
            Align::Right => {
                let _ = write!(out, "{:>width$}", row[idx], width = widths[idx]);
            }
        }
    }
    out.push('\n');
}

fn push_aligned_row_dyn(out: &mut String, row: &[&str], widths: &[usize], align: &[Align]) {
    for idx in 0..row.len() {
        if idx > 0 {
            out.push_str("  ");
        }
        match align[idx] {
            Align::Left => {
                let _ = write!(out, "{:<width$}", row[idx], width = widths[idx]);
            }
            Align::Right => {
                let _ = write!(out, "{:>width$}", row[idx], width = widths[idx]);
            }
        }
    }
    out.push('\n');
}
