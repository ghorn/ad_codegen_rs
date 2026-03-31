use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::model::{ProblemRunRecord, RunStatus};
use crate::runner::RunResults;

pub fn write_transcript_artifacts(results: &mut RunResults, output_dir: &Path) -> Result<()> {
    let transcript_dir = output_dir.join("transcripts");
    fs::create_dir_all(&transcript_dir)?;
    for record in &mut results.records {
        let Some(text) = record.console_output.as_deref() else {
            record.console_output_path = None;
            continue;
        };
        let basename = format!(
            "{}__{}__{}",
            slugify(&record.id),
            slugify(record.solver.label()),
            slugify(record.options.label()),
        );
        let txt_filename = format!("{basename}.txt");
        let html_filename = format!("{basename}.html");
        let txt_path = transcript_dir.join(&txt_filename);
        let html_path = transcript_dir.join(&html_filename);
        fs::write(&txt_path, text)?;
        fs::write(
            &html_path,
            render_transcript_html(record, text, &txt_filename),
        )?;
        record.console_output_path = Some(format!("transcripts/{html_filename}"));
    }
    Ok(())
}

fn render_transcript_html(
    record: &ProblemRunRecord,
    transcript: &str,
    txt_filename: &str,
) -> String {
    let (status_class, status_text) = status_badge(record.status);
    let reason = transcript_reason(record);
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 24px; background: #0f172a; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 20px; max-width: 1100px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    .meta {{ color: #94a3b8; margin: 0 0 16px; }}
    .links {{ margin: 0 0 16px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 0 0 18px; }}
    .summary-card {{ background: #0b1220; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; }}
    .summary-label {{ color: #94a3b8; font-size: 12px; margin-bottom: 4px; }}
    .summary-value {{ font-weight: 700; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 700; }}
    .ok {{ background: #dcfce7; color: #166534; }}
    .warn {{ background: #fef3c7; color: #92400e; }}
    .bad {{ background: #fee2e2; color: #991b1b; }}
    .skip {{ background: #e5e7eb; color: #374151; }}
    a {{ color: #93c5fd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    pre {{
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      white-space: pre;
      background: #020617;
      border: 1px solid #1e293b;
      border-radius: 12px;
      color: #e2e8f0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <p class="meta">{solver} / {jit}</p>
    <p class="links"><a href="../report.html">report</a> · <a href="../dashboard.html">dashboard</a> · <a href="{txt_filename}">raw txt</a></p>
    <div class="summary">
      <div class="summary-card"><div class="summary-label">Status</div><div class="summary-value"><span class="badge {status_class}">{status_text}</span></div></div>
      <div class="summary-card"><div class="summary-label">Reason</div><div class="summary-value">{reason}</div></div>
      <div class="summary-card"><div class="summary-label">Termination thresholds</div><div class="summary-value">{solver_thresholds}</div></div>
      <div class="summary-card"><div class="summary-label">Validation thresholds</div><div class="summary-value">{validation_thresholds}</div></div>
    </div>
    <pre>{transcript}</pre>
  </div>
</body>
</html>"#,
        title = html_escape(&record.id),
        solver = html_escape(record.solver.label()),
        jit = html_escape(record.options.label()),
        txt_filename = html_escape(txt_filename),
        status_class = status_class,
        status_text = status_text,
        reason = html_escape(&reason),
        solver_thresholds = html_escape(record.solver_thresholds.as_deref().unwrap_or("--")),
        validation_thresholds = html_escape(&record.validation.tolerance),
        transcript = html_escape(transcript),
    )
}

fn status_badge(status: RunStatus) -> (&'static str, &'static str) {
    match status {
        RunStatus::Passed => ("ok", "PASS"),
        RunStatus::ReducedAccuracy => ("warn", "REDUCED"),
        RunStatus::FailedValidation => ("bad", "FAIL"),
        RunStatus::SolveError => ("bad", "ERROR"),
        RunStatus::Skipped => ("skip", "SKIP"),
    }
}

fn transcript_reason(record: &ProblemRunRecord) -> String {
    match record.status {
        RunStatus::Passed => "--".to_string(),
        RunStatus::ReducedAccuracy | RunStatus::FailedValidation => record.validation.detail.clone(),
        RunStatus::SolveError => record
            .error
            .clone()
            .unwrap_or_else(|| "solve error".to_string()),
        RunStatus::Skipped => "skipped".to_string(),
    }
}

fn slugify(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
