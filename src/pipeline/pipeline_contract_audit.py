from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet"
CONTEXT_FILE = PROJECT_ROOT / "data" / "context" / "prematch_context.parquet"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "pipeline_contract_audit.json"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "pipeline_contract_issues.csv"


def issue(code: str, severity: str, message: str, rows: int | None = None) -> dict:
    return {"code": code, "severity": severity, "message": message, "rows": rows}


def main() -> None:
    issues: list[dict] = []
    if not REPORT_FILE.exists():
        issues.append(issue("REPORT_MISSING", "error", str(REPORT_FILE)))
        report = pd.DataFrame()
    else:
        report = pd.read_parquet(REPORT_FILE).copy()

    if not report.empty:
        required = [
            "match_id",
            "kickoff_utc",
            "home_team",
            "away_team",
            "home_probability",
            "draw_probability",
            "away_probability",
        ]
        missing = [column for column in required if column not in report]
        if missing:
            issues.append(
                issue(
                    "REPORT_COLUMNS_MISSING",
                    "error",
                    ", ".join(missing),
                )
            )
        if "match_id" in report:
            duplicates = int(report["match_id"].astype(str).duplicated().sum())
            if duplicates:
                issues.append(
                    issue(
                        "DUPLICATE_MATCH_ID",
                        "error",
                        "Prematch report contains duplicate match IDs.",
                        duplicates,
                    )
                )
        if "kickoff_utc" in report:
            invalid_kickoff = int(
                pd.to_datetime(report["kickoff_utc"], utc=True, errors="coerce")
                .isna()
                .sum()
            )
            if invalid_kickoff:
                issues.append(
                    issue(
                        "INVALID_KICKOFF",
                        "error",
                        "Kickoff timestamps could not be parsed.",
                        invalid_kickoff,
                    )
                )
        probability_columns = [
            "home_probability",
            "draw_probability",
            "away_probability",
        ]
        if all(column in report for column in probability_columns):
            probabilities = report[probability_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
            invalid = (
                ~np.isfinite(probabilities).all(axis=1)
                | (probabilities < 0.0).any(axis=1)
                | (probabilities > 1.0).any(axis=1)
            )
            invalid_count = int(invalid.sum())
            if invalid_count:
                issues.append(
                    issue(
                        "INVALID_PROBABILITY",
                        "error",
                        "Probabilities must be finite and between zero and one.",
                        invalid_count,
                    )
                )
            sums = probabilities.sum(axis=1)
            bad_sums = int((np.abs(sums - 1.0) > 0.002).sum())
            if bad_sums:
                issues.append(
                    issue(
                        "PROBABILITY_SUM",
                        "error",
                        "Home/draw/away probabilities do not sum to one.",
                        bad_sums,
                    )
                )
        if {"home_team", "away_team"}.issubset(report.columns):
            same_team = int(
                report["home_team"].astype(str).eq(report["away_team"].astype(str)).sum()
            )
            if same_team:
                issues.append(
                    issue(
                        "SAME_TEAM_FIXTURE",
                        "error",
                        "A fixture has the same home and away team.",
                        same_team,
                    )
                )
        if "llm_safe_to_display" in report:
            safe = report["llm_safe_to_display"].fillna(False).astype(str).str.casefold().isin(
                {"true", "1", "yes"}
            )
            unsafe = int((~safe).sum())
            if unsafe:
                issues.append(
                    issue(
                        "LLM_BRIEF_BLOCKED",
                        "warning",
                        "Some AI narratives were blocked; deterministic summaries remain available.",
                        unsafe,
                    )
                )

    context_rows = None
    if CONTEXT_FILE.exists():
        context = pd.read_parquet(CONTEXT_FILE)
        context_rows = int(len(context))
        if not report.empty and "match_id" in report and "match_id" in context:
            report_ids = set(report["match_id"].astype(str))
            context_ids = set(context["match_id"].astype(str))
            missing_context = len(report_ids - context_ids)
            extra_context = len(context_ids - report_ids)
            if missing_context:
                issues.append(
                    issue(
                        "CONTEXT_MISSING_FOR_MATCH",
                        "warning",
                        "Some prediction rows have no context row.",
                        missing_context,
                    )
                )
            if extra_context:
                issues.append(
                    issue(
                        "ORPHAN_CONTEXT_ROW",
                        "warning",
                        "Some context rows are not in the current report.",
                        extra_context,
                    )
                )
    else:
        issues.append(issue("CONTEXT_MISSING", "warning", str(CONTEXT_FILE)))

    errors = sum(item["severity"] == "error" for item in issues)
    warnings = sum(item["severity"] == "warning" for item in issues)
    status = "blocked" if errors else "attention_required" if warnings else "healthy"
    payload = {
        "status": status,
        "report_rows": int(len(report)),
        "context_rows": context_rows,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
        "production_model_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        issues, columns=["code", "severity", "message", "rows"]
    ).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("Pipeline contract audit completed.")
    print(f"Status: {status}; errors: {errors}; warnings: {warnings}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
