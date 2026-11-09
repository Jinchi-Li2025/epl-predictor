from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet"
LOG_DIR = PROJECT_ROOT / "logs" / "matchday"


STEPS = [
    ("weather", "src/context/fetch_weather.py"),
    ("referee", "src/context/fetch_referee.py"),
    ("availability", "src/context/fetch_availability.py"),
    ("lineups", "src/context/fetch_lineups.py"),
    ("football-data lineups", "src/context/fetch_lineups_football_data.py"),
    ("manual lineup overrides", "src/context/apply_lineup_overrides.py"),
    ("team news", "src/context/fetch_team_news.py"),
    ("team news context", "src/context/build_team_news_context.py"),
    ("prematch report", "src/context/build_prematch_report.py"),
    ("LLM evidence validation", "src/context/validate_llm_evidence.py"),
    ("match briefs", "src/reporting/build_match_briefs.py"),
    ("squad features", "src/features/build_squad_features.py"),
    ("squad snapshot", "src/evaluation/archive_squad_snapshot.py"),
    ("registry quality", "src/evaluation/registry_quality_audit.py"),
    ("pipeline contracts", "src/pipeline/pipeline_contract_audit.py"),
    ("system health", "src/pipeline/system_health_check.py"),
    ("operational readiness", "src/pipeline/operational_readiness.py"),
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh context only when a fixture is close to kickoff."
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-hours", type=float, default=3.5)
    parser.add_argument("--post-kickoff-minutes", type=int, default=15)
    return parser.parse_args()


def due_matches(window_hours: float, post_minutes: int) -> pd.DataFrame:
    if not REPORT_FILE.exists():
        return pd.DataFrame()
    report = pd.read_parquet(REPORT_FILE).copy()
    if "kickoff_utc" not in report:
        return pd.DataFrame()
    report["kickoff_utc"] = pd.to_datetime(
        report["kickoff_utc"], utc=True, errors="coerce"
    )
    now = pd.Timestamp.now(tz="UTC")
    lower = now - pd.Timedelta(minutes=post_minutes)
    upper = now + pd.Timedelta(hours=window_hours)
    return report.loc[report["kickoff_utc"].between(lower, upper)].copy()


def run_step(label: str, relative_path: str, log) -> None:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        message = f"SKIPPED missing optional step: {relative_path}"
        print(message)
        log.write(message + "\n")
        return
    print(f"START {label}: {relative_path}")
    log.write(f"START {label}: {relative_path}\n")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, str(path)],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Matchday step failed: {label} ({return_code})")
    log.write(f"DONE {label}\n")


def main() -> None:
    options = arguments()
    due = due_matches(options.window_hours, options.post_kickoff_minutes)
    if due.empty and not options.force:
        print("No fixture is inside the matchday refresh window.")
        print("No files or production probabilities were changed.")
        return
    if not due.empty:
        columns = [
            column
            for column in ["kickoff_utc", "home_team", "away_team"]
            if column in due
        ]
        print("Fixtures inside refresh window:")
        print(due[columns].to_string(index=False))
    else:
        print("Forced matchday refresh requested.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        "refresh_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log"
    )
    with log_path.open("w", encoding="utf-8") as log:
        for label, relative_path in STEPS:
            run_step(label, relative_path, log)
    print("Matchday context refresh completed.")
    print(f"Log: {log_path}")
    print("The production model and frozen probabilities were not retrained.")


if __name__ == "__main__":
    main()
