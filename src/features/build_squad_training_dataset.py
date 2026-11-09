from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "history" / "squad_snapshots"
MANIFEST_FILE = SNAPSHOT_ROOT / "manifest.csv"
SETTLED_FILE = PROJECT_ROOT / "data" / "history" / "settled_predictions.parquet"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "squad_challenger_training.parquet"
)
AUDIT_FILE = PROJECT_ROOT / "reports" / "squad_training_dataset_audit.json"


def probability_columns(frame: pd.DataFrame) -> dict[str, str]:
    candidates = [
        {
            "baseline_home_probability": "home_probability",
            "baseline_draw_probability": "draw_probability",
            "baseline_away_probability": "away_probability",
        },
        {
            "baseline_home_probability": "published_home_probability",
            "baseline_draw_probability": "published_draw_probability",
            "baseline_away_probability": "published_away_probability",
        },
    ]
    for mapping in candidates:
        if all(source in frame.columns for source in mapping.values()):
            return mapping
    return {}


def actual_result_column(frame: pd.DataFrame) -> str | None:
    for column in ["actual_result", "actual_result_code", "result"]:
        if column in frame.columns:
            return column
    return None


def empty_audit(reason: str, snapshot_count: int = 0) -> None:
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.write_text(
        json.dumps(
            {
                "status": "insufficient_history",
                "reason": reason,
                "snapshot_count": snapshot_count,
                "training_rows": 0,
                "post_kickoff_rows_excluded": 0,
                "production_model_changed": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(reason)


def main() -> None:
    if not MANIFEST_FILE.exists():
        empty_audit("No squad snapshot manifest is available yet.")
        return
    if not SETTLED_FILE.exists():
        empty_audit("No settled prediction history is available yet.")
        return

    manifest = pd.read_csv(MANIFEST_FILE)
    snapshot_rows = []
    excluded_post_kickoff = 0
    for _, item in manifest.iterrows():
        snapshot_id = str(item.get("snapshot_id") or "").strip()
        snapshot_dir = SNAPSHOT_ROOT / snapshot_id
        feature_file = snapshot_dir / "squad_features.parquet"
        report_file = snapshot_dir / "prematch_report.parquet"
        if not feature_file.exists() or not report_file.exists():
            continue

        features = pd.read_parquet(feature_file).copy()
        report = pd.read_parquet(report_file).copy()
        if "match_id" not in features or "match_id" not in report:
            continue
        mapping = probability_columns(report)
        report_columns = [
            column
            for column in [
                "match_id",
                "kickoff_utc",
                "home_team",
                "away_team",
                *mapping.values(),
            ]
            if column in report.columns
        ]
        merged = features.merge(
            report[report_columns].drop_duplicates("match_id"),
            on="match_id",
            how="left",
            suffixes=("", "_report"),
        )
        for target, source in mapping.items():
            merged[target] = pd.to_numeric(merged[source], errors="coerce")
        merged["snapshot_at_utc"] = pd.to_datetime(
            item.get("created_at_utc"), utc=True, errors="coerce"
        )
        kickoff_source = (
            "kickoff_utc"
            if "kickoff_utc" in merged.columns
            else "kickoff_utc_report"
        )
        merged["kickoff_utc"] = pd.to_datetime(
            merged[kickoff_source], utc=True, errors="coerce"
        )
        pre_kickoff = merged["snapshot_at_utc"].lt(merged["kickoff_utc"])
        excluded_post_kickoff += int((~pre_kickoff).sum())
        snapshot_rows.append(merged.loc[pre_kickoff].copy())

    if not snapshot_rows:
        empty_audit(
            "Snapshots exist, but no valid pre-kickoff rows were found.",
            snapshot_count=len(manifest),
        )
        return

    combined = pd.concat(snapshot_rows, ignore_index=True)
    combined = combined.sort_values(["match_id", "snapshot_at_utc"])
    combined = combined.drop_duplicates("match_id", keep="last")

    settled = pd.read_parquet(SETTLED_FILE).copy()
    result_column = actual_result_column(settled)
    if result_column is None or "match_id" not in settled:
        empty_audit(
            "Settled history does not contain match_id and actual result fields.",
            snapshot_count=len(manifest),
        )
        return
    settled = settled[["match_id", result_column]].drop_duplicates(
        "match_id", keep="last"
    )
    settled = settled.rename(columns={result_column: "actual_result"})
    output = combined.merge(settled, on="match_id", how="inner")
    output["actual_result"] = output["actual_result"].astype(str).str.upper()
    output = output.loc[output["actual_result"].isin(["H", "D", "A"])].copy()

    required_probabilities = [
        "baseline_home_probability",
        "baseline_draw_probability",
        "baseline_away_probability",
    ]
    if not all(column in output for column in required_probabilities):
        empty_audit(
            "Snapshot reports do not contain the baseline probability columns.",
            snapshot_count=len(manifest),
        )
        return
    output = output.dropna(subset=required_probabilities + ["kickoff_utc"])
    output = output.sort_values("kickoff_utc").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_FILE, index=False)
    unique_dates = int(output["snapshot_at_utc"].dt.date.nunique())
    candidate_features = [
        column for column in output.columns if column.startswith("squad_diff_")
    ]
    audit = {
        "status": (
            "candidate_backtest_ready"
            if len(output) >= 80 and unique_dates >= 8
            else "insufficient_history"
        ),
        "snapshot_count": int(len(manifest)),
        "unique_snapshot_dates": unique_dates,
        "training_rows": int(len(output)),
        "candidate_feature_count": len(candidate_features),
        "post_kickoff_rows_excluded": excluded_post_kickoff,
        "selection_rule": "latest snapshot strictly before kickoff",
        "production_model_changed": False,
    }
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Squad challenger training dataset built.")
    print(f"Rows: {len(output)}")
    print(f"Candidate features: {len(candidate_features)}")
    print(f"Status: {audit['status']}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
