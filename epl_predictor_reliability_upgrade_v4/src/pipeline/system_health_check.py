from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "system_health.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "system_health.md"

CHECKS = {
    "player_registry": PROJECT_ROOT / "data" / "assets" / "player_registry.parquet",
    "prematch_context": PROJECT_ROOT / "data" / "context" / "prematch_context.parquet",
    "prematch_report": PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet",
    "prediction_ledger": PROJECT_ROOT / "reports" / "prediction_ledger.csv",
    "model_center": PROJECT_ROOT / "pages" / "1_模型中心.py",
    "player_center": PROJECT_ROOT / "pages" / "3_球员资料库.py",
    "llm_evidence_audit": PROJECT_ROOT / "reports" / "llm_evidence_audit.csv",
    "model_monitor": PROJECT_ROOT / "reports" / "model_monitor.json",
}

MAX_AGE_HOURS = {
    "player_registry": 36.0,
    "prematch_context": 12.0,
    "prematch_report": 12.0,
    "prediction_ledger": 72.0,
    "llm_evidence_audit": 12.0,
    "model_monitor": 36.0,
}


def load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


def row_count(path: Path) -> int | None:
    try:
        if path.suffix == ".parquet":
            return int(len(pd.read_parquet(path)))
        if path.suffix == ".csv":
            return int(len(pd.read_csv(path)))
    except Exception:
        return None
    return None


def main() -> None:
    load_dotenv()
    now = datetime.now(timezone.utc)
    file_checks = []
    for name, path in CHECKS.items():
        exists = path.exists()
        modified = (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if exists
            else None
        )
        age_hours = (
            (now - modified).total_seconds() / 3600.0 if modified else None
        )
        if not exists:
            status = "missing"
        elif name in MAX_AGE_HOURS and age_hours is not None and age_hours > MAX_AGE_HOURS[name]:
            status = "stale"
        else:
            status = "ok"
        file_checks.append(
            {
                "name": name,
                "path": str(path),
                "exists": exists,
                "rows": row_count(path) if exists else None,
                "modified_at_utc": modified.isoformat() if modified else None,
                "age_hours": round(age_hours, 2) if age_hours is not None else None,
                "maximum_age_hours": MAX_AGE_HOURS.get(name),
                "status": status,
            }
        )

    environment = {
        "deepseek_api_key_configured": bool(
            os.getenv("DEEPSEEK_API_KEY", "").strip()
        ),
        "football_data_api_key_configured": bool(
            os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
            or os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
        ),
        "api_football_key_configured": bool(
            os.getenv("API_FOOTBALL_KEY", "").strip()
        ),
    }
    attention_required = [
        item["name"] for item in file_checks if item["status"] != "ok"
    ]
    payload = {
        "status": "healthy" if not attention_required else "attention_required",
        "generated_at_utc": now.isoformat(),
        "file_checks": file_checks,
        "environment": environment,
        "attention_required": attention_required,
        "production_model_changed": False,
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# EPL Predictor system health",
        "",
        f"- Status: **{payload['status']}**",
        f"- Generated: {payload['generated_at_utc']}",
        "- This check never changes the production model.",
        "",
        "## Files",
        "",
        "| Name | Status | Rows | Age hours |",
        "|---|---:|---:|---:|",
    ]
    for item in file_checks:
        lines.append(
            f"| {item['name']} | {item['status']} | "
            f"{item['rows'] if item['rows'] is not None else '-'} | "
            f"{item['age_hours'] if item['age_hours'] is not None else '-'} |"
        )
    lines.extend(["", "## API configuration", ""])
    for key, configured in environment.items():
        lines.append(f"- {key}: {'yes' if configured else 'no'}")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("System health check completed.")
    print(f"Status: {payload['status']}")
    print(f"JSON: {OUTPUT_JSON}")
    print("No API key values were printed.")


if __name__ == "__main__":
    main()
