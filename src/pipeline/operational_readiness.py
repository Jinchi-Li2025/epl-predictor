from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
OUTPUT_JSON = REPORTS / "operational_readiness.json"
OUTPUT_MD = REPORTS / "operational_readiness.md"


def read_json(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    health = read_json("system_health.json")
    monitor = read_json("model_monitor.json")
    registry = read_json("player_registry_quality.json")
    contract = read_json("pipeline_contract_audit.json")
    calibration = read_json("calibration_challenger_decision.json")
    snapshots = read_json("squad_snapshot_readiness.json")
    squad = read_json("squad_challenger_decision.json")

    evidence_file = REPORTS / "llm_evidence_audit.csv"
    blocked_briefs = None
    if evidence_file.exists():
        evidence = pd.read_csv(evidence_file)
        if "safe_to_display" in evidence:
            safe = evidence["safe_to_display"].astype(str).str.casefold().isin(
                {"true", "1", "yes"}
            )
            blocked_briefs = int((~safe).sum())

    blockers: list[str] = []
    warnings: list[str] = []
    if not health:
        blockers.append("System health report is missing.")
    elif health.get("status") != "healthy":
        blockers.append("System health requires attention.")
    if contract.get("status") == "blocked":
        blockers.append("Pipeline data-contract audit found errors.")
    elif contract.get("status") == "attention_required":
        warnings.append("Pipeline data-contract audit contains warnings.")
    if registry.get("status") == "attention_required":
        warnings.append("Player registry quality requires review.")
    if monitor.get("status") == "attention_required":
        warnings.append("Model drift/calibration monitoring requires attention.")
    if blocked_briefs:
        warnings.append(
            f"{blocked_briefs} AI brief(s) are safely hidden pending evidence review."
        )

    serving_status = "blocked" if blockers else "ready_with_warnings" if warnings else "ready"
    experiment_status = {
        "calibration": calibration.get("status", "not_available"),
        "squad_features": squad.get("status", "not_available"),
        "snapshots": snapshots.get("status", "not_available"),
    }
    payload = {
        "serving_status": serving_status,
        "safe_to_serve_existing_predictions": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "blocked_ai_briefs": blocked_briefs,
        "experiment_status": experiment_status,
        "production_model_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# EPL Predictor operational readiness",
        "",
        f"- Serving status: **{serving_status}**",
        f"- Existing predictions safe to serve: **{not blockers}**",
        "- Production model changed: **False**",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- {item}" for item in blockers] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.extend(["", "## Shadow experiments", ""])
    for name, status in experiment_status.items():
        lines.append(f"- {name}: {status}")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Operational readiness gate completed.")
    print(f"Serving status: {serving_status}")
    print(f"Blockers: {len(blockers)}; warnings: {len(warnings)}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
