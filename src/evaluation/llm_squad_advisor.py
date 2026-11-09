from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORTS = PROJECT_ROOT / "reports"
OUTPUT_JSON = REPORTS / "llm_squad_advice.json"
OUTPUT_MD = REPORTS / "llm_squad_advice.md"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save(payload: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# DeepSeek squad-feature research advice",
        "",
        f"- Status: **{payload.get('status', 'unknown')}**",
        "- Production model changed: **no**",
        "",
        str(payload.get("summary_zh") or payload.get("reason") or "No advice."),
    ]
    for experiment in payload.get("recommended_experiments", []):
        lines.extend(
            [
                "",
                f"## {experiment.get('title', 'Experiment')}",
                f"- Hypothesis: {experiment.get('hypothesis', '—')}",
                f"- Metric: {experiment.get('primary_metric', 'Log Loss')}",
                f"- Leakage guard: {experiment.get('leakage_guard', 'Chronological snapshots only')}",
            ]
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = read_json(REPORTS / "squad_training_dataset_audit.json")
    decision = read_json(REPORTS / "squad_challenger_decision.json")
    readiness = read_json(REPORTS / "squad_snapshot_readiness.json")
    try:
        from src.llm.deepseek_client import DeepSeekClient

        client = DeepSeekClient()
        if not client.available:
            raise RuntimeError(client.unavailable_reason())
        result = client.json_completion(
            system_prompt=(
                "你是足球概率模型实验设计顾问。只根据输入的审计和评估事实返回JSON，"
                "不得修改生产模型，不得宣称因果，不得建议使用开球后的信息。返回字段："
                "summary_zh、recommended_experiments。recommended_experiments是对象数组，"
                "每项含title、hypothesis、primary_metric、leakage_guard、risk。"
            ),
            user_payload={
                "dataset_audit": audit,
                "challenger_decision": decision,
                "snapshot_readiness": readiness,
            },
            temperature=0.1,
            max_tokens=1800,
        )
        payload = {
            "status": "success",
            **result,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "production_changed": False,
        }
    except Exception as error:
        payload = {
            "status": "skipped",
            "reason": str(error),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "production_changed": False,
        }
    save(payload)
    print(f"Squad advisor status: {payload['status']}")
    print(f"JSON: {OUTPUT_JSON}")
    print("Production model was not changed.")


if __name__ == "__main__":
    main()
