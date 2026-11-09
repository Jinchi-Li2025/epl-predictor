from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.deepseek_client import DeepSeekClient  # noqa: E402


METRICS_FILE = PROJECT_ROOT / "reports" / "champion_challenger_metrics.csv"
DECISION_FILE = PROJECT_ROOT / "reports" / "champion_challenger_decision.json"
PRODUCTION_FILE = PROJECT_ROOT / "configs" / "production_model.json"
POLICY_FILE = PROJECT_ROOT / "configs" / "champion_challenger_policy.json"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "llm_model_advice.json"
OUTPUT_MARKDOWN = PROJECT_ROOT / "reports" / "llm_model_advice.md"


SYSTEM_PROMPT = """
你是足球概率模型的实验设计审阅员。请根据输入的Champion/Challenger指标、决策门槛和生产配置，提出下一轮可验证实验。

强制边界：
1. 你只能提出实验建议，不能修改文件、模型权重、训练数据、生产配置或已发布概率。
2. 不得建议绕过时间顺序验证、最终留出集、bootstrap置信区间或人工晋升流程。
3. current_external_validation是样本外评估，不得用于直接调参后再宣称为样本外结果。
4. 每项实验必须写明假设、涉及文件、主要评价指标、风险和优先级。
5. 结论必须严格服从输入中的promotion_gates和promotion_mode。
6. 只返回简体中文JSON，不要输出Markdown或额外说明。

必须返回：
{
  "summary_zh": "string",
  "diagnosis": ["string"],
  "recommended_experiments": [
    {
      "title": "string",
      "hypothesis": "string",
      "files": ["string"],
      "primary_metric": "string",
      "risk": "string",
      "priority": "high|medium|low"
    }
  ],
  "do_not_change": ["string"],
  "promotion_recommendation": "string"
}
""".strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON根节点不是对象：{path}")
    return value


def safe_string_list(value: Any, limit: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        " ".join(item.split()).strip()[:500]
        for item in value
        if isinstance(item, str) and item.strip()
    ][:limit]


def validate_result(value: dict[str, Any]) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    raw_experiments = value.get("recommended_experiments", [])
    if isinstance(raw_experiments, list):
        for item in raw_experiments[:8]:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority", "medium")).lower()
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            experiments.append(
                {
                    "title": str(item.get("title", "未命名实验"))[:160],
                    "hypothesis": str(item.get("hypothesis", ""))[:700],
                    "files": safe_string_list(item.get("files"), limit=10),
                    "primary_metric": str(
                        item.get("primary_metric", "Log Loss")
                    )[:200],
                    "risk": str(item.get("risk", ""))[:500],
                    "priority": priority,
                }
            )

    return {
        "summary_zh": str(value.get("summary_zh", ""))[:1200],
        "diagnosis": safe_string_list(value.get("diagnosis")),
        "recommended_experiments": experiments,
        "do_not_change": safe_string_list(value.get("do_not_change")),
        "promotion_recommendation": str(
            value.get("promotion_recommendation", "保持当前生产模型。")
        )[:600],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    status = payload.get("status", "unknown")
    lines = [
        "# DeepSeek 模型实验建议",
        "",
        f"- 状态：`{status}`",
        f"- 生成时间：{payload.get('generated_at_utc', '')}",
        "- 安全边界：本报告不会自动修改权重、生产配置或已发布概率。",
        "",
    ]
    if status not in {"success", "cache_hit"}:
        lines.extend(
            [
                "DeepSeek本次未生成建议。",
                "",
                f"原因：{payload.get('reason', '')}",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## 总结",
            "",
            str(payload.get("summary_zh", "")),
            "",
            "## 诊断",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload.get("diagnosis", []))
    lines.extend(["", "## 推荐实验", ""])
    for index, experiment in enumerate(
        payload.get("recommended_experiments", []), start=1
    ):
        files = ", ".join(experiment.get("files", [])) or "待确定"
        lines.extend(
            [
                f"### {index}. {experiment.get('title', '')}",
                "",
                f"- 优先级：{experiment.get('priority', '')}",
                f"- 假设：{experiment.get('hypothesis', '')}",
                f"- 涉及文件：{files}",
                f"- 主要指标：{experiment.get('primary_metric', '')}",
                f"- 风险：{experiment.get('risk', '')}",
                "",
            ]
        )
    lines.extend(["## 不应自动改变", ""])
    lines.extend(f"- {item}" for item in payload.get("do_not_change", []))
    lines.extend(
        [
            "",
            "## 晋升意见",
            "",
            str(payload.get("promotion_recommendation", "")),
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUTPUT_MARKDOWN.write_text(render_markdown(payload), encoding="utf-8")


def main() -> None:
    required = [METRICS_FILE, DECISION_FILE, PRODUCTION_FILE, POLICY_FILE]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少模型顾问输入文件：\n"
            + "\n".join(str(path) for path in missing)
        )

    client = DeepSeekClient()
    generated_at = datetime.now(timezone.utc).isoformat()
    if not client.available:
        payload = {
            "status": "disabled",
            "reason": client.unavailable_reason(),
            "model": client.model,
            "generated_at_utc": generated_at,
            "production_changed": False,
        }
        write_outputs(payload)
        print(
            "DeepSeek模型顾问已跳过："
            + client.unavailable_reason()
        )
        print(f"状态文件：{OUTPUT_JSON}")
        return

    metrics_frame = pd.read_csv(METRICS_FILE)
    metric_records = []
    for record in metrics_frame.to_dict(orient="records"):
        metric_records.append(
            {
                key: (value if pd.notna(value) else None)
                for key, value in record.items()
            }
        )
    input_payload = {
        "champion_challenger_metrics": metric_records,
        "champion_challenger_decision": load_json(DECISION_FILE),
        "production_model": load_json(PRODUCTION_FILE),
        "promotion_policy": load_json(POLICY_FILE),
        "immutable_execution_policy": {
            "production_changes_allowed": False,
            "weight_changes_allowed": False,
            "published_probability_changes_allowed": False,
            "advice_only": True,
        },
    }
    try:
        raw_result = client.json_completion(
            system_prompt=SYSTEM_PROMPT,
            user_payload=input_payload,
            temperature=0.1,
            max_tokens=6000,
        )
        result = validate_result(raw_result)
        payload = {
            "status": (
                "cache_hit"
                if raw_result.get("_deepseek_cache_hit")
                else "success"
            ),
            "model": client.model,
            "generated_at_utc": generated_at,
            "production_changed": False,
            **result,
        }
    except RuntimeError as error:
        payload = {
            "status": "failed",
            "reason": str(error),
            "model": client.model,
            "generated_at_utc": generated_at,
            "production_changed": False,
        }

    write_outputs(payload)
    print("DeepSeek模型实验建议处理完成。")
    print(f"状态：{payload['status']}")
    print("生产模型未被修改。")
    print(f"JSON：{OUTPUT_JSON}")
    print(f"可读报告：{OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
