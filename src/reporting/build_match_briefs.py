from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "match_briefs"
INDEX_FILE = OUTPUT_DIR / "README.md"


def safe(value: Any, fallback: str = "—") -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    return text if text and text.casefold() not in {"nan", "none"} else fallback


def probability(row: pd.Series, candidates: list[str]) -> str:
    for column in candidates:
        if column in row and pd.notna(row.get(column)):
            value = float(row.get(column))
            if value > 1.0:
                value /= 100.0
            return f"{value:.1%}"
    return "—"


def sources(row: pd.Series) -> list[dict]:
    try:
        value = json.loads(str(row.get("llm_evidence_sources_json") or "[]"))
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        return []


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_markdown(row: pd.Series) -> str:
    home = safe(row.get("home_team"))
    away = safe(row.get("away_team"))
    evidence_status = safe(row.get("llm_evidence_status"), "not_checked")
    llm_safe = str(row.get("llm_safe_to_display", True)).lower() != "false"
    lines = [
        f"# {home} vs {away}",
        "",
        f"- Kickoff: {safe(row.get('kickoff_utc'))}",
        f"- Venue: {safe(row.get('venue_name'))}",
        f"- Prediction: **{safe(row.get('predicted_outcome'))}**",
        f"- Most likely score: **{safe(row.get('most_likely_score'))}**",
        f"- Home / Draw / Away: **{probability(row, ['home_probability', 'published_home_probability'])} / {probability(row, ['draw_probability', 'published_draw_probability'])} / {probability(row, ['away_probability', 'published_away_probability'])}**",
        "",
        "## Structured pre-match summary",
        "",
        safe(row.get("context_summary"), "No structured summary is available."),
        "",
        f"Evidence status: `{evidence_status}`",
    ]
    if llm_safe and safe(row.get("llm_prematch_brief"), ""):
        lines.extend(["", "## DeepSeek evidence-bounded interpretation", "", safe(row.get("llm_prematch_brief"))])
    elif not llm_safe:
        lines.extend(["", "> The AI narrative was hidden because cross-match entity contamination was detected."])
    source_items = sources(row)
    if source_items:
        lines.extend(["", "## Sources", ""])
        for item in source_items:
            title = safe(item.get("title"), safe(item.get("source"), "Source"))
            url = safe(item.get("url"), "")
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
    lines.extend(["", "---", "Probabilities are frozen model outputs; narrative context does not modify them."])
    return "\n".join(lines) + "\n"


def main() -> None:
    if not REPORT_FILE.exists():
        print("Match brief generation skipped: prematch report is missing.")
        return
    report = pd.read_parquet(REPORT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_lines = ["# EPL Predictor match briefs", ""]
    for _, row in report.iterrows():
        home = safe(row.get("home_team"))
        away = safe(row.get("away_team"))
        filename = f"{slug(home)}-vs-{slug(away)}.md"
        (OUTPUT_DIR / filename).write_text(build_markdown(row), encoding="utf-8")
        index_lines.append(f"- [{home} vs {away}]({filename})")
    INDEX_FILE.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Match briefs generated: {len(report)}")
    print(f"Index: {INDEX_FILE}")


if __name__ == "__main__":
    main()
