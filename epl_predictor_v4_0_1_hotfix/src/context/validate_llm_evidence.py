from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet"
REPORT_CSV = PROJECT_ROOT / "reports" / "prematch_report.csv"
REPORT_JSON = PROJECT_ROOT / "reports" / "prematch_report.json"
REGISTRY_FILE = PROJECT_ROOT / "data" / "assets" / "player_registry.parquet"
AUDIT_FILE = PROJECT_ROOT / "reports" / "llm_evidence_audit.csv"
OVERRIDE_FILE = PROJECT_ROOT / "data" / "manual" / "player_team_overrides.csv"


def normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_items(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value or "[]"))
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def source_records(row: pd.Series) -> list[dict]:
    sources: list[dict] = []
    for column in ["home_news_items_json", "away_news_items_json"]:
        for item in parse_items(row.get(column)):
            url = str(item.get("url") or item.get("link") or "").strip()
            title = str(item.get("title") or item.get("headline") or "").strip()
            if url or title:
                sources.append(
                    {
                        "title": title,
                        "url": url,
                        "source": str(item.get("source") or "").strip(),
                        "published_at": str(item.get("published_at") or "").strip(),
                    }
                )
    unique: dict[tuple[str, str], dict] = {}
    for source in sources:
        unique[(source["url"], source["title"])] = source
    return list(unique.values())


def player_aliases(registry: pd.DataFrame) -> dict[str, set[str]]:
    candidates: dict[str, list[tuple[int, str]]] = {}
    for _, player in registry.iterrows():
        team = str(player.get("team") or "").strip()
        if not team:
            continue
        fpl_id = player.get("fpl_id")
        has_fpl = fpl_id is not None and not pd.isna(fpl_id)
        registration_verified = str(
            player.get("registration_verified", False)
        ).strip().casefold() in {"true", "1", "yes"}
        profile_id = player.get("football_data_player_id")
        has_profile = profile_id is not None and not pd.isna(profile_id)
        priority = 30 if has_fpl else 20 if registration_verified else 10 if has_profile else 1
        names = {
            str(player.get(column) or "").strip()
            for column in ["display_name", "registered_name", "fpl_full_name", "web_name"]
        }
        for name in names:
            key = normalise(name)
            if len(key) >= 4:
                candidates.setdefault(key, []).append((priority, team))
            parts = key.split()
            if parts and len(parts[-1]) >= 5:
                candidates.setdefault(parts[-1], []).append((priority, team))

    if OVERRIDE_FILE.exists():
        overrides = pd.read_csv(OVERRIDE_FILE, encoding="utf-8-sig")
        if {"player_name", "team"}.issubset(overrides.columns):
            for _, row in overrides.iterrows():
                team = str(row.get("team") or "").strip()
                key = normalise(row.get("player_name"))
                if not team or not key:
                    continue
                candidates.setdefault(key, []).append((100, team))
                parts = key.split()
                if parts and len(parts[-1]) >= 5:
                    candidates.setdefault(parts[-1], []).append((100, team))

    aliases: dict[str, set[str]] = {}
    for alias, rows in candidates.items():
        maximum_priority = max(priority for priority, _ in rows)
        aliases[alias] = {
            team for priority, team in rows if priority == maximum_priority
        }
    return aliases


def mentioned(text_key: str, alias: str) -> bool:
    return re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", text_key) is not None


def main() -> None:
    if not REPORT_FILE.exists() or not REGISTRY_FILE.exists():
        print("LLM evidence validation skipped: report or player registry is missing.")
        return
    report = pd.read_parquet(REPORT_FILE).copy()
    registry = pd.read_parquet(REGISTRY_FILE).copy()
    aliases = player_aliases(registry)
    all_teams = sorted(registry["team"].dropna().astype(str).unique())
    audit_rows = []
    column_defaults = {
        "llm_evidence_status": "not_checked",
        "llm_safe_to_display": True,
        "llm_evidence_issues_json": "[]",
        "llm_evidence_sources_json": "[]",
        "llm_evidence_checked_at_utc": "",
    }
    for column, default in column_defaults.items():
        if column not in report.columns:
            report[column] = default

    for index, row in report.iterrows():
        home_team = str(row.get("home_team") or "")
        away_team = str(row.get("away_team") or "")
        permitted_teams = {home_team, away_team}
        brief = str(row.get("llm_prematch_brief") or "")
        points = str(row.get("llm_key_points_json") or "")
        text_key = normalise(brief + " " + points)
        issues: list[dict] = []

        for alias, teams in aliases.items():
            if not mentioned(text_key, alias):
                continue
            if teams & permitted_teams:
                continue
            if len(teams) == 1:
                player_team = next(iter(teams))
                issues.append(
                    {
                        "type": "foreign_player_entity",
                        "entity": alias,
                        "registered_team": player_team,
                    }
                )
        for team in all_teams:
            if team not in permitted_teams and mentioned(text_key, normalise(team)):
                issues.append(
                    {"type": "foreign_team_entity", "entity": team}
                )

        unique_issues = list({json.dumps(item, sort_keys=True): item for item in issues}.values())
        sources = source_records(row)
        if unique_issues:
            status = "blocked_cross_match_contamination"
            safe = False
        elif sources:
            status = "checked_with_sources"
            safe = True
        else:
            status = "checked_structured_data_only"
            safe = True

        report.at[index, "llm_evidence_status"] = status
        report.at[index, "llm_safe_to_display"] = safe
        report.at[index, "llm_evidence_issues_json"] = json.dumps(
            unique_issues, ensure_ascii=False
        )
        report.at[index, "llm_evidence_sources_json"] = json.dumps(
            sources, ensure_ascii=False
        )
        report.at[index, "llm_evidence_checked_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        audit_rows.append(
            {
                "match_id": row.get("match_id"),
                "home_team": home_team,
                "away_team": away_team,
                "status": status,
                "safe_to_display": safe,
                "issue_count": len(unique_issues),
                "source_count": len(sources),
                "issues": json.dumps(unique_issues, ensure_ascii=False),
            }
        )

    report.to_parquet(REPORT_FILE, index=False)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    REPORT_JSON.write_text(
        report.to_json(
            orient="records",
            force_ascii=False,
            indent=2,
            date_format="iso",
        ),
        encoding="utf-8",
    )
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")
    blocked = sum(not item["safe_to_display"] for item in audit_rows)
    print("LLM evidence validation completed.")
    print(f"Matches checked: {len(audit_rows)}")
    print(f"Briefs blocked for cross-match contamination: {blocked}")
    for item in audit_rows:
        if item["safe_to_display"]:
            continue
        print(
            "BLOCKED: "
            f"{item['home_team']} vs {item['away_team']} | "
            f"{item['issues']}"
        )
    print("Prediction probabilities were not modified.")


if __name__ == "__main__":
    main()
