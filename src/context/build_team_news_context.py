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


CONTEXT_FILE = PROJECT_ROOT / "data" / "context" / "prematch_context.parquet"
CONTEXT_CSV_FILE = PROJECT_ROOT / "data" / "context" / "prematch_context.csv"
NEWS_FILE = PROJECT_ROOT / "data" / "raw" / "team_news" / "team_news.json"
AUDIT_FILE = PROJECT_ROOT / "reports" / "team_news_context_audit.csv"
MAX_PLAYERS_PER_TEAM = 4


SYSTEM_PROMPT = """
你是英超赛前情报编辑器。把输入JSON中的结构化伤停事实和公开新闻标题整理成简洁中文摘要。

严格规则：
1. 只能使用输入JSON明确提供的事实，不得补充常识、猜测、传闻细节或虚构球员、教练、事件。
2. 新闻只有标题、来源和链接，必须使用“据公开标题显示”“相关报道关注”等谨慎措辞，不得假装读过正文。
3. 不评价或修改模型概率、预测结果、模型权重、阵容强度数值或生产配置。
4. used_source_urls只能从输入的source_articles[].url中选择。
5. 信息不足时明确写“暂无足够公开新闻信息”，不要猜测。
6. 所有文字使用简体中文，保持中性、可核查。

只返回JSON对象，必须包含：
{
  "home_summary_zh": "string",
  "away_summary_zh": "string",
  "match_brief_zh": "string",
  "key_points": ["string"],
  "used_source_urls": ["string"]
}
""".strip()


def parse_json_list(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def first_available(item: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def format_player_news(item: dict) -> str:
    player_name = first_available(
        item, ("player_name", "player", "name", "web_name")
    ) or "未命名球员"
    status = first_available(item, ("status_description", "status"))
    chance = first_available(
        item,
        ("chance_of_playing_next_round", "chance", "playing_chance"),
    )
    news = first_available(item, ("news", "reason", "description"))
    details: list[str] = []
    if news is not None:
        details.append(str(news).strip())
    if chance is not None:
        try:
            details.append(f"下轮出场概率约{int(float(chance))}%")
        except (TypeError, ValueError):
            pass
    if not details and status is not None:
        status_mapping = {
            "a": "当前可出场",
            "d": "出场存疑",
            "i": "伤病缺阵",
            "n": "暂时无法出场",
            "s": "停赛",
            "u": "状态未知",
        }
        details.append(status_mapping.get(str(status).lower(), str(status)))
    if not details:
        details.append("存在可用性风险")
    return f"{player_name}：" + "；".join(details)


def create_team_summary(
    absence_json: Any,
    injury_impact: Any,
) -> tuple[str, int]:
    absences = parse_json_list(absence_json)
    if absences:
        summaries = [
            format_player_news(item)
            for item in absences[:MAX_PLAYERS_PER_TEAM]
        ]
        return "；".join(summaries), len(absences)
    try:
        impact = float(injury_impact)
    except (TypeError, ValueError):
        impact = 0.0
    if pd.isna(impact):
        impact = 0.0
    if impact > 0:
        return (
            "检测到一定球员可用性风险，但当前数据没有提供可展示的球员明细。",
            0,
        )
    return "当前结构化数据未标记关键伤停或停赛。", 0


def load_public_news() -> dict[str, list[dict]]:
    if not NEWS_FILE.exists():
        return {}
    try:
        payload = json.loads(NEWS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    teams = payload.get("teams", {}) if isinstance(payload, dict) else {}
    if not isinstance(teams, dict):
        return {}
    return {
        str(team): [item for item in items if isinstance(item, dict)]
        for team, items in teams.items()
        if isinstance(items, list)
    }


def clean_output_text(value: Any, fallback: str, max_length: int) -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split()).strip()
    return cleaned[:max_length] if cleaned else fallback


def validate_llm_result(
    result: dict[str, Any],
    *,
    fallback_home: str,
    fallback_away: str,
    allowed_urls: set[str],
) -> dict[str, Any]:
    key_points = result.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = []
    cleaned_points = [
        " ".join(point.split()).strip()[:220]
        for point in key_points
        if isinstance(point, str) and point.strip()
    ][:6]
    used_urls = result.get("used_source_urls", [])
    if not isinstance(used_urls, list):
        used_urls = []
    cleaned_urls = list(
        dict.fromkeys(
            url
            for url in used_urls
            if isinstance(url, str) and url in allowed_urls
        )
    )
    return {
        "home_summary_zh": clean_output_text(
            result.get("home_summary_zh"), fallback_home, 600
        ),
        "away_summary_zh": clean_output_text(
            result.get("away_summary_zh"), fallback_away, 600
        ),
        "match_brief_zh": clean_output_text(
            result.get("match_brief_zh"),
            "DeepSeek未生成可用的赛前情报解读。",
            900,
        ),
        "key_points": cleaned_points,
        "used_source_urls": cleaned_urls,
    }


def optional_value(row: pd.Series, key: str) -> Any:
    value = row.get(key)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(f"找不到赛前上下文文件：{CONTEXT_FILE}")
    context = pd.read_parquet(CONTEXT_FILE).copy()
    required_columns = {
        "match_id",
        "home_team",
        "away_team",
        "home_key_absences_json",
        "away_key_absences_json",
        "home_injury_impact",
        "away_injury_impact",
    }
    missing_columns = required_columns - set(context.columns)
    if missing_columns:
        raise RuntimeError(f"上下文缺少必要字段：{sorted(missing_columns)}")

    checked_at = datetime.now(timezone.utc).isoformat()
    public_news = load_public_news()
    client = DeepSeekClient()
    audit_rows: list[dict] = []
    print("开始生成球队新闻与DeepSeek赛前摘要……")
    if not client.available:
        print(
            "DeepSeek未启用（"
            + client.unavailable_reason()
            + "），本次使用确定性摘要继续运行。"
        )

    for position, (index, row) in enumerate(context.iterrows(), start=1):
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        print(f"{position}/{len(context)} {home_team} vs {away_team}")
        fallback_home, home_players = create_team_summary(
            row["home_key_absences_json"], row["home_injury_impact"]
        )
        fallback_away, away_players = create_team_summary(
            row["away_key_absences_json"], row["away_injury_impact"]
        )
        home_articles = public_news.get(home_team, [])
        away_articles = public_news.get(away_team, [])
        source_articles = home_articles + away_articles
        allowed_urls = {
            str(item.get("url"))
            for item in source_articles
            if item.get("url")
        }
        llm_status = client.unavailable_reason()
        llm_error = ""
        llm_result = {
            "home_summary_zh": fallback_home,
            "away_summary_zh": fallback_away,
            "match_brief_zh": "",
            "key_points": [],
            "used_source_urls": [],
        }

        if client.available:
            try:
                raw_result = client.json_completion(
                    system_prompt=SYSTEM_PROMPT,
                    user_payload={
                        "match": {
                            "match_id": str(row["match_id"]),
                            "home_team": home_team,
                            "away_team": away_team,
                            "kickoff_utc": optional_value(row, "kickoff_utc"),
                            "venue_name": optional_value(row, "venue_name"),
                        },
                        "structured_availability": {
                            "home_summary": fallback_home,
                            "away_summary": fallback_away,
                            "home_injury_impact": optional_value(
                                row, "home_injury_impact"
                            ),
                            "away_injury_impact": optional_value(
                                row, "away_injury_impact"
                            ),
                        },
                        "other_verified_context": {
                            "lineup_status": optional_value(row, "lineup_status"),
                            "weather_description": optional_value(
                                row, "weather_description"
                            ),
                            "importance_reason": optional_value(
                                row, "importance_reason"
                            ),
                            "referee_name": optional_value(row, "referee_name"),
                        },
                        "source_articles": source_articles,
                        "non_negotiable_policy": {
                            "probability_changes_allowed": False,
                            "model_weight_changes_allowed": False,
                        },
                    },
                    max_tokens=3000,
                )
                llm_result = validate_llm_result(
                    raw_result,
                    fallback_home=fallback_home,
                    fallback_away=fallback_away,
                    allowed_urls=allowed_urls,
                )
                llm_status = (
                    "cache_hit"
                    if raw_result.get("_deepseek_cache_hit")
                    else "success"
                )
            except RuntimeError as error:
                llm_status = "failed_fallback"
                llm_error = str(error)

        context.at[index, "home_news_summary"] = llm_result[
            "home_summary_zh"
        ]
        context.at[index, "away_news_summary"] = llm_result[
            "away_summary_zh"
        ]
        context.at[index, "home_news_items_json"] = json.dumps(
            home_articles, ensure_ascii=False
        )
        context.at[index, "away_news_items_json"] = json.dumps(
            away_articles, ensure_ascii=False
        )
        context.at[index, "llm_prematch_brief"] = llm_result[
            "match_brief_zh"
        ]
        context.at[index, "llm_key_points_json"] = json.dumps(
            llm_result["key_points"], ensure_ascii=False
        )
        context.at[index, "llm_used_source_urls_json"] = json.dumps(
            llm_result["used_source_urls"], ensure_ascii=False
        )
        context.at[index, "llm_status"] = llm_status
        context.at[index, "llm_model"] = client.model
        context.at[index, "llm_checked_at_utc"] = checked_at
        context.at[index, "home_news_impact"] = 0.0
        context.at[index, "away_news_impact"] = 0.0
        context.at[index, "news_source"] = (
            "FPL availability + Google News RSS headlines"
        )
        context.at[index, "news_checked_at_utc"] = checked_at

        audit_rows.append(
            {
                "match_id": row["match_id"],
                "home_team": home_team,
                "away_team": away_team,
                "home_reported_players": home_players,
                "away_reported_players": away_players,
                "home_article_count": len(home_articles),
                "away_article_count": len(away_articles),
                "home_news_summary": llm_result["home_summary_zh"],
                "away_news_summary": llm_result["away_summary_zh"],
                "llm_status": llm_status,
                "llm_model": client.model,
                "llm_used_source_urls_json": json.dumps(
                    llm_result["used_source_urls"], ensure_ascii=False
                ),
                "llm_error": llm_error,
                "checked_at_utc": checked_at,
            }
        )

    context.to_parquet(CONTEXT_FILE, index=False)
    context.to_csv(CONTEXT_CSV_FILE, index=False, encoding="utf-8-sig")
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        AUDIT_FILE, index=False, encoding="utf-8-sig"
    )
    status_counts = pd.Series(
        [row["llm_status"] for row in audit_rows]
    ).value_counts()
    print("\n球队新闻上下文更新完成。")
    print(f"比赛数量：{len(context)}")
    print("DeepSeek状态：")
    print(status_counts.to_string())
    print(
        "安全边界：新闻和AI摘要仅用于解释，"
        "home_news_impact与away_news_impact固定为0，"
        "不会修改预测概率。"
    )
    print(f"上下文文件：{CONTEXT_FILE}")
    print(f"新闻检查报告：{AUDIT_FILE}")


if __name__ == "__main__":
    main()
