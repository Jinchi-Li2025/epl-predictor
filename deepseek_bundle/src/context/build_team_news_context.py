from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "context"
    / "prematch_context.parquet"
)

CONTEXT_CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "context"
    / "prematch_context.csv"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "team_news_context_audit.csv"
)

MAX_PLAYERS_PER_TEAM = 4


def parse_json_list(value: Any) -> list[dict]:
    """安全读取伤停球员JSON列表。"""

    if value is None:
        return []

    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [
        item
        for item in parsed
        if isinstance(item, dict)
    ]


def first_available(
    item: dict,
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        value = item.get(key)

        if value is not None and str(value).strip():
            return value

    return None


def format_player_news(item: dict) -> str:
    """将一名球员的结构化状态转换为简短说明。"""

    player_name = first_available(
        item,
        (
            "player_name",
            "player",
            "name",
            "web_name",
        ),
    )

    if player_name is None:
        player_name = "未命名球员"

    status = first_available(
        item,
        (
            "status_description",
            "status",
        ),
    )

    chance = first_available(
        item,
        (
            "chance_of_playing_next_round",
            "chance",
            "playing_chance",
        ),
    )

    news = first_available(
        item,
        (
            "news",
            "reason",
            "description",
        ),
    )

    details = []

    if news is not None:
        details.append(str(news).strip())

    if chance is not None:
        try:
            chance_value = int(float(chance))
            details.append(
                f"下轮出场概率约{chance_value}%"
            )
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

        status_text = status_mapping.get(
            str(status).lower(),
            str(status),
        )

        details.append(status_text)

    if not details:
        details.append("存在可用性风险")

    return (
        f"{player_name}："
        + "；".join(details)
    )


def create_team_summary(
    absence_json: Any,
    injury_impact: Any,
) -> tuple[str, int]:
    absences = parse_json_list(absence_json)

    if absences:
        summaries = [
            format_player_news(item)
            for item in absences[
                :MAX_PLAYERS_PER_TEAM
            ]
        ]

        return (
            "；".join(summaries),
            len(absences),
        )

    try:
        impact = float(injury_impact)
    except (TypeError, ValueError):
        impact = 0.0

    if pd.isna(impact):
        impact = 0.0

    if impact > 0:
        return (
            "检测到一定球员可用性风险，"
            "但当前数据没有提供可展示的球员明细。",
            0,
        )

    return (
        "当前结构化数据未标记关键伤停或停赛。",
        0,
    )


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到赛前上下文文件：{CONTEXT_FILE}"
        )

    context = pd.read_parquet(
        CONTEXT_FILE
    ).copy()

    required_columns = {
        "match_id",
        "home_team",
        "away_team",
        "home_key_absences_json",
        "away_key_absences_json",
        "home_injury_impact",
        "away_injury_impact",
    }

    missing_columns = (
        required_columns - set(context.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "上下文缺少必要字段："
            f"{sorted(missing_columns)}"
        )

    checked_at = datetime.now(
        timezone.utc
    ).isoformat()

    audit_rows = []

    print(
        "开始生成结构化球队新闻摘要……"
    )

    for index, row in context.iterrows():
        print(
            f"{index + 1}/{len(context)} "
            f"{row['home_team']} vs "
            f"{row['away_team']}"
        )

        home_summary, home_players = (
            create_team_summary(
                row["home_key_absences_json"],
                row["home_injury_impact"],
            )
        )

        away_summary, away_players = (
            create_team_summary(
                row["away_key_absences_json"],
                row["away_injury_impact"],
            )
        )

        context.at[
            index,
            "home_news_summary",
        ] = home_summary

        context.at[
            index,
            "away_news_summary",
        ] = away_summary

        # 新闻摘要与伤停来自同一数据源，
        # 不增加第二次独立概率影响。
        context.at[
            index,
            "home_news_impact",
        ] = 0.0

        context.at[
            index,
            "away_news_impact",
        ] = 0.0

        context.at[
            index,
            "news_source",
        ] = (
            "Fantasy Premier League "
            "availability summary"
        )

        context.at[
            index,
            "news_checked_at_utc",
        ] = checked_at

        audit_rows.append(
            {
                "match_id": row["match_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_reported_players": (
                    home_players
                ),
                "away_reported_players": (
                    away_players
                ),
                "home_news_summary": (
                    home_summary
                ),
                "away_news_summary": (
                    away_summary
                ),
                "news_source": (
                    "Fantasy Premier League "
                    "availability summary"
                ),
                "checked_at_utc": checked_at,
            }
        )

    context.to_parquet(
        CONTEXT_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    audit = pd.DataFrame(audit_rows)

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n球队新闻上下文更新完成。"
    )
    print(f"比赛数量：{len(context)}")
    print(
        "说明：新闻摘要与伤停数据同源，"
        "不会重复调整预测概率。"
    )
    print(f"上下文文件：{CONTEXT_FILE}")
    print(f"新闻检查报告：{AUDIT_FILE}")

    print("\n球队新闻摘要：")

    print(
        context[
            [
                "home_team",
                "away_team",
                "home_news_summary",
                "away_news_summary",
                "news_source",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()