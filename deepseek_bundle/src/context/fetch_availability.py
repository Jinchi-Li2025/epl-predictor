from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests


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

RAW_FPL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fpl"
    / "bootstrap_static.json"
)

AVAILABILITY_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "availability_context_audit.csv"
)

CONTEXT_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_context_audit.csv"
)

FPL_URL = (
    "https://fantasy.premierleague.com/"
    "api/bootstrap-static/"
)

TEAM_ALIASES = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Leeds": "Leeds United",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

STATUS_NAMES = {
    "a": "可出场",
    "d": "出场存疑",
    "i": "受伤",
    "n": "无法出场",
    "s": "停赛",
    "u": "不可用",
}


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None or pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    return int(
        round(
            safe_float(value, default)
        )
    )


def canonical_team_name(
    team_name: str,
) -> str:
    name = str(team_name).strip()
    return TEAM_ALIASES.get(name, name)


def availability_risk(
    status: str,
    chance: Any,
) -> float:
    if chance is not None and not pd.isna(chance):
        chance_value = max(
            0.0,
            min(100.0, float(chance)),
        )
        return 1.0 - chance_value / 100.0

    default_risks = {
        "a": 0.0,
        "d": 0.5,
        "i": 0.9,
        "n": 1.0,
        "s": 1.0,
        "u": 1.0,
    }

    return default_risks.get(status, 0.5)


def fetch_fpl_data() -> dict[str, Any]:
    try:
        response = requests.get(
            FPL_URL,
            headers={
                "User-Agent": (
                    "EPL-Predictor/"
                    "1.0 personal research"
                )
            },
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"连接英超Fantasy数据失败：{error}"
        ) from error

    if not response.ok:
        raise RuntimeError(
            "英超Fantasy接口返回"
            f"HTTP {response.status_code}。"
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            "英超Fantasy接口没有返回有效JSON。"
        ) from error

    if not payload.get("teams"):
        raise RuntimeError(
            "Fantasy数据中没有球队列表。"
        )

    if not payload.get("elements"):
        raise RuntimeError(
            "Fantasy数据中没有球员列表。"
        )

    return payload


def is_roster_exit_news(
    news_value: Any,
) -> bool:
    """
    判断新闻是否表示球员已经永久转会或租借离队。

    这些球员已经不属于当前可用阵容，
    不应作为伤停或停赛计算。
    """

    text = str(
        news_value or ""
    ).strip().lower()

    if not text:
        return False

    loan_phrases = (
        "on loan",
        "loan until",
        "loan for the rest",
        "loaned to",
        "loan move",
    )

    permanent_exit = (
        "permanently" in text
        and (
            "joined" in text
            or "signed for" in text
            or "moved to" in text
        )
    )

    other_exit_phrases = (
        "has left the club",
        "left the club",
        "has departed the club",
        "departed the club",
        "as a free agent",
        "released by the club",
        "contract terminated",
        "transferred to",
)

    return (
        permanent_exit
        or any(
            phrase in text
            for phrase in loan_phrases
        )
        or any(
            phrase in text
            for phrase in other_exit_phrases
        )
    )

def build_team_availability(
    players: list[dict[str, Any]],
) -> dict[str, Any]:
    # 先排除已经永久转会或租借离队的球员。
    # 他们不能进入球队有效阵容总分钟数，
    # 也不能作为当前伤停计算。
    active_players = [
        player
        for player in players
        if not is_roster_exit_news(
            player.get("news")
        )
    ]

    total_team_minutes = sum(
        safe_float(player.get("minutes"))
        for player in active_players
    )

    denominator = max(
        total_team_minutes,
        1.0,
    )

    flagged_players = []

    for player in active_players:
        status = str(
            player.get("status") or "a"
        ).lower()

        chance = player.get(
            "chance_of_playing_next_round"
        )

        risk = availability_risk(
            status=status,
            chance=chance,
        )

        news = str(
            player.get("news") or ""
        ).strip()

        if (
            status == "a"
            and risk <= 0.0
        ):
            continue

        minutes = safe_float(
            player.get("minutes")
        )

        minute_share = (
            minutes / denominator
        )

        impact_contribution = (
            risk * minute_share
        )

        first_name = str(
            player.get("first_name") or ""
        ).strip()

        second_name = str(
            player.get("second_name") or ""
        ).strip()

        full_name = (
            f"{first_name} {second_name}"
        ).strip()

        if not full_name:
            full_name = str(
                player.get("web_name")
                or "Unknown"
            )

        flagged_players.append(
            {
                "player": full_name,
                "display_name": (
                    player.get("web_name")
                ),
                "status": status,
                "status_description": (
                    STATUS_NAMES.get(
                        status,
                        status,
                    )
                ),
                "chance_of_playing": (
                    None
                    if chance is None
                    else safe_int(chance)
                ),
                "news": news or None,
                "news_added": (
                    player.get("news_added")
                ),
                "minutes": safe_int(minutes),
                "starts": safe_int(
                    player.get("starts")
                ),
                "total_points": safe_int(
                    player.get("total_points")
                ),
                "availability_risk": round(
                    risk,
                    4,
                ),
                "impact_contribution": round(
                    impact_contribution,
                    4,
                ),
                "source": (
                    "Fantasy Premier League"
                ),
            }
        )

    flagged_players.sort(
        key=lambda player: (
            player["impact_contribution"],
            player["minutes"],
        ),
        reverse=True,
    )

    team_impact = min(
        sum(
            player["impact_contribution"]
            for player in flagged_players
        ),
        1.0,
    )

    return {
        "players": flagged_players[:10],
        "impact": round(team_impact, 4),
        "flagged_player_count": len(
            flagged_players
        ),
    }


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到上下文文件：{CONTEXT_FILE}"
        )

    context = pd.read_parquet(
        CONTEXT_FILE
    )

    if context.empty:
        raise RuntimeError(
            "上下文文件为空。"
        )

    print(
        "正在下载英超官方Fantasy球员状态……"
    )

    payload = fetch_fpl_data()

    RAW_FPL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_FPL_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    team_id_to_name = {
        int(team["id"]): canonical_team_name(
            team["name"]
        )
        for team in payload["teams"]
    }

    players_by_team: dict[
        str,
        list[dict[str, Any]],
    ] = {
        team_name: []
        for team_name in team_id_to_name.values()
    }

    for player in payload["elements"]:
        team_id = safe_int(
            player.get("team")
        )

        team_name = team_id_to_name.get(
            team_id
        )

        if team_name is not None:
            players_by_team.setdefault(
                team_name,
                [],
            ).append(player)

    availability_by_team = {
        team_name: build_team_availability(
            players
        )
        for team_name, players
        in players_by_team.items()
    }

    checked_at = pd.Timestamp.now(
        tz="UTC"
    )

    if "availability_source" not in context:
        context["availability_source"] = None

    if (
        "availability_checked_at_utc"
        not in context
    ):
        context[
            "availability_checked_at_utc"
        ] = None

    audit_rows = []

    print("开始写入比赛伤停上下文……")

    for index, row in context.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        print(
            f"{index + 1}/{len(context)} "
            f"{home_team} vs {away_team}"
        )

        home_data = availability_by_team.get(
            home_team
        )

        away_data = availability_by_team.get(
            away_team
        )

        status = "available"

        if home_data is None or away_data is None:
            status = "team_not_found"

        if home_data is not None:
            context.at[
                index,
                "home_key_absences_json",
            ] = json.dumps(
                home_data["players"],
                ensure_ascii=False,
            )

            context.at[
                index,
                "home_injury_impact",
            ] = home_data["impact"]

        if away_data is not None:
            context.at[
                index,
                "away_key_absences_json",
            ] = json.dumps(
                away_data["players"],
                ensure_ascii=False,
            )

            context.at[
                index,
                "away_injury_impact",
            ] = away_data["impact"]

        if status == "available":
            context.at[
                index,
                "availability_source",
            ] = (
                "Fantasy Premier League "
                "bootstrap-static"
            )

            context.at[
                index,
                "availability_checked_at_utc",
            ] = checked_at.isoformat()

            context.at[
                index,
                "updated_at_utc",
            ] = checked_at.isoformat()

        audit_rows.append(
            {
                "match_id": row["match_id"],
                "home_team": home_team,
                "away_team": away_team,
                "home_team_found": (
                    home_data is not None
                ),
                "away_team_found": (
                    away_data is not None
                ),
                "home_flagged_players": (
                    None
                    if home_data is None
                    else home_data[
                        "flagged_player_count"
                    ]
                ),
                "away_flagged_players": (
                    None
                    if away_data is None
                    else away_data[
                        "flagged_player_count"
                    ]
                ),
                "home_injury_impact": (
                    None
                    if home_data is None
                    else home_data["impact"]
                ),
                "away_injury_impact": (
                    None
                    if away_data is None
                    else away_data["impact"]
                ),
                "status": status,
                "checked_at_utc": (
                    checked_at.isoformat()
                ),
            }
        )

    weather_complete = (
        context["temperature_c"].notna()
        & context["wind_speed_kph"].notna()
        & context["weather_source"].notna()
    )

    referee_complete = (
        context["referee_name"].notna()
        & context["referee_source"].notna()
    )

    importance_complete = (
        context["home_importance"].notna()
        & context["away_importance"].notna()
        & context["importance_reason"].notna()
    )

    availability_complete = (
        context["availability_source"].notna()
        & context[
            "availability_checked_at_utc"
        ].notna()
    )

    completed_sections = (
        weather_complete.astype(int)
        + referee_complete.astype(int)
        + importance_complete.astype(int)
        + availability_complete.astype(int)
    )

    context["context_confidence"] = (
        completed_sections / 7
    )

    context.loc[
        completed_sections > 0,
        "context_status",
    ] = "partial"

    context.to_parquet(
        CONTEXT_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    availability_audit = pd.DataFrame(
        audit_rows
    )

    availability_audit.to_csv(
        AVAILABILITY_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    context_audit = context[
        [
            "home_team",
            "away_team",
        ]
    ].copy()

    context_audit["completed_sections"] = (
        completed_sections
    )
    context_audit["total_sections"] = 7
    context_audit["completeness"] = (
        completed_sections / 7
    )

    context_audit.to_csv(
        CONTEXT_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    available_matches = int(
        availability_audit["status"]
        .eq("available")
        .sum()
    )

    missing_matches = int(
        availability_audit["status"]
        .eq("team_not_found")
        .sum()
    )

    print("\n伤停上下文更新完成。")
    print(f"比赛数量：{len(context)}")
    print(
        f"球队全部匹配：{available_matches}"
    )
    print(
        f"存在未匹配球队：{missing_matches}"
    )
    print(
        f"上下文文件：{CONTEXT_FILE}"
    )
    print(
        f"伤停检查报告："
        f"{AVAILABILITY_AUDIT_FILE}"
    )

    print("\n伤停影响：")
    print(
        context[
            [
                "home_team",
                "away_team",
                "home_injury_impact",
                "away_injury_impact",
                "availability_source",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()