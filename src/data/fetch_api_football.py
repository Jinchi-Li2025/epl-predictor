from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "api_football"
    / "epl_2026_27_fixtures.json"
)

NORMALIZED_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "api_football_epl_2026_27.parquet"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "api_football_audit.csv"
)

API_URL = "https://v3.football.api-sports.io/fixtures"
PREMIER_LEAGUE_ID = 39
API_SEASON = 2026
PROJECT_SEASON = "2026-27"

COMPLETED_STATUSES = {
    "FT",
    "AET",
    "PEN",
}

UNPLAYED_STATUSES = {
    "TBD",
    "NS",
    "PST",
    "CANC",
    "ABD",
    "AWD",
    "WO",
    "SUSP",
    "INT",
}


def normalise_team_name(value: Any) -> str:
    name = re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()

    name = re.sub(
        r"\s+(?:FC|AFC)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    if not name:
        raise ValueError(
            f"无法识别球队名称：{value!r}"
        )

    return name


def make_team_id(team_name: str) -> str:
    value = team_name.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def make_match_id(
    match_date: str,
    home_team: str,
    away_team: str,
) -> str:
    raw_value = (
        f"EPL|{PROJECT_SEASON}|{match_date}|"
        f"{home_team}|{away_team}"
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"epl-{digest}"


def parse_matchweek(value: Any) -> int | None:
    if value is None:
        return None

    numbers = re.findall(
        r"\d+",
        str(value),
    )

    if not numbers:
        return None

    return int(numbers[-1])


def determine_result(
    home_goals: int,
    away_goals: int,
) -> str:
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def request_fixtures(api_key: str) -> tuple[dict, dict]:
    response = requests.get(
        API_URL,
        headers={
            "x-apisports-key": api_key,
        },
        params={
            "league": PREMIER_LEAGUE_ID,
            "season": API_SEASON,
            "timezone": "Europe/London",
        },
        timeout=30,
    )

    quota_information = {
        "daily_limit": response.headers.get(
            "x-ratelimit-requests-limit"
        ),
        "daily_remaining": response.headers.get(
            "x-ratelimit-requests-remaining"
        ),
    }

    if response.status_code != 200:
        raise RuntimeError(
            "API请求失败："
            f"HTTP {response.status_code}\n"
            f"{response.text[:500]}"
        )

    payload = response.json()

    errors = payload.get("errors")

    if errors:
        raise RuntimeError(
            f"API返回错误：{errors}"
        )

    fixtures = payload.get("response")

    if not isinstance(fixtures, list):
        raise RuntimeError(
            "API响应中没有有效的response列表。"
        )

    return payload, quota_information


def normalise_fixture(
    item: dict,
) -> dict:
    fixture = item.get("fixture") or {}
    league = item.get("league") or {}
    teams = item.get("teams") or {}
    goals = item.get("goals") or {}

    home_data = teams.get("home") or {}
    away_data = teams.get("away") or {}
    status_data = fixture.get("status") or {}
    venue_data = fixture.get("venue") or {}

    kickoff_utc = pd.to_datetime(
        fixture.get("date"),
        utc=True,
        errors="raise",
    )

    match_date = kickoff_utc.date().isoformat()

    home_team = normalise_team_name(
        home_data.get("name")
    )

    away_team = normalise_team_name(
        away_data.get("name")
    )

    status_short = str(
        status_data.get("short") or "UNKNOWN"
    )

    home_goals_raw = goals.get("home")
    away_goals_raw = goals.get("away")

    if status_short in COMPLETED_STATUSES:
        if (
            home_goals_raw is None
            or away_goals_raw is None
        ):
            raise RuntimeError(
                "已完赛比赛缺少比分："
                f"{home_team} vs {away_team}"
            )

        home_goals = int(home_goals_raw)
        away_goals = int(away_goals_raw)

        result = determine_result(
            home_goals,
            away_goals,
        )

        model_status = "completed"
    else:
        home_goals = pd.NA
        away_goals = pd.NA
        result = pd.NA
        model_status = "scheduled"

    return {
        "match_id": make_match_id(
            match_date,
            home_team,
            away_team,
        ),
        "provider_fixture_id": fixture.get("id"),
        "competition": "EPL",
        "provider_league_id": league.get("id"),
        "season": PROJECT_SEASON,
        "provider_season": league.get("season"),
        "matchweek": parse_matchweek(
            league.get("round")
        ),
        "match_date": match_date,
        "kickoff_utc": kickoff_utc,
        "kickoff_time_known": True,
        "home_team_id": make_team_id(home_team),
        "home_provider_team_id": home_data.get("id"),
        "home_team": home_team,
        "away_team_id": make_team_id(away_team),
        "away_provider_team_id": away_data.get("id"),
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "status": model_status,
        "provider_status": status_short,
        "provider_status_long": status_data.get(
            "long"
        ),
        "referee": fixture.get("referee"),
        "venue_name": venue_data.get("name"),
        "venue_city": venue_data.get("city"),
        "source": "API-Football",
    }


def build_audit(
    data: pd.DataFrame,
    quota: dict,
) -> pd.DataFrame:
    completed = data.loc[
        data["status"] == "completed"
    ]

    scheduled = data.loc[
        data["status"] == "scheduled"
    ]

    team_names = pd.concat(
        [
            data["home_team"],
            data["away_team"],
        ],
        ignore_index=True,
    )

    rows = [
        {
            "check": "total_matches",
            "value": len(data),
        },
        {
            "check": "completed_matches",
            "value": len(completed),
        },
        {
            "check": "scheduled_matches",
            "value": len(scheduled),
        },
        {
            "check": "unique_teams",
            "value": team_names.nunique(),
        },
        {
            "check": "duplicate_match_ids",
            "value": data[
                "match_id"
            ].duplicated().sum(),
        },
        {
            "check": "earliest_date",
            "value": data[
                "match_date"
            ].min(),
        },
        {
            "check": "latest_date",
            "value": data[
                "match_date"
            ].max(),
        },
        {
            "check": "api_daily_limit",
            "value": quota.get("daily_limit"),
        },
        {
            "check": "api_daily_remaining",
            "value": quota.get(
                "daily_remaining"
            ),
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "API_FOOTBALL_KEY",
        "",
    ).strip()

    if len(api_key) <= 10:
        raise RuntimeError(
            "没有在.env中读取到有效的"
            "API_FOOTBALL_KEY。"
        )

    print("正在连接API-Football……")

    try:
        payload, quota = request_fixtures(
            api_key
        )
    except RuntimeError as error:
        error_text = str(error)

        if (
            "Free plans do not have access"
            in error_text
            or "'plan'" in error_text
        ):
            print(
                "当前API-Football免费方案不提供"
                "2026赛季数据。"
            )
            print(
                "已跳过官方赛程下载；"
                "预计阵容和其他赛前功能仍可继续运行。"
            )
            return

        raise

    raw_fixtures = payload["response"]

    print(
        f"API返回比赛数量："
        f"{len(raw_fixtures)}"
    )

    if not raw_fixtures:
        raise RuntimeError(
            "API没有返回英超比赛。"
            "请检查免费方案是否支持2026赛季。"
        )

    records = [
        normalise_fixture(item)
        for item in raw_fixtures
    ]

    data = pd.DataFrame(records)

    data["match_date"] = pd.to_datetime(
        data["match_date"],
        errors="raise",
    )

    data["kickoff_utc"] = pd.to_datetime(
        data["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    data["matchweek"] = data[
        "matchweek"
    ].astype("Int64")

    data["home_goals"] = data[
        "home_goals"
    ].astype("Int64")

    data["away_goals"] = data[
        "away_goals"
    ].astype("Int64")

    data = data.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    duplicate_match_ids = int(
        data["match_id"].duplicated().sum()
    )

    if duplicate_match_ids:
        raise RuntimeError(
            "API数据出现重复比赛ID："
            f"{duplicate_match_ids}"
        )

    audit = build_audit(data, quota)

    RAW_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    NORMALIZED_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    data.to_parquet(
        NORMALIZED_OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    completed = data.loc[
        data["status"] == "completed"
    ]

    scheduled = data.loc[
        data["status"] == "scheduled"
    ]

    print("\nAPI-Football数据获取成功。")
    print(f"总比赛数：{len(data)}")
    print(f"已完赛：{len(completed)}")
    print(f"未完赛：{len(scheduled)}")
    print(
        f"球队数："
        f"{pd.concat([data['home_team'], data['away_team']]).nunique()}"
    )
    print(
        "日期范围："
        f"{data['match_date'].min().date()} "
        "至 "
        f"{data['match_date'].max().date()}"
    )
    print(
        "今日剩余请求："
        f"{quota.get('daily_remaining')}"
    )

    if not scheduled.empty:
        print("\n最近的未完赛比赛：")

        print(
            scheduled[
                [
                    "matchweek",
                    "match_date",
                    "home_team",
                    "away_team",
                    "provider_status",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print(f"\n原始文件：{RAW_OUTPUT_FILE}")
    print(f"标准化文件：{NORMALIZED_OUTPUT_FILE}")
    print(f"检查报告：{AUDIT_FILE}")


if __name__ == "__main__":
    main()
