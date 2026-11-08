from __future__ import annotations

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
    / "football_data"
    / "epl_current_fixtures.json"
)

NORMALIZED_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "football_data_epl_current.parquet"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "football_data_audit.csv"
)

BASE_URL = "https://api.football-data.org/v4"
COMPETITION_CODE = "PL"

COMPLETED_STATUSES = {"FINISHED"}


def normalise_team_name(team_name: str) -> str:
    name = str(team_name).strip()

    # 删除 football-data.org 队名末尾的 FC 或 AFC。
    # 不影响开头的 AFC Bournemouth。
    name = re.sub(
        r"\s+(FC|AFC)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    replacements = {
        "Wolverhampton Wanderers": "Wolverhampton Wanderers",
        "Brighton Hove Albion": "Brighton & Hove Albion",
        "Tottenham Hotspur": "Tottenham Hotspur",
        "Manchester United": "Manchester United",
        "Manchester City": "Manchester City",
        "Newcastle United": "Newcastle United",
        "Nottingham Forest": "Nottingham Forest",
        "West Ham United": "West Ham United",
    }

    return replacements.get(name, name)


def request_json(
    endpoint: str,
    token: str,
) -> tuple[dict[str, Any], requests.Response]:
    url = f"{BASE_URL}{endpoint}"

    try:
        response = requests.get(
            url,
            headers={
                "X-Auth-Token": token,
                "Accept": "application/json",
            },
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"连接 football-data.org 失败：{error}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            "football-data.org 没有返回有效的 JSON 数据。"
        ) from error

    if response.status_code == 401:
        raise RuntimeError(
            "football-data.org Token 无效。"
            "请检查 .env 中的 FOOTBALL_DATA_TOKEN。"
        )

    if response.status_code == 403:
        raise RuntimeError(
            "football-data.org 拒绝访问。"
            f"服务器信息：{payload}"
        )

    if response.status_code == 429:
        raise RuntimeError(
            "请求速度超过免费套餐限制。"
            "请等待一分钟后重新运行。"
        )

    if not response.ok:
        raise RuntimeError(
            f"football-data.org 返回 HTTP {response.status_code}："
            f"{payload}"
        )

    return payload, response


def season_label_from_competition(
    competition_payload: dict[str, Any],
) -> str:
    current_season = competition_payload.get("currentSeason") or {}
    start_date = str(current_season.get("startDate") or "")

    if len(start_date) < 4:
        raise RuntimeError(
            "API 返回的数据中缺少当前赛季开始日期。"
        )

    start_year = int(start_date[:4])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def extract_referee(match: dict[str, Any]) -> str | None:
    referees = match.get("referees") or []

    if not referees:
        return None

    first_referee = referees[0] or {}
    return first_referee.get("name")


def determine_result(
    home_goals: int | None,
    away_goals: int | None,
) -> str | None:
    if home_goals is None or away_goals is None:
        return None

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def normalise_matches(
    matches: list[dict[str, Any]],
    season: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for match in matches:
        status = str(match.get("status") or "").upper()
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}

        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        if home_goals is not None:
            home_goals = int(home_goals)

        if away_goals is not None:
            away_goals = int(away_goals)

        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}

        kickoff_utc = pd.to_datetime(
            match.get("utcDate"),
            utc=True,
            errors="coerce",
        )

        if pd.isna(kickoff_utc):
            match_date = None
        else:
            match_date = kickoff_utc.date().isoformat()

        provider_match_id = match.get("id")

        rows.append(
            {
                "match_id": f"football_data_{provider_match_id}",
                "provider": "football-data.org",
                "provider_match_id": provider_match_id,
                "competition": "Premier League",
                "competition_code": COMPETITION_CODE,
                "season": season,
                "matchweek": match.get("matchday"),
                "kickoff_utc": kickoff_utc,
                "match_date": match_date,
                "home_team": normalise_team_name(
                    home_team.get("name") or ""
                ),
                "away_team": normalise_team_name(
                    away_team.get("name") or ""
                ),
                "provider_home_team_id": home_team.get("id"),
                "provider_away_team_id": away_team.get("id"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": determine_result(
                    home_goals,
                    away_goals,
                ),
                "is_completed": status in COMPLETED_STATUSES,
                "provider_status": status,
                "venue": match.get("venue"),
                "referee": extract_referee(match),
                "last_updated": match.get("lastUpdated"),
            }
        )

    data = pd.DataFrame(rows)

    if data.empty:
        raise RuntimeError(
            "football-data.org 没有返回任何英超比赛。"
        )

    data = data.sort_values(
        ["kickoff_utc", "provider_match_id"],
        na_position="last",
    ).reset_index(drop=True)

    return data


def build_audit(
    data: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    completed = data["is_completed"].fillna(False)
    unplayed = ~completed

    audit_rows = [
        {
            "source": "football-data.org",
            "season": season,
            "total_matches": len(data),
            "completed_matches": int(completed.sum()),
            "unplayed_matches": int(unplayed.sum()),
            "team_count": int(
                pd.concat(
                    [
                        data["home_team"],
                        data["away_team"],
                    ]
                ).nunique()
            ),
            "duplicate_provider_ids": int(
                data["provider_match_id"].duplicated().sum()
            ),
            "missing_kickoff": int(
                data["kickoff_utc"].isna().sum()
            ),
            "missing_home_team": int(
                data["home_team"].eq("").sum()
            ),
            "missing_away_team": int(
                data["away_team"].eq("").sum()
            ),
            "earliest_kickoff": data["kickoff_utc"].min(),
            "latest_kickoff": data["kickoff_utc"].max(),
            "fetched_at_utc": pd.Timestamp.now(tz="UTC"),
        }
    ]

    return pd.DataFrame(audit_rows)


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()

    if not token:
        raise RuntimeError(
            "没有读取到 FOOTBALL_DATA_TOKEN。"
            "请检查项目根目录中的 .env 文件。"
        )

    print("正在连接 football-data.org……")

    competition_payload, _ = request_json(
        f"/competitions/{COMPETITION_CODE}",
        token,
    )

    season = season_label_from_competition(
        competition_payload
    )

    print(f"API识别到的当前英超赛季：{season}")
    print("正在下载当前赛季英超赛程与赛果……")

    fixtures_payload, response = request_json(
        f"/competitions/{COMPETITION_CODE}/matches",
        token,
    )

    matches = fixtures_payload.get("matches") or []

    data = normalise_matches(
        matches=matches,
        season=season,
    )

    audit = build_audit(
        data=data,
        season=season,
    )

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

    RAW_OUTPUT_FILE.write_text(
        json.dumps(
            fixtures_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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

    completed_count = int(
        data["is_completed"].sum()
    )
    unplayed_count = len(data) - completed_count

    teams = pd.concat(
        [
            data["home_team"],
            data["away_team"],
        ]
    ).dropna()

    now_utc = pd.Timestamp.now(tz="UTC")

    upcoming = data[
        (~data["is_completed"])
        & (data["kickoff_utc"] >= now_utc)
    ].copy()

    print("\nfootball-data.org 数据下载完成。")
    print(f"赛季：{season}")
    print(f"总比赛数：{len(data)}")
    print(f"已完赛：{completed_count}")
    print(f"未完赛：{unplayed_count}")
    print(f"球队数量：{teams.nunique()}")
    print(
        "重复比赛ID："
        f"{data['provider_match_id'].duplicated().sum()}"
    )

    remaining_requests = response.headers.get(
        "X-Requests-Available-Minute"
    )

    if remaining_requests is not None:
        print(
            "本分钟剩余请求次数："
            f"{remaining_requests}"
        )

    if upcoming.empty:
        print("\n没有找到未来未开赛比赛。")
    else:
        nearest_matchweek = upcoming["matchweek"].dropna()

        if nearest_matchweek.empty:
            nearest_upcoming = upcoming.head(10)
        else:
            next_matchweek = int(nearest_matchweek.min())
            nearest_upcoming = upcoming[
                upcoming["matchweek"] == next_matchweek
            ]

            print(
                f"\n最近的未来轮次：第{next_matchweek}轮"
            )

        print(
            nearest_upcoming[
                [
                    "matchweek",
                    "kickoff_utc",
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