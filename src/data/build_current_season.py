from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openfootball"
    / "2026-27"
    / "en.1.json"
)

COMPLETED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches.parquet"
)

FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_fixtures.parquet"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_season_audit.csv"
)

COMPETITION = "EPL"
SEASON = "2026-27"
LONDON_TIMEZONE = ZoneInfo("Europe/London")


def extract_team_name(value: Any) -> str:
    """从字符串或字典中提取球队名称。"""

    if isinstance(value, str):
        name = value
    elif isinstance(value, dict):
        name = (
            value.get("name")
            or value.get("title")
            or value.get("team")
            or ""
        )
    else:
        name = str(value or "")

    name = re.sub(r"\s+", " ", name).strip()

    # 与历史数据保持一致，删除球队名称末尾的 FC 或 AFC。
    name = re.sub(r"\s+(?:FC|AFC)$", "", name, flags=re.IGNORECASE)

    if not name:
        raise ValueError(f"无法识别球队名称：{value!r}")

    return name


def make_team_id(team_name: str) -> str:
    """生成稳定的球队 ID。"""

    team_id = team_name.lower()
    team_id = team_id.replace("&", " and ")
    team_id = re.sub(r"[^a-z0-9]+", "-", team_id)
    team_id = team_id.strip("-")

    return team_id


def make_match_id(
    season: str,
    match_date: str,
    home_team: str,
    away_team: str,
) -> str:
    """生成稳定且可重复的比赛 ID。"""

    raw_value = (
        f"{COMPETITION}|{season}|{match_date}|"
        f"{home_team}|{away_team}"
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"epl-{digest}"


def parse_matchweek(value: Any) -> int | None:
    """从 round 字段提取轮次数字。"""

    if value is None:
        return None

    match = re.search(r"\d+", str(value))

    if match is None:
        return None

    return int(match.group())


def parse_integer(value: Any) -> int | None:
    """安全地将比分转换为整数。"""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number) or not number.is_integer():
        return None

    return int(number)


def parse_final_score(score: Any) -> tuple[int, int] | None:
    """读取 OpenFootball 的全场比分。"""

    candidate = score

    if isinstance(score, dict):
        candidate = None

        for key in (
            "ft",
            "fulltime",
            "full_time",
            "full-time",
            "final",
        ):
            if key in score:
                candidate = score[key]
                break

        if candidate is None:
            home_value = (
                score.get("home")
                or score.get("team1")
                or score.get("score1")
            )
            away_value = (
                score.get("away")
                or score.get("team2")
                or score.get("score2")
            )

            if home_value is not None and away_value is not None:
                candidate = [home_value, away_value]

    if not isinstance(candidate, (list, tuple)):
        return None

    if len(candidate) < 2:
        return None

    home_goals = parse_integer(candidate[0])
    away_goals = parse_integer(candidate[1])

    if home_goals is None or away_goals is None:
        return None

    if home_goals < 0 or away_goals < 0:
        return None

    return home_goals, away_goals


def determine_result(home_goals: int, away_goals: int) -> str:
    """将比分转换为 H、D、A。"""

    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def parse_kickoff(
    date_value: Any,
    time_value: Any,
) -> tuple[str, pd.Timestamp, bool]:
    """
    将英国当地开球时间转换为 UTC。

    如果没有开球时间，使用当地中午 12:00 作为排序代理值，
    同时将 kickoff_time_known 标记为 False。
    """

    date_text = str(date_value or "").strip()

    date_match = re.search(
        r"(\d{4})-(\d{2})-(\d{2})",
        date_text,
    )

    if date_match is None:
        raise ValueError(f"无法识别比赛日期：{date_value!r}")

    year, month, day = map(int, date_match.groups())
    match_date = f"{year:04d}-{month:02d}-{day:02d}"

    time_text = str(time_value or "").strip()
    time_match = re.search(
        r"(\d{1,2}):(\d{2})",
        time_text,
    )

    # 有些数据会把时间直接写进 date 字段。
    if time_match is None:
        time_match = re.search(
            r"[T\s](\d{1,2}):(\d{2})",
            date_text,
        )

    if time_match is None:
        hour = 12
        minute = 0
        kickoff_time_known = False
    else:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        kickoff_time_known = True

    local_kickoff = datetime(
        year,
        month,
        day,
        hour,
        minute,
        tzinfo=LONDON_TIMEZONE,
    )

    kickoff_utc = pd.Timestamp(local_kickoff).tz_convert("UTC")

    return match_date, kickoff_utc, kickoff_time_known


def extract_referee(value: Any) -> Any:
    """读取裁判字段；当前数据没有时保留为空。"""

    if value is None:
        return pd.NA

    if isinstance(value, dict):
        value = value.get("name") or value.get("title")

    if value is None:
        return pd.NA

    text = str(value).strip()

    return text if text else pd.NA


def build_record(match: dict[str, Any]) -> dict[str, Any]:
    """把一场原始比赛转换成统一结构。"""

    home_team = extract_team_name(match.get("team1"))
    away_team = extract_team_name(match.get("team2"))

    if home_team == away_team:
        raise ValueError(
            f"主客队名称相同：{home_team}"
        )

    match_date, kickoff_utc, kickoff_time_known = parse_kickoff(
        match.get("date"),
        match.get("time"),
    )

    final_score = parse_final_score(match.get("score"))

    if final_score is None:
        home_goals = pd.NA
        away_goals = pd.NA
        result = pd.NA
        status = "scheduled"
    else:
        home_goals, away_goals = final_score
        result = determine_result(home_goals, away_goals)
        status = "completed"

    return {
        "match_id": make_match_id(
            SEASON,
            match_date,
            home_team,
            away_team,
        ),
        "competition": COMPETITION,
        "season": SEASON,
        "match_date": match_date,
        "kickoff_utc": kickoff_utc,
        "kickoff_time_known": kickoff_time_known,
        "matchweek": parse_matchweek(match.get("round")),
        "home_team_id": make_team_id(home_team),
        "home_team": home_team,
        "away_team_id": make_team_id(away_team),
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "status": status,
        "referee": extract_referee(match.get("referee")),
        "source": str(SOURCE_FILE.relative_to(PROJECT_ROOT)),
    }


def prepare_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """创建并规范化 DataFrame。"""

    frame = pd.DataFrame(records)

    if frame.empty:
        raise RuntimeError("当前赛季文件中没有任何比赛记录。")

    frame["match_date"] = pd.to_datetime(
        frame["match_date"],
        errors="raise",
    )

    frame["kickoff_utc"] = pd.to_datetime(
        frame["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    frame["matchweek"] = frame["matchweek"].astype("Int64")
    frame["home_goals"] = frame["home_goals"].astype("Int64")
    frame["away_goals"] = frame["away_goals"].astype("Int64")
    frame["kickoff_time_known"] = (
        frame["kickoff_time_known"].astype(bool)
    )

    frame = frame.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    return frame


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            "找不到当前赛季数据文件：\n"
            f"{SOURCE_FILE}"
        )

    with SOURCE_FILE.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        raw_data = json.load(file)

    raw_matches = raw_data.get("matches")

    if not isinstance(raw_matches, list):
        raise RuntimeError(
            "JSON 文件中没有有效的 matches 列表。"
        )

    records = [build_record(match) for match in raw_matches]
    all_matches = prepare_dataframe(records)

    duplicate_match_ids = int(
        all_matches["match_id"].duplicated().sum()
    )

    if duplicate_match_ids != 0:
        duplicated = all_matches.loc[
            all_matches["match_id"].duplicated(
                keep=False
            ),
            [
                "match_id",
                "match_date",
                "home_team",
                "away_team",
            ],
        ]

        raise RuntimeError(
            "检测到重复比赛 ID：\n"
            f"{duplicated.to_string(index=False)}"
        )

    completed = all_matches.loc[
        all_matches["status"] == "completed"
    ].copy()

    fixtures = all_matches.loc[
        all_matches["status"] == "scheduled"
    ].copy()

    team_names = pd.concat(
        [
            all_matches["home_team"],
            all_matches["away_team"],
        ],
        ignore_index=True,
    )

    unique_teams = int(team_names.nunique())
    known_kickoff_times = int(
        all_matches["kickoff_time_known"].sum()
    )

    COMPLETED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIXTURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed.to_parquet(
        COMPLETED_FILE,
        index=False,
    )

    fixtures.to_parquet(
        FIXTURES_FILE,
        index=False,
    )

    audit_rows = [
        {"check": "season", "value": SEASON},
        {
            "check": "total_records",
            "value": len(all_matches),
        },
        {
            "check": "completed_matches",
            "value": len(completed),
        },
        {
            "check": "scheduled_fixtures",
            "value": len(fixtures),
        },
        {
            "check": "unique_teams",
            "value": unique_teams,
        },
        {
            "check": "duplicate_match_ids",
            "value": duplicate_match_ids,
        },
        {
            "check": "known_kickoff_times",
            "value": known_kickoff_times,
        },
        {
            "check": "unknown_kickoff_times",
            "value": (
                len(all_matches) - known_kickoff_times
            ),
        },
        {
            "check": "earliest_date",
            "value": all_matches["match_date"]
            .min()
            .date()
            .isoformat(),
        },
        {
            "check": "latest_date",
            "value": all_matches["match_date"]
            .max()
            .date()
            .isoformat(),
        },
    ]

    audit = pd.DataFrame(audit_rows)

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n2026-27 当前赛季数据整理完成。")
    print(f"原始记录总数：{len(all_matches)}")
    print(f"已完赛比赛：{len(completed)}")
    print(f"未来赛程：{len(fixtures)}")
    print(f"球队数量：{unique_teams}")
    print(f"重复比赛 ID：{duplicate_match_ids}")

    print(
        "日期范围："
        f"{all_matches['match_date'].min().date()} "
        "至 "
        f"{all_matches['match_date'].max().date()}"
    )

    print(f"\n已完赛文件：{COMPLETED_FILE}")
    print(f"未来赛程文件：{FIXTURES_FILE}")
    print(f"检查报告：{AUDIT_FILE}")

    if len(all_matches) != 380:
        print(
            "\n注意：当前原始文件不是完整的380场。"
            "这不一定是错误，可能是数据源目前只提供了"
            "已经公布或已经完成的比赛。"
        )

    if unique_teams != 20:
        print(
            "\n注意：当前识别出的球队数量不是20支，"
            "运行下一步前需要检查。"
        )

    if len(completed) == 0:
        print(
            "\n注意：没有识别到已完赛比赛，"
            "需要检查原始比分格式。"
        )

    if len(fixtures) == 0:
        print(
            "\n注意：数据源暂时没有提供未来赛程。"
        )


if __name__ == "__main__":
    main()