from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "openfootball"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "matches.parquet"
AUDIT_FILE = PROJECT_ROOT / "reports" / "data_audit.csv"

# 需要读取的16个完整英超赛季
SEASONS = [
    f"{year}-{str(year + 1)[-2:]}"
    for year in range(2010, 2026)
]


def normalise_team_name(raw_name: str) -> str:
    """统一球队名称。"""
    name = " ".join(raw_name.strip().split())

    # 删除名称末尾的 FC 或 AFC
    name = re.sub(
        r"\s+(?:F\.?C\.?|A\.?F\.?C\.?)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name.strip()


def create_team_id(team_name: str) -> str:
    """生成稳定的球队ID。"""
    team_id = team_name.lower()
    team_id = team_id.replace("&", "and")
    team_id = re.sub(r"[^a-z0-9]+", "-", team_id)
    return team_id.strip("-")


def parse_score(score_value: object) -> tuple[int, int]:
    """读取OpenFootball的两种比分格式。"""
    full_time_score = score_value

    if isinstance(score_value, dict):
        full_time_score = (
            score_value.get("ft")
            or score_value.get("fulltime")
            or score_value.get("full_time")
        )

    if (
        not isinstance(full_time_score, (list, tuple))
        or len(full_time_score) != 2
    ):
        raise ValueError(f"无法识别比分格式：{score_value!r}")

    home_goals = int(full_time_score[0])
    away_goals = int(full_time_score[1])

    if home_goals < 0 or away_goals < 0:
        raise ValueError(f"发现负数比分：{full_time_score!r}")

    return home_goals, away_goals


def parse_matchweek(round_value: object) -> int | None:
    """从Matchday 1中提取数字1。"""
    if round_value is None:
        return None

    match = re.search(r"\d+", str(round_value))
    return int(match.group()) if match else None


def parse_kickoff(
    date_text: str,
    time_text: str | None,
) -> tuple[pd.Timestamp, bool]:
    """
    把英国当地开球时间转换成UTC。

    如果旧比赛缺少具体时间，暂时使用中午12点排序，
    同时把kickoff_time_known记录为False。
    """
    time_known = bool(time_text)
    usable_time = time_text if time_known else "12:00"

    local_datetime = datetime.fromisoformat(
        f"{date_text}T{usable_time}"
    ).replace(tzinfo=ZoneInfo("Europe/London"))

    utc_datetime = local_datetime.astimezone(ZoneInfo("UTC"))
    return pd.Timestamp(utc_datetime), time_known


def create_match_id(
    season: str,
    match_date: str,
    home_team: str,
    away_team: str,
) -> str:
    """生成唯一且可以重复得到的比赛ID。"""
    identity = "|".join(
        ["EPL", season, match_date, home_team, away_team]
    )

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return f"epl-{digest}"


def load_season(season: str) -> list[dict]:
    """读取一个赛季。"""
    input_file = RAW_DATA_DIR / season / "en.1.json"

    if not input_file.exists():
        raise FileNotFoundError(f"找不到赛季文件：{input_file}")

    with input_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    rows: list[dict] = []

    for match in payload["matches"]:
        date_text = match["date"]
        home_team = normalise_team_name(match["team1"])
        away_team = normalise_team_name(match["team2"])

        home_goals, away_goals = parse_score(match["score"])

        kickoff_utc, time_known = parse_kickoff(
            date_text,
            match.get("time"),
        )

        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

        rows.append(
            {
                "match_id": create_match_id(
                    season,
                    date_text,
                    home_team,
                    away_team,
                ),
                "competition": "EPL",
                "season": season,
                "match_date": pd.Timestamp(date_text),
                "kickoff_utc": kickoff_utc,
                "kickoff_time_known": time_known,
                "matchweek": parse_matchweek(match.get("round")),
                "home_team_id": create_team_id(home_team),
                "home_team": home_team,
                "away_team_id": create_team_id(away_team),
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "referee": pd.NA,
                "source": "openfootball",
            }
        )

    return rows


def validate_data(matches: pd.DataFrame) -> pd.DataFrame:
    """检查比赛数据是否完整。"""
    if matches["match_id"].duplicated().any():
        raise ValueError("发现重复比赛。")

    required_columns = [
        "season",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]

    if matches[required_columns].isna().any().any():
        raise ValueError("关键字段中存在缺失值。")

    audit_rows: list[dict] = []

    for season, season_matches in matches.groupby("season"):
        teams = pd.concat(
            [
                season_matches["home_team"],
                season_matches["away_team"],
            ],
            ignore_index=True,
        )

        appearances = teams.value_counts()
        invalid_appearances = appearances[appearances != 38]

        audit_rows.append(
            {
                "season": season,
                "matches": len(season_matches),
                "teams": teams.nunique(),
                "unknown_kickoff_times": int(
                    (~season_matches["kickoff_time_known"]).sum()
                ),
                "all_teams_played_38": invalid_appearances.empty,
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values("season")

    invalid_seasons = audit[
        (audit["matches"] != 380)
        | (audit["teams"] != 20)
        | (~audit["all_teams_played_38"])
    ]

    if not invalid_seasons.empty:
        raise ValueError(
            "以下赛季未通过完整性检查：\n"
            + invalid_seasons.to_string(index=False)
        )

    if len(matches) != 6080:
        raise ValueError(
            f"总场次应为6080，实际为{len(matches)}。"
        )

    return audit


def main() -> None:
    """建立完整历史比赛数据集。"""
    all_rows: list[dict] = []

    for season in SEASONS:
        print(f"正在读取 {season}……")
        all_rows.extend(load_season(season))

    matches = pd.DataFrame(all_rows)

    matches = matches.sort_values(
        ["kickoff_utc", "home_team", "away_team"]
    ).reset_index(drop=True)

    audit = validate_data(matches)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    matches.to_parquet(OUTPUT_FILE, index=False)
    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n数据构建成功。")
    print(f"总比赛数：{len(matches)}")
    print(f"处理后文件：{OUTPUT_FILE}")
    print(f"检查报告：{AUDIT_FILE}")
    print("\n各赛季检查结果：")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()