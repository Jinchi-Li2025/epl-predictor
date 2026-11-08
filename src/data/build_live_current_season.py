from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "football_data_epl_current.parquet"
)

OLD_COMPLETED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches.parquet"
)

OLD_FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_fixtures.parquet"
)

CANDIDATE_COMPLETED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches_candidate.parquet"
)

CANDIDATE_FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_fixtures_candidate.parquet"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "live_source_reconciliation.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "live_source_match_comparison.csv"
)

OUTPUT_COLUMNS = [
    "match_id",
    "competition",
    "season",
    "match_date",
    "kickoff_utc",
    "kickoff_time_known",
    "matchweek",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "status",
    "referee",
    "source",
]


def slugify_team_name(team_name: str) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(team_name),
    )
    text = text.encode(
        "ascii",
        "ignore",
    ).decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def make_pair_key(
    season: str,
    home_team_id: str,
    away_team_id: str,
) -> str:
    return (
        f"{season}|"
        f"{home_team_id}|"
        f"{away_team_id}"
    )


def make_match_id(pair_key: str) -> str:
    digest = hashlib.sha256(
        pair_key.encode("utf-8")
    ).hexdigest()[:16]

    return f"epl-{digest}"


def map_status(provider_status: str) -> str:
    status = str(provider_status).upper()

    mapping = {
        "FINISHED": "completed",
        "SCHEDULED": "scheduled",
        "TIMED": "scheduled",
        "IN_PLAY": "live",
        "PAUSED": "live",
        "POSTPONED": "postponed",
        "SUSPENDED": "suspended",
        "CANCELLED": "cancelled",
    }

    return mapping.get(status, status.lower())


def load_old_current_data() -> pd.DataFrame:
    completed = pd.read_parquet(
        OLD_COMPLETED_FILE
    )
    fixtures = pd.read_parquet(
        OLD_FIXTURES_FILE
    )

    old_data = pd.concat(
        [completed, fixtures],
        ignore_index=True,
    )

    old_data["pair_key"] = old_data.apply(
        lambda row: make_pair_key(
            str(row["season"]),
            str(row["home_team_id"]),
            str(row["away_team_id"]),
        ),
        axis=1,
    )

    if old_data["pair_key"].duplicated().any():
        duplicates = old_data.loc[
            old_data["pair_key"].duplicated(
                keep=False
            ),
            [
                "season",
                "home_team",
                "away_team",
            ],
        ]

        raise RuntimeError(
            "旧数据中存在重复的主客队组合：\n"
            f"{duplicates.to_string(index=False)}"
        )

    return old_data


def convert_api_data(
    api_data: pd.DataFrame,
    old_data: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "season",
        "matchweek",
        "kickoff_utc",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "is_completed",
        "provider_status",
        "referee",
    }

    missing_columns = (
        required_columns - set(api_data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "API文件缺少必要字段："
            f"{sorted(missing_columns)}"
        )

    converted = api_data.copy()

    converted["home_team_id"] = (
        converted["home_team"]
        .map(slugify_team_name)
    )

    converted["away_team_id"] = (
        converted["away_team"]
        .map(slugify_team_name)
    )

    converted["pair_key"] = converted.apply(
        lambda row: make_pair_key(
            str(row["season"]),
            str(row["home_team_id"]),
            str(row["away_team_id"]),
        ),
        axis=1,
    )

    if converted["pair_key"].duplicated().any():
        raise RuntimeError(
            "API数据中存在重复的主客队组合。"
        )

    old_id_map = (
        old_data
        .set_index("pair_key")["match_id"]
        .to_dict()
    )

    converted["match_id"] = (
        converted["pair_key"].map(old_id_map)
    )

    missing_id_mask = converted[
        "match_id"
    ].isna()

    converted.loc[
        missing_id_mask,
        "match_id",
    ] = converted.loc[
        missing_id_mask,
        "pair_key",
    ].map(make_match_id)

    converted["competition"] = "EPL"

    converted["kickoff_utc"] = pd.to_datetime(
        converted["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    converted["match_date"] = pd.to_datetime(
        converted["match_date"],
        errors="coerce",
    )

    converted["kickoff_time_known"] = (
        converted["kickoff_utc"].notna()
    )

    converted["matchweek"] = pd.to_numeric(
        converted["matchweek"],
        errors="coerce",
    ).astype("Int64")

    converted["home_goals"] = pd.to_numeric(
        converted["home_goals"],
        errors="coerce",
    ).astype("Int64")

    converted["away_goals"] = pd.to_numeric(
        converted["away_goals"],
        errors="coerce",
    ).astype("Int64")

    converted["result"] = (
        converted["result"].astype("string")
    )

    converted["status"] = (
        converted["provider_status"]
        .map(map_status)
        .astype("string")
    )

    converted["source"] = (
        "football-data.org:v4"
    )

    completed_mask = converted[
        "is_completed"
    ].fillna(False)

    missing_completed_scores = (
        completed_mask
        & (
            converted["home_goals"].isna()
            | converted["away_goals"].isna()
        )
    )

    if missing_completed_scores.any():
        raise RuntimeError(
            "API中有已完赛比赛缺少比分。"
        )

    converted = converted.sort_values(
        [
            "kickoff_utc",
            "matchweek",
            "match_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    return converted


def build_comparison(
    old_data: pd.DataFrame,
    converted: pd.DataFrame,
) -> pd.DataFrame:
    old_columns = [
        "pair_key",
        "match_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        "home_goals",
        "away_goals",
        "status",
    ]

    new_columns = [
        "pair_key",
        "match_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        "home_goals",
        "away_goals",
        "status",
    ]

    comparison = old_data[
        old_columns
    ].merge(
        converted[new_columns],
        on="pair_key",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    old_kickoff = pd.to_datetime(
        comparison["kickoff_utc_old"],
        utc=True,
        errors="coerce",
    )

    new_kickoff = pd.to_datetime(
        comparison["kickoff_utc_new"],
        utc=True,
        errors="coerce",
    )

    comparison["kickoff_difference_minutes"] = (
        (new_kickoff - old_kickoff)
        .abs()
        .dt.total_seconds()
        .div(60)
    )

    old_completed = (
        comparison["status_old"] == "completed"
    )

    new_completed = (
        comparison["status_new"] == "completed"
    )

    both_completed = old_completed & new_completed

    old_home_goals = pd.to_numeric(
        comparison["home_goals_old"],
        errors="coerce",
    )

    new_home_goals = pd.to_numeric(
        comparison["home_goals_new"],
        errors="coerce",
    )

    old_away_goals = pd.to_numeric(
        comparison["away_goals_old"],
        errors="coerce",
    )

    new_away_goals = pd.to_numeric(
        comparison["away_goals_new"],
        errors="coerce",
    )

    comparison["score_disagreement"] = (
        both_completed
        & (
            old_home_goals.ne(new_home_goals)
            | old_away_goals.ne(new_away_goals)
        )
    )

    return comparison


def main() -> None:
    print(
        "正在转换 football-data.org 当前赛季数据……"
    )

    api_data = pd.read_parquet(API_FILE)
    old_data = load_old_current_data()

    converted = convert_api_data(
        api_data=api_data,
        old_data=old_data,
    )

    comparison = build_comparison(
        old_data=old_data,
        converted=converted,
    )

    completed_mask = (
        converted["status"] == "completed"
    )

    completed = (
        converted.loc[
            completed_mask,
            OUTPUT_COLUMNS,
        ]
        .copy()
        .reset_index(drop=True)
    )

    fixtures = (
        converted.loc[
            ~completed_mask,
            OUTPUT_COLUMNS,
        ]
        .copy()
        .reset_index(drop=True)
    )

    CANDIDATE_COMPLETED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed.to_parquet(
        CANDIDATE_COMPLETED_FILE,
        index=False,
    )

    fixtures.to_parquet(
        CANDIDATE_FIXTURES_FILE,
        index=False,
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    overlap_count = int(
        comparison["_merge"].eq("both").sum()
    )

    missing_in_api = int(
        comparison["_merge"]
        .eq("left_only")
        .sum()
    )

    new_in_api = int(
        comparison["_merge"]
        .eq("right_only")
        .sum()
    )

    score_disagreements = int(
        comparison[
            "score_disagreement"
        ].sum()
    )

    kickoff_changes = int(
        comparison[
            "kickoff_difference_minutes"
        ]
        .gt(1)
        .fillna(False)
        .sum()
    )

    reused_ids = int(
        converted["match_id"]
        .isin(old_data["match_id"])
        .sum()
    )

    old_teams = set(
        pd.concat(
            [
                old_data["home_team_id"],
                old_data["away_team_id"],
            ]
        )
    )

    new_teams = set(
        pd.concat(
            [
                converted["home_team_id"],
                converted["away_team_id"],
            ]
        )
    )

    audit = pd.DataFrame(
        [
            {
                "old_total_matches": len(old_data),
                "api_total_matches": len(converted),
                "old_completed_matches": int(
                    (
                        old_data["status"]
                        == "completed"
                    ).sum()
                ),
                "api_completed_matches": len(
                    completed
                ),
                "api_uncompleted_matches": len(
                    fixtures
                ),
                "matched_home_away_pairs": (
                    overlap_count
                ),
                "missing_in_api": missing_in_api,
                "new_in_api": new_in_api,
                "reused_match_ids": reused_ids,
                "score_disagreements": (
                    score_disagreements
                ),
                "kickoff_changes_over_1_minute": (
                    kickoff_changes
                ),
                "team_sets_equal": (
                    old_teams == new_teams
                ),
                "api_duplicate_match_ids": int(
                    converted["match_id"]
                    .duplicated()
                    .sum()
                ),
            }
        ]
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n转换与核对完成。")
    print(f"旧数据比赛：{len(old_data)}")
    print(f"API数据比赛：{len(converted)}")
    print(f"主客队组合匹配：{overlap_count}")
    print(f"API缺失比赛：{missing_in_api}")
    print(f"API新增组合：{new_in_api}")
    print(f"复用旧比赛ID：{reused_ids}")
    print(f"旧数据已完赛：{int((old_data['status'] == 'completed').sum())}")
    print(f"API已完赛：{len(completed)}")
    print(f"API未完赛：{len(fixtures)}")
    print(f"已完赛比分冲突：{score_disagreements}")
    print(f"开球时间发生变化：{kickoff_changes}")
    print(
        "双方球队集合一致："
        f"{old_teams == new_teams}"
    )
    print(
        "重复比赛ID："
        f"{converted['match_id'].duplicated().sum()}"
    )

    print(
        f"\n候选已完赛文件："
        f"{CANDIDATE_COMPLETED_FILE}"
    )
    print(
        f"候选赛程文件："
        f"{CANDIDATE_FIXTURES_FILE}"
    )
    print(f"核对摘要：{AUDIT_FILE}")
    print(f"逐场对比：{COMPARISON_FILE}")

    if (
        len(converted) == 380
        and overlap_count == 380
        and missing_in_api == 0
        and new_in_api == 0
        and score_disagreements == 0
        and old_teams == new_teams
    ):
        print(
            "\n核对通过：候选数据可以进入正式切换步骤。"
        )
    else:
        print(
            "\n核对未完全通过：暂时不要覆盖正式文件。"
        )


if __name__ == "__main__":
    main()