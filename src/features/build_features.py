from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCHES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "matches.parquet"
)
ELO_FILE = (
    PROJECT_ROOT / "data" / "processed" / "elo_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
)
AUDIT_FILE = (
    PROJECT_ROOT / "reports" / "feature_audit.csv"
)


def empty_standing() -> dict[str, int]:
    return {
        "played": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
    }


def rolling_summary(
    records: list[dict],
    window: int,
) -> dict[str, float]:
    """计算最近N场表现。"""
    selected = records[-window:]

    if not selected:
        return {
            "ppg": np.nan,
            "goals_for": np.nan,
            "goals_against": np.nan,
            "goal_difference": np.nan,
            "win_rate": np.nan,
        }

    games = len(selected)

    points = sum(
        record["points"]
        for record in selected
    )

    goals_for = sum(
        record["goals_for"]
        for record in selected
    )

    goals_against = sum(
        record["goals_against"]
        for record in selected
    )

    wins = sum(
        record["points"] == 3
        for record in selected
    )

    return {
        "ppg": points / games,
        "goals_for": goals_for / games,
        "goals_against": goals_against / games,
        "goal_difference": (
            goals_for - goals_against
        ) / games,
        "win_rate": wins / games,
    }


def standing_snapshot(
    standings: dict[str, dict],
) -> tuple[
    dict[str, float],
    int,
    int,
    int,
]:
    """
    生成比赛日开始前的积分榜快照。

    完全相同的积分、净胜球和进球数使用平均排名，
    避免按球队名称任意打破平局。
    """
    def sorting_key(team_id: str) -> tuple:
        standing = standings[team_id]

        goal_difference = (
            standing["goals_for"]
            - standing["goals_against"]
        )

        return (
            -standing["points"],
            -goal_difference,
            -standing["goals_for"],
            team_id,
        )

    ordered_teams = sorted(
        standings,
        key=sorting_key,
    )

    ranks: dict[str, float] = {}

    index = 0

    while index < len(ordered_teams):
        team_id = ordered_teams[index]
        standing = standings[team_id]

        comparison_key = (
            standing["points"],
            standing["goals_for"]
            - standing["goals_against"],
            standing["goals_for"],
        )

        end_index = index + 1

        while end_index < len(ordered_teams):
            next_team = ordered_teams[end_index]
            next_standing = standings[next_team]

            next_key = (
                next_standing["points"],
                next_standing["goals_for"]
                - next_standing[
                    "goals_against"
                ],
                next_standing["goals_for"],
            )

            if next_key != comparison_key:
                break

            end_index += 1

        # 实际名次从1开始
        average_rank = (
            (index + 1) + end_index
        ) / 2.0

        for tied_index in range(
            index,
            end_index,
        ):
            ranks[
                ordered_teams[tied_index]
            ] = average_rank

        index = end_index

    leader_points = standings[
        ordered_teams[0]
    ]["points"]

    fourth_points = standings[
        ordered_teams[3]
    ]["points"]

    seventeenth_points = standings[
        ordered_teams[16]
    ]["points"]

    return (
        ranks,
        leader_points,
        fourth_points,
        seventeenth_points,
    )


def calculate_pressures(
    standing: dict,
    rank: float,
    leader_points: int,
    fourth_points: int,
    seventeenth_points: int,
) -> dict[str, float]:
    """
    根据当时积分榜计算比赛重要性。

    只使用赛前积分和剩余比赛数量，
    不查看赛季最终排名。
    """
    played = standing["played"]
    points = standing["points"]

    remaining_matches = max(
        38 - played,
        0,
    )

    progress = played / 38.0
    late_season_weight = progress ** 1.5

    mathematically_alive_for_title = (
        points + 3 * remaining_matches
        >= leader_points
    )

    if mathematically_alive_for_title:
        title_pressure = (
            late_season_weight
            * np.exp(
                -max(
                    leader_points - points,
                    0,
                )
                / 6.0
            )
        )
    else:
        title_pressure = 0.0

    europe_pressure = (
        late_season_weight
        * np.exp(
            -abs(points - fourth_points)
            / 6.0
        )
    )

    # 低于安全线的球队保持高保级压力；
    # 高于安全线越多，压力越小。
    relegation_pressure = (
        late_season_weight
        * np.exp(
            -max(
                points - seventeenth_points,
                0,
            )
            / 6.0
        )
    )

    return {
        "title_pressure": float(
            title_pressure
        ),
        "europe_pressure": float(
            europe_pressure
        ),
        "relegation_pressure": float(
            relegation_pressure
        ),
        "season_progress": progress,
        "rank": rank,
    }


def build_team_features(
    team_id: str,
    current_date: pd.Timestamp,
    venue: str,
    histories: dict[str, list[dict]],
    standings: dict[str, dict],
    ranks: dict[str, float],
    leader_points: int,
    fourth_points: int,
    seventeenth_points: int,
) -> dict[str, float]:
    """生成一支球队在开球前可知的特征。"""
    records = histories[team_id]
    standing = standings[team_id]

    if records:
        rest_days = (
            current_date
            - records[-1]["date"]
        ).days
    else:
        rest_days = np.nan

    matches_last_14_days = sum(
        0
        < (
            current_date - record["date"]
        ).days
        <= 14
        for record in records
    )

    matches_last_30_days = sum(
        0
        < (
            current_date - record["date"]
        ).days
        <= 30
        for record in records
    )

    last_5 = rolling_summary(
        records,
        5,
    )

    last_10 = rolling_summary(
        records,
        10,
    )

    venue_records = [
        record
        for record in records
        if record["venue"] == venue
    ]

    venue_last_5 = rolling_summary(
        venue_records,
        5,
    )

    goal_difference = (
        standing["goals_for"]
        - standing["goals_against"]
    )

    if standing["played"] > 0:
        season_ppg = (
            standing["points"]
            / standing["played"]
        )

        season_goals_for_per_game = (
            standing["goals_for"]
            / standing["played"]
        )

        season_goals_against_per_game = (
            standing["goals_against"]
            / standing["played"]
        )
    else:
        season_ppg = np.nan
        season_goals_for_per_game = np.nan
        season_goals_against_per_game = (
            np.nan
        )

    pressures = calculate_pressures(
        standing,
        ranks[team_id],
        leader_points,
        fourth_points,
        seventeenth_points,
    )

    return {
        "matches_before": len(records),
        "rest_days": rest_days,
        "league_matches_last_14_days": (
            matches_last_14_days
        ),
        "league_matches_last_30_days": (
            matches_last_30_days
        ),
        "last5_ppg": last_5["ppg"],
        "last5_goals_for": (
            last_5["goals_for"]
        ),
        "last5_goals_against": (
            last_5["goals_against"]
        ),
        "last5_goal_difference": (
            last_5["goal_difference"]
        ),
        "last5_win_rate": (
            last_5["win_rate"]
        ),
        "last10_ppg": last_10["ppg"],
        "last10_goals_for": (
            last_10["goals_for"]
        ),
        "last10_goals_against": (
            last_10["goals_against"]
        ),
        "last10_goal_difference": (
            last_10["goal_difference"]
        ),
        "last10_win_rate": (
            last_10["win_rate"]
        ),
        "venue_last5_ppg": (
            venue_last_5["ppg"]
        ),
        "venue_last5_goals_for": (
            venue_last_5["goals_for"]
        ),
        "venue_last5_goals_against": (
            venue_last_5[
                "goals_against"
            ]
        ),
        "season_matches_played": (
            standing["played"]
        ),
        "season_points": (
            standing["points"]
        ),
        "season_ppg": season_ppg,
        "season_goals_for": (
            standing["goals_for"]
        ),
        "season_goals_against": (
            standing["goals_against"]
        ),
        "season_goal_difference": (
            goal_difference
        ),
        "season_goals_for_per_game": (
            season_goals_for_per_game
        ),
        "season_goals_against_per_game": (
            season_goals_against_per_game
        ),
        **pressures,
    }


def add_prefix(
    values: dict,
    prefix: str,
) -> dict:
    return {
        f"{prefix}_{key}": value
        for key, value in values.items()
    }


def result_points(
    home_goals: int,
    away_goals: int,
) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0

    if home_goals < away_goals:
        return 0, 3

    return 1, 1


def update_after_match(
    match,
    histories: dict[str, list[dict]],
    standings: dict[str, dict],
) -> None:
    """
    比赛完成后更新状态。

    此函数只在当天所有比赛的特征已经生成后调用。
    """
    home_points, away_points = (
        result_points(
            match.home_goals,
            match.away_goals,
        )
    )

    histories[match.home_team_id].append(
        {
            "date": match.match_date,
            "points": home_points,
            "goals_for": match.home_goals,
            "goals_against": match.away_goals,
            "venue": "home",
        }
    )

    histories[match.away_team_id].append(
        {
            "date": match.match_date,
            "points": away_points,
            "goals_for": match.away_goals,
            "goals_against": match.home_goals,
            "venue": "away",
        }
    )

    home_standing = standings[
        match.home_team_id
    ]

    away_standing = standings[
        match.away_team_id
    ]

    home_standing["played"] += 1
    home_standing["points"] += home_points
    home_standing["goals_for"] += (
        match.home_goals
    )
    home_standing["goals_against"] += (
        match.away_goals
    )

    away_standing["played"] += 1
    away_standing["points"] += away_points
    away_standing["goals_for"] += (
        match.away_goals
    )
    away_standing["goals_against"] += (
        match.home_goals
    )


def build_historical_features(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    按赛季、按比赛日期生成赛前特征。

    同一天的全部比赛先生成特征，再统一更新结果，
    防止同日比赛之间发生信息泄漏。
    """
    histories: dict[str, list[dict]] = (
        defaultdict(list)
    )

    feature_rows: list[dict] = []

    for season in sorted(
        matches["season"].unique()
    ):
        season_matches = matches[
            matches["season"] == season
        ].sort_values(
            [
                "match_date",
                "kickoff_utc",
                "match_id",
            ]
        )

        season_teams = sorted(
            set(season_matches["home_team_id"])
            | set(
                season_matches["away_team_id"]
            )
        )

        standings = {
            team_id: empty_standing()
            for team_id in season_teams
        }

        for (
            match_date,
            date_matches,
        ) in season_matches.groupby(
            "match_date",
            sort=True,
        ):
            (
                ranks,
                leader_points,
                fourth_points,
                seventeenth_points,
            ) = standing_snapshot(standings)

            day_matches = list(
                date_matches.itertuples(
                    index=False
                )
            )

            # 先生成当日所有比赛特征
            for match in day_matches:
                home_features = (
                    build_team_features(
                        match.home_team_id,
                        match_date,
                        "home",
                        histories,
                        standings,
                        ranks,
                        leader_points,
                        fourth_points,
                        seventeenth_points,
                    )
                )

                away_features = (
                    build_team_features(
                        match.away_team_id,
                        match_date,
                        "away",
                        histories,
                        standings,
                        ranks,
                        leader_points,
                        fourth_points,
                        seventeenth_points,
                    )
                )

                home_pressures = [
                    home_features[
                        "title_pressure"
                    ],
                    home_features[
                        "europe_pressure"
                    ],
                    home_features[
                        "relegation_pressure"
                    ],
                ]

                away_pressures = [
                    away_features[
                        "title_pressure"
                    ],
                    away_features[
                        "europe_pressure"
                    ],
                    away_features[
                        "relegation_pressure"
                    ],
                ]

                match_importance = max(
                    max(home_pressures),
                    max(away_pressures),
                )

                feature_rows.append(
                    {
                        "match_id": match.match_id,
                        "competition": (
                            match.competition
                        ),
                        "season": match.season,
                        "match_date": (
                            match.match_date
                        ),
                        "kickoff_utc": (
                            match.kickoff_utc
                        ),
                        "features_as_of": (
                            match.kickoff_utc
                        ),
                        "matchweek": (
                            match.matchweek
                        ),
                        "home_team_id": (
                            match.home_team_id
                        ),
                        "home_team": (
                            match.home_team
                        ),
                        "away_team_id": (
                            match.away_team_id
                        ),
                        "away_team": (
                            match.away_team
                        ),
                        **add_prefix(
                            home_features,
                            "home",
                        ),
                        **add_prefix(
                            away_features,
                            "away",
                        ),
                        "match_importance": (
                            match_importance
                        ),
                        "referee_known": (
                            pd.notna(
                                match.referee
                            )
                        ),
                        "home_goals": (
                            match.home_goals
                        ),
                        "away_goals": (
                            match.away_goals
                        ),
                        "result": match.result,
                    }
                )

            # 当天特征全部生成后才更新结果
            for match in day_matches:
                update_after_match(
                    match,
                    histories,
                    standings,
                )

    return pd.DataFrame(feature_rows)


def add_difference_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """生成主队减客队的差值特征。"""
    comparable_features = [
        "rest_days",
        "league_matches_last_14_days",
        "league_matches_last_30_days",
        "last5_ppg",
        "last5_goals_for",
        "last5_goals_against",
        "last5_goal_difference",
        "last5_win_rate",
        "last10_ppg",
        "last10_goals_for",
        "last10_goals_against",
        "last10_goal_difference",
        "last10_win_rate",
        "venue_last5_ppg",
        "venue_last5_goals_for",
        "venue_last5_goals_against",
        "season_matches_played",
        "season_points",
        "season_ppg",
        "season_goal_difference",
        "season_goals_for_per_game",
        "season_goals_against_per_game",
        "rank",
        "title_pressure",
        "europe_pressure",
        "relegation_pressure",
    ]

    for feature_name in comparable_features:
        features[
            f"{feature_name}_difference"
        ] = (
            features[
                f"home_{feature_name}"
            ]
            - features[
                f"away_{feature_name}"
            ]
        )

    return features


def merge_elo_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    elo = pd.read_parquet(
        ELO_FILE
    )[
        [
            "match_id",
            "home_elo",
            "away_elo",
            "elo_difference",
            "absolute_elo_difference",
            "elo_expected_home_score",
            "home_team_was_new",
            "away_team_was_new",
        ]
    ]

    return features.merge(
        elo,
        on="match_id",
        how="left",
        validate="one_to_one",
    )


def validate_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    if len(features) != 6080:
        raise ValueError(
            f"应有6080场，实际为"
            f"{len(features)}场。"
        )

    if features["match_id"].duplicated().any():
        raise ValueError(
            "特征表中存在重复比赛。"
        )

    season_counts = (
        features.groupby("season")
        .size()
    )

    invalid_seasons = season_counts[
        season_counts != 380
    ]

    if not invalid_seasons.empty:
        raise ValueError(
            "以下赛季场次不是380：\n"
            + invalid_seasons.to_string()
        )

    required_columns = [
        "match_id",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "home_elo",
        "away_elo",
    ]

    if (
        features[required_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "关键字段存在缺失值。"
        )

    audit_rows = [
        {
            "check": "total_matches",
            "value": len(features),
        },
        {
            "check": "seasons",
            "value": (
                features["season"].nunique()
            ),
        },
        {
            "check": "duplicate_match_ids",
            "value": int(
                features[
                    "match_id"
                ].duplicated().sum()
            ),
        },
        {
            "check": "earliest_season",
            "value": (
                features["season"].min()
            ),
        },
        {
            "check": "latest_season",
            "value": (
                features["season"].max()
            ),
        },
        {
            "check": "feature_columns",
            "value": len(features.columns),
        },
        {
            "check": (
                "missing_home_rest_days"
            ),
            "value": int(
                features[
                    "home_rest_days"
                ].isna().sum()
            ),
        },
        {
            "check": (
                "missing_away_rest_days"
            ),
            "value": int(
                features[
                    "away_rest_days"
                ].isna().sum()
            ),
        },
    ]

    return pd.DataFrame(audit_rows)


def main() -> None:
    matches = pd.read_parquet(
        MATCHES_FILE
    )

    print(
        "正在按时间顺序生成无泄漏历史特征……"
    )

    features = build_historical_features(
        matches
    )

    features = add_difference_features(
        features
    )

    features = merge_elo_features(
        features
    )

    features = features.sort_values(
        ["kickoff_utc", "match_id"]
    ).reset_index(drop=True)

    audit = validate_features(
        features
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n历史特征构建成功。")
    print(f"比赛总数：{len(features)}")
    print(f"总字段数：{len(features.columns)}")
    print(f"特征文件：{OUTPUT_FILE}")
    print(f"检查报告：{AUDIT_FILE}")
    print(
        "\n注意：当前休息天数和赛程密度"
        "只统计英超比赛。杯赛和欧战将在"
        "后续数据扩展层加入。"
    )
    print("\n检查结果：")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()