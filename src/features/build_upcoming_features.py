# pyright: reportMissingImports=false

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = PROJECT_ROOT / "src" / "models"
FEATURES_DIRECTORY = PROJECT_ROOT / "src" / "features"

for directory in (
    MODELS_DIRECTORY,
    FEATURES_DIRECTORY,
):
    directory_text = str(directory)

    if directory_text not in sys.path:
        sys.path.insert(0, directory_text)


from build_features import (  # noqa: E402
    add_difference_features,
    build_historical_features,
)
from elo import build_pre_match_elo_features  # noqa: E402
from gradient_boosting import (  # noqa: E402
    select_feature_columns,
)


HISTORICAL_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches.parquet"
)

CURRENT_COMPLETED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches.parquet"
)

CURRENT_FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_fixtures.parquet"
)

FROZEN_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

UPCOMING_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "upcoming_features.parquet"
)

UPCOMING_FIXTURES_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "upcoming_fixtures.csv"
)


def check_required_files() -> None:
    required_files = [
        HISTORICAL_MATCHES_FILE,
        CURRENT_COMPLETED_FILE,
        CURRENT_FIXTURES_FILE,
        FROZEN_FEATURES_FILE,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            str(path) for path in missing_files
        )

        raise FileNotFoundError(
            "缺少生成未来特征所需的文件：\n"
            f"{missing_text}"
        )


def align_to_historical_schema(
    data: pd.DataFrame,
    historical_columns: list[str],
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in historical_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "当前赛季数据缺少历史字段：\n"
            f"{missing_columns}"
        )

    return data[historical_columns].copy()


def load_completed_matches() -> pd.DataFrame:
    historical = pd.read_parquet(
        HISTORICAL_MATCHES_FILE
    )

    current_completed = pd.read_parquet(
        CURRENT_COMPLETED_FILE
    )

    current_completed = align_to_historical_schema(
        current_completed,
        historical.columns.tolist(),
    )

    completed = pd.concat(
        [
            historical,
            current_completed,
        ],
        ignore_index=True,
    )

    completed["match_date"] = pd.to_datetime(
        completed["match_date"],
        errors="raise",
    )

    completed["kickoff_utc"] = pd.to_datetime(
        completed["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    completed["home_goals"] = pd.to_numeric(
        completed["home_goals"],
        errors="raise",
    ).astype(int)

    completed["away_goals"] = pd.to_numeric(
        completed["away_goals"],
        errors="raise",
    ).astype(int)

    if completed["match_id"].duplicated().any():
        raise RuntimeError(
            "历史比赛与当前已完赛比赛中存在重复ID。"
        )

    completed = completed.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    return completed


def select_next_fixtures(
    completed: pd.DataFrame,
) -> pd.DataFrame:
    fixtures = pd.read_parquet(
        CURRENT_FIXTURES_FILE
    )

    if fixtures.empty:
        raise RuntimeError(
            "未来赛程文件为空。"
        )

    fixtures["match_date"] = pd.to_datetime(
        fixtures["match_date"],
        errors="raise",
    )

    fixtures["kickoff_utc"] = pd.to_datetime(
        fixtures["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    now_utc = pd.Timestamp.now(tz="UTC")

    if "status" in fixtures.columns:
        scheduled_mask = (
            fixtures["status"]
            .astype("string")
            .str.lower()
            .eq("scheduled")
        )
    else:
        scheduled_mask = pd.Series(
            True,
            index=fixtures.index,
        )

    future_candidates = fixtures.loc[
        scheduled_mask
        & (fixtures["kickoff_utc"] > now_utc)
    ].copy()

    if future_candidates.empty:
        raise RuntimeError(
            "没有找到尚未开赛的未来比赛。"
        )

    future_candidates = future_candidates.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    available_matchweeks = (
        pd.to_numeric(
            future_candidates["matchweek"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    if available_matchweeks:
        first_matchweek = available_matchweeks[0]

        first_matchweek_count = int(
            (
                future_candidates["matchweek"]
                == first_matchweek
            ).sum()
        )

        selected_matchweeks = [first_matchweek]

        # 如果当前轮只剩少量比赛，
        # 同时加入下一个完整轮次。
        if (
            first_matchweek_count < 5
            and len(available_matchweeks) > 1
        ):
            selected_matchweeks.append(
                available_matchweeks[1]
            )

        selected = future_candidates.loc[
            future_candidates["matchweek"].isin(
                selected_matchweeks
            )
        ].copy()

        if len(selected_matchweeks) == 1:
            selection_method = (
                f"第{first_matchweek}轮"
            )
        else:
            selection_method = (
                f"第{selected_matchweeks[0]}轮剩余比赛"
                f" + 第{selected_matchweeks[1]}轮"
            )
    else:
        selected = future_candidates.head(10).copy()
        selection_method = "最近10场未来比赛"

    selected = selected.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    print(
        f"当前UTC时间：{now_utc}"
    )
    print(
        f"赛程选择方式：{selection_method}"
    )
    print(
        f"本次准备预测：{len(selected)}场"
    )

    return selected


def create_synthetic_fixture(
    fixture: pd.Series,
    historical_columns: list[str],
) -> pd.DataFrame:
    """
    创建仅用于计算赛前特征的临时比赛行。

    临时比分不会进入最终输出，也不会影响其他比赛，
    因为每一场未来比赛都会独立计算。
    """

    fixture_frame = fixture.to_frame().T

    fixture_frame["home_goals"] = 0
    fixture_frame["away_goals"] = 0
    fixture_frame["result"] = "D"

    fixture_frame = align_to_historical_schema(
        fixture_frame,
        historical_columns,
    )

    fixture_frame["home_goals"] = (
        fixture_frame["home_goals"].astype(int)
    )

    fixture_frame["away_goals"] = (
        fixture_frame["away_goals"].astype(int)
    )

    fixture_frame["match_date"] = pd.to_datetime(
        fixture_frame["match_date"],
        errors="raise",
    )

    fixture_frame["kickoff_utc"] = pd.to_datetime(
        fixture_frame["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    return fixture_frame


def build_one_fixture_features(
    completed: pd.DataFrame,
    fixture: pd.Series,
    frozen_features: pd.DataFrame,
) -> pd.DataFrame:
    synthetic_fixture = create_synthetic_fixture(
        fixture,
        completed.columns.tolist(),
    )

    fixture_match_id = str(
        synthetic_fixture.iloc[0]["match_id"]
    )

    calculation_matches = pd.concat(
        [
            completed,
            synthetic_fixture,
        ],
        ignore_index=True,
    )

    calculation_matches = calculation_matches.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    base_features = build_historical_features(
        calculation_matches
    )

    elo_features = build_pre_match_elo_features(
        calculation_matches
    )

    direct_elo_columns = [
        column
        for column in elo_features.columns
        if (
            column != "match_id"
            and column in frozen_features.columns
            and column not in base_features.columns
        )
    ]

    features = base_features.merge(
        elo_features[
            ["match_id", *direct_elo_columns]
        ],
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    features = add_difference_features(features)

    fixture_features = features.loc[
        features["match_id"] == fixture_match_id
    ].copy()

    if len(fixture_features) != 1:
        raise RuntimeError(
            f"比赛{fixture_match_id}应生成1行特征，"
            f"实际生成{len(fixture_features)}行。"
        )

    # 删除临时赛果，防止被误认为真实结果。
    for column in (
        "home_goals",
        "away_goals",
        "result",
    ):
        if column in fixture_features.columns:
            fixture_features[column] = pd.NA

    return fixture_features


def main() -> None:
    check_required_files()

    print("开始生成下一轮比赛的无泄漏赛前特征。")

    completed = load_completed_matches()
    selected_fixtures = select_next_fixtures(
        completed
    )

    frozen_features = pd.read_parquet(
        FROZEN_FEATURES_FILE
    )

    frozen_feature_columns = select_feature_columns(
        frozen_features
    )

    print(
        "冻结的XGBoost特征数量："
        f"{len(frozen_feature_columns)}"
    )

    feature_rows = []

    for index, fixture in selected_fixtures.iterrows():
        print(
            f"{index + 1}/{len(selected_fixtures)} "
            f"{fixture['home_team']} vs "
            f"{fixture['away_team']}"
        )

        fixture_features = build_one_fixture_features(
            completed=completed,
            fixture=fixture,
            frozen_features=frozen_features,
        )

        feature_rows.append(fixture_features)

    upcoming_features = pd.concat(
        feature_rows,
        ignore_index=True,
    )

    missing_feature_columns = [
        column
        for column in frozen_feature_columns
        if column not in upcoming_features.columns
    ]

    if missing_feature_columns:
        raise RuntimeError(
            "未来比赛特征缺少冻结模型字段：\n"
            f"{missing_feature_columns}"
        )

    if upcoming_features[
        "match_id"
    ].duplicated().any():
        raise RuntimeError(
            "未来比赛特征中存在重复比赛ID。"
        )

    if len(upcoming_features) != len(
        selected_fixtures
    ):
        raise RuntimeError(
            "未来比赛数与特征行数不一致。"
        )

    UPCOMING_FEATURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    UPCOMING_FIXTURES_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    upcoming_features.to_parquet(
        UPCOMING_FEATURES_FILE,
        index=False,
    )

    report_columns = [
        "match_id",
        "season",
        "matchweek",
        "match_date",
        "kickoff_utc",
        "home_team",
        "away_team",
    ]

    selected_fixtures[
        report_columns
    ].to_csv(
        UPCOMING_FIXTURES_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n下一轮赛前特征生成成功。")
    print(f"比赛数量：{len(upcoming_features)}")
    print(
        f"每场XGBoost特征数量："
        f"{len(frozen_feature_columns)}"
    )
    print(f"特征文件：{UPCOMING_FEATURES_FILE}")
    print(
        f"赛程报告：{UPCOMING_FIXTURES_REPORT}"
    )

    print("\n即将预测的比赛：")

    print(
        selected_fixtures[
            [
                "matchweek",
                "match_date",
                "home_team",
                "away_team",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()