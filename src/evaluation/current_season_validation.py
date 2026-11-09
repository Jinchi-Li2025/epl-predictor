# pyright: reportMissingImports=false

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = PROJECT_ROOT / "src" / "models"
FEATURES_DIRECTORY = PROJECT_ROOT / "src" / "features"
EVALUATION_DIRECTORY = PROJECT_ROOT / "src" / "evaluation"

for directory in (
    MODELS_DIRECTORY,
    FEATURES_DIRECTORY,
    EVALUATION_DIRECTORY,
):
    directory_text = str(directory)

    if directory_text not in sys.path:
        sys.path.insert(0, directory_text)


from build_features import (  # noqa: E402
    add_difference_features,
    build_historical_features,
    validate_features,
)
from elo import build_pre_match_elo_features  # noqa: E402
from final_holdout import (  # noqa: E402
    add_overall_metrics,
    evaluate_holdout_season,
)
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

FROZEN_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

ENSEMBLE_CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "ensemble_weights.json"
)

CURRENT_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_features.parquet"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_external_validation_predictions.parquet"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_external_validation_metrics.csv"
)

TEST_SEASON = "2026-27"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_required_files() -> None:
    required_files = [
        HISTORICAL_MATCHES_FILE,
        CURRENT_COMPLETED_FILE,
        FROZEN_FEATURES_FILE,
        ENSEMBLE_CONFIG_FILE,
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
            "缺少外部验证所需文件：\n"
            f"{missing_text}"
        )


def align_current_schema(
    historical_matches: pd.DataFrame,
    current_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    让当前赛季数据采用与历史比赛完全相同的字段顺序。

    current_completed_matches 中额外的 status 字段不会进入
    历史训练数据。
    """

    missing_columns = [
        column
        for column in historical_matches.columns
        if column not in current_matches.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "当前赛季数据缺少历史比赛字段：\n"
            f"{missing_columns}"
        )

    aligned = current_matches[
        historical_matches.columns
    ].copy()

    return aligned


def combine_matches() -> pd.DataFrame:
    historical_matches = pd.read_parquet(
        HISTORICAL_MATCHES_FILE
    )

    current_matches = pd.read_parquet(
        CURRENT_COMPLETED_FILE
    )

    if current_matches.empty:
        raise RuntimeError(
            "当前赛季没有已完赛比赛，无法进行外部验证。"
        )

    current_seasons = set(
        current_matches["season"]
        .dropna()
        .astype(str)
        .unique()
    )

    if current_seasons != {TEST_SEASON}:
        raise RuntimeError(
            "当前比赛文件的赛季不符合预期："
            f"{sorted(current_seasons)}"
        )

    if current_matches[
        ["home_goals", "away_goals", "result"]
    ].isna().any().any():
        raise RuntimeError(
            "已完赛文件中存在缺失的比分或赛果。"
        )

    aligned_current = align_current_schema(
        historical_matches,
        current_matches,
    )

    combined = pd.concat(
        [
            historical_matches,
            aligned_current,
        ],
        ignore_index=True,
    )

    combined["match_date"] = pd.to_datetime(
        combined["match_date"],
        errors="raise",
    )

    combined["kickoff_utc"] = pd.to_datetime(
        combined["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    combined["home_goals"] = pd.to_numeric(
        combined["home_goals"],
        errors="raise",
    ).astype(int)

    combined["away_goals"] = pd.to_numeric(
        combined["away_goals"],
        errors="raise",
    ).astype(int)

    duplicate_match_ids = int(
        combined["match_id"].duplicated().sum()
    )

    if duplicate_match_ids:
        raise RuntimeError(
            "合并数据中出现重复比赛 ID："
            f"{duplicate_match_ids}"
        )

    combined = combined.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    return combined


def build_external_features(
    combined_matches: pd.DataFrame,
    frozen_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用与历史模型相同的方法重建全部时间序列特征。

    特征计算遵守时间顺序，每场比赛只能看到比赛开始前的资料。
    """

    print("\n正在重建无泄漏历史特征……")

    base_features = build_historical_features(
        combined_matches
    )

    print("正在重建赛前Elo特征……")

    elo_features = build_pre_match_elo_features(
        combined_matches
    )

    if base_features["match_id"].duplicated().any():
        raise RuntimeError(
            "基础特征中出现重复比赛 ID。"
        )

    if elo_features["match_id"].duplicated().any():
        raise RuntimeError(
            "Elo特征中出现重复比赛 ID。"
        )

    # 只合并冻结模型过去实际使用过的Elo字段，
    # 防止新字段意外进入模型。
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
    expected_rows = len(combined_matches)

    if len(features) != expected_rows:
        raise RuntimeError(
            "重建特征数量不正确："
            f"应有{expected_rows}行，"
            f"实际为{len(features)}行。"
        )

    if features["match_id"].isna().any():
        raise RuntimeError(
            "重建特征中存在空比赛ID。"
        )

    if features["match_id"].duplicated().any():
        raise RuntimeError(
            "重建特征中存在重复比赛ID。"
        )

    current_match_ids = set(
        combined_matches.loc[
            combined_matches["season"] == TEST_SEASON,
            "match_id",
        ]
    )

    current_features = features.loc[
        features["match_id"].isin(current_match_ids)
    ].copy()

    if len(current_features) != len(current_match_ids):
        raise RuntimeError(
            "当前赛季特征行数与已完赛比赛数不一致："
            f"特征={len(current_features)}，"
            f"比赛={len(current_match_ids)}"
        )

    return features, elo_features


def select_frozen_feature_columns(
    frozen_features: pd.DataFrame,
    rebuilt_features: pd.DataFrame,
) -> list[str]:
    """
    特征名单只从原有冻结历史特征中选择。

    2026-27数据不能参与特征选择。
    """

    feature_columns = select_feature_columns(
        frozen_features
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in rebuilt_features.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "重建后的特征缺少冻结模型字段：\n"
            f"{missing_columns}"
        )

    print(
        f"冻结的XGBoost特征数量："
        f"{len(feature_columns)}"
    )

    return feature_columns


def print_metric_table(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        print("\n没有生成指标结果。")
        return

    overall = metrics.loc[
        metrics["season"] == "OVERALL"
    ].copy()

    if overall.empty:
        overall = metrics.loc[
            metrics["season"] == TEST_SEASON
        ].copy()

    overall = overall.sort_values(
        "log_loss",
        ascending=True,
    )

    display_columns = [
        "model",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "accuracy",
    ]

    print("\n2026-27外部验证结果：")
    print(
        overall[display_columns].to_string(
            index=False
        )
    )

    best_row = overall.iloc[0]

    current_match_count = int(
        best_row["test_matches"]
    )

    print(
        f"\n当前{current_match_count}场中"
        "Log Loss最好的模型："
        f"{best_row['model']} "
        f"({best_row['log_loss']:.4f})"
    )

    ensemble_rows = overall.loc[
        overall["model"] == "ensemble"
    ]

    league_rows = overall.loc[
        overall["model"] == "league_average"
    ]

    if (
        not ensemble_rows.empty
        and not league_rows.empty
    ):
        ensemble_log_loss = float(
            ensemble_rows.iloc[0]["log_loss"]
        )

        league_log_loss = float(
            league_rows.iloc[0]["log_loss"]
        )

        improvement = (
            league_log_loss - ensemble_log_loss
        )

        improvement_percent = (
            improvement / league_log_loss * 100
        )

        print(
            "冻结集成相对联赛平均Log Loss改善："
            f"{improvement:.4f} "
            f"({improvement_percent:.2f}%)"
        )


def main() -> None:
    check_required_files()

    print("开始2026-27赛季外部验证。")
    print(
        "本次不会修改2024-25和2025-26的"
        "最终留出测试结果。"
    )

    combined_matches = combine_matches()

    historical_count = int(
        (
            combined_matches["season"]
            != TEST_SEASON
        ).sum()
    )

    test_count = int(
        (
            combined_matches["season"]
            == TEST_SEASON
        ).sum()
    )

    print(f"\n历史训练比赛：{historical_count}")
    print(f"外部验证比赛：{test_count}")

    frozen_features = pd.read_parquet(
        FROZEN_FEATURES_FILE
    )

    ensemble_config = load_json(
        ENSEMBLE_CONFIG_FILE
    )

    rebuilt_features, elo_features = (
        build_external_features(
            combined_matches,
            frozen_features,
        )
    )

    feature_columns = (
        select_frozen_feature_columns(
            frozen_features,
            rebuilt_features,
        )
    )

    print(
        "\n开始使用截至2025-26的数据训练模型，"
        "并预测2026-27已完赛比赛……"
    )

    predictions, metric_rows = (
        evaluate_holdout_season(
            test_season=TEST_SEASON,
            matches=combined_matches,
            features=rebuilt_features,
            elo_features=elo_features,
            feature_columns=feature_columns,
            ensemble_config=ensemble_config,
            expected_test_matches=test_count,
        )
    )

    add_overall_metrics(
        predictions,
        metric_rows,
    )

    metrics = pd.DataFrame(metric_rows)

    current_features = rebuilt_features.loc[
        rebuilt_features["season"]
        == TEST_SEASON
    ].copy()

    CURRENT_FEATURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_features.to_parquet(
        CURRENT_FEATURES_FILE,
        index=False,
    )

    predictions.to_parquet(
        PREDICTIONS_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_metric_table(metrics)

    print("\n外部验证完成。")
    print(f"特征文件：{CURRENT_FEATURES_FILE}")
    print(f"预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")

    print(
        f"\n注意：目前只有{test_count}场，结果会比760场"
        "最终留出测试更容易波动。"
    )


if __name__ == "__main__":
    main()
