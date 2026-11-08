from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from baseline import (
    CLASS_ORDER,
    PROBABILITY_COLUMNS,
    calculate_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
)
CONFIG_FILE = PROJECT_ROOT / "configs" / "backtest.yaml"

BASELINE_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "baseline_metrics.csv"
)
POISSON_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "poisson_metrics.csv"
)
ELO_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "elo_metrics.csv"
)
DIXON_COLES_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "dixon_coles_metrics.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "xgboost_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "xgboost_metrics.csv"
)
IMPORTANCE_FILE = (
    PROJECT_ROOT / "reports" / "xgboost_feature_importance.csv"
)

TARGET_TO_NUMBER = {
    "H": 0,
    "D": 1,
    "A": 2,
}

EXCLUDED_COLUMNS = {
    "match_id",
    "competition",
    "season",
    "match_date",
    "kickoff_utc",
    "features_as_of",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
}


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    只选择数值或布尔类型的赛前特征。

    比分和比赛结果被明确排除，
    防止目标泄漏。
    """
    feature_columns: list[str] = []

    for column in data.columns:
        if column in EXCLUDED_COLUMNS:
            continue

        if (
            pd.api.types.is_numeric_dtype(
                data[column]
            )
            or pd.api.types.is_bool_dtype(
                data[column]
            )
        ):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError("没有找到可用特征。")

    return feature_columns


def prepare_feature_matrix(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    转换成XGBoost需要的数值矩阵。

    缺失值保留为NaN，由树模型原生处理。
    """
    matrix = data[
        feature_columns
    ].copy()

    for column in feature_columns:
        if pd.api.types.is_bool_dtype(
            matrix[column]
        ):
            matrix[column] = (
                matrix[column].astype(float)
            )

    matrix = matrix.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return matrix.astype("float32")


def prepare_labels(
    data: pd.DataFrame,
) -> np.ndarray:
    labels = (
        data["result"]
        .map(TARGET_TO_NUMBER)
    )

    if labels.isna().any():
        raise ValueError(
            "发现无法识别的比赛结果。"
        )

    return labels.to_numpy(dtype=int)


def make_dmatrix(
    data: pd.DataFrame,
    feature_columns: list[str],
    include_labels: bool,
) -> xgb.DMatrix:
    matrix = prepare_feature_matrix(
        data,
        feature_columns,
    )

    if include_labels:
        labels = prepare_labels(data)

        return xgb.DMatrix(
            matrix,
            label=labels,
            feature_names=feature_columns,
            missing=np.nan,
        )

    return xgb.DMatrix(
        matrix,
        feature_names=feature_columns,
        missing=np.nan,
    )


def xgboost_parameters() -> dict:
    """
    第一版使用保守参数，降低过拟合。

    参数调优只会在开发赛季内进行，
    不接触最终封存赛季。
    """
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "eta": 0.03,
        "max_depth": 3,
        "min_child_weight": 10.0,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.20,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "seed": 42,
        "nthread": 0,
        "verbosity": 0,
    }


def choose_boosting_rounds(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: list[str],
) -> int:
    """
    使用紧邻测试赛季之前的一个赛季选择树数量。

    验证赛季不参与这一步模型拟合，
    从而避免使用测试赛季信息。
    """
    training_matrix = make_dmatrix(
        training_data,
        feature_columns,
        include_labels=True,
    )

    validation_matrix = make_dmatrix(
        validation_data,
        feature_columns,
        include_labels=True,
    )

    selection_model = xgb.train(
        params=xgboost_parameters(),
        dtrain=training_matrix,
        num_boost_round=1200,
        evals=[
            (
                validation_matrix,
                "validation",
            )
        ],
        early_stopping_rounds=60,
        verbose_eval=False,
    )

    best_rounds = (
        selection_model.best_iteration + 1
    )

    return max(best_rounds, 1)


def train_final_model(
    training_data: pd.DataFrame,
    feature_columns: list[str],
    boosting_rounds: int,
) -> xgb.Booster:
    """
    确定树数量后，使用测试赛季之前的
    全部历史数据重新训练。
    """
    training_matrix = make_dmatrix(
        training_data,
        feature_columns,
        include_labels=True,
    )

    return xgb.train(
        params=xgboost_parameters(),
        dtrain=training_matrix,
        num_boost_round=boosting_rounds,
        verbose_eval=False,
    )


def evaluate_xgboost(
    data: pd.DataFrame,
    feature_columns: list[str],
    test_seasons: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    feature_gains: dict[str, list[float]] = (
        defaultdict(list)
    )

    for test_season in test_seasons:
        all_training_data = data[
            data["season"] < test_season
        ].copy()

        test_data = data[
            data["season"] == test_season
        ].copy()

        earlier_seasons = sorted(
            all_training_data[
                "season"
            ].unique()
        )

        if len(earlier_seasons) < 2:
            raise ValueError(
                f"{test_season}之前的赛季不足。"
            )

        validation_season = (
            earlier_seasons[-1]
        )

        selection_training_data = (
            all_training_data[
                all_training_data["season"]
                < validation_season
            ].copy()
        )

        validation_data = (
            all_training_data[
                all_training_data["season"]
                == validation_season
            ].copy()
        )

        print(
            f"\n正在回测 {test_season}："
            f"使用{validation_season}"
            "选择树数量……"
        )

        boosting_rounds = (
            choose_boosting_rounds(
                selection_training_data,
                validation_data,
                feature_columns,
            )
        )

        print(
            f"最佳树数量：{boosting_rounds}；"
            "正在使用全部历史重新训练……"
        )

        model = train_final_model(
            all_training_data,
            feature_columns,
            boosting_rounds,
        )

        test_matrix = make_dmatrix(
            test_data,
            feature_columns,
            include_labels=False,
        )

        probabilities = model.predict(
            test_matrix
        )

        if probabilities.shape != (
            len(test_data),
            3,
        ):
            raise ValueError(
                "XGBoost概率输出形状异常："
                f"{probabilities.shape}"
            )

        # 输出顺序固定为0、1、2，即H、D、A
        metrics = calculate_metrics(
            test_data["result"],
            probabilities,
        )

        predictions = test_data[
            [
                "match_id",
                "season",
                "match_date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "result",
            ]
        ].copy()

        predictions = predictions.rename(
            columns={
                "result": "actual_result"
            }
        )

        predictions["p_home"] = (
            probabilities[:, 0]
        )
        predictions["p_draw"] = (
            probabilities[:, 1]
        )
        predictions["p_away"] = (
            probabilities[:, 2]
        )

        predicted_indices = np.argmax(
            probabilities,
            axis=1,
        )

        predictions["predicted_result"] = [
            CLASS_ORDER[index]
            for index in predicted_indices
        ]

        predictions["boosting_rounds"] = (
            boosting_rounds
        )

        predictions["model"] = (
            "xgboost_historical_features"
        )

        metric_rows.append(
            {
                "season": test_season,
                "training_matches": len(
                    all_training_data
                ),
                "validation_season": (
                    validation_season
                ),
                "test_matches": len(
                    test_data
                ),
                "boosting_rounds": (
                    boosting_rounds
                ),
                **metrics,
            }
        )

        gain_scores = model.get_score(
            importance_type="gain"
        )

        for feature in feature_columns:
            feature_gains[feature].append(
                float(
                    gain_scores.get(
                        feature,
                        0.0,
                    )
                )
            )

        all_predictions.append(predictions)

        print(
            f"{test_season}完成："
            f"Log Loss={metrics['log_loss']:.4f}，"
            f"Accuracy={metrics['accuracy']:.2%}"
        )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_probabilities = (
        combined_predictions[
            PROBABILITY_COLUMNS
        ].to_numpy(dtype=float)
    )

    overall_metrics = calculate_metrics(
        combined_predictions["actual_result"],
        combined_probabilities,
    )

    metric_rows.append(
        {
            "season": "OVERALL",
            "training_matches": pd.NA,
            "validation_season": pd.NA,
            "test_matches": len(
                combined_predictions
            ),
            "boosting_rounds": pd.NA,
            **overall_metrics,
        }
    )

    importance_rows: list[dict] = []

    for feature, gains in feature_gains.items():
        importance_rows.append(
            {
                "feature": feature,
                "mean_gain": float(
                    np.mean(gains)
                ),
                "folds_used": int(
                    np.count_nonzero(gains)
                ),
            }
        )

    importance_table = pd.DataFrame(
        importance_rows
    ).sort_values(
        "mean_gain",
        ascending=False,
    )

    return (
        combined_predictions,
        pd.DataFrame(metric_rows),
        importance_table,
    )


def read_overall_metric(
    file_path: Path,
    metric_name: str,
) -> float:
    table = pd.read_csv(file_path)

    return float(
        table.loc[
            table["season"] == "OVERALL",
            metric_name,
        ].iloc[0]
    )


def main() -> None:
    config = load_config()

    development_seasons = config[
        "backtest"
    ]["development_test_seasons"]

    holdout_seasons = set(
        config[
            "backtest"
        ]["final_holdout_seasons"]
    )

    if holdout_seasons.intersection(
        development_seasons
    ):
        raise ValueError(
            "开发集与封存测试集发生重叠。"
        )

    data = pd.read_parquet(
        FEATURES_FILE
    )

    feature_columns = (
        select_feature_columns(data)
    )

    print(
        f"可用赛前特征数量："
        f"{len(feature_columns)}"
    )

    (
        predictions,
        metrics,
        importance,
    ) = evaluate_xgboost(
        data,
        feature_columns,
        development_seasons,
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

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    xgboost_overall = metrics.loc[
        metrics["season"] == "OVERALL"
    ].iloc[0]

    baseline_log_loss = read_overall_metric(
        BASELINE_METRICS_FILE,
        "log_loss",
    )

    poisson_log_loss = read_overall_metric(
        POISSON_METRICS_FILE,
        "log_loss",
    )

    elo_log_loss = read_overall_metric(
        ELO_METRICS_FILE,
        "log_loss",
    )

    dc_log_loss = read_overall_metric(
        DIXON_COLES_METRICS_FILE,
        "log_loss",
    )

    print("\nXGBoost回测完成。")
    print(f"开发回测场次：{len(predictions)}")
    print(
        f"XGBoost Log Loss："
        f"{xgboost_overall['log_loss']:.4f}"
    )
    print(
        f"Dixon-Coles Log Loss："
        f"{dc_log_loss:.4f}"
    )
    print(
        f"独立泊松Log Loss："
        f"{poisson_log_loss:.4f}"
    )
    print(
        f"Elo Log Loss："
        f"{elo_log_loss:.4f}"
    )
    print(
        f"联赛平均Log Loss："
        f"{baseline_log_loss:.4f}"
    )
    print(
        f"XGBoost Brier Score："
        f"{xgboost_overall['brier_score']:.4f}"
    )
    print(
        f"XGBoost RPS："
        f"{xgboost_overall['ranked_probability_score']:.4f}"
    )
    print(
        f"XGBoost Accuracy："
        f"{xgboost_overall['accuracy']:.2%}"
    )

    print("\n平均增益最高的10个特征：")
    print(
        importance.head(10).to_string(
            index=False
        )
    )

    print(f"\n预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(f"特征重要性：{IMPORTANCE_FILE}")
    print(
        "\n2024-25和2025-26仍处于封存状态。"
    )


if __name__ == "__main__":
    main()