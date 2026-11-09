from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCHES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "matches.parquet"
)
CONFIG_FILE = PROJECT_ROOT / "configs" / "backtest.yaml"

PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "baseline_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "baseline_metrics.csv"
)

# 概率列的固定顺序
CLASS_ORDER = ["H", "D", "A"]
PROBABILITY_COLUMNS = ["p_home", "p_draw", "p_away"]


def load_config() -> dict:
    """读取回测配置。"""
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_one_hot(actual_results: pd.Series) -> np.ndarray:
    """把H、D、A转换成评估指标需要的0/1矩阵。"""
    class_to_index = {
        result: index
        for index, result in enumerate(CLASS_ORDER)
    }

    one_hot = np.zeros(
        (len(actual_results), len(CLASS_ORDER)),
        dtype=float,
    )

    for row_index, result in enumerate(actual_results):
        one_hot[
            row_index,
            class_to_index[result],
        ] = 1.0

    return one_hot


def calculate_metrics(
    actual_results: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """计算概率预测指标。"""
    one_hot = create_one_hot(actual_results)

    class_to_index = {
        result: index
        for index, result in enumerate(CLASS_ORDER)
    }

    actual_indices = np.array(
        [class_to_index[result] for result in actual_results],
        dtype=int,
    )

    safe_probabilities = np.clip(
        probabilities,
        1e-15,
        1.0,
    )

    # 多分类Log Loss：越低越好
    log_loss = -np.mean(
        np.log(
            safe_probabilities[
                np.arange(len(actual_indices)),
                actual_indices,
            ]
        )
    )

    # 多分类Brier Score：越低越好
    brier_score = np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1,
        )
    )

    # Ranked Probability Score：越低越好
    predicted_cumulative = np.cumsum(
        probabilities,
        axis=1,
    )[:, :-1]

    actual_cumulative = np.cumsum(
        one_hot,
        axis=1,
    )[:, :-1]

    ranked_probability_score = np.mean(
        np.mean(
            (
                predicted_cumulative
                - actual_cumulative
            )
            ** 2,
            axis=1,
        )
    )

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = np.mean(
        predicted_indices == actual_indices
    )

    return {
        "log_loss": float(log_loss),
        "brier_score": float(brier_score),
        "ranked_probability_score": float(
            ranked_probability_score
        ),
        "accuracy": float(accuracy),
    }


def evaluate_baseline(
    matches: pd.DataFrame,
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    进行按赛季滚动回测。

    每个测试赛季只能使用该赛季之前的比赛计算
    英超平均主胜、平局和客胜概率。
    """
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    for test_season in test_seasons:
        training_matches = matches[
            matches["season"] < test_season
        ].copy()

        test_matches = matches[
            matches["season"] == test_season
        ].copy()

        if training_matches.empty:
            raise ValueError(
                f"{test_season}之前没有训练数据。"
            )

        if len(test_matches) != 380:
            raise ValueError(
                f"{test_season}测试集应为380场，"
                f"实际为{len(test_matches)}场。"
            )

        result_counts = (
            training_matches["result"]
            .value_counts()
            .reindex(CLASS_ORDER, fill_value=0)
        )

        league_probabilities = (
            result_counts / result_counts.sum()
        ).to_numpy(dtype=float)

        probabilities = np.tile(
            league_probabilities,
            (len(test_matches), 1),
        )

        metrics = calculate_metrics(
            test_matches["result"],
            probabilities,
        )

        metric_rows.append(
            {
                "season": test_season,
                "training_matches": len(training_matches),
                "test_matches": len(test_matches),
                "p_home": league_probabilities[0],
                "p_draw": league_probabilities[1],
                "p_away": league_probabilities[2],
                **metrics,
            }
        )

        predictions = test_matches[
            [
                "match_id",
                "season",
                "match_date",
                "home_team",
                "away_team",
                "result",
            ]
        ].copy()

        predictions = predictions.rename(
            columns={"result": "actual_result"}
        )

        predictions["p_home"] = probabilities[:, 0]
        predictions["p_draw"] = probabilities[:, 1]
        predictions["p_away"] = probabilities[:, 2]

        predictions["predicted_result"] = [
            CLASS_ORDER[index]
            for index in np.argmax(
                probabilities,
                axis=1,
            )
        ]

        predictions["model"] = (
            "league_average_baseline"
        )

        all_predictions.append(predictions)

        print(
            f"{test_season}："
            f"主胜={league_probabilities[0]:.3f}，"
            f"平局={league_probabilities[1]:.3f}，"
            f"客胜={league_probabilities[2]:.3f}，"
            f"Log Loss={metrics['log_loss']:.4f}"
        )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_probabilities = combined_predictions[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    overall_metrics = calculate_metrics(
        combined_predictions["actual_result"],
        combined_probabilities,
    )

    metric_rows.append(
        {
            "season": "OVERALL",
            "training_matches": pd.NA,
            "test_matches": len(combined_predictions),
            "p_home": pd.NA,
            "p_draw": pd.NA,
            "p_away": pd.NA,
            **overall_metrics,
        }
    )

    metrics_table = pd.DataFrame(metric_rows)

    return combined_predictions, metrics_table


def main() -> None:
    """运行联赛平均概率基准回测。"""
    config = load_config()

    development_seasons = config[
        "backtest"
    ]["development_test_seasons"]

    holdout_seasons = set(
        config["backtest"]["final_holdout_seasons"]
    )

    if holdout_seasons.intersection(development_seasons):
        raise ValueError(
            "开发测试集与最终保留测试集发生重叠。"
        )

    matches = pd.read_parquet(MATCHES_FILE)

    predictions, metrics = evaluate_baseline(
        matches,
        development_seasons,
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    overall = metrics[
        metrics["season"] == "OVERALL"
    ].iloc[0]

    print("\n基准模型回测完成。")
    print(f"开发回测场次：{len(predictions)}")
    print(
        f"总体Log Loss："
        f"{overall['log_loss']:.4f}"
    )
    print(
        f"总体Brier Score："
        f"{overall['brier_score']:.4f}"
    )
    print(
        f"总体RPS："
        f"{overall['ranked_probability_score']:.4f}"
    )
    print(
        f"总体Accuracy："
        f"{overall['accuracy']:.2%}"
    )
    print(f"预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(
        "\n2024-25和2025-26仍处于封存状态，"
        "本程序没有使用它们。"
    )


if __name__ == "__main__":
    main()