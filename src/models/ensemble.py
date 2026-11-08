from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import (
    minimize,
    minimize_scalar,
)

from baseline import (
    CLASS_ORDER,
    PROBABILITY_COLUMNS,
    calculate_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = PROJECT_ROOT / "configs" / "backtest.yaml"

DIXON_COLES_FILE = (
    PROJECT_ROOT / "reports" / "dixon_coles_predictions.parquet"
)
XGBOOST_FILE = (
    PROJECT_ROOT / "reports" / "xgboost_predictions.parquet"
)
ELO_FILE = (
    PROJECT_ROOT / "reports" / "elo_predictions.parquet"
)
POISSON_FILE = (
    PROJECT_ROOT / "reports" / "poisson_predictions.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "reports" / "ensemble_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "ensemble_metrics.csv"
)
ENSEMBLE_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "ensemble_weights.json"
)

MODEL_NAMES = [
    "dixon_coles",
    "xgboost",
    "elo",
    "poisson",
]

WEIGHT_REGULARIZATION = 0.0001


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def rename_probability_columns(
    data: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    return data.rename(
        columns={
            "p_home": f"{model_name}_p_home",
            "p_draw": f"{model_name}_p_draw",
            "p_away": f"{model_name}_p_away",
        }
    )


def load_and_merge_predictions() -> pd.DataFrame:
    """
    按match_id合并四个模型的滚动回测概率。
    """
    dixon_coles = pd.read_parquet(
        DIXON_COLES_FILE
    )

    dixon_coles = rename_probability_columns(
        dixon_coles,
        "dixon_coles",
    )

    keep_dc_columns = [
        "match_id",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "actual_home_goals",
        "actual_away_goals",
        "actual_result",
        "home_expected_goals",
        "away_expected_goals",
        "most_likely_score",
        "top_5_scores",
        "dixon_coles_p_home",
        "dixon_coles_p_draw",
        "dixon_coles_p_away",
    ]

    merged = dixon_coles[
        keep_dc_columns
    ].copy()

    model_files = {
        "xgboost": XGBOOST_FILE,
        "elo": ELO_FILE,
        "poisson": POISSON_FILE,
    }

    for model_name, file_path in model_files.items():
        model_data = pd.read_parquet(
            file_path
        )

        model_data = rename_probability_columns(
            model_data,
            model_name,
        )

        probability_columns = [
            "match_id",
            f"{model_name}_p_home",
            f"{model_name}_p_draw",
            f"{model_name}_p_away",
        ]

        merged = merged.merge(
            model_data[probability_columns],
            on="match_id",
            how="inner",
            validate="one_to_one",
        )

    merged = merged.sort_values(
        ["match_date", "match_id"]
    ).reset_index(drop=True)

    if len(merged) != 3040:
        raise ValueError(
            f"合并后应有3040场，"
            f"实际为{len(merged)}场。"
        )

    if merged["match_id"].duplicated().any():
        raise ValueError(
            "集成数据中出现重复比赛。"
        )

    return merged


def create_probability_cube(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    返回形状：

    比赛数 × 模型数 × 3种结果。
    """
    model_probabilities: list[np.ndarray] = []

    for model_name in MODEL_NAMES:
        columns = [
            f"{model_name}_p_home",
            f"{model_name}_p_draw",
            f"{model_name}_p_away",
        ]

        probabilities = data[
            columns
        ].to_numpy(dtype=float)

        if not np.isfinite(
            probabilities
        ).all():
            raise ValueError(
                f"{model_name}包含无效概率。"
            )

        row_sums = probabilities.sum(
            axis=1
        )

        if not np.allclose(
            row_sums,
            1.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{model_name}概率之和不为1。"
            )

        model_probabilities.append(
            probabilities
        )

    return np.stack(
        model_probabilities,
        axis=1,
    )


def result_indices(
    actual_results: pd.Series,
) -> np.ndarray:
    result_to_index = {
        result: index
        for index, result in enumerate(
            CLASS_ORDER
        )
    }

    return np.array(
        [
            result_to_index[result]
            for result in actual_results
        ],
        dtype=int,
    )


def weighted_probabilities(
    probability_cube: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    probabilities = np.einsum(
        "nmc,m->nc",
        probability_cube,
        weights,
    )

    return (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )


def probability_log_loss(
    probabilities: np.ndarray,
    actual_indices: np.ndarray,
) -> float:
    safe_probabilities = np.clip(
        probabilities,
        1e-15,
        1.0,
    )

    return float(
        -np.mean(
            np.log(
                safe_probabilities[
                    np.arange(
                        len(actual_indices)
                    ),
                    actual_indices,
                ]
            )
        )
    )


def optimise_weights(
    probability_cube: np.ndarray,
    actual_indices: np.ndarray,
) -> np.ndarray:
    """
    寻找非负且总和为1的集成权重。
    """
    initial_weights = np.array(
        [0.40, 0.30, 0.10, 0.20],
        dtype=float,
    )

    def objective(
        weights: np.ndarray,
    ) -> float:
        probabilities = weighted_probabilities(
            probability_cube,
            weights,
        )

        log_loss = probability_log_loss(
            probabilities,
            actual_indices,
        )

        regularisation = (
            WEIGHT_REGULARIZATION
            * np.sum(weights**2)
        )

        return log_loss + regularisation

    constraints = [
        {
            "type": "eq",
            "fun": (
                lambda weights:
                weights.sum() - 1.0
            ),
        }
    ]

    bounds = [
        (0.0, 1.0)
        for _ in MODEL_NAMES
    ]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "集成权重优化失败："
            f"{result.message}"
        )

    weights = np.clip(
        result.x,
        0.0,
        None,
    )

    return weights / weights.sum()


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    温度大于1会降低过度自信，
    小于1会增强概率差异。
    """
    log_probabilities = np.log(
        np.clip(
            probabilities,
            1e-15,
            1.0,
        )
    )

    scaled_logits = (
        log_probabilities / temperature
    )

    scaled_logits = (
        scaled_logits
        - scaled_logits.max(
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        scaled_logits
    )

    return (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )


def optimise_temperature(
    probabilities: np.ndarray,
    actual_indices: np.ndarray,
) -> float:
    """
    在开发回测集上寻找概率温度。
    """
    def objective(
        log_temperature: float,
    ) -> float:
        temperature = float(
            np.exp(log_temperature)
        )

        calibrated = apply_temperature(
            probabilities,
            temperature,
        )

        return probability_log_loss(
            calibrated,
            actual_indices,
        )

    result = minimize_scalar(
        objective,
        bounds=(
            np.log(0.50),
            np.log(2.00),
        ),
        method="bounded",
        options={
            "xatol": 1e-8,
        },
    )

    if not result.success:
        raise RuntimeError(
            "温度校准失败。"
        )

    return float(
        np.exp(result.x)
    )


def calculate_metrics_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    metric_rows: list[dict] = []

    for season in sorted(
        data["season"].unique()
    ):
        season_data = data[
            data["season"] == season
        ]

        probabilities = season_data[
            PROBABILITY_COLUMNS
        ].to_numpy(dtype=float)

        metrics = calculate_metrics(
            season_data["actual_result"],
            probabilities,
        )

        metric_rows.append(
            {
                "season": season,
                "test_matches": len(
                    season_data
                ),
                **metrics,
            }
        )

    overall_probabilities = data[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    overall_metrics = calculate_metrics(
        data["actual_result"],
        overall_probabilities,
    )

    metric_rows.append(
        {
            "season": "OVERALL",
            "test_matches": len(data),
            **overall_metrics,
        }
    )

    return pd.DataFrame(metric_rows)


def component_metrics(
    data: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    probabilities = data[
        [
            f"{model_name}_p_home",
            f"{model_name}_p_draw",
            f"{model_name}_p_away",
        ]
    ].to_numpy(dtype=float)

    return calculate_metrics(
        data["actual_result"],
        probabilities,
    )


def main() -> None:
    config = load_config()

    development_seasons = config[
        "backtest"
    ]["development_test_seasons"]

    holdout_seasons = config[
        "backtest"
    ]["final_holdout_seasons"]

    data = load_and_merge_predictions()

    actual_seasons = sorted(
        data["season"].unique()
    )

    if actual_seasons != sorted(
        development_seasons
    ):
        raise ValueError(
            "集成数据的赛季范围与配置不一致。"
        )

    probability_cube = (
        create_probability_cube(data)
    )

    actual_indices = result_indices(
        data["actual_result"]
    )

    weights = optimise_weights(
        probability_cube,
        actual_indices,
    )

    uncalibrated_probabilities = (
        weighted_probabilities(
            probability_cube,
            weights,
        )
    )

    uncalibrated_log_loss = (
        probability_log_loss(
            uncalibrated_probabilities,
            actual_indices,
        )
    )

    temperature = optimise_temperature(
        uncalibrated_probabilities,
        actual_indices,
    )

    calibrated_probabilities = (
        apply_temperature(
            uncalibrated_probabilities,
            temperature,
        )
    )

    calibrated_log_loss = (
        probability_log_loss(
            calibrated_probabilities,
            actual_indices,
        )
    )

    data["p_home"] = (
        calibrated_probabilities[:, 0]
    )

    data["p_draw"] = (
        calibrated_probabilities[:, 1]
    )

    data["p_away"] = (
        calibrated_probabilities[:, 2]
    )

    predicted_indices = np.argmax(
        calibrated_probabilities,
        axis=1,
    )

    data["predicted_result"] = [
        CLASS_ORDER[index]
        for index in predicted_indices
    ]

    data["model"] = (
        "calibrated_weighted_ensemble"
    )

    metrics = calculate_metrics_table(
        data
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    component_results = {
        model_name: component_metrics(
            data,
            model_name,
        )
        for model_name in MODEL_NAMES
    }

    ensemble_config = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_names": MODEL_NAMES,
        "weights": {
            model_name: float(weight)
            for model_name, weight
            in zip(MODEL_NAMES, weights)
        },
        "temperature": temperature,
        "fitted_on_seasons": (
            development_seasons
        ),
        "final_holdout_seasons": (
            holdout_seasons
        ),
        "primary_metric": "log_loss",
        "uncalibrated_log_loss": (
            uncalibrated_log_loss
        ),
        "calibrated_log_loss": (
            calibrated_log_loss
        ),
        "component_log_losses": {
            model_name: (
                values["log_loss"]
            )
            for model_name, values
            in component_results.items()
        },
    }

    with ENSEMBLE_CONFIG_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            ensemble_config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    overall = metrics[
        metrics["season"] == "OVERALL"
    ].iloc[0]

    print("集成权重优化完成。\n")

    for model_name, weight in zip(
        MODEL_NAMES,
        weights,
    ):
        print(
            f"{model_name:15s}："
            f"{weight:.2%}"
        )

    print(
        f"\n温度参数：{temperature:.4f}"
    )

    print(
        f"校准前Log Loss："
        f"{uncalibrated_log_loss:.4f}"
    )

    print(
        f"校准后Log Loss："
        f"{calibrated_log_loss:.4f}"
    )

    print("\n开发集总体表现：")
    print(
        f"Log Loss："
        f"{overall['log_loss']:.4f}"
    )
    print(
        f"Brier Score："
        f"{overall['brier_score']:.4f}"
    )
    print(
        f"RPS："
        f"{overall['ranked_probability_score']:.4f}"
    )
    print(
        f"Accuracy："
        f"{overall['accuracy']:.2%}"
    )

    print("\n各组件Log Loss：")

    for model_name in MODEL_NAMES:
        print(
            f"{model_name:15s}："
            f"{component_results[model_name]['log_loss']:.4f}"
        )

    print(f"\n预测文件：{OUTPUT_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(
        f"冻结权重文件："
        f"{ENSEMBLE_CONFIG_FILE}"
    )

    print(
        "\n注意：这些权重是在开发回测集上选择的。"
        "下一阶段将冻结权重，并首次开启"
        "2024-25和2025-26最终测试集。"
    )


if __name__ == "__main__":
    main()