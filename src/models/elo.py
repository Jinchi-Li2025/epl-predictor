from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from baseline import (
    CLASS_ORDER,
    PROBABILITY_COLUMNS,
    calculate_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCHES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "matches.parquet"
)
CONFIG_FILE = PROJECT_ROOT / "configs" / "backtest.yaml"

BASELINE_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "baseline_metrics.csv"
)
POISSON_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "poisson_metrics.csv"
)

ELO_FEATURES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "elo_features.parquet"
)
PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "elo_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "elo_metrics.csv"
)

INITIAL_RATING = 1500.0
HOME_ADVANTAGE = 65.0
K_FACTOR = 25.0
SEASON_REVERSION = 0.25
SOFTMAX_L2 = 0.05


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def expected_home_score(
    home_rating: float,
    away_rating: float,
) -> float:
    rating_difference = (
        home_rating
        + HOME_ADVANTAGE
        - away_rating
    )

    return 1.0 / (
        1.0
        + 10.0 ** (-rating_difference / 400.0)
    )


def actual_home_score(result: str) -> float:
    if result == "H":
        return 1.0

    if result == "D":
        return 0.5

    return 0.0


def goal_margin_multiplier(
    home_goals: int,
    away_goals: int,
) -> float:
    goal_difference = abs(
        home_goals - away_goals
    )

    if goal_difference <= 1:
        return 1.0

    return float(
        1.0 + 0.25 * np.log(goal_difference)
    )


def regress_ratings_to_mean(
    ratings: dict[str, float],
) -> None:
    for team_id in ratings:
        ratings[team_id] = (
            INITIAL_RATING
            + (
                ratings[team_id]
                - INITIAL_RATING
            )
            * (1.0 - SEASON_REVERSION)
        )


def build_pre_match_elo_features(
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    先记录赛前Elo，再使用本场结果更新，
    因此不会使用未来比赛信息。
    """
    ordered_matches = matches.sort_values(
        ["kickoff_utc", "match_id"]
    ).reset_index(drop=True)

    ratings: dict[str, float] = {}
    feature_rows: list[dict] = []

    current_season: str | None = None

    for match in ordered_matches.itertuples(
        index=False
    ):
        if current_season is None:
            current_season = match.season

        elif match.season != current_season:
            regress_ratings_to_mean(ratings)
            current_season = match.season

        home_is_new = (
            match.home_team_id not in ratings
        )

        away_is_new = (
            match.away_team_id not in ratings
        )

        home_rating = ratings.get(
            match.home_team_id,
            INITIAL_RATING,
        )

        away_rating = ratings.get(
            match.away_team_id,
            INITIAL_RATING,
        )

        elo_difference = (
            home_rating
            + HOME_ADVANTAGE
            - away_rating
        )

        expected_score = expected_home_score(
            home_rating,
            away_rating,
        )

        feature_rows.append(
            {
                "match_id": match.match_id,
                "season": match.season,
                "match_date": match.match_date,
                "kickoff_utc": match.kickoff_utc,
                "home_team_id": match.home_team_id,
                "home_team": match.home_team,
                "away_team_id": match.away_team_id,
                "away_team": match.away_team,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "actual_result": match.result,
                "home_elo": home_rating,
                "away_elo": away_rating,
                "elo_difference": elo_difference,
                "absolute_elo_difference": abs(
                    elo_difference
                ),
                "elo_expected_home_score": (
                    expected_score
                ),
                "home_team_was_new": home_is_new,
                "away_team_was_new": away_is_new,
            }
        )

        actual_score = actual_home_score(
            match.result
        )

        multiplier = goal_margin_multiplier(
            match.home_goals,
            match.away_goals,
        )

        rating_change = (
            K_FACTOR
            * multiplier
            * (actual_score - expected_score)
        )

        ratings[match.home_team_id] = (
            home_rating + rating_change
        )

        ratings[match.away_team_id] = (
            away_rating - rating_change
        )

    return pd.DataFrame(feature_rows)


def create_model_features(
    elo_features: pd.DataFrame,
) -> np.ndarray:
    return np.column_stack(
        [
            elo_features[
                "elo_difference"
            ].to_numpy(dtype=float)
            / 400.0,
            elo_features[
                "absolute_elo_difference"
            ].to_numpy(dtype=float)
            / 400.0,
        ]
    )


def add_intercept(features: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(features)),
            features,
        ]
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    stable_logits = (
        logits
        - logits.max(axis=1, keepdims=True)
    )

    exponentials = np.exp(stable_logits)

    return (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )


def fit_softmax_classifier(
    features: np.ndarray,
    results: pd.Series,
) -> dict:
    """
    用SciPy拟合H、D、A三分类概率。

    客胜A作为参考类别，避免参数不可识别。
    """
    design_matrix = add_intercept(features)

    class_to_index = {
        result: index
        for index, result in enumerate(CLASS_ORDER)
    }

    result_indices = np.array(
        [
            class_to_index[result]
            for result in results
        ],
        dtype=int,
    )

    one_hot = np.zeros(
        (len(results), len(CLASS_ORDER)),
        dtype=float,
    )

    one_hot[
        np.arange(len(results)),
        result_indices,
    ] = 1.0

    number_of_features = design_matrix.shape[1]

    initial_parameters = np.zeros(
        2 * number_of_features,
        dtype=float,
    )

    def objective(
        flat_parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        coefficients = flat_parameters.reshape(
            2,
            number_of_features,
        )

        home_logits = (
            design_matrix @ coefficients[0]
        )

        draw_logits = (
            design_matrix @ coefficients[1]
        )

        away_logits = np.zeros(
            len(design_matrix)
        )

        logits = np.column_stack(
            [
                home_logits,
                draw_logits,
                away_logits,
            ]
        )

        probabilities = softmax(logits)

        safe_probabilities = np.clip(
            probabilities,
            1e-15,
            1.0,
        )

        data_loss = -np.mean(
            np.log(
                safe_probabilities[
                    np.arange(len(results)),
                    result_indices,
                ]
            )
        )

        # 截距不进行正则化
        regularisation_loss = (
            0.5
            * SOFTMAX_L2
            * np.sum(
                coefficients[:, 1:] ** 2
            )
        )

        total_loss = (
            data_loss + regularisation_loss
        )

        errors = probabilities - one_hot

        gradient = (
            errors[:, :2].T
            @ design_matrix
        ) / len(results)

        gradient[:, 1:] += (
            SOFTMAX_L2
            * coefficients[:, 1:]
        )

        return (
            float(total_loss),
            gradient.ravel(),
        )

    result = minimize(
        objective,
        initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Elo概率校准优化失败："
            f"{result.message}"
        )

    return {
        "coefficients": result.x.reshape(
            2,
            number_of_features,
        )
    }


def predict_softmax_probabilities(
    model: dict,
    features: np.ndarray,
) -> np.ndarray:
    design_matrix = add_intercept(features)
    coefficients = model["coefficients"]

    home_logits = (
        design_matrix @ coefficients[0]
    )

    draw_logits = (
        design_matrix @ coefficients[1]
    )

    away_logits = np.zeros(
        len(design_matrix)
    )

    logits = np.column_stack(
        [
            home_logits,
            draw_logits,
            away_logits,
        ]
    )

    # 顺序固定为H、D、A
    return softmax(logits)


def evaluate_elo(
    elo_features: pd.DataFrame,
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    for test_season in test_seasons:
        training_data = elo_features[
            elo_features["season"] < test_season
        ].copy()

        test_data = elo_features[
            elo_features["season"] == test_season
        ].copy()

        if len(test_data) != 380:
            raise ValueError(
                f"{test_season}应有380场，"
                f"实际为{len(test_data)}场。"
            )

        classifier = fit_softmax_classifier(
            create_model_features(training_data),
            training_data["actual_result"],
        )

        probabilities = (
            predict_softmax_probabilities(
                classifier,
                create_model_features(test_data),
            )
        )

        metrics = calculate_metrics(
            test_data["actual_result"],
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
                "actual_result",
                "home_elo",
                "away_elo",
                "elo_difference",
                "elo_expected_home_score",
                "home_team_was_new",
                "away_team_was_new",
            ]
        ].copy()

        predictions["p_home"] = probabilities[:, 0]
        predictions["p_draw"] = probabilities[:, 1]
        predictions["p_away"] = probabilities[:, 2]

        predicted_indices = np.argmax(
            probabilities,
            axis=1,
        )

        predictions["predicted_result"] = [
            CLASS_ORDER[index]
            for index in predicted_indices
        ]

        predictions["model"] = (
            "elo_softmax_calibrated"
        )

        metric_rows.append(
            {
                "season": test_season,
                "training_matches": len(
                    training_data
                ),
                "test_matches": len(test_data),
                "mean_home_elo": (
                    test_data["home_elo"].mean()
                ),
                "mean_away_elo": (
                    test_data["away_elo"].mean()
                ),
                **metrics,
            }
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
            "test_matches": len(
                combined_predictions
            ),
            "mean_home_elo": (
                combined_predictions[
                    "home_elo"
                ].mean()
            ),
            "mean_away_elo": (
                combined_predictions[
                    "away_elo"
                ].mean()
            ),
            **overall_metrics,
        }
    )

    return (
        combined_predictions,
        pd.DataFrame(metric_rows),
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

    matches = pd.read_parquet(
        MATCHES_FILE
    )

    print("正在逐场生成赛前Elo评分……")

    elo_features = (
        build_pre_match_elo_features(matches)
    )

    ELO_FEATURES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    elo_features.to_parquet(
        ELO_FEATURES_FILE,
        index=False,
    )

    predictions, metrics = evaluate_elo(
        elo_features,
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

    elo_overall = metrics.loc[
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

    print("\nElo模型回测完成。")
    print(f"开发回测场次：{len(predictions)}")
    print(
        f"Elo模型Log Loss："
        f"{elo_overall['log_loss']:.4f}"
    )
    print(
        f"联赛平均基准Log Loss："
        f"{baseline_log_loss:.4f}"
    )
    print(
        f"独立泊松Log Loss："
        f"{poisson_log_loss:.4f}"
    )
    print(
        f"Elo模型Brier Score："
        f"{elo_overall['brier_score']:.4f}"
    )
    print(
        f"Elo模型RPS："
        f"{elo_overall['ranked_probability_score']:.4f}"
    )
    print(
        f"Elo模型Accuracy："
        f"{elo_overall['accuracy']:.2%}"
    )
    print(f"Elo特征文件：{ELO_FEATURES_FILE}")
    print(f"预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(
        "\n2024-25和2025-26仍处于封存状态。"
    )


if __name__ == "__main__":
    main()