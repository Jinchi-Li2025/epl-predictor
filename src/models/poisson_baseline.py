from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

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
PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "poisson_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "poisson_metrics.csv"
)

MAX_GOALS = 10
REGULARIZATION = 0.01


def load_config() -> dict:
    """读取回测配置。"""
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def unpack_parameters(
    parameters: np.ndarray,
    number_of_teams: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    将参数向量拆分成：

    - 主场优势
    - 联赛平均进球水平
    - 各球队进攻能力
    - 各球队防守弱度

    最后一支球队的参数由零和约束确定，
    避免模型出现不可识别问题。
    """
    home_advantage = parameters[0]
    intercept = parameters[1]

    attack_start = 2
    attack_end = attack_start + number_of_teams - 1

    attack_free = parameters[
        attack_start:attack_end
    ]

    defence_free = parameters[
        attack_end:
        attack_end + number_of_teams - 1
    ]

    attack = np.append(
        attack_free,
        -attack_free.sum(),
    )

    defence = np.append(
        defence_free,
        -defence_free.sum(),
    )

    return (
        home_advantage,
        intercept,
        attack,
        defence,
    )


def fit_poisson_model(
    training_matches: pd.DataFrame,
) -> dict:
    """用历史比赛拟合球队进攻和防守能力。"""
    teams = sorted(
        set(training_matches["home_team_id"])
        | set(training_matches["away_team_id"])
    )

    team_to_index = {
        team: index
        for index, team in enumerate(teams)
    }

    number_of_teams = len(teams)

    home_indices = (
        training_matches["home_team_id"]
        .map(team_to_index)
        .to_numpy(dtype=int)
    )

    away_indices = (
        training_matches["away_team_id"]
        .map(team_to_index)
        .to_numpy(dtype=int)
    )

    home_goals = training_matches[
        "home_goals"
    ].to_numpy(dtype=float)

    away_goals = training_matches[
        "away_goals"
    ].to_numpy(dtype=float)

    mean_home_goals = max(home_goals.mean(), 0.01)
    mean_away_goals = max(away_goals.mean(), 0.01)

    number_of_parameters = (
        2 + 2 * (number_of_teams - 1)
    )

    initial_parameters = np.zeros(
        number_of_parameters,
        dtype=float,
    )

    initial_parameters[0] = np.log(
        mean_home_goals / mean_away_goals
    )

    initial_parameters[1] = np.log(
        mean_away_goals
    )

    def negative_log_likelihood(
        parameters: np.ndarray,
    ) -> float:
        (
            home_advantage,
            intercept,
            attack,
            defence,
        ) = unpack_parameters(
            parameters,
            number_of_teams,
        )

        home_log_rates = (
            intercept
            + home_advantage
            + attack[home_indices]
            + defence[away_indices]
        )

        away_log_rates = (
            intercept
            + attack[away_indices]
            + defence[home_indices]
        )

        # 防止优化过程中指数溢出
        home_log_rates = np.clip(
            home_log_rates,
            -5.0,
            5.0,
        )

        away_log_rates = np.clip(
            away_log_rates,
            -5.0,
            5.0,
        )

        home_rates = np.exp(home_log_rates)
        away_rates = np.exp(away_log_rates)

        home_loss = (
            home_rates
            - home_goals * home_log_rates
            + gammaln(home_goals + 1)
        ).sum()

        away_loss = (
            away_rates
            - away_goals * away_log_rates
            + gammaln(away_goals + 1)
        ).sum()

        penalty = REGULARIZATION * (
            np.sum(attack**2)
            + np.sum(defence**2)
        )

        return float(home_loss + away_loss + penalty)

    optimisation_result = minimize(
        negative_log_likelihood,
        initial_parameters,
        method="L-BFGS-B",
        options={
            "maxiter": 5000,
            "maxfun": 500000,
            "maxls": 50,
            "ftol": 1e-8,
        },
    )

    if not optimisation_result.success:
        raise RuntimeError(
            "泊松模型优化失败："
            f"{optimisation_result.message}"
        )

    (
        home_advantage,
        intercept,
        attack,
        defence,
    ) = unpack_parameters(
        optimisation_result.x,
        number_of_teams,
    )

    return {
        "team_to_index": team_to_index,
        "home_advantage": home_advantage,
        "intercept": intercept,
        "attack": attack,
        "defence": defence,
        "objective": float(optimisation_result.fun),
    }


def get_team_strength(
    model: dict,
    team_id: str,
) -> tuple[float, float, bool]:
    """
    返回球队进攻和防守参数。

    如果升班马以前从未参加过英超，
    暂时使用联赛平均值0。
    """
    team_to_index = model["team_to_index"]

    if team_id not in team_to_index:
        return 0.0, 0.0, True

    index = team_to_index[team_id]

    return (
        float(model["attack"][index]),
        float(model["defence"][index]),
        False,
    )


def predict_expected_goals(
    model: dict,
    home_team_id: str,
    away_team_id: str,
) -> tuple[float, float, bool]:
    """预测双方预期进球。"""
    (
        home_attack,
        home_defence,
        home_unseen,
    ) = get_team_strength(
        model,
        home_team_id,
    )

    (
        away_attack,
        away_defence,
        away_unseen,
    ) = get_team_strength(
        model,
        away_team_id,
    )

    home_log_rate = (
        model["intercept"]
        + model["home_advantage"]
        + home_attack
        + away_defence
    )

    away_log_rate = (
        model["intercept"]
        + away_attack
        + home_defence
    )

    home_expected_goals = float(
        np.exp(np.clip(home_log_rate, -5.0, 5.0))
    )

    away_expected_goals = float(
        np.exp(np.clip(away_log_rate, -5.0, 5.0))
    )

    return (
        home_expected_goals,
        away_expected_goals,
        home_unseen or away_unseen,
    )


def create_score_matrix(
    home_expected_goals: float,
    away_expected_goals: float,
) -> np.ndarray:
    """生成0比0至10比10的比分概率矩阵。"""
    goals = np.arange(MAX_GOALS + 1)

    home_probabilities = poisson.pmf(
        goals,
        home_expected_goals,
    )

    away_probabilities = poisson.pmf(
        goals,
        away_expected_goals,
    )

    score_matrix = np.outer(
        home_probabilities,
        away_probabilities,
    )

    # 把10球以上被截断的极小概率重新归一化
    score_matrix = score_matrix / score_matrix.sum()

    return score_matrix


def score_matrix_to_prediction(
    score_matrix: np.ndarray,
) -> tuple[float, float, float, str]:
    """从比分矩阵得到胜平负概率和最可能比分。"""
    p_home = float(
        np.tril(score_matrix, k=-1).sum()
    )

    p_draw = float(
        np.trace(score_matrix)
    )

    p_away = float(
        np.triu(score_matrix, k=1).sum()
    )

    top_index = np.unravel_index(
        np.argmax(score_matrix),
        score_matrix.shape,
    )

    most_likely_score = (
        f"{top_index[0]}-{top_index[1]}"
    )

    return (
        p_home,
        p_draw,
        p_away,
        most_likely_score,
    )


def evaluate_poisson(
    matches: pd.DataFrame,
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按赛季滚动回测独立泊松模型。"""
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    for test_season in test_seasons:
        training_matches = matches[
            matches["season"] < test_season
        ].copy()

        test_matches = matches[
            matches["season"] == test_season
        ].copy()

        print(
            f"\n正在拟合 {test_season}："
            f"使用此前{len(training_matches)}场比赛……"
        )

        model = fit_poisson_model(
            training_matches
        )

        prediction_rows: list[dict] = []
        unseen_matches = 0

        for match in test_matches.itertuples(
            index=False
        ):
            (
                home_expected_goals,
                away_expected_goals,
                contains_unseen_team,
            ) = predict_expected_goals(
                model,
                match.home_team_id,
                match.away_team_id,
            )

            if contains_unseen_team:
                unseen_matches += 1

            score_matrix = create_score_matrix(
                home_expected_goals,
                away_expected_goals,
            )

            (
                p_home,
                p_draw,
                p_away,
                most_likely_score,
            ) = score_matrix_to_prediction(
                score_matrix
            )

            probabilities = np.array(
                [p_home, p_draw, p_away]
            )

            predicted_result = CLASS_ORDER[
                int(np.argmax(probabilities))
            ]

            prediction_rows.append(
                {
                    "match_id": match.match_id,
                    "season": match.season,
                    "match_date": match.match_date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "actual_home_goals": (
                        match.home_goals
                    ),
                    "actual_away_goals": (
                        match.away_goals
                    ),
                    "actual_result": match.result,
                    "home_expected_goals": (
                        home_expected_goals
                    ),
                    "away_expected_goals": (
                        away_expected_goals
                    ),
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                    "predicted_result": (
                        predicted_result
                    ),
                    "most_likely_score": (
                        most_likely_score
                    ),
                    "contains_unseen_team": (
                        contains_unseen_team
                    ),
                    "model": (
                        "independent_poisson"
                    ),
                }
            )

        season_predictions = pd.DataFrame(
            prediction_rows
        )

        probabilities = season_predictions[
            PROBABILITY_COLUMNS
        ].to_numpy(dtype=float)

        metrics = calculate_metrics(
            season_predictions["actual_result"],
            probabilities,
        )

        metric_rows.append(
            {
                "season": test_season,
                "training_matches": len(
                    training_matches
                ),
                "test_matches": len(
                    test_matches
                ),
                "unseen_team_matches": unseen_matches,
                "mean_home_xg": (
                    season_predictions[
                        "home_expected_goals"
                    ].mean()
                ),
                "mean_away_xg": (
                    season_predictions[
                        "away_expected_goals"
                    ].mean()
                ),
                **metrics,
            }
        )

        all_predictions.append(
            season_predictions
        )

        print(
            f"{test_season}完成："
            f"Log Loss={metrics['log_loss']:.4f}，"
            f"Accuracy={metrics['accuracy']:.2%}，"
            f"新球队比赛={unseen_matches}"
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
            "unseen_team_matches": int(
                combined_predictions[
                    "contains_unseen_team"
                ].sum()
            ),
            "mean_home_xg": (
                combined_predictions[
                    "home_expected_goals"
                ].mean()
            ),
            "mean_away_xg": (
                combined_predictions[
                    "away_expected_goals"
                ].mean()
            ),
            **overall_metrics,
        }
    )

    metrics_table = pd.DataFrame(
        metric_rows
    )

    return (
        combined_predictions,
        metrics_table,
    )


def main() -> None:
    """运行独立泊松模型的滚动回测。"""
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
            "开发集与最终封存测试集发生重叠。"
        )

    matches = pd.read_parquet(
        MATCHES_FILE
    )

    predictions, metrics = evaluate_poisson(
        matches,
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

    poisson_overall = metrics[
        metrics["season"] == "OVERALL"
    ].iloc[0]

    baseline_metrics = pd.read_csv(
        BASELINE_METRICS_FILE
    )

    baseline_overall = baseline_metrics[
        baseline_metrics["season"] == "OVERALL"
    ].iloc[0]

    log_loss_improvement = (
        baseline_overall["log_loss"]
        - poisson_overall["log_loss"]
    )

    print("\n独立泊松模型回测完成。")
    print(f"开发回测场次：{len(predictions)}")
    print(
        f"泊松模型Log Loss："
        f"{poisson_overall['log_loss']:.4f}"
    )
    print(
        f"联赛平均基准Log Loss："
        f"{baseline_overall['log_loss']:.4f}"
    )
    print(
        f"Log Loss改善："
        f"{log_loss_improvement:.4f}"
    )
    print(
        f"泊松模型Brier Score："
        f"{poisson_overall['brier_score']:.4f}"
    )
    print(
        f"泊松模型RPS："
        f"{poisson_overall['ranked_probability_score']:.4f}"
    )
    print(
        f"泊松模型Accuracy："
        f"{poisson_overall['accuracy']:.2%}"
    )
    print(f"预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(
        "\n2024-25和2025-26仍处于封存状态。"
    )


if __name__ == "__main__":
    main()