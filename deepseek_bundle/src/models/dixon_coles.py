from __future__ import annotations

import json
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
from poisson_baseline import fit_poisson_model


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
ELO_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "elo_metrics.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "dixon_coles_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "dixon_coles_metrics.csv"
)

MAX_GOALS = 10
REGULARIZATION = 0.01

# 每天的时间衰减率，对应大约385天半衰期
DECAY_RATE_PER_DAY = 0.0018


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def unpack_parameters(
    parameters: np.ndarray,
    number_of_teams: int,
) -> tuple[
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    参数包括：

    - 主场优势
    - 联赛平均进球水平
    - Dixon-Coles低比分相关参数rho
    - 各球队进攻能力
    - 各球队防守弱度
    """
    home_advantage = parameters[0]
    intercept = parameters[1]
    rho = parameters[2]

    attack_start = 3
    attack_end = (
        attack_start + number_of_teams - 1
    )

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
        rho,
        attack,
        defence,
    )


def calculate_dc_tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_rates: np.ndarray,
    away_rates: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    Dixon-Coles对四种低比分进行相关性修正：

    0-0、0-1、1-0、1-1。
    """
    tau = np.ones(
        len(home_goals),
        dtype=float,
    )

    mask_00 = (
        (home_goals == 0)
        & (away_goals == 0)
    )

    mask_01 = (
        (home_goals == 0)
        & (away_goals == 1)
    )

    mask_10 = (
        (home_goals == 1)
        & (away_goals == 0)
    )

    mask_11 = (
        (home_goals == 1)
        & (away_goals == 1)
    )

    tau[mask_00] = (
        1.0
        - home_rates[mask_00]
        * away_rates[mask_00]
        * rho
    )

    tau[mask_01] = (
        1.0
        + home_rates[mask_01] * rho
    )

    tau[mask_10] = (
        1.0
        + away_rates[mask_10] * rho
    )

    tau[mask_11] = 1.0 - rho

    return tau


def calculate_time_weights(
    training_matches: pd.DataFrame,
) -> np.ndarray:
    """
    越接近训练截止日的比赛权重越高。
    """
    latest_time = training_matches[
        "kickoff_utc"
    ].max()

    age_days = (
        latest_time
        - training_matches["kickoff_utc"]
    ).dt.total_seconds().to_numpy() / 86400.0

    raw_weights = np.exp(
        -DECAY_RATE_PER_DAY * age_days
    )

    # 保持平均权重为1，便于优化和正则化
    return raw_weights / raw_weights.mean()


def create_initial_parameters(
    training_matches: pd.DataFrame,
    teams: list[str],
    team_to_index: dict[str, int],
) -> np.ndarray:
    """
    使用已经验证成功的独立泊松模型作为初始值，
    提高Dixon-Coles优化稳定性。
    """
    poisson_model = fit_poisson_model(
        training_matches
    )

    number_of_teams = len(teams)

    attack_initial = np.zeros(
        number_of_teams
    )

    defence_initial = np.zeros(
        number_of_teams
    )

    for team in teams:
        new_index = team_to_index[team]
        old_index = poisson_model[
            "team_to_index"
        ][team]

        attack_initial[new_index] = (
            poisson_model["attack"][old_index]
        )

        defence_initial[new_index] = (
            poisson_model["defence"][old_index]
        )

    parameters = np.zeros(
        3 + 2 * (number_of_teams - 1),
        dtype=float,
    )

    parameters[0] = poisson_model[
        "home_advantage"
    ]

    parameters[1] = poisson_model[
        "intercept"
    ]

    # 典型Dixon-Coles初值为一个小负数
    parameters[2] = -0.05

    attack_start = 3
    attack_end = (
        attack_start + number_of_teams - 1
    )

    parameters[
        attack_start:attack_end
    ] = attack_initial[:-1]

    parameters[
        attack_end:
        attack_end + number_of_teams - 1
    ] = defence_initial[:-1]

    return parameters


def fit_dixon_coles_model(
    training_matches: pd.DataFrame,
) -> dict:
    """拟合带时间衰减的Dixon-Coles模型。"""
    training_matches = training_matches.sort_values(
        ["kickoff_utc", "match_id"]
    ).reset_index(drop=True)

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

    time_weights = calculate_time_weights(
        training_matches
    )

    initial_parameters = (
        create_initial_parameters(
            training_matches,
            teams,
            team_to_index,
        )
    )

    def negative_log_likelihood(
        parameters: np.ndarray,
    ) -> float:
        (
            home_advantage,
            intercept,
            rho,
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

        tau = calculate_dc_tau(
            home_goals,
            away_goals,
            home_rates,
            away_rates,
            rho,
        )

        if np.any(tau <= 0.0):
            return 1e12

        home_log_probability = (
            home_goals * home_log_rates
            - home_rates
            - gammaln(home_goals + 1)
        )

        away_log_probability = (
            away_goals * away_log_rates
            - away_rates
            - gammaln(away_goals + 1)
        )

        joint_log_probability = (
            home_log_probability
            + away_log_probability
            + np.log(tau)
        )

        weighted_loss = -np.sum(
            time_weights
            * joint_log_probability
        )

        penalty = REGULARIZATION * (
            np.sum(attack**2)
            + np.sum(defence**2)
        )

        return float(
            weighted_loss + penalty
        )

    bounds = (
        [
            (-1.5, 1.5),   # 主场优势
            (-1.5, 1.5),   # 联赛进球水平
            (-0.20, 0.10), # rho
        ]
        + [
            (-3.0, 3.0)
            for _ in range(
                2 * (number_of_teams - 1)
            )
        ]
    )

    result = minimize(
        negative_log_likelihood,
        initial_parameters,
        method="L-BFGS-B",
        bounds=bounds,
        options={
    "maxiter": 3000,
    "maxfun": 200000,
    "maxls": 50,
    "ftol": 1e-8,
},
    )

    if not result.success:
        raise RuntimeError(
            "Dixon-Coles优化失败："
            f"{result.message}"
        )

    (
        home_advantage,
        intercept,
        rho,
        attack,
        defence,
    ) = unpack_parameters(
        result.x,
        number_of_teams,
    )

    return {
        "team_to_index": team_to_index,
        "home_advantage": home_advantage,
        "intercept": intercept,
        "rho": rho,
        "attack": attack,
        "defence": defence,
        "objective": float(result.fun),
    }


def get_team_strength(
    model: dict,
    team_id: str,
) -> tuple[float, float, bool]:
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
        np.exp(
            np.clip(home_log_rate, -5.0, 5.0)
        )
    )

    away_expected_goals = float(
        np.exp(
            np.clip(away_log_rate, -5.0, 5.0)
        )
    )

    return (
        home_expected_goals,
        away_expected_goals,
        home_unseen or away_unseen,
    )


def create_dc_score_matrix(
    home_expected_goals: float,
    away_expected_goals: float,
    rho: float,
) -> np.ndarray:
    """生成经过Dixon-Coles修正的比分矩阵。"""
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

    corrections = {
        (0, 0): (
            1.0
            - home_expected_goals
            * away_expected_goals
            * rho
        ),
        (0, 1): (
            1.0
            + home_expected_goals * rho
        ),
        (1, 0): (
            1.0
            + away_expected_goals * rho
        ),
        (1, 1): 1.0 - rho,
    }

    for score, correction in corrections.items():
        score_matrix[score] *= max(
            correction,
            1e-12,
        )

    score_matrix = np.clip(
        score_matrix,
        0.0,
        None,
    )

    return score_matrix / score_matrix.sum()


def score_matrix_summary(
    score_matrix: np.ndarray,
) -> tuple[float, float, float, str, str]:
    p_home = float(
        np.tril(score_matrix, k=-1).sum()
    )

    p_draw = float(
        np.trace(score_matrix)
    )

    p_away = float(
        np.triu(score_matrix, k=1).sum()
    )

    top_indices = np.argsort(
        score_matrix.ravel()
    )[::-1][:5]

    top_scores: list[dict] = []

    for flat_index in top_indices:
        home_goals, away_goals = (
            np.unravel_index(
                flat_index,
                score_matrix.shape,
            )
        )

        top_scores.append(
            {
                "score": (
                    f"{home_goals}-{away_goals}"
                ),
                "probability": float(
                    score_matrix[
                        home_goals,
                        away_goals,
                    ]
                ),
            }
        )

    most_likely_score = top_scores[0]["score"]

    top_scores_json = json.dumps(
        top_scores,
        ensure_ascii=False,
    )

    return (
        p_home,
        p_draw,
        p_away,
        most_likely_score,
        top_scores_json,
    )


def evaluate_dixon_coles(
    matches: pd.DataFrame,
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        model = fit_dixon_coles_model(
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

            score_matrix = (
                create_dc_score_matrix(
                    home_expected_goals,
                    away_expected_goals,
                    model["rho"],
                )
            )

            (
                p_home,
                p_draw,
                p_away,
                most_likely_score,
                top_scores_json,
            ) = score_matrix_summary(
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
                    "top_5_scores": top_scores_json,
                    "rho": model["rho"],
                    "contains_unseen_team": (
                        contains_unseen_team
                    ),
                    "model": "dixon_coles",
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
                "rho": model["rho"],
                "unseen_team_matches": (
                    unseen_matches
                ),
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
            f"rho={model['rho']:.4f}，"
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
            "rho": pd.NA,
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

    predictions, metrics = (
        evaluate_dixon_coles(
            matches,
            development_seasons,
        )
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

    dc_overall = metrics.loc[
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

    improvement_vs_poisson = (
        poisson_log_loss
        - dc_overall["log_loss"]
    )

    half_life_days = (
        np.log(2.0)
        / DECAY_RATE_PER_DAY
    )

    print("\nDixon-Coles模型回测完成。")
    print(f"开发回测场次：{len(predictions)}")
    print(
        f"Dixon-Coles Log Loss："
        f"{dc_overall['log_loss']:.4f}"
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
        f"联赛平均基准Log Loss："
        f"{baseline_log_loss:.4f}"
    )
    print(
        f"相对独立泊松改善："
        f"{improvement_vs_poisson:.4f}"
    )
    print(
        f"Dixon-Coles Brier Score："
        f"{dc_overall['brier_score']:.4f}"
    )
    print(
        f"Dixon-Coles RPS："
        f"{dc_overall['ranked_probability_score']:.4f}"
    )
    print(
        f"Dixon-Coles Accuracy："
        f"{dc_overall['accuracy']:.2%}"
    )
    print(
        f"时间权重半衰期："
        f"{half_life_days:.1f}天"
    )
    print(f"预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(
        "\n2024-25和2025-26仍处于封存状态。"
    )


if __name__ == "__main__":
    main()