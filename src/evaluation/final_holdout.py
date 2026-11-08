# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = PROJECT_ROOT / "src" / "models"

# 允许本文件导入src/models中的模型函数
sys.path.insert(
    0,
    str(MODELS_DIRECTORY),
)

from baseline import (  # noqa: E402
    CLASS_ORDER,
    calculate_metrics,
)
from dixon_coles import (  # noqa: E402
    create_dc_score_matrix,
    fit_dixon_coles_model,
    predict_expected_goals as predict_dc_xg,
    score_matrix_summary,
)
from elo import (  # noqa: E402
    create_model_features as create_elo_features,
    fit_softmax_classifier,
    predict_softmax_probabilities,
)
from ensemble import (  # noqa: E402
    apply_temperature,
)
from gradient_boosting import (  # noqa: E402
    choose_boosting_rounds,
    make_dmatrix,
    select_feature_columns,
    train_final_model,
)
from poisson_baseline import (  # noqa: E402
    create_score_matrix,
    fit_poisson_model,
    predict_expected_goals as predict_poisson_xg,
    score_matrix_to_prediction,
)


MATCHES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "matches.parquet"
)
FEATURES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
)
ELO_FEATURES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "elo_features.parquet"
)

BACKTEST_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "backtest.yaml"
)
ENSEMBLE_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "ensemble_weights.json"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "final_holdout_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "final_holdout_metrics.csv"
)
FREEZE_MANIFEST_FILE = (
    PROJECT_ROOT / "configs" / "model_freeze_manifest.json"
)

MODEL_ORDER = [
    "dixon_coles",
    "xgboost",
    "elo",
    "poisson",
]

RESULT_TO_INDEX = {
    "H": 0,
    "D": 1,
    "A": 2,
}


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def calculate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def create_freeze_manifest(
    holdout_seasons: list[str],
    ensemble_config: dict,
) -> None:
    """
    在读取最终成绩前记录模型代码和权重摘要。
    """
    files_to_freeze = [
        BACKTEST_CONFIG_FILE,
        ENSEMBLE_CONFIG_FILE,
        MODELS_DIRECTORY / "baseline.py",
        MODELS_DIRECTORY / "poisson_baseline.py",
        MODELS_DIRECTORY / "elo.py",
        MODELS_DIRECTORY / "dixon_coles.py",
        MODELS_DIRECTORY / "gradient_boosting.py",
        MODELS_DIRECTORY / "ensemble.py",
        Path(__file__).resolve(),
    ]

    manifest = {
        "frozen_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "holdout_seasons": holdout_seasons,
        "ensemble_weights": (
            ensemble_config["weights"]
        ),
        "temperature": (
            ensemble_config["temperature"]
        ),
        "file_sha256": {
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            ): calculate_file_hash(path)
            for path in files_to_freeze
        },
        "rule": (
            "Model parameters are frozen before "
            "final holdout results are evaluated."
        ),
    }

    with FREEZE_MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )


def order_by_match_ids(
    data: pd.DataFrame,
    match_ids: pd.Series,
) -> pd.DataFrame:
    """
    按基础比赛表的match_id顺序重新排列。
    """
    indexed = data.set_index("match_id")

    ordered = indexed.loc[
        match_ids.tolist()
    ].reset_index()

    return ordered


def predict_league_average(
    training_matches: pd.DataFrame,
    number_of_matches: int,
) -> np.ndarray:
    counts = (
        training_matches["result"]
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0,
        )
    )

    probabilities = (
        counts / counts.sum()
    ).to_numpy(dtype=float)

    return np.tile(
        probabilities,
        (number_of_matches, 1),
    )


def predict_poisson_season(
    training_matches: pd.DataFrame,
    test_matches: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    model = fit_poisson_model(
        training_matches
    )

    probability_rows: list[list[float]] = []
    extra_rows: list[dict] = []

    for match in test_matches.itertuples(
        index=False
    ):
        (
            home_xg,
            away_xg,
            unseen_team,
        ) = predict_poisson_xg(
            model,
            match.home_team_id,
            match.away_team_id,
        )

        score_matrix = create_score_matrix(
            home_xg,
            away_xg,
        )

        (
            p_home,
            p_draw,
            p_away,
            most_likely_score,
        ) = score_matrix_to_prediction(
            score_matrix
        )

        probability_rows.append(
            [p_home, p_draw, p_away]
        )

        extra_rows.append(
            {
                "poisson_home_xg": home_xg,
                "poisson_away_xg": away_xg,
                "poisson_top_score": (
                    most_likely_score
                ),
                "poisson_unseen_team": (
                    unseen_team
                ),
            }
        )

    return (
        np.asarray(
            probability_rows,
            dtype=float,
        ),
        extra_rows,
    )


def predict_dixon_coles_season(
    training_matches: pd.DataFrame,
    test_matches: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    model = fit_dixon_coles_model(
        training_matches
    )

    probability_rows: list[list[float]] = []
    extra_rows: list[dict] = []

    for match in test_matches.itertuples(
        index=False
    ):
        (
            home_xg,
            away_xg,
            unseen_team,
        ) = predict_dc_xg(
            model,
            match.home_team_id,
            match.away_team_id,
        )

        score_matrix = create_dc_score_matrix(
            home_xg,
            away_xg,
            model["rho"],
        )

        (
            p_home,
            p_draw,
            p_away,
            most_likely_score,
            top_5_scores,
        ) = score_matrix_summary(
            score_matrix
        )

        probability_rows.append(
            [p_home, p_draw, p_away]
        )

        extra_rows.append(
            {
                "home_expected_goals": home_xg,
                "away_expected_goals": away_xg,
                "most_likely_score": (
                    most_likely_score
                ),
                "top_5_scores": top_5_scores,
                "dixon_coles_rho": (
                    model["rho"]
                ),
                "dixon_coles_unseen_team": (
                    unseen_team
                ),
            }
        )

    return (
        np.asarray(
            probability_rows,
            dtype=float,
        ),
        extra_rows,
    )


def predict_elo_season(
    elo_features: pd.DataFrame,
    test_season: str,
    ordered_match_ids: pd.Series,
) -> np.ndarray:
    training_data = elo_features[
        elo_features["season"] < test_season
    ].copy()

    test_data = elo_features[
        elo_features["season"] == test_season
    ].copy()

    test_data = order_by_match_ids(
        test_data,
        ordered_match_ids,
    )

    model = fit_softmax_classifier(
        create_elo_features(
            training_data
        ),
        training_data["actual_result"],
    )

    return predict_softmax_probabilities(
        model,
        create_elo_features(test_data),
    )


def predict_xgboost_season(
    features: pd.DataFrame,
    test_season: str,
    ordered_match_ids: pd.Series,
    feature_columns: list[str],
) -> tuple[np.ndarray, int]:
    all_training_data = features[
        features["season"] < test_season
    ].copy()

    test_data = features[
        features["season"] == test_season
    ].copy()

    test_data = order_by_match_ids(
        test_data,
        ordered_match_ids,
    )

    earlier_seasons = sorted(
        all_training_data[
            "season"
        ].unique()
    )

    validation_season = earlier_seasons[-1]

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

    boosting_rounds = (
        choose_boosting_rounds(
            selection_training_data,
            validation_data,
            feature_columns,
        )
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

    return (
        probabilities,
        boosting_rounds,
    )


def blend_frozen_probabilities(
    component_probabilities: dict[
        str,
        np.ndarray,
    ],
    ensemble_config: dict,
) -> np.ndarray:
    weights = np.array(
        [
            ensemble_config[
                "weights"
            ][model_name]
            for model_name in MODEL_ORDER
        ],
        dtype=float,
    )

    probability_cube = np.stack(
        [
            component_probabilities[
                model_name
            ]
            for model_name in MODEL_ORDER
        ],
        axis=1,
    )

    blended = np.einsum(
        "nmc,m->nc",
        probability_cube,
        weights,
    )

    blended = (
        blended
        / blended.sum(
            axis=1,
            keepdims=True,
        )
    )

    return apply_temperature(
        blended,
        float(
            ensemble_config[
                "temperature"
            ]
        ),
    )


def add_probability_columns(
    data: pd.DataFrame,
    prefix: str,
    probabilities: np.ndarray,
) -> None:
    data[
        f"{prefix}_p_home"
    ] = probabilities[:, 0]

    data[
        f"{prefix}_p_draw"
    ] = probabilities[:, 1]

    data[
        f"{prefix}_p_away"
    ] = probabilities[:, 2]


def append_metrics(
    metric_rows: list[dict],
    model_name: str,
    season: str,
    actual_results: pd.Series,
    probabilities: np.ndarray,
) -> None:
    metrics = calculate_metrics(
        actual_results,
        probabilities,
    )

    metric_rows.append(
        {
            "model": model_name,
            "season": season,
            "test_matches": len(
                actual_results
            ),
            **metrics,
        }
    )


def evaluate_holdout_season(
    test_season: str,
    matches: pd.DataFrame,
    features: pd.DataFrame,
    elo_features: pd.DataFrame,
    feature_columns: list[str],
    ensemble_config: dict,
    expected_test_matches: int = 380,
) -> tuple[
    pd.DataFrame,
    list[dict],
]:
    training_matches = matches[
        matches["season"] < test_season
    ].copy()

    test_matches = matches[
        matches["season"] == test_season
    ].sort_values(
        ["kickoff_utc", "match_id"]
    ).reset_index(drop=True)

    if len(test_matches) != expected_test_matches:
        raise ValueError(
            f"{test_season}应有{expected_test_matches}场，"
            f"实际为{len(test_matches)}场。"
        )

    print(
        f"\n正在进行最终测试：{test_season}"
    )
    print(
        f"训练比赛：{len(training_matches)}"
    )

    print("1/4 拟合独立泊松模型……")

    (
        poisson_probabilities,
        poisson_extras,
    ) = predict_poisson_season(
        training_matches,
        test_matches,
    )

    print("2/4 拟合Dixon-Coles模型……")

    (
        dc_probabilities,
        dc_extras,
    ) = predict_dixon_coles_season(
        training_matches,
        test_matches,
    )

    print("3/4 拟合Elo概率模型……")

    elo_probabilities = (
        predict_elo_season(
            elo_features,
            test_season,
            test_matches["match_id"],
        )
    )

    print("4/4 拟合XGBoost模型……")

    (
        xgboost_probabilities,
        boosting_rounds,
    ) = predict_xgboost_season(
        features,
        test_season,
        test_matches["match_id"],
        feature_columns,
    )

    baseline_probabilities = (
        predict_league_average(
            training_matches,
            len(test_matches),
        )
    )

    component_probabilities = {
        "dixon_coles": dc_probabilities,
        "xgboost": xgboost_probabilities,
        "elo": elo_probabilities,
        "poisson": poisson_probabilities,
    }

    ensemble_probabilities = (
        blend_frozen_probabilities(
            component_probabilities,
            ensemble_config,
        )
    )

    predictions = test_matches[
        [
            "match_id",
            "season",
            "match_date",
            "kickoff_utc",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            "result": "actual_result",
        }
    )

    add_probability_columns(
        predictions,
        "league_average",
        baseline_probabilities,
    )

    add_probability_columns(
        predictions,
        "poisson",
        poisson_probabilities,
    )

    add_probability_columns(
        predictions,
        "elo",
        elo_probabilities,
    )

    add_probability_columns(
        predictions,
        "xgboost",
        xgboost_probabilities,
    )

    add_probability_columns(
        predictions,
        "dixon_coles",
        dc_probabilities,
    )

    add_probability_columns(
        predictions,
        "ensemble",
        ensemble_probabilities,
    )

    # 网页默认使用集成概率
    predictions["p_home"] = (
        ensemble_probabilities[:, 0]
    )
    predictions["p_draw"] = (
        ensemble_probabilities[:, 1]
    )
    predictions["p_away"] = (
        ensemble_probabilities[:, 2]
    )

    predicted_indices = np.argmax(
        ensemble_probabilities,
        axis=1,
    )

    predictions["predicted_result"] = [
        CLASS_ORDER[index]
        for index in predicted_indices
    ]

    predictions["xgboost_rounds"] = (
        boosting_rounds
    )

    predictions["model"] = (
        "frozen_final_ensemble"
    )

    poisson_extras_table = pd.DataFrame(
        poisson_extras
    )

    dc_extras_table = pd.DataFrame(
        dc_extras
    )

    predictions = pd.concat(
        [
            predictions,
            poisson_extras_table,
            dc_extras_table,
        ],
        axis=1,
    )

    metric_rows: list[dict] = []

    model_probabilities = {
        "league_average": (
            baseline_probabilities
        ),
        "poisson": poisson_probabilities,
        "elo": elo_probabilities,
        "xgboost": xgboost_probabilities,
        "dixon_coles": dc_probabilities,
        "ensemble": ensemble_probabilities,
    }

    for model_name, probabilities in (
        model_probabilities.items()
    ):
        append_metrics(
            metric_rows,
            model_name,
            test_season,
            predictions["actual_result"],
            probabilities,
        )

    ensemble_metrics = calculate_metrics(
        predictions["actual_result"],
        ensemble_probabilities,
    )

    print(
        f"{test_season}完成："
        f"集成Log Loss="
        f"{ensemble_metrics['log_loss']:.4f}，"
        f"Accuracy="
        f"{ensemble_metrics['accuracy']:.2%}"
    )

    return predictions, metric_rows


def add_overall_metrics(
    predictions: pd.DataFrame,
    metric_rows: list[dict],
) -> None:
    model_prefixes = {
        "league_average": (
            "league_average"
        ),
        "poisson": "poisson",
        "elo": "elo",
        "xgboost": "xgboost",
        "dixon_coles": "dixon_coles",
        "ensemble": "ensemble",
    }

    for model_name, prefix in (
        model_prefixes.items()
    ):
        probabilities = predictions[
            [
                f"{prefix}_p_home",
                f"{prefix}_p_draw",
                f"{prefix}_p_away",
            ]
        ].to_numpy(dtype=float)

        append_metrics(
            metric_rows,
            model_name,
            "OVERALL",
            predictions["actual_result"],
            probabilities,
        )


def main() -> None:
    backtest_config = load_yaml(
        BACKTEST_CONFIG_FILE
    )

    ensemble_config = load_json(
        ENSEMBLE_CONFIG_FILE
    )

    holdout_seasons = backtest_config[
        "backtest"
    ]["final_holdout_seasons"]

    if (
        ensemble_config[
            "final_holdout_seasons"
        ]
        != holdout_seasons
    ):
        raise ValueError(
            "集成配置中的封存赛季不一致。"
        )

    # 在最终测试结果被计算前创建冻结记录
    create_freeze_manifest(
        holdout_seasons,
        ensemble_config,
    )

    print(
        "模型和集成权重已冻结。"
    )

    matches = pd.read_parquet(
        MATCHES_FILE
    )

    features = pd.read_parquet(
        FEATURES_FILE
    )

    elo_features = pd.read_parquet(
        ELO_FEATURES_FILE
    )

    feature_columns = (
        select_feature_columns(features)
    )

    all_predictions: list[pd.DataFrame] = []
    all_metric_rows: list[dict] = []

    for test_season in holdout_seasons:
        predictions, metric_rows = (
            evaluate_holdout_season(
                test_season,
                matches,
                features,
                elo_features,
                feature_columns,
                ensemble_config,
            )
        )

        all_predictions.append(predictions)
        all_metric_rows.extend(metric_rows)

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    add_overall_metrics(
        combined_predictions,
        all_metric_rows,
    )

    metrics = pd.DataFrame(
        all_metric_rows
    )

    combined_predictions.to_parquet(
        PREDICTIONS_FILE,
        index=False,
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics = metrics[
        metrics["season"] == "OVERALL"
    ].sort_values("log_loss")

    print("\n最终封存测试完成。")
    print(
        f"最终测试比赛数："
        f"{len(combined_predictions)}"
    )

    print("\n各模型最终测试表现：")
    print(
        overall_metrics[
            [
                "model",
                "log_loss",
                "brier_score",
                "ranked_probability_score",
                "accuracy",
            ]
        ].to_string(index=False)
    )

    ensemble_overall = overall_metrics[
        overall_metrics["model"]
        == "ensemble"
    ].iloc[0]

    baseline_overall = overall_metrics[
        overall_metrics["model"]
        == "league_average"
    ].iloc[0]

    improvement = (
        baseline_overall["log_loss"]
        - ensemble_overall["log_loss"]
    )

    print(
        f"\n集成模型相对联赛平均"
        f"Log Loss改善：{improvement:.4f}"
    )

    if improvement > 0:
        print(
            "结论：最终系统在未见测试集上"
            "成功击败联赛平均基准。"
        )
    else:
        print(
            "结论：最终系统没有击败基准，"
            "必须如实报告，不能事后修改"
            "并继续称其为未见测试。"
        )

    print(f"\n预测文件：{PREDICTIONS_FILE}")
    print(f"指标文件：{METRICS_FILE}")
    print(
        f"冻结记录：{FREEZE_MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()