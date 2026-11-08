# pyright: reportMissingImports=false

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = PROJECT_ROOT / "src" / "models"

if str(MODELS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODELS_DIRECTORY))


from dixon_coles import (  # noqa: E402
    create_dc_score_matrix,
    fit_dixon_coles_model,
    predict_expected_goals,
    score_matrix_summary,
)
from gradient_boosting import (  # noqa: E402
    choose_boosting_rounds,
    make_dmatrix,
    prepare_labels,
    select_feature_columns,
    train_final_model,
)


HISTORICAL_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "matches.parquet"
)

CURRENT_COMPLETED_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches.parquet"
)

HISTORICAL_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

CURRENT_COMPLETED_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_features.parquet"
)

UPCOMING_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "upcoming_features.parquet"
)

PRODUCTION_CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "production_model.json"
)

PREDICTIONS_PARQUET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "upcoming_predictions.parquet"
)

PREDICTIONS_CSV_FILE = (
    PROJECT_ROOT
    / "reports"
    / "upcoming_predictions.csv"
)

PREDICTIONS_JSON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "upcoming_predictions.json"
)

XGBOOST_MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "xgboost_production.json"
)

ROUND_SELECTION_SEASON = "2025-26"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_required_files() -> None:
    required_files = [
        HISTORICAL_MATCHES_FILE,
        CURRENT_COMPLETED_MATCHES_FILE,
        HISTORICAL_FEATURES_FILE,
        CURRENT_COMPLETED_FEATURES_FILE,
        UPCOMING_FEATURES_FILE,
        PRODUCTION_CONFIG_FILE,
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
            "缺少生产预测所需文件：\n"
            f"{missing_text}"
        )


def validate_production_config(
    config: dict,
) -> None:
    primary_model = config.get(
        "primary_outcome_model"
    )

    scoreline_model = config.get(
        "scoreline_model"
    )

    if primary_model != "xgboost":
        raise RuntimeError(
            "生产配置中的胜平负模型不是xgboost。"
        )

    if scoreline_model != "dixon_coles":
        raise RuntimeError(
            "生产配置中的比分模型不是dixon_coles。"
        )


def combine_completed_matches() -> pd.DataFrame:
    historical = pd.read_parquet(
        HISTORICAL_MATCHES_FILE
    )

    current = pd.read_parquet(
        CURRENT_COMPLETED_MATCHES_FILE
    )

    missing_columns = [
        column
        for column in historical.columns
        if column not in current.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "当前已完赛数据缺少历史字段：\n"
            f"{missing_columns}"
        )

    current = current[
        historical.columns
    ].copy()

    matches = pd.concat(
        [
            historical,
            current,
        ],
        ignore_index=True,
    )

    matches["match_date"] = pd.to_datetime(
        matches["match_date"],
        errors="raise",
    )

    matches["kickoff_utc"] = pd.to_datetime(
        matches["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    matches["home_goals"] = pd.to_numeric(
        matches["home_goals"],
        errors="raise",
    ).astype(int)

    matches["away_goals"] = pd.to_numeric(
        matches["away_goals"],
        errors="raise",
    ).astype(int)

    if matches["match_id"].duplicated().any():
        raise RuntimeError(
            "生产训练比赛中存在重复比赛ID。"
        )

    matches = matches.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    return matches


def combine_training_features() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    historical_features = pd.read_parquet(
        HISTORICAL_FEATURES_FILE
    )

    current_features = pd.read_parquet(
        CURRENT_COMPLETED_FEATURES_FILE
    )

    missing_columns = [
        column
        for column in historical_features.columns
        if column not in current_features.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "当前已完赛特征缺少历史字段：\n"
            f"{missing_columns}"
        )

    current_features = current_features[
        historical_features.columns
    ].copy()

    training_features = pd.concat(
        [
            historical_features,
            current_features,
        ],
        ignore_index=True,
    )

    if training_features[
        "match_id"
    ].duplicated().any():
        raise RuntimeError(
            "生产训练特征中存在重复比赛ID。"
        )

    return training_features, historical_features


def choose_production_rounds(
    historical_features: pd.DataFrame,
    feature_columns: list[str],
) -> int:
    """
    只用完整历史赛季选择训练轮数。

    2025-26作为早停验证期，之后再用全部6130场
    重新训练正式生产模型。
    """

    validation_data = historical_features.loc[
        historical_features["season"]
        == ROUND_SELECTION_SEASON
    ].copy()

    training_data = historical_features.loc[
        historical_features["season"]
        < ROUND_SELECTION_SEASON
    ].copy()

    if len(validation_data) != 380:
        raise RuntimeError(
            f"{ROUND_SELECTION_SEASON}应有380行特征，"
            f"实际为{len(validation_data)}行。"
        )

    if training_data.empty:
        raise RuntimeError(
            "选择XGBoost轮数时训练数据为空。"
        )

    print(
        f"早停训练比赛：{len(training_data)}"
    )

    print(
        f"早停验证比赛：{len(validation_data)}"
    )

    boosting_rounds = choose_boosting_rounds(
        training_data=training_data,
        validation_data=validation_data,
        feature_columns=feature_columns,
    )

    if boosting_rounds <= 0:
        raise RuntimeError(
            f"无效的XGBoost轮数：{boosting_rounds}"
        )

    return int(boosting_rounds)


def determine_class_indexes() -> dict[str, int]:
    """
    自动确认H、D、A在XGBoost概率矩阵中的列号，
    避免依赖人工假设。
    """

    label_check = pd.DataFrame(
        {
            "result": [
                "H",
                "D",
                "A",
            ]
        }
    )

    encoded_labels = prepare_labels(label_check)

    class_indexes = {
        "H": int(encoded_labels[0]),
        "D": int(encoded_labels[1]),
        "A": int(encoded_labels[2]),
    }

    if set(class_indexes.values()) != {0, 1, 2}:
        raise RuntimeError(
            "无法识别XGBoost类别编码："
            f"{class_indexes}"
        )

    return class_indexes


def predict_xgboost(
    training_features: pd.DataFrame,
    upcoming_features: pd.DataFrame,
    feature_columns: list[str],
    boosting_rounds: int,
) -> tuple[np.ndarray, object]:
    print(
        "\n正在使用全部已完赛数据训练"
        "生产版XGBoost……"
    )

    model = train_final_model(
        training_data=training_features,
        feature_columns=feature_columns,
        boosting_rounds=boosting_rounds,
    )

    prediction_matrix = make_dmatrix(
        data=upcoming_features,
        feature_columns=feature_columns,
        include_labels=False,
    )

    raw_probabilities = np.asarray(
        model.predict(prediction_matrix),
        dtype=float,
    )

    number_of_matches = len(upcoming_features)

    if raw_probabilities.ndim == 1:
        raw_probabilities = raw_probabilities.reshape(
            number_of_matches,
            3,
        )

    if raw_probabilities.shape != (
        number_of_matches,
        3,
    ):
        raise RuntimeError(
            "XGBoost概率矩阵形状不正确："
            f"{raw_probabilities.shape}"
        )

    probability_sums = raw_probabilities.sum(
        axis=1,
        keepdims=True,
    )

    if (
        not np.isfinite(raw_probabilities).all()
        or (raw_probabilities < 0).any()
        or (probability_sums <= 0).any()
    ):
        raise RuntimeError(
            "XGBoost生成了无效概率。"
        )

    raw_probabilities = (
        raw_probabilities / probability_sums
    )

    return raw_probabilities, model


def predict_dixon_coles(
    completed_matches: pd.DataFrame,
    upcoming_features: pd.DataFrame,
) -> pd.DataFrame:
    print(
        "\n正在使用全部已完赛比赛拟合"
        "生产版Dixon-Coles……"
    )

    model = fit_dixon_coles_model(
        completed_matches
    )

    if "rho" not in model:
        raise RuntimeError(
            "Dixon-Coles模型中缺少rho参数。"
        )

    result_rows = []

    for _, fixture in upcoming_features.iterrows():
        home_expected_goals, away_expected_goals, new_team = (
            predict_expected_goals(
                model=model,
                home_team_id=str(
                    fixture["home_team_id"]
                ),
                away_team_id=str(
                    fixture["away_team_id"]
                ),
            )
        )

        score_matrix = create_dc_score_matrix(
            home_expected_goals=home_expected_goals,
            away_expected_goals=away_expected_goals,
            rho=float(model["rho"]),
        )

        (
            home_probability,
            draw_probability,
            away_probability,
            most_likely_score,
            top_scorelines,
        ) = score_matrix_summary(score_matrix)

        result_rows.append(
            {
                "match_id": fixture["match_id"],
                "expected_home_goals": float(
                    home_expected_goals
                ),
                "expected_away_goals": float(
                    away_expected_goals
                ),
                "dc_home_probability": float(
                    home_probability
                ),
                "dc_draw_probability": float(
                    draw_probability
                ),
                "dc_away_probability": float(
                    away_probability
                ),
                "most_likely_score": (
                    most_likely_score
                ),
                "top_scorelines": top_scorelines,
                "dc_new_team_warning": bool(
                    new_team
                ),
            }
        )

    return pd.DataFrame(result_rows)


def outcome_label(
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> str:
    probabilities = {
        "主胜": home_probability,
        "平局": draw_probability,
        "客胜": away_probability,
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


def build_prediction_output(
    upcoming_features: pd.DataFrame,
    xgboost_probabilities: np.ndarray,
    dixon_coles_predictions: pd.DataFrame,
    class_indexes: dict[str, int],
    boosting_rounds: int,
) -> pd.DataFrame:
    metadata_columns = [
        "match_id",
        "season",
        "matchweek",
        "match_date",
        "kickoff_utc",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
    ]

    missing_metadata = [
        column
        for column in metadata_columns
        if column not in upcoming_features.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            "未来特征缺少比赛信息：\n"
            f"{missing_metadata}"
        )

    output = upcoming_features[
        metadata_columns
    ].copy()

    output["home_probability"] = (
        xgboost_probabilities[
            :,
            class_indexes["H"],
        ]
    )

    output["draw_probability"] = (
        xgboost_probabilities[
            :,
            class_indexes["D"],
        ]
    )

    output["away_probability"] = (
        xgboost_probabilities[
            :,
            class_indexes["A"],
        ]
    )

    output["predicted_outcome"] = output.apply(
        lambda row: outcome_label(
            float(row["home_probability"]),
            float(row["draw_probability"]),
            float(row["away_probability"]),
        ),
        axis=1,
    )

    output["maximum_probability"] = output[
        [
            "home_probability",
            "draw_probability",
            "away_probability",
        ]
    ].max(axis=1)

    output = output.merge(
        dixon_coles_predictions,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    output["xgboost_rounds"] = boosting_rounds
    output["outcome_model"] = "xgboost"
    output["scoreline_model"] = "dixon_coles"

    output["generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
        "dc_home_probability",
        "dc_draw_probability",
        "dc_away_probability",
    ]

    for column in probability_columns:
        if not output[column].between(
            0,
            1,
            inclusive="both",
        ).all():
            raise RuntimeError(
                f"{column}存在无效概率。"
            )

    output = output.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    return output


def save_outputs(
    predictions: pd.DataFrame,
    xgboost_model,
) -> None:
    PREDICTIONS_PARQUET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_CSV_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    XGBOOST_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_parquet(
        PREDICTIONS_PARQUET_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    predictions.to_json(
        PREDICTIONS_JSON_FILE,
        orient="records",
        date_format="iso",
        force_ascii=False,
        indent=2,
    )

    xgboost_model.save_model(
        XGBOOST_MODEL_FILE
    )


def print_predictions(
    predictions: pd.DataFrame,
) -> None:
    print("\n下一轮正式预测结果：")

    for _, row in predictions.iterrows():
        print(
            "\n"
            f"{row['home_team']} vs "
            f"{row['away_team']}"
        )

        print(
            "  XGBoost胜平负："
            f"主胜 {row['home_probability']:.1%} | "
            f"平局 {row['draw_probability']:.1%} | "
            f"客胜 {row['away_probability']:.1%}"
        )

        print(
            "  主预测："
            f"{row['predicted_outcome']}"
        )

        print(
            "  Dixon-Coles预期进球："
            f"{row['expected_home_goals']:.2f} - "
            f"{row['expected_away_goals']:.2f}"
        )

        print(
            "  最可能比分："
            f"{row['most_likely_score']}"
        )

        print(
            "  比分候选："
            f"{row['top_scorelines']}"
        )


def main() -> None:
    check_required_files()

    production_config = load_json(
        PRODUCTION_CONFIG_FILE
    )

    validate_production_config(
        production_config
    )

    print("生产预测配置检查通过。")

    completed_matches = combine_completed_matches()

    (
        training_features,
        historical_features,
    ) = combine_training_features()

    upcoming_features = pd.read_parquet(
        UPCOMING_FEATURES_FILE
    )

    if upcoming_features.empty:
        raise RuntimeError(
            "未来比赛特征为空。"
        )

    feature_columns = select_feature_columns(
        historical_features
    )

    missing_feature_columns = [
        column
        for column in feature_columns
        if column not in upcoming_features.columns
    ]

    if missing_feature_columns:
        raise RuntimeError(
            "未来比赛缺少XGBoost字段：\n"
            f"{missing_feature_columns}"
        )

    print(f"生产训练比赛：{len(training_features)}")
    print(f"即将预测比赛：{len(upcoming_features)}")
    print(f"XGBoost特征数量：{len(feature_columns)}")

    boosting_rounds = choose_production_rounds(
        historical_features=historical_features,
        feature_columns=feature_columns,
    )

    print(
        f"最终XGBoost训练轮数："
        f"{boosting_rounds}"
    )

    class_indexes = determine_class_indexes()

    xgboost_probabilities, xgboost_model = (
        predict_xgboost(
            training_features=training_features,
            upcoming_features=upcoming_features,
            feature_columns=feature_columns,
            boosting_rounds=boosting_rounds,
        )
    )

    dixon_coles_predictions = (
        predict_dixon_coles(
            completed_matches=completed_matches,
            upcoming_features=upcoming_features,
        )
    )

    predictions = build_prediction_output(
        upcoming_features=upcoming_features,
        xgboost_probabilities=(
            xgboost_probabilities
        ),
        dixon_coles_predictions=(
            dixon_coles_predictions
        ),
        class_indexes=class_indexes,
        boosting_rounds=boosting_rounds,
    )

    save_outputs(
        predictions=predictions,
        xgboost_model=xgboost_model,
    )

    print_predictions(predictions)

    print("\n生产预测生成完成。")
    print(
        f"Parquet：{PREDICTIONS_PARQUET_FILE}"
    )
    print(f"CSV：{PREDICTIONS_CSV_FILE}")
    print(f"JSON：{PREDICTIONS_JSON_FILE}")
    print(f"XGBoost模型：{XGBOOST_MODEL_FILE}")


if __name__ == "__main__":
    main()