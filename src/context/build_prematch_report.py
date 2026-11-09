from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "upcoming_predictions.parquet"
)

CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "context"
    / "prematch_context.parquet"
)

CONTEXT_CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "context"
    / "prematch_context.csv"
)

REPORT_PARQUET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prematch_report.parquet"
)

REPORT_CSV_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_report.csv"
)

REPORT_JSON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_report.json"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_report_audit.csv"
)


def has_text(
    values: pd.Series,
) -> pd.Series:
    text = (
        values
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return ~text.isin(
        [
            "",
            "None",
            "nan",
            "<NA>",
            "[]",
        ]
    )


def boolean_values(
    values: pd.Series,
) -> pd.Series:
    return values.fillna(False).map(
        lambda value: (
            value is True
            or str(value).strip().lower()
            == "true"
        )
    )


def validate_inputs(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
) -> None:
    required_prediction_columns = {
        "match_id",
        "season",
        "matchweek",
        "kickoff_utc",
        "home_team",
        "away_team",
        "home_probability",
        "draw_probability",
        "away_probability",
        "predicted_outcome",
        "expected_home_goals",
        "expected_away_goals",
        "dc_home_probability",
        "dc_draw_probability",
        "dc_away_probability",
        "most_likely_score",
    }

    required_context_columns = {
        "match_id",
        "kickoff_utc",
        "home_team",
        "away_team",
        "lineups_confirmed",
        "lineup_status",
        "home_predicted_xi_json",
        "away_predicted_xi_json",
        "home_starting_xi_json",
        "away_starting_xi_json",
        "home_substitutes_json",
        "away_substitutes_json",
        "home_lineup_changes_json",
        "away_lineup_changes_json",
        "lineup_explanation",
        "home_formation",
        "away_formation",
        "home_injury_impact",
        "away_injury_impact",
        "availability_source",
        "home_importance",
        "away_importance",
        "importance_reason",
        "referee_name",
        "referee_source",
        "temperature_c",
        "precipitation_mm",
        "wind_speed_kph",
        "weather_description",
        "weather_source",
        "home_news_summary",
        "away_news_summary",
        "news_source",
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    missing_context_columns = (
        required_context_columns
        - set(context.columns)
    )

    if missing_prediction_columns:
        raise RuntimeError(
            "预测文件缺少字段："
            f"{sorted(missing_prediction_columns)}"
        )

    if missing_context_columns:
        raise RuntimeError(
            "上下文文件缺少字段："
            f"{sorted(missing_context_columns)}"
        )

    if predictions["match_id"].duplicated().any():
        raise RuntimeError(
            "预测文件存在重复比赛ID。"
        )

    if context["match_id"].duplicated().any():
        raise RuntimeError(
            "上下文文件存在重复比赛ID。"
        )

    prediction_ids = set(
        predictions["match_id"]
    )

    context_ids = set(
        context["match_id"]
    )

    if prediction_ids != context_ids:
        missing_in_context = sorted(
            prediction_ids - context_ids
        )

        missing_in_predictions = sorted(
            context_ids - prediction_ids
        )

        raise RuntimeError(
            "预测与上下文的比赛集合不一致。"
            f"上下文缺少：{missing_in_context}；"
            f"预测缺少：{missing_in_predictions}"
        )

    identity_check = predictions[
        [
            "match_id",
            "home_team",
            "away_team",
            "kickoff_utc",
        ]
    ].merge(
        context[
            [
                "match_id",
                "home_team",
                "away_team",
                "kickoff_utc",
            ]
        ],
        on="match_id",
        how="inner",
        suffixes=(
            "_prediction",
            "_context",
        ),
        validate="one_to_one",
    )

    team_mismatch = (
        (
            identity_check[
                "home_team_prediction"
            ]
            != identity_check[
                "home_team_context"
            ]
        )
        | (
            identity_check[
                "away_team_prediction"
            ]
            != identity_check[
                "away_team_context"
            ]
        )
    )

    if team_mismatch.any():
        raise RuntimeError(
            "预测与上下文存在球队名称不一致。"
        )

    prediction_probability_sum = (
        predictions["home_probability"]
        + predictions["draw_probability"]
        + predictions["away_probability"]
    )

    if not prediction_probability_sum.between(
        0.999,
        1.001,
    ).all():
        raise RuntimeError(
            "部分胜平负概率之和不等于1。"
        )


def dc_predicted_outcome(
    report: pd.DataFrame,
) -> pd.Series:
    dc_columns = [
        "dc_home_probability",
        "dc_draw_probability",
        "dc_away_probability",
    ]

    mapping = {
        "dc_home_probability": "主胜",
        "dc_draw_probability": "平局",
        "dc_away_probability": "客胜",
    }

    return (
        report[dc_columns]
        .idxmax(axis=1)
        .map(mapping)
    )


def probability_margin(
    row: pd.Series,
) -> float:
    probabilities = sorted(
        [
            float(row["home_probability"]),
            float(row["draw_probability"]),
            float(row["away_probability"]),
        ],
        reverse=True,
    )

    return (
        probabilities[0]
        - probabilities[1]
    )


def probability_separation_label(
    margin: float,
) -> str:
    if margin >= 0.15:
        return "相对明确"

    if margin >= 0.07:
        return "中等区分"

    return "结果接近"


def context_status_label(
    confidence: float,
) -> str:
    if confidence >= 0.80:
        return "信息较完整"

    if confidence >= 0.55:
        return "部分信息可用"

    return "信息有限"


def build_context_summary(
    row: pd.Series,
) -> str:
    notes = []

    home_injury = row.get(
        "home_injury_impact"
    )

    away_injury = row.get(
        "away_injury_impact"
    )

    if (
        pd.notna(home_injury)
        and pd.notna(away_injury)
    ):
        injury_difference = (
            float(home_injury)
            - float(away_injury)
        )

        if injury_difference >= 0.08:
            notes.append(
                "主队伤停负担明显更高"
            )
        elif injury_difference <= -0.08:
            notes.append(
                "客队伤停负担明显更高"
            )
        else:
            notes.append(
                "双方伤停负担差距有限"
            )

    if bool(row.get("weather_available")):
        precipitation = row.get(
            "precipitation_mm"
        )

        wind_speed = row.get(
            "wind_speed_kph"
        )

        severe_weather = (
            (
                pd.notna(precipitation)
                and float(precipitation)
                >= 2.0
            )
            or (
                pd.notna(wind_speed)
                and float(wind_speed)
                >= 30.0
            )
        )

        if severe_weather:
            notes.append(
                "天气条件可能影响比赛节奏"
            )
    else:
        notes.append(
            "天气预报尚不完整"
        )

    if not bool(
        row.get("referee_available")
    ):
        notes.append(
            "裁判安排尚未公布"
        )

    if bool(row.get("lineup_available")):
        notes.append(
            "双方官方首发已经确认"
        )
    elif bool(row.get("predicted_lineup_available")):
        notes.append(
            "官方首发尚未确认，当前展示预计阵容"
        )
    else:
        notes.append(
            "首发阵容尚未确认"
        )

    if bool(row.get("models_agree")):
        notes.append(
            "XGBoost与Dixon-Coles胜平负方向一致"
        )
    else:
        notes.append(
            "两个模型的胜平负方向存在分歧"
        )

    if bool(
        row.get("dc_new_team_warning", False)
    ):
        notes.append(
            "Dixon-Coles包含新球队估计风险"
        )

    if not notes:
        return "暂无额外上下文提示。"

    return "；".join(notes) + "。"


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"找不到预测文件：{PREDICTIONS_FILE}"
        )

    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到上下文文件：{CONTEXT_FILE}"
        )

    predictions = pd.read_parquet(
        PREDICTIONS_FILE
    ).copy()

    context = pd.read_parquet(
        CONTEXT_FILE
    ).copy()

    validate_inputs(
        predictions,
        context,
    )

    context_payload = context.drop(
        columns=[
            "season",
            "matchweek",
            "kickoff_utc",
            "home_team",
            "away_team",
        ],
        errors="ignore",
    )

    report = predictions.merge(
        context_payload,
        on="match_id",
        how="left",
        validate="one_to_one",
    )

    report["weather_available"] = (
        report["temperature_c"].notna()
        & report["weather_source"].notna()
        & has_text(
            report["weather_description"]
        )
        & (
            report["weather_description"]
            != "未知天气"
        )
    )

    report["availability_available"] = (
        report["home_injury_impact"].notna()
        & report["away_injury_impact"].notna()
        & report[
            "availability_source"
        ].notna()
    )

    report["importance_available"] = (
        report["home_importance"].notna()
        & report["away_importance"].notna()
        & has_text(
            report["importance_reason"]
        )
    )

    report["referee_available"] = (
        has_text(report["referee_name"])
        & has_text(report["referee_source"])
    )

    report["news_available"] = (
        has_text(
            report["home_news_summary"]
        )
        & has_text(
            report["away_news_summary"]
        )
        & has_text(report["news_source"])
    )

    report["lineup_available"] = (
        boolean_values(
            report["lineups_confirmed"]
        )
        & has_text(
            report["home_starting_xi_json"]
        )
        & has_text(
            report["away_starting_xi_json"]
        )
    )

    report["predicted_lineup_available"] = (
        has_text(
            report["home_predicted_xi_json"]
        )
        & has_text(
            report["away_predicted_xi_json"]
        )
    )

    report["formation_available"] = (
        has_text(report["home_formation"])
        & has_text(report["away_formation"])
    )

    section_columns = [
        "weather_available",
        "availability_available",
        "importance_available",
        "referee_available",
        "news_available",
        "lineup_available",
        "formation_available",
    ]

    report[
        "context_sections_completed"
    ] = (
        report[section_columns]
        .sum(axis=1)
        .astype(int)
    )

    report["context_sections_total"] = len(
        section_columns
    )

    report["context_completeness"] = (
        report[
            "context_sections_completed"
        ]
        / report["context_sections_total"]
    )

    section_weights = {
        "weather_available": 0.15,
        "availability_available": 0.20,
        "importance_available": 0.15,
        "referee_available": 0.10,
        "news_available": 0.10,
        "lineup_available": 0.20,
        "formation_available": 0.10,
    }

    report["context_confidence"] = 0.0

    for column, weight in (
        section_weights.items()
    ):
        report["context_confidence"] += (
            report[column].astype(float)
            * weight
        )

    report["context_status"] = (
        report["context_confidence"].map(
            context_status_label
        )
    )

    report["dc_predicted_outcome"] = (
        dc_predicted_outcome(report)
    )

    report["models_agree"] = (
        report["predicted_outcome"]
        == report["dc_predicted_outcome"]
    )

    report["probability_margin"] = (
        report.apply(
            probability_margin,
            axis=1,
        )
    )

    report[
        "probability_separation"
    ] = report[
        "probability_margin"
    ].map(
        probability_separation_label
    )

    # 当前不进行未经回测的概率修正。
    report["published_home_probability"] = (
        report["home_probability"]
    )

    report["published_draw_probability"] = (
        report["draw_probability"]
    )

    report["published_away_probability"] = (
        report["away_probability"]
    )

    report[
        "context_probability_adjustment_applied"
    ] = False

    report[
        "context_probability_adjustment_note"
    ] = (
        "上下文当前仅用于解释和风险提示；"
        "尚未经过历史回测，因此不修改冻结模型概率。"
    )

    report["context_summary"] = (
        report.apply(
            build_context_summary,
            axis=1,
        )
    )

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    report[
        "prematch_report_generated_at_utc"
    ] = generated_at

    report = report.sort_values(
        [
            "kickoff_utc",
            "match_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    REPORT_PARQUET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_CSV_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_parquet(
        REPORT_PARQUET_FILE,
        index=False,
    )

    report.to_csv(
        REPORT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    report.to_json(
        REPORT_JSON_FILE,
        orient="records",
        force_ascii=False,
        date_format="iso",
        indent=2,
    )

    context_metrics = report.set_index(
        "match_id"
    )

    context["context_confidence"] = (
        context["match_id"].map(
            context_metrics[
                "context_confidence"
            ]
        )
    )

    context["context_status"] = (
        context["match_id"].map(
            context_metrics[
                "context_status"
            ]
        )
    )

    context["updated_at_utc"] = generated_at

    context.to_parquet(
        CONTEXT_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    audit_columns = [
        "match_id",
        "home_team",
        "away_team",
        "kickoff_utc",
        "predicted_outcome",
        "dc_predicted_outcome",
        "models_agree",
        "probability_margin",
        "probability_separation",
        "context_sections_completed",
        "context_sections_total",
        "context_completeness",
        "context_confidence",
        "context_status",
        "weather_available",
        "availability_available",
        "importance_available",
        "referee_available",
        "news_available",
        "lineup_status",
        "lineup_available",
        "predicted_lineup_available",
        "formation_available",
        "context_summary",
    ]

    audit_columns.extend(
        column
        for column in [
            "llm_status",
            "llm_model",
            "llm_prematch_brief",
            "llm_used_source_urls_json",
            "llm_checked_at_utc",
        ]
        if column in report.columns
    )

    report[audit_columns].to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n统一赛前报告生成完成。")
    print(f"比赛数量：{len(report)}")
    print(
        "上下文概率调整：未启用"
    )
    print(
        f"Parquet：{REPORT_PARQUET_FILE}"
    )
    print(f"CSV：{REPORT_CSV_FILE}")
    print(f"JSON：{REPORT_JSON_FILE}")
    print(f"检查报告：{AUDIT_FILE}")

    print("\n赛前报告摘要：")

    print(
        report[
            [
                "home_team",
                "away_team",
                "predicted_outcome",
                "most_likely_score",
                "models_agree",
                "context_sections_completed",
                "context_sections_total",
                "context_confidence",
                "context_status",
                "context_summary",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
