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

CONTEXT_PARQUET_FILE = (
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

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_context_audit.csv"
)


CONTEXT_COLUMNS = [
    "match_id",
    "season",
    "matchweek",
    "kickoff_utc",
    "home_team",
    "away_team",

    "context_status",
    "context_confidence",

    "lineups_confirmed",
    "lineup_checked_at_utc",
    "lineup_source",

    "home_starting_xi_json",
    "away_starting_xi_json",

    "home_formation",
    "away_formation",
    "home_usual_formation",
    "away_usual_formation",
    "home_formation_changed",
    "away_formation_changed",

    "home_rotation_index",
    "away_rotation_index",
    "home_lineup_strength_delta",
    "away_lineup_strength_delta",

    "home_key_absences_json",
    "away_key_absences_json",
    "home_injury_impact",
    "away_injury_impact",

    "home_importance",
    "away_importance",
    "importance_reason",

    "referee_name",
    "referee_cards_per_match",
    "referee_penalties_per_match",
    "referee_source",
    "referee_checked_at_utc",

    "venue_name",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kph",
    "weather_code",
    "weather_description",
    "weather_source",
    "weather_checked_at_utc",

    "home_news_summary",
    "away_news_summary",
    "home_news_impact",
    "away_news_impact",
    "news_source",
    "news_checked_at_utc",

    "adjustment_applied",
    "adjusted_home_probability",
    "adjusted_draw_probability",
    "adjusted_away_probability",
    "adjustment_explanation",

    "created_at_utc",
    "updated_at_utc",
]


NUMERIC_COLUMNS = [
    "context_confidence",
    "home_rotation_index",
    "away_rotation_index",
    "home_lineup_strength_delta",
    "away_lineup_strength_delta",
    "home_injury_impact",
    "away_injury_impact",
    "home_importance",
    "away_importance",
    "referee_cards_per_match",
    "referee_penalties_per_match",
    "temperature_c",
    "precipitation_mm",
    "wind_speed_kph",
    "home_news_impact",
    "away_news_impact",
    "adjusted_home_probability",
    "adjusted_draw_probability",
    "adjusted_away_probability",
]


BOOLEAN_COLUMNS = [
    "lineups_confirmed",
    "home_formation_changed",
    "away_formation_changed",
    "adjustment_applied",
]


def current_utc_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_default_row(
    prediction: pd.Series,
    timestamp: str,
) -> dict:
    return {
        "match_id": prediction["match_id"],
        "season": prediction["season"],
        "matchweek": prediction["matchweek"],
        "kickoff_utc": prediction["kickoff_utc"],
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],

        "context_status": "pending",
        "context_confidence": 0.0,

        "lineups_confirmed": False,
        "lineup_checked_at_utc": pd.NA,
        "lineup_source": pd.NA,

        "home_starting_xi_json": "[]",
        "away_starting_xi_json": "[]",

        "home_formation": pd.NA,
        "away_formation": pd.NA,
        "home_usual_formation": pd.NA,
        "away_usual_formation": pd.NA,
        "home_formation_changed": pd.NA,
        "away_formation_changed": pd.NA,

        "home_rotation_index": pd.NA,
        "away_rotation_index": pd.NA,
        "home_lineup_strength_delta": pd.NA,
        "away_lineup_strength_delta": pd.NA,

        "home_key_absences_json": "[]",
        "away_key_absences_json": "[]",
        "home_injury_impact": pd.NA,
        "away_injury_impact": pd.NA,

        "home_importance": pd.NA,
        "away_importance": pd.NA,
        "importance_reason": pd.NA,

        "referee_name": pd.NA,
        "referee_cards_per_match": pd.NA,
        "referee_penalties_per_match": pd.NA,
        "referee_source": pd.NA,
        "referee_checked_at_utc": pd.NA,

        "venue_name": pd.NA,
        "temperature_c": pd.NA,
        "precipitation_mm": pd.NA,
        "wind_speed_kph": pd.NA,
        "weather_code": pd.NA,
        "weather_description": pd.NA,
        "weather_source": pd.NA,
        "weather_checked_at_utc": pd.NA,

        "home_news_summary": pd.NA,
        "away_news_summary": pd.NA,
        "home_news_impact": pd.NA,
        "away_news_impact": pd.NA,
        "news_source": pd.NA,
        "news_checked_at_utc": pd.NA,

        "adjustment_applied": False,
        "adjusted_home_probability": pd.NA,
        "adjusted_draw_probability": pd.NA,
        "adjusted_away_probability": pd.NA,
        "adjustment_explanation": pd.NA,

        "created_at_utc": timestamp,
        "updated_at_utc": timestamp,
    }


def preserve_existing_values(
    template: pd.DataFrame,
) -> pd.DataFrame:
    """
    再次运行程序时保留已经收集的赛前数据，
    防止天气、阵容等内容被覆盖。
    """

    if not CONTEXT_PARQUET_FILE.exists():
        return template

    existing = pd.read_parquet(
        CONTEXT_PARQUET_FILE
    )

    if existing.empty:
        return template

    existing = existing.drop_duplicates(
        subset=["match_id"],
        keep="last",
    )

    template = template.set_index("match_id")
    existing = existing.set_index("match_id")

    common_match_ids = template.index.intersection(
        existing.index
    )

    common_columns = [
        column
        for column in existing.columns
        if column in template.columns
    ]

    for column in common_columns:
        old_values = existing.loc[
            common_match_ids,
            column,
        ]

        template.loc[
            common_match_ids,
            column,
        ] = old_values

    return template.reset_index()


def normalise_types(
    context: pd.DataFrame,
) -> pd.DataFrame:
    context["kickoff_utc"] = pd.to_datetime(
        context["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        context[column] = pd.to_numeric(
            context[column],
            errors="coerce",
        ).astype("Float64")

    for column in BOOLEAN_COLUMNS:
        context[column] = context[
            column
        ].astype("boolean")

    return context


def build_audit(
    context: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows = []

    for _, row in context.iterrows():
        lineup_complete = bool(
            row["lineups_confirmed"]
        )

        formation_complete = (
            pd.notna(row["home_formation"])
            and pd.notna(row["away_formation"])
        )

        rotation_complete = (
            pd.notna(row["home_rotation_index"])
            and pd.notna(row["away_rotation_index"])
        )

        referee_complete = pd.notna(
            row["referee_name"]
        )

        weather_complete = (
            pd.notna(row["temperature_c"])
            and pd.notna(row["wind_speed_kph"])
        )

        importance_complete = (
            pd.notna(row["home_importance"])
            and pd.notna(row["away_importance"])
        )

        news_complete = (
            pd.notna(row["home_news_summary"])
            and pd.notna(row["away_news_summary"])
        )

        completed_sections = sum(
            [
                lineup_complete,
                formation_complete,
                rotation_complete,
                referee_complete,
                weather_complete,
                importance_complete,
                news_complete,
            ]
        )

        completeness = completed_sections / 7

        audit_rows.append(
            {
                "match_id": row["match_id"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "lineup_complete": lineup_complete,
                "formation_complete": formation_complete,
                "rotation_complete": rotation_complete,
                "referee_complete": referee_complete,
                "weather_complete": weather_complete,
                "importance_complete": importance_complete,
                "news_complete": news_complete,
                "completed_sections": completed_sections,
                "total_sections": 7,
                "completeness": completeness,
            }
        )

    return pd.DataFrame(audit_rows)


def main() -> None:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "找不到未来预测文件：\n"
            f"{PREDICTIONS_FILE}"
        )

    predictions = pd.read_parquet(
        PREDICTIONS_FILE
    )

    if predictions.empty:
        raise RuntimeError(
            "未来预测文件为空。"
        )

    if predictions["match_id"].duplicated().any():
        raise RuntimeError(
            "未来预测中存在重复比赛ID。"
        )

    timestamp = current_utc_text()

    context = pd.DataFrame(
        [
            build_default_row(
                prediction,
                timestamp,
            )
            for _, prediction in predictions.iterrows()
        ]
    )

    context = context[CONTEXT_COLUMNS]

    context = preserve_existing_values(
        context
    )

    context = normalise_types(context)

    context = context.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    audit = build_audit(context)

    CONTEXT_PARQUET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTEXT_CSV_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    context.to_parquet(
        CONTEXT_PARQUET_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n赛前上下文模板生成完成。")
    print(f"比赛数量：{len(context)}")
    print(f"字段数量：{len(context.columns)}")
    print(f"上下文文件：{CONTEXT_PARQUET_FILE}")
    print(f"可查看CSV：{CONTEXT_CSV_FILE}")
    print(f"完整度报告：{AUDIT_FILE}")

    print("\n当前完整度：")

    print(
        audit[
            [
                "home_team",
                "away_team",
                "completed_sections",
                "total_sections",
                "completeness",
            ]
        ].to_string(index=False)
    )

    print(
        "\n当前全部显示0%属于正常现象，"
        "下一步将首先自动填入天气数据。"
    )


if __name__ == "__main__":
    main()