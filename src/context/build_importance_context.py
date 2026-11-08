from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

UPCOMING_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "upcoming_features.parquet"
)

IMPORTANCE_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "importance_context_audit.csv"
)

CONTEXT_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_context_audit.csv"
)


def numeric_value(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None or pd.isna(value):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def importance_level(value: float) -> str:
    if value >= 0.65:
        return "高"

    if value >= 0.35:
        return "中等"

    return "较低"


def pressure_description(
    row: pd.Series,
    prefix: str,
) -> tuple[float, str]:
    pressures = {
        "争冠": numeric_value(
            row[f"{prefix}_title_pressure"]
        ),
        "欧战资格": numeric_value(
            row[f"{prefix}_europe_pressure"]
        ),
        "保级": numeric_value(
            row[f"{prefix}_relegation_pressure"]
        ),
    }

    importance = max(
        pressures.values()
    )

    if importance < 0.05:
        pressure_focus = "常规联赛"
    else:
        maximum_pressure = max(
            pressures.values()
        )

        relevant_pressures = [
            name
            for name, value in pressures.items()
            if (
                value >= maximum_pressure * 0.75
                and value >= 0.05
            )
        ]

        pressure_focus = "、".join(
            relevant_pressures
        )

    rank = numeric_value(
        row[f"{prefix}_rank"],
        default=float("nan"),
    )

    points = numeric_value(
        row[f"{prefix}_season_points"],
        default=float("nan"),
    )

    rest_days = numeric_value(
        row[f"{prefix}_rest_days"],
        default=float("nan"),
    )

    if pd.isna(rank):
        rank_text = "排名未知"
    else:
        rank_text = f"排名第{int(rank)}"

    if pd.isna(points):
        points_text = "积分未知"
    else:
        points_text = f"{int(points)}分"

    if pd.isna(rest_days):
        rest_text = "休息天数未知"
    elif rest_days <= 3:
        rest_text = (
            f"休息{rest_days:.0f}天，"
            "赛程非常紧密"
        )
    elif rest_days <= 5:
        rest_text = (
            f"休息{rest_days:.0f}天，"
            "恢复时间一般"
        )
    else:
        rest_text = (
            f"休息{rest_days:.0f}天，"
            "恢复时间相对充足"
        )

    description = (
        f"{pressure_focus}压力"
        f"{importance_level(importance)}"
        f"（{importance:.2f}），"
        f"{rank_text}，{points_text}，"
        f"{rest_text}"
    )

    return importance, description


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到上下文文件：{CONTEXT_FILE}"
        )

    if not UPCOMING_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"找不到未来比赛特征："
            f"{UPCOMING_FEATURES_FILE}"
        )

    context = pd.read_parquet(
        CONTEXT_FILE
    )

    features = pd.read_parquet(
        UPCOMING_FEATURES_FILE
    )

    required_columns = {
        "match_id",
        "home_title_pressure",
        "home_europe_pressure",
        "home_relegation_pressure",
        "home_rank",
        "home_season_points",
        "home_rest_days",
        "away_title_pressure",
        "away_europe_pressure",
        "away_relegation_pressure",
        "away_rank",
        "away_season_points",
        "away_rest_days",
        "match_importance",
    }

    missing_columns = (
        required_columns
        - set(features.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "未来比赛特征缺少字段："
            f"{sorted(missing_columns)}"
        )

    if features["match_id"].duplicated().any():
        raise RuntimeError(
            "未来比赛特征中存在重复比赛ID。"
        )

    feature_lookup = (
        features
        .set_index("match_id")
    )

    checked_at = pd.Timestamp.now(
        tz="UTC"
    )

    first_matchweek = int(
        pd.to_numeric(
            context["matchweek"],
            errors="coerce",
        )
        .dropna()
        .min()
    )

    audit_rows: list[dict[str, Any]] = []

    print("开始生成比赛重要性上下文……")

    for index, row in context.iterrows():
        match_id = row["match_id"]
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        print(
            f"{index + 1}/{len(context)} "
            f"{home_team} vs {away_team}"
        )

        if match_id not in feature_lookup.index:
            audit_rows.append(
                {
                    "match_id": match_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "status": "feature_not_found",
                }
            )
            continue

        feature_row = feature_lookup.loc[
            match_id
        ]

        if isinstance(
            feature_row,
            pd.DataFrame,
        ):
            raise RuntimeError(
                f"比赛ID重复：{match_id}"
            )

        home_importance, home_reason = (
            pressure_description(
                feature_row,
                "home",
            )
        )

        away_importance, away_reason = (
            pressure_description(
                feature_row,
                "away",
            )
        )

        model_match_importance = (
            numeric_value(
                feature_row[
                    "match_importance"
                ]
            )
        )

        matchweek = int(
            numeric_value(
                row["matchweek"]
            )
        )

        preliminary = (
            matchweek > first_matchweek
        )

        reason_parts = [
            f"主队：{home_reason}。",
            f"客队：{away_reason}。",
            (
                "模型综合比赛重要性："
                f"{model_match_importance:.2f}。"
            ),
        ]

        if preliminary:
            reason_parts.append(
                "该场属于下一轮初步预测；"
                "当前轮尚未完全结束，"
                "积分、排名和休息天数将在"
                "下一次数据刷新后更新。"
            )
        else:
            reason_parts.append(
                "该场属于当前最近未开赛比赛。"
            )

        importance_reason = "".join(
            reason_parts
        )

        context.at[
            index,
            "home_importance",
        ] = home_importance

        context.at[
            index,
            "away_importance",
        ] = away_importance

        context.at[
            index,
            "importance_reason",
        ] = importance_reason

        context.at[
            index,
            "updated_at_utc",
        ] = checked_at.isoformat()

        audit_rows.append(
            {
                "match_id": match_id,
                "matchweek": matchweek,
                "home_team": home_team,
                "away_team": away_team,
                "home_importance": (
                    home_importance
                ),
                "away_importance": (
                    away_importance
                ),
                "model_match_importance": (
                    model_match_importance
                ),
                "home_rest_days": (
                    numeric_value(
                        feature_row[
                            "home_rest_days"
                        ],
                        default=float("nan"),
                    )
                ),
                "away_rest_days": (
                    numeric_value(
                        feature_row[
                            "away_rest_days"
                        ],
                        default=float("nan"),
                    )
                ),
                "prediction_stage": (
                    "preliminary"
                    if preliminary
                    else "current"
                ),
                "status": "available",
                "importance_reason": (
                    importance_reason
                ),
            }
        )

    weather_complete = (
        context["temperature_c"].notna()
        & context["wind_speed_kph"].notna()
        & context["weather_source"].notna()
    )

    referee_complete = (
        context["referee_name"].notna()
        & context["referee_source"].notna()
    )

    importance_complete = (
        context["home_importance"].notna()
        & context["away_importance"].notna()
        & context["importance_reason"].notna()
    )

    completed_sections = (
        weather_complete.astype(int)
        + referee_complete.astype(int)
        + importance_complete.astype(int)
    )

    context["context_confidence"] = (
        completed_sections / 7
    )

    context.loc[
        completed_sections > 0,
        "context_status",
    ] = "partial"

    context.loc[
        completed_sections == 0,
        "context_status",
    ] = "pending"

    context.to_parquet(
        CONTEXT_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    importance_audit = pd.DataFrame(
        audit_rows
    )

    importance_audit.to_csv(
        IMPORTANCE_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    context_audit = context[
        [
            "home_team",
            "away_team",
        ]
    ].copy()

    context_audit["completed_sections"] = (
        completed_sections
    )
    context_audit["total_sections"] = 7
    context_audit["completeness"] = (
        completed_sections / 7
    )

    context_audit.to_csv(
        CONTEXT_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    available_count = int(
        importance_audit["status"]
        .eq("available")
        .sum()
    )

    preliminary_count = int(
        importance_audit[
            "prediction_stage"
        ]
        .eq("preliminary")
        .sum()
    )

    print("\n重要性上下文更新完成。")
    print(f"比赛数量：{len(context)}")
    print(f"成功生成：{available_count}")
    print(
        f"当前轮正式分析："
        f"{available_count - preliminary_count}"
    )
    print(
        f"下一轮初步分析："
        f"{preliminary_count}"
    )
    print(
        f"上下文文件：{CONTEXT_FILE}"
    )
    print(
        f"重要性报告："
        f"{IMPORTANCE_AUDIT_FILE}"
    )

    print("\n比赛重要性：")
    print(
        context[
            [
                "home_team",
                "away_team",
                "home_importance",
                "away_importance",
                "importance_reason",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()