from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LEDGER_FILE = (
    PROJECT_ROOT
    / "data"
    / "history"
    / "prediction_ledger.parquet"
)

COMPLETED_MATCHES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches.parquet"
)

SETTLED_PARQUET_FILE = (
    PROJECT_ROOT
    / "data"
    / "history"
    / "settled_predictions.parquet"
)

SETTLED_CSV_FILE = (
    PROJECT_ROOT
    / "reports"
    / "settled_predictions.csv"
)

SUMMARY_CSV_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_performance_summary.csv"
)

SUMMARY_JSON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_performance_summary.json"
)

MATCHWEEK_REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_performance_by_matchweek.csv"
)


RESULT_ALIASES = {
    "H": "H",
    "HOME": "H",
    "HOME WIN": "H",
    "主胜": "H",
    "D": "D",
    "DRAW": "D",
    "平": "D",
    "平局": "D",
    "A": "A",
    "AWAY": "A",
    "AWAY WIN": "A",
    "客胜": "A",
}


def normalise_result_code(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    return RESULT_ALIASES.get(text)


def normalise_score(value: object) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .replace("–", "-")
        .replace("—", "-")
        .replace(":", "-")
        .replace(" ", "")
    )


def probability_columns(
    ledger: pd.DataFrame,
) -> tuple[str, str, str]:
    published_columns = (
        "published_home_probability",
        "published_draw_probability",
        "published_away_probability",
    )

    if all(
        column in ledger.columns
        for column in published_columns
    ):
        return published_columns

    return (
        "home_probability",
        "draw_probability",
        "away_probability",
    )


def validate_inputs(
    ledger: pd.DataFrame,
    completed: pd.DataFrame,
) -> None:
    required_ledger_columns = {
        "match_id",
        "kickoff_utc",
        "generated_at_utc",
        "home_team",
        "away_team",
        "most_likely_score",
    }

    required_completed_columns = {
        "match_id",
        "kickoff_utc",
        "home_goals",
        "away_goals",
    }

    missing_ledger = (
        required_ledger_columns
        - set(ledger.columns)
    )

    missing_completed = (
        required_completed_columns
        - set(completed.columns)
    )

    if missing_ledger:
        raise RuntimeError(
            "预测账本缺少字段："
            f"{sorted(missing_ledger)}"
        )

    if missing_completed:
        raise RuntimeError(
            "已完赛文件缺少字段："
            f"{sorted(missing_completed)}"
        )


def main() -> None:
    if not LEDGER_FILE.exists():
        print(
            "预测历史账本尚不存在，"
            "当前没有可结算预测。"
        )
        return

    if not COMPLETED_MATCHES_FILE.exists():
        raise FileNotFoundError(
            "找不到当前赛季已完赛文件："
            f"{COMPLETED_MATCHES_FILE}"
        )

    ledger = pd.read_parquet(
        LEDGER_FILE
    ).copy()

    completed = pd.read_parquet(
        COMPLETED_MATCHES_FILE
    ).copy()

    if ledger.empty:
        print("预测历史账本为空，当前没有可结算预测。")
        return

    if completed.empty:
        print("当前赛季尚无已完赛比赛。")
        return

    validate_inputs(
        ledger,
        completed,
    )

    home_column, draw_column, away_column = (
        probability_columns(ledger)
    )

    for column in (
        home_column,
        draw_column,
        away_column,
    ):
        if column not in ledger.columns:
            raise RuntimeError(
                f"预测账本缺少概率字段：{column}"
            )

    ledger["generated_at_utc"] = (
        pd.to_datetime(
            ledger["generated_at_utc"],
            utc=True,
            errors="coerce",
        )
    )

    # generated_at_utc 决定预测是否在开球前产生；
    # archived_at_utc 只在生成时间相同时用于选择最后一次归档。
    if "archived_at_utc" in ledger.columns:
        ledger["_archive_order_utc"] = pd.to_datetime(
            ledger["archived_at_utc"],
            utc=True,
            errors="coerce",
        )
    else:
        ledger["_archive_order_utc"] = ledger[
            "generated_at_utc"
        ]

    ledger["prediction_kickoff_utc"] = (
        pd.to_datetime(
            ledger["kickoff_utc"],
            utc=True,
            errors="coerce",
        )
    )

    completed["actual_kickoff_utc"] = (
        pd.to_datetime(
            completed["kickoff_utc"],
            utc=True,
            errors="coerce",
        )
    )

    completed["home_goals"] = pd.to_numeric(
        completed["home_goals"],
        errors="coerce",
    )
    completed["away_goals"] = pd.to_numeric(
        completed["away_goals"],
        errors="coerce",
    )

    # 只把拥有完整比分的记录视为完赛；赛果直接由比分推导，
    # 避免上游 result 字段语言或格式不同造成漏结算。
    completed = completed.loc[
        completed["match_id"].notna()
        & completed["actual_kickoff_utc"].notna()
        & completed["home_goals"].notna()
        & completed["away_goals"].notna()
    ].copy()

    completed["result"] = np.select(
        [
            completed["home_goals"] > completed["away_goals"],
            completed["home_goals"] == completed["away_goals"],
        ],
        ["H", "D"],
        default="A",
    )

    # 同一比赛若在上游文件中出现多次，只保留最后一条完整记录。
    completed = (
        completed
        .sort_values("actual_kickoff_utc", kind="stable")
        .drop_duplicates(subset=["match_id"], keep="last")
        .reset_index(drop=True)
    )

    if completed.empty:
        print("当前赛季尚无带完整比分的已完赛比赛。")
        return

    completed_results = completed[
        [
            "match_id",
            "actual_kickoff_utc",
            "home_goals",
            "away_goals",
            "result",
        ]
    ].rename(
        columns={
            "home_goals": (
                "actual_home_goals"
            ),
            "away_goals": (
                "actual_away_goals"
            ),
            "result": "actual_result",
        }
    )

    merged = ledger.merge(
        completed_results,
        on="match_id",
        how="inner",
        validate="many_to_one",
    )

    valid_predictions = merged.loc[
        merged["generated_at_utc"].notna()
        & merged["actual_kickoff_utc"].notna()
        & (
            merged["generated_at_utc"]
            <= merged["actual_kickoff_utc"]
        )
    ].copy()

    if valid_predictions.empty:
        print(
            "当前预测账本中尚无已经完赛且"
            "在开球前生成的有效预测。"
        )
        return

    # 每场比赛只使用开球前最后一次预测。
    latest_predictions = (
        valid_predictions
        .sort_values(
            [
                "match_id",
                "generated_at_utc",
                "_archive_order_utc",
            ],
            kind="stable",
        )
        .groupby(
            "match_id",
            as_index=False,
            sort=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    probabilities = latest_predictions[
        [
            home_column,
            draw_column,
            away_column,
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_probability_mask = (
        np.isfinite(probabilities).all(axis=1)
        & (probabilities >= 0.0).all(axis=1)
        & (probabilities.sum(axis=1) > 0.0)
    )

    removed_probability_rows = int(
        (~finite_probability_mask).sum()
    )

    latest_predictions = latest_predictions.loc[
        finite_probability_mask
    ].reset_index(drop=True)
    probabilities = probabilities[finite_probability_mask]

    if latest_predictions.empty:
        print("已匹配比赛中没有完整有效的胜平负概率。")
        return

    probabilities = np.clip(
        probabilities,
        1e-15,
        1.0,
    )

    probability_sums = probabilities.sum(
        axis=1,
        keepdims=True,
    )

    probabilities = (
        probabilities / probability_sums
    )

    result_mapping = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    actual_result_codes = latest_predictions[
        "actual_result"
    ].map(normalise_result_code)

    valid_result_mask = (
        actual_result_codes.isin(
            result_mapping
        )
    )

    latest_predictions = (
        latest_predictions.loc[
            valid_result_mask
        ]
        .reset_index(drop=True)
    )

    probabilities = probabilities[
        valid_result_mask.to_numpy()
    ]

    actual_result_codes = (
        actual_result_codes.loc[
            valid_result_mask
        ]
        .reset_index(drop=True)
    )

    if latest_predictions.empty:
        print(
            "已匹配比赛中没有有效的H/D/A赛果。"
        )
        return

    actual_indices = (
        actual_result_codes
        .map(result_mapping)
        .astype(int)
        .to_numpy()
    )

    one_hot_actual = np.eye(3)[
        actual_indices
    ]

    selected_probabilities = probabilities[
        np.arange(
            len(probabilities)
        ),
        actual_indices,
    ]

    latest_predictions[
        "settled_home_probability"
    ] = probabilities[:, 0]

    latest_predictions[
        "settled_draw_probability"
    ] = probabilities[:, 1]

    latest_predictions[
        "settled_away_probability"
    ] = probabilities[:, 2]

    latest_predictions["log_loss"] = (
        -np.log(selected_probabilities)
    )

    latest_predictions["brier_score"] = (
        (
            probabilities
            - one_hot_actual
        )
        ** 2
    ).sum(axis=1)

    cumulative_probabilities = (
        np.cumsum(
            probabilities,
            axis=1,
        )[:, :2]
    )

    cumulative_actual = (
        np.cumsum(
            one_hot_actual,
            axis=1,
        )[:, :2]
    )

    latest_predictions[
        "ranked_probability_score"
    ] = (
        (
            cumulative_probabilities
            - cumulative_actual
        )
        ** 2
    ).mean(axis=1)

    predicted_indices = (
        probabilities.argmax(axis=1)
    )

    code_by_index = np.array(
        [
            "H",
            "D",
            "A",
        ]
    )

    latest_predictions[
        "predicted_result_code"
    ] = code_by_index[
        predicted_indices
    ]

    latest_predictions[
        "outcome_correct"
    ] = (
        latest_predictions[
            "predicted_result_code"
        ].to_numpy()
        == actual_result_codes.to_numpy()
    )

    latest_predictions[
        "actual_score"
    ] = (
        latest_predictions[
            "actual_home_goals"
        ]
        .astype(int)
        .astype(str)
        + "-"
        + latest_predictions[
            "actual_away_goals"
        ]
        .astype(int)
        .astype(str)
    )

    latest_predictions[
        "exact_score_correct"
    ] = (
        latest_predictions[
            "most_likely_score"
        ].map(normalise_score)
        == latest_predictions[
            "actual_score"
        ].map(normalise_score)
    )

    latest_predictions[
        "prediction_lead_hours"
    ] = (
        (
            latest_predictions[
                "actual_kickoff_utc"
            ]
            - latest_predictions[
                "generated_at_utc"
            ]
        )
        .dt.total_seconds()
        / 3600.0
    )

    if (
        "expected_home_goals"
        in latest_predictions.columns
    ):
        latest_predictions[
            "home_goal_absolute_error"
        ] = (
            latest_predictions[
                "expected_home_goals"
            ].astype(float)
            - latest_predictions[
                "actual_home_goals"
            ].astype(float)
        ).abs()

    if (
        "expected_away_goals"
        in latest_predictions.columns
    ):
        latest_predictions[
            "away_goal_absolute_error"
        ] = (
            latest_predictions[
                "expected_away_goals"
            ].astype(float)
            - latest_predictions[
                "actual_away_goals"
            ].astype(float)
        ).abs()

    settled_at = datetime.now(
        timezone.utc
    ).isoformat()

    latest_predictions[
        "settled_at_utc"
    ] = settled_at

    # 对完全相同的历史预测保留首次结算时间，使重复执行真正幂等。
    if SETTLED_PARQUET_FILE.exists():
        try:
            previous_settled = pd.read_parquet(
                SETTLED_PARQUET_FILE
            )
            required_previous_columns = {
                "match_id",
                "generated_at_utc",
                "settled_at_utc",
            }
            if required_previous_columns.issubset(
                previous_settled.columns
            ):
                previous_settled["generated_at_utc"] = pd.to_datetime(
                    previous_settled["generated_at_utc"],
                    utc=True,
                    errors="coerce",
                )
                previous_times = (
                    previous_settled
                    .dropna(
                        subset=[
                            "match_id",
                            "generated_at_utc",
                            "settled_at_utc",
                        ]
                    )
                    .drop_duplicates(
                        subset=["match_id", "generated_at_utc"],
                        keep="last",
                    )[
                        [
                            "match_id",
                            "generated_at_utc",
                            "settled_at_utc",
                        ]
                    ]
                    .rename(
                        columns={
                            "settled_at_utc": "_previous_settled_at_utc"
                        }
                    )
                )
                latest_predictions = latest_predictions.merge(
                    previous_times,
                    on=["match_id", "generated_at_utc"],
                    how="left",
                    validate="one_to_one",
                )
                latest_predictions["settled_at_utc"] = (
                    latest_predictions["_previous_settled_at_utc"]
                    .fillna(latest_predictions["settled_at_utc"])
                )
                latest_predictions = latest_predictions.drop(
                    columns=["_previous_settled_at_utc"]
                )
        except (OSError, ValueError, KeyError):
            # 旧结算文件损坏或结构过旧时，当前结果仍可重新生成。
            pass

    latest_predictions = latest_predictions.drop(
        columns=["_archive_order_utc"],
        errors="ignore",
    )

    latest_predictions = (
        latest_predictions
        .sort_values(
            [
                "actual_kickoff_utc",
                "match_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    SETTLED_PARQUET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SETTLED_CSV_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_predictions.to_parquet(
        SETTLED_PARQUET_FILE,
        index=False,
    )

    latest_predictions.to_csv(
        SETTLED_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "settled_matches": int(
            len(latest_predictions)
        ),
        "log_loss": float(
            latest_predictions[
                "log_loss"
            ].mean()
        ),
        "brier_score": float(
            latest_predictions[
                "brier_score"
            ].mean()
        ),
        "ranked_probability_score": float(
            latest_predictions[
                "ranked_probability_score"
            ].mean()
        ),
        "accuracy": float(
            latest_predictions[
                "outcome_correct"
            ].mean()
        ),
        "exact_score_accuracy": float(
            latest_predictions[
                "exact_score_correct"
            ].mean()
        ),
        "mean_prediction_lead_hours": float(
            latest_predictions[
                "prediction_lead_hours"
            ].mean()
        ),
        "generated_at_utc": settled_at,
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    SUMMARY_JSON_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if "matchweek" in latest_predictions:
        matchweek_summary = (
            latest_predictions
            .groupby(
                "matchweek",
                dropna=False,
            )
            .agg(
                matches=(
                    "match_id",
                    "count",
                ),
                log_loss=(
                    "log_loss",
                    "mean",
                ),
                brier_score=(
                    "brier_score",
                    "mean",
                ),
                ranked_probability_score=(
                    "ranked_probability_score",
                    "mean",
                ),
                accuracy=(
                    "outcome_correct",
                    "mean",
                ),
                exact_score_accuracy=(
                    "exact_score_correct",
                    "mean",
                ),
            )
            .reset_index()
        )

        matchweek_summary.to_csv(
            MATCHWEEK_REPORT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    print("\n历史预测自动结算完成。")
    print(
        f"已结算比赛："
        f"{summary['settled_matches']}"
    )
    print(
        f"Log Loss："
        f"{summary['log_loss']:.6f}"
    )
    print(
        f"Brier Score："
        f"{summary['brier_score']:.6f}"
    )
    print(
        "Ranked Probability Score："
        f"{summary['ranked_probability_score']:.6f}"
    )
    print(
        f"胜平负准确率："
        f"{summary['accuracy']:.2%}"
    )
    print(
        f"精确比分率："
        f"{summary['exact_score_accuracy']:.2%}"
    )
    print(
        f"平均提前预测："
        f"{summary['mean_prediction_lead_hours']:.2f}小时"
    )
    print(
        f"结算明细：{SETTLED_CSV_FILE}"
    )
    print(
        f"累计指标：{SUMMARY_CSV_FILE}"
    )
    if removed_probability_rows:
        print(
            "已跳过概率不完整的比赛："
            f"{removed_probability_rows}场"
        )


if __name__ == "__main__":
    main()
