from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SETTLED_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "history"
    / "settled_predictions.parquet"
)

PREDICTION_LEDGER_FILE = (
    PROJECT_ROOT
    / "data"
    / "history"
    / "prediction_ledger.parquet"
)

PREMATCH_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prematch_report.parquet"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_performance_summary.csv"
)

BY_MATCHWEEK_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_performance_by_matchweek.csv"
)

FINAL_HOLDOUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "final_holdout_metrics.csv"
)

CURRENT_VALIDATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_external_validation_metrics.csv"
)


st.set_page_config(
    page_title="模型表现 | EPL Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(0, 255, 135, 0.055),
                transparent 32%
            ),
            #080d17;
        color: #f6f8fc;
    }

    [data-testid="stSidebar"] {
        background: #101522;
        border-right: 1px solid #252c3c;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 23, 0.88);
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .performance-header {
        padding: 28px 30px;
        margin-bottom: 24px;
        border: 1px solid #273149;
        border-radius: 16px;
        background:
            linear-gradient(
                120deg,
                rgba(72, 15, 125, 0.95),
                rgba(26, 91, 112, 0.90)
            );
    }

    .performance-title {
        margin: 0;
        color: white;
        font-size: 2rem;
        font-weight: 800;
    }

    .performance-subtitle {
        margin-top: 8px;
        color: #d8e2f0;
        font-size: 0.92rem;
    }

    .status-card {
        height: 100%;
        min-height: 116px;
        padding: 18px;
        border: 1px solid #273149;
        border-radius: 12px;
        background:
            linear-gradient(
                145deg,
                #131b2d,
                #0d1422
            );
    }

    .status-label {
        color: #8496b4;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .status-value {
        margin-top: 10px;
        color: #ffffff;
        font-size: 1.65rem;
        font-weight: 800;
    }

    .status-detail {
        margin-top: 6px;
        color: #9cacbf;
        font-size: 0.77rem;
    }

    .green-value {
        color: #00ff87;
    }

    .empty-panel {
        padding: 26px;
        margin-top: 20px;
        border: 1px solid #2a354c;
        border-radius: 14px;
        background: #111827;
    }

    .empty-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 800;
    }

    .empty-text {
        margin-top: 8px;
        color: #aebbd0;
        line-height: 1.7;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #263148;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        padding: 18px;
        border: 1px solid #273149;
        border-radius: 12px;
        background: #111827;
    }

    [data-testid="stMetricValue"] {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_optional_parquet(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


@st.cache_data
def read_optional_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def first_existing_column(
    data: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in data.columns:
            return candidate

    return None


def numeric_mean(
    data: pd.DataFrame,
    candidates: list[str],
) -> float | None:
    column = first_existing_column(
        data,
        candidates,
    )

    if column is None or data.empty:
        return None

    values = pd.to_numeric(
        data[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.mean())


def metric_text(
    value: float | None,
    percentage: bool = False,
    decimals: int = 3,
) -> str:
    if value is None:
        return "—"

    if percentage:
        return f"{value:.1%}"

    return f"{value:.{decimals}f}"


ledger = read_optional_parquet(
    PREDICTION_LEDGER_FILE
)

settled = read_optional_parquet(
    SETTLED_PREDICTIONS_FILE
)

upcoming = read_optional_parquet(
    PREMATCH_REPORT_FILE
)

summary = read_optional_csv(
    SUMMARY_FILE
)

by_matchweek = read_optional_csv(
    BY_MATCHWEEK_FILE
)

final_holdout = read_optional_csv(
    FINAL_HOLDOUT_FILE
)

current_validation = read_optional_csv(
    CURRENT_VALIDATION_FILE
)


st.markdown(
    """
    <div class="performance-header">
        <div class="performance-title">
            模型表现中心
        </div>
        <div class="performance-subtitle">
            追踪真实赛前预测、比赛结果结算、
            概率质量和模型稳定性
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


ledger_versions = len(ledger)
settled_matches = len(settled)
upcoming_matches = len(upcoming)

if (
    not ledger.empty
    and "match_id" in ledger.columns
):
    ledger_fixtures = int(
        ledger["match_id"].nunique()
    )
else:
    ledger_fixtures = 0


status_columns = st.columns(4)

status_cards = [
    (
        "预测历史版本",
        str(ledger_versions),
        "每次更新都会保留赛前快照",
        True,
    ),
    (
        "账本覆盖比赛",
        str(ledger_fixtures),
        "按match_id去重后的比赛数量",
        False,
    ),
    (
        "已经结算",
        str(settled_matches),
        "必须同时满足完赛和开球前预测",
        settled_matches > 0,
    ),
    (
        "当前待预测",
        str(upcoming_matches),
        "当前赛前报告中的未来比赛",
        False,
    ),
]

for column, card in zip(
    status_columns,
    status_cards,
):
    label, value, detail, green = card

    value_class = (
        "status-value green-value"
        if green
        else "status-value"
    )

    with column:
        st.markdown(
            f"""
            <div class="status-card">
                <div class="status-label">
                    {label}
                </div>
                <div class="{value_class}">
                    {value}
                </div>
                <div class="status-detail">
                    {detail}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if settled.empty:
    st.markdown(
        """
        <div class="empty-panel">
            <div class="empty-title">
                正在等待首批预测结算
            </div>
            <div class="empty-text">
                当前预测账本已经开始积累，但账本内的比赛尚未全部结束，
                或现有预测是在开球后才首次归档，因此暂时不会计算正式表现。
                <br><br>
                自动更新程序会在比赛结束后同步真实结果，
                并只选取开球前最后一次有效预测进行评分。
                这是正常状态，不需要手动填写结果。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 冻结验证结果")

    if not final_holdout.empty:
        st.markdown("### 最终留出测试")
        st.dataframe(
            final_holdout,
            hide_index=True,
            width="stretch",
        )

    if not current_validation.empty:
        st.markdown("### 当前赛季外部验证")
        st.dataframe(
            current_validation,
            hide_index=True,
            width="stretch",
        )

    st.info(
        "首场拥有有效赛前归档的比赛结束后，"
        "这里会自动显示真实生产表现。"
    )

    st.stop()


log_loss = numeric_mean(
    settled,
    [
        "log_loss",
        "prediction_log_loss",
    ],
)

brier_score = numeric_mean(
    settled,
    [
        "brier_score",
        "multiclass_brier_score",
    ],
)

rps = numeric_mean(
    settled,
    [
        "ranked_probability_score",
        "rps",
    ],
)

outcome_accuracy = numeric_mean(
    settled,
    [
        "outcome_correct",
        "is_outcome_correct",
        "correct_prediction",
    ],
)

exact_score_accuracy = numeric_mean(
    settled,
    [
        "exact_score_correct",
        "is_exact_score_correct",
    ],
)

average_lead_time = numeric_mean(
    settled,
    [
        "lead_time_hours",
        "prediction_lead_hours",
    ],
)


st.markdown("## 真实生产表现")

metric_columns = st.columns(6)

metric_values = [
    (
        "Log Loss",
        metric_text(log_loss),
    ),
    (
        "Brier Score",
        metric_text(brier_score),
    ),
    (
        "RPS",
        metric_text(rps),
    ),
    (
        "胜平负准确率",
        metric_text(
            outcome_accuracy,
            percentage=True,
        ),
    ),
    (
        "精确比分命中率",
        metric_text(
            exact_score_accuracy,
            percentage=True,
        ),
    ),
    (
        "平均提前时间",
        (
            "—"
            if average_lead_time is None
            else f"{average_lead_time:.1f}小时"
        ),
    ),
]

for column, (
    label,
    value,
) in zip(
    metric_columns,
    metric_values,
):
    column.metric(
        label,
        value,
    )


if not by_matchweek.empty:
    st.markdown("## 分轮表现")

    matchweek_column = (
        first_existing_column(
            by_matchweek,
            [
                "matchweek",
                "round",
            ],
        )
    )

    numeric_columns = [
        column
        for column in by_matchweek.columns
        if (
            column != matchweek_column
            and pd.api.types.is_numeric_dtype(
                by_matchweek[column]
            )
        )
    ]

    chart_columns = [
        column
        for column in numeric_columns
        if column in {
            "log_loss",
            "brier_score",
            "ranked_probability_score",
            "rps",
            "accuracy",
            "outcome_accuracy",
        }
    ]

    if (
        matchweek_column is not None
        and chart_columns
    ):
        chart_data = (
            by_matchweek.set_index(
                matchweek_column
            )[chart_columns]
        )

        st.line_chart(
            chart_data
        )

    st.dataframe(
        by_matchweek,
        hide_index=True,
        width="stretch",
    )


st.markdown("## 最近结算的预测")

preferred_columns = [
    "match_date",
    "matchweek",
    "home_team",
    "away_team",
    "predicted_outcome",
    "actual_outcome",
    "home_probability",
    "draw_probability",
    "away_probability",
    "most_likely_score",
    "actual_score",
    "outcome_correct",
    "exact_score_correct",
    "log_loss",
]

visible_columns = [
    column
    for column in preferred_columns
    if column in settled.columns
]

recent_settled = settled.copy()

sort_column = first_existing_column(
    recent_settled,
    [
        "kickoff_utc",
        "match_date",
    ],
)

if sort_column is not None:
    recent_settled = (
        recent_settled.sort_values(
            sort_column,
            ascending=False,
            kind="stable",
        )
    )

if visible_columns:
    recent_settled = recent_settled[
        visible_columns
    ]

st.dataframe(
    recent_settled.head(30),
    hide_index=True,
    width="stretch",
)


st.markdown("## 验证基准")

validation_columns = st.columns(2)

with validation_columns[0]:
    st.markdown("### 最终留出测试")

    if final_holdout.empty:
        st.caption(
            "暂未找到最终留出测试报告。"
        )
    else:
        st.dataframe(
            final_holdout,
            hide_index=True,
            width="stretch",
        )

with validation_columns[1]:
    st.markdown("### 当前赛季外部验证")

    if current_validation.empty:
        st.caption(
            "暂未找到当前赛季外部验证报告。"
        )
    else:
        st.dataframe(
            current_validation,
            hide_index=True,
            width="stretch",
        )


if not summary.empty:
    with st.expander(
        "查看原始表现摘要",
        expanded=False,
    ):
        st.dataframe(
            summary,
            hide_index=True,
            width="stretch",
        )