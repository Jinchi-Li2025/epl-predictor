from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="赛果复盘 · EPL Predictor",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_external_validation_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_external_validation_metrics.csv"
)
LEDGER_FILE = PROJECT_ROOT / "reports" / "prediction_ledger.csv"

RESULT_LABELS = {
    "H": "主胜",
    "D": "平局",
    "A": "客胜",
    "主胜": "主胜",
    "平局": "平局",
    "客胜": "客胜",
}


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 92% 8%, rgba(0, 232, 157, .08), transparent 28%),
            #070c16;
    }
    [data-testid="stHeader"] {
        background: rgba(7, 12, 22, .82);
    }
    .review-hero {
        padding: 30px 34px;
        margin-bottom: 24px;
        border: 1px solid #283a58;
        border-radius: 20px;
        background:
            linear-gradient(120deg, rgba(75, 14, 130, .93), rgba(16, 80, 110, .90));
        box-shadow: 0 20px 55px rgba(0, 0, 0, .23);
    }
    .review-hero-kicker {
        margin-bottom: 9px;
        color: #4cf2bd;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .review-hero h1 {
        margin: 0 0 8px;
        color: white;
        font-size: clamp(30px, 4vw, 48px);
    }
    .review-hero p {
        max-width: 840px;
        margin: 0;
        color: #d6dcf5;
        line-height: 1.7;
    }
    .status-note {
        padding: 14px 16px;
        margin: 8px 0 20px;
        border-left: 4px solid #00e89d;
        border-radius: 8px;
        color: #cdd9ec;
        background: #111a2a;
    }
    .status-note strong {
        color: #ffffff;
    }
    div[data-testid="stMetric"] {
        min-height: 126px;
        padding: 18px 20px;
        border: 1px solid #263754;
        border-radius: 15px;
        background: linear-gradient(145deg, #111b2d, #0d1524);
    }
    div[data-testid="stMetricLabel"] {
        color: #8fa8cc;
    }
    div[data-testid="stMetricValue"] {
        color: #f4f8ff;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .small-muted {
        color: #8194b2;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_validation(file_mtime_ns: int) -> pd.DataFrame:
    del file_mtime_ns
    data = pd.read_parquet(VALIDATION_FILE)
    data = data.copy()
    data["match_date"] = pd.to_datetime(
        data["match_date"],
        errors="coerce",
    )
    data["kickoff_utc"] = pd.to_datetime(
        data["kickoff_utc"],
        errors="coerce",
        utc=True,
    )

    for column in ["p_home", "p_draw", "p_away"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["prediction_correct"] = (
        data["predicted_result"].astype(str)
        == data["actual_result"].astype(str)
    )
    data["actual_score"] = (
        data["home_goals"].astype("Int64").astype(str)
        + "-"
        + data["away_goals"].astype("Int64").astype(str)
    )
    data["score_correct"] = (
        data["most_likely_score"].astype(str)
        == data["actual_score"].astype(str)
    )
    data["predicted_result_label"] = (
        data["predicted_result"].map(RESULT_LABELS).fillna("未知")
    )
    data["actual_result_label"] = (
        data["actual_result"].map(RESULT_LABELS).fillna("未知")
    )

    data["actual_probability"] = np.select(
        [
            data["actual_result"].eq("H"),
            data["actual_result"].eq("D"),
            data["actual_result"].eq("A"),
        ],
        [data["p_home"], data["p_draw"], data["p_away"]],
        default=np.nan,
    )

    result_order = {"H": 0, "D": 1, "A": 2}
    actual_index = data["actual_result"].map(result_order)
    probability_array = data[["p_home", "p_draw", "p_away"]].to_numpy(
        dtype=float
    )
    target_array = np.zeros_like(probability_array)
    valid_index = actual_index.notna().to_numpy()
    target_array[
        np.arange(len(data))[valid_index],
        actual_index[valid_index].astype(int).to_numpy(),
    ] = 1.0
    data["row_brier"] = np.sum(
        (probability_array - target_array) ** 2,
        axis=1,
    )
    data["row_log_loss"] = -np.log(
        data["actual_probability"].clip(lower=1e-15)
    )
    data["maximum_probability"] = data[
        ["p_home", "p_draw", "p_away"]
    ].max(axis=1)
    return data.sort_values("kickoff_utc").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics(file_mtime_ns: int) -> pd.DataFrame:
    del file_mtime_ns
    metrics = pd.read_csv(METRICS_FILE)
    return metrics.copy()


@st.cache_data(show_spinner=False)
def load_ledger(file_mtime_ns: int) -> pd.DataFrame:
    del file_mtime_ns
    ledger = pd.read_csv(LEDGER_FILE)
    ledger = ledger.copy()
    ledger["kickoff_utc"] = pd.to_datetime(
        ledger["kickoff_utc"],
        errors="coerce",
        utc=True,
    )

    timestamp_column = None
    for candidate in [
        "archived_at_utc",
        "prematch_report_generated_at_utc",
        "generated_at_utc",
    ]:
        if candidate in ledger.columns:
            ledger[candidate] = pd.to_datetime(
                ledger[candidate],
                errors="coerce",
                utc=True,
            )
            if timestamp_column is None:
                timestamp_column = candidate

    if timestamp_column:
        ledger = ledger.sort_values(timestamp_column)
    ledger = ledger.drop_duplicates(subset=["match_id"], keep="last")
    return ledger.sort_values("kickoff_utc").reset_index(drop=True)


def metric_value(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def format_probability(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


st.markdown(
    """
    <section class="review-hero">
        <div class="review-hero-kicker">Prediction review</div>
        <h1>赛果与模型复盘</h1>
        <p>
            当前赛季样本外验证、概率质量和已发布赛前预测统一展示。
            历史复盘不会重新计算或覆盖当时公布的概率。
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

missing_files = [
    path
    for path in [VALIDATION_FILE, METRICS_FILE, LEDGER_FILE]
    if not path.exists()
]
if missing_files:
    st.error(
        "缺少复盘所需文件：\n"
        + "\n".join(f"- {path}" for path in missing_files)
    )
    st.stop()

validation = load_validation(VALIDATION_FILE.stat().st_mtime_ns)
metrics = load_metrics(METRICS_FILE.stat().st_mtime_ns)
ledger = load_ledger(LEDGER_FILE.stat().st_mtime_ns)

st.markdown(
    f"""
    <div class="status-note">
        <strong>数据口径：</strong>
        当前赛季复盘来自严格样本外验证，共 {len(validation)} 场；
        已发布预测账本已从 {pd.read_csv(LEDGER_FILE, usecols=['match_id']).shape[0]}
        条快照去重为 {len(ledger)} 场比赛。
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 复盘筛选")

    season_options = sorted(validation["season"].dropna().astype(str).unique())
    selected_seasons = st.multiselect(
        "赛季",
        season_options,
        default=season_options,
    )

    team_options = sorted(
        set(validation["home_team"].dropna())
        | set(validation["away_team"].dropna())
    )
    selected_team = st.selectbox(
        "球队",
        ["全部球队", *team_options],
    )

    outcome_filter = st.selectbox(
        "预测结果",
        ["全部", "预测正确", "预测错误"],
    )

    minimum_confidence = st.slider(
        "最低预测置信度",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.0f%%",
    )

filtered = validation[validation["season"].astype(str).isin(selected_seasons)]
if selected_team != "全部球队":
    filtered = filtered[
        filtered["home_team"].eq(selected_team)
        | filtered["away_team"].eq(selected_team)
    ]
if outcome_filter == "预测正确":
    filtered = filtered[filtered["prediction_correct"]]
elif outcome_filter == "预测错误":
    filtered = filtered[~filtered["prediction_correct"]]
filtered = filtered[
    filtered["maximum_probability"].ge(minimum_confidence)
].copy()

total_matches = len(filtered)
accuracy = metric_value(filtered, "prediction_correct")
score_accuracy = metric_value(filtered, "score_correct")
log_loss = metric_value(filtered, "row_log_loss")
brier_score = metric_value(filtered, "row_brier")

metric_columns = st.columns(5)
metric_columns[0].metric("复盘比赛", f"{total_matches} 场")
metric_columns[1].metric(
    "胜平负准确率",
    "—" if np.isnan(accuracy) else f"{accuracy:.1%}",
)
metric_columns[2].metric(
    "精确比分命中率",
    "—" if np.isnan(score_accuracy) else f"{score_accuracy:.1%}",
)
metric_columns[3].metric(
    "Log Loss",
    "—" if np.isnan(log_loss) else f"{log_loss:.4f}",
    help="越低越好；同时评价预测方向与概率置信度。",
)
metric_columns[4].metric(
    "Brier Score",
    "—" if np.isnan(brier_score) else f"{brier_score:.4f}",
    help="越低越好；衡量三分类概率与真实赛果之间的误差。",
)

st.markdown("## 当前赛季表现趋势")
trend_column, confidence_column = st.columns([1.35, 1])

with trend_column:
    if filtered.empty:
        st.info("当前筛选条件下没有比赛。")
    else:
        trend = filtered.sort_values("kickoff_utc").copy()
        trend["10场滚动准确率"] = (
            trend["prediction_correct"]
            .astype(float)
            .rolling(10, min_periods=1)
            .mean()
        )
        trend_chart = trend.set_index("match_date")[["10场滚动准确率"]]
        st.line_chart(
            trend_chart,
            height=310,
            color=["#00e89d"],
            width="stretch",
        )
        st.caption("每个点表示截至该场比赛的最近10场胜平负预测准确率。")

with confidence_column:
    if filtered.empty:
        st.info("暂无置信度分组数据。")
    else:
        confidence = filtered.copy()
        confidence["置信度区间"] = pd.cut(
            confidence["maximum_probability"],
            bins=[0.0, 0.40, 0.50, 0.60, 0.70, 1.0],
            labels=["≤40%", "40–50%", "50–60%", "60–70%", ">70%"],
            include_lowest=True,
        )
        calibration = (
            confidence.groupby("置信度区间", observed=False)
            .agg(
                比赛数量=("match_id", "count"),
                实际准确率=("prediction_correct", "mean"),
            )
            .reset_index()
        )
        calibration = calibration[calibration["比赛数量"].gt(0)]
        st.dataframe(
            calibration,
            hide_index=True,
            width="stretch",
            column_config={
                "实际准确率": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption("用于观察模型的高置信度预测是否真的更可靠。")

st.markdown("## 模型横向比较")
overall_metrics = metrics[metrics["season"].astype(str).eq("OVERALL")].copy()
if overall_metrics.empty:
    overall_metrics = metrics.copy()

metric_display = overall_metrics[
    [
        "model",
        "test_matches",
        "accuracy",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
    ]
].rename(
    columns={
        "model": "模型",
        "test_matches": "比赛数",
        "accuracy": "准确率",
        "log_loss": "Log Loss",
        "brier_score": "Brier Score",
        "ranked_probability_score": "RPS",
    }
)
st.dataframe(
    metric_display,
    hide_index=True,
    width="stretch",
    column_config={
        "准确率": st.column_config.NumberColumn(format="%.1f%%"),
        "Log Loss": st.column_config.NumberColumn(format="%.4f"),
        "Brier Score": st.column_config.NumberColumn(format="%.4f"),
        "RPS": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption("概率模型优先比较 Log Loss、Brier Score 与 RPS，准确率仅作辅助。")

st.markdown("## 当前赛季逐场复盘")
if filtered.empty:
    st.info("当前筛选条件下没有可展示的比赛。")
else:
    match_display = filtered.sort_values("kickoff_utc", ascending=False).copy()
    match_display["状态"] = np.where(
        match_display["prediction_correct"],
        "✅ 正确",
        "❌ 错误",
    )
    match_display["比赛日期"] = match_display["match_date"].dt.strftime("%Y-%m-%d")
    match_display["主胜"] = match_display["p_home"].map(format_probability)
    match_display["平局"] = match_display["p_draw"].map(format_probability)
    match_display["客胜"] = match_display["p_away"].map(format_probability)
    match_display = match_display.rename(
        columns={
            "home_team": "主队",
            "away_team": "客队",
            "actual_score": "实际比分",
            "most_likely_score": "预测比分",
            "predicted_result_label": "预测赛果",
            "actual_result_label": "实际赛果",
        }
    )
    st.dataframe(
        match_display[
            [
                "状态",
                "比赛日期",
                "主队",
                "实际比分",
                "客队",
                "预测赛果",
                "实际赛果",
                "预测比分",
                "主胜",
                "平局",
                "客胜",
            ]
        ],
        hide_index=True,
        width="stretch",
        height=480,
    )

st.markdown("## 已发布预测 · 等待赛果")
if ledger.empty:
    st.info("预测账本目前为空。")
else:
    waiting = ledger.copy()
    waiting["开球时间"] = waiting["kickoff_utc"].dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    waiting["主胜"] = waiting["published_home_probability"].map(
        format_probability
    )
    waiting["平局"] = waiting["published_draw_probability"].map(
        format_probability
    )
    waiting["客胜"] = waiting["published_away_probability"].map(
        format_probability
    )
    waiting["状态"] = "等待结算"
    waiting = waiting.rename(
        columns={
            "home_team": "主队",
            "away_team": "客队",
            "predicted_outcome": "预测赛果",
            "most_likely_score": "预测比分",
            "context_status": "情报状态",
        }
    )
    st.dataframe(
        waiting[
            [
                "状态",
                "开球时间",
                "主队",
                "客队",
                "预测赛果",
                "预测比分",
                "主胜",
                "平局",
                "客胜",
                "情报状态",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

st.markdown(
    """
    <p class="small-muted">
        当前赛季验证结果属于样本外评估；“已发布预测”来自赛前冻结账本。
        页面不会用赛后结果反向覆盖赛前概率。
    </p>
    """,
    unsafe_allow_html=True,
)

