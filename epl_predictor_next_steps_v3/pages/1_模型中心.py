from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="模型中心 · EPL Predictor",
    page_icon="🧠",
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
SETTLED_FILE = (
    PROJECT_ROOT
    / "data"
    / "history"
    / "settled_predictions.parquet"
)
CHALLENGER_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "champion_challenger_metrics.csv"
)
CHALLENGER_DECISION_FILE = (
    PROJECT_ROOT / "reports" / "champion_challenger_decision.json"
)
LLM_ADVICE_FILE = PROJECT_ROOT / "reports" / "llm_model_advice.json"
SQUAD_READINESS_FILE = (
    PROJECT_ROOT / "reports" / "squad_snapshot_readiness.json"
)
SQUAD_DATASET_AUDIT_FILE = (
    PROJECT_ROOT / "reports" / "squad_training_dataset_audit.json"
)
SQUAD_CHALLENGER_METRICS_FILE = (
    PROJECT_ROOT / "reports" / "squad_challenger_metrics.csv"
)
SQUAD_CHALLENGER_DECISION_FILE = (
    PROJECT_ROOT / "reports" / "squad_challenger_decision.json"
)
SYSTEM_HEALTH_FILE = PROJECT_ROOT / "reports" / "system_health.json"
LLM_SQUAD_ADVICE_FILE = PROJECT_ROOT / "reports" / "llm_squad_advice.json"

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
    [data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #273550;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #dce7f8 !important;
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
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: #ffffff !important;
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


@st.cache_data(show_spinner=False)
def load_settled(file_mtime_ns: int) -> pd.DataFrame:
    del file_mtime_ns
    settled = pd.read_parquet(SETTLED_FILE).copy()

    for column in [
        "actual_kickoff_utc",
        "generated_at_utc",
        "settled_at_utc",
    ]:
        if column in settled.columns:
            settled[column] = pd.to_datetime(
                settled[column],
                errors="coerce",
                utc=True,
            )

    for column in [
        "settled_home_probability",
        "settled_draw_probability",
        "settled_away_probability",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "prediction_lead_hours",
    ]:
        if column in settled.columns:
            settled[column] = pd.to_numeric(
                settled[column],
                errors="coerce",
            )

    for column in ["outcome_correct", "exact_score_correct"]:
        if column in settled.columns:
            settled[column] = settled[column].fillna(False).astype(bool)

    settled["predicted_result_label"] = (
        settled["predicted_result_code"]
        .map(RESULT_LABELS)
        .fillna("未知")
    )
    settled["actual_result_label"] = (
        settled["actual_result"]
        .map(RESULT_LABELS)
        .fillna("未知")
    )
    settled["maximum_probability"] = settled[
        [
            "settled_home_probability",
            "settled_draw_probability",
            "settled_away_probability",
        ]
    ].max(axis=1)

    return settled.sort_values(
        "actual_kickoff_utc",
        ascending=False,
    ).reset_index(drop=True)


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
        <div class="review-hero-kicker">🧠 MODEL LAB · PREDICTION REVIEW</div>
        <h1>模型表现与赛果复盘</h1>
        <p>
            把模型表现、Champion/Challenger治理、样本外验证、概率质量、
            已发布预测结算与DeepSeek实验顾问统一放在一个模型中心。
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

if SETTLED_FILE.exists():
    settled = load_settled(SETTLED_FILE.stat().st_mtime_ns)
else:
    settled = pd.DataFrame()

settled_match_ids = (
    set(settled["match_id"].dropna().astype(str))
    if not settled.empty
    else set()
)
waiting = ledger.loc[
    ~ledger["match_id"].astype(str).isin(settled_match_ids)
].copy()

ledger_snapshot_count = pd.read_csv(
    LEDGER_FILE,
    usecols=["match_id"],
).shape[0]

st.markdown(
    f"""
    <div class="status-note">
        <strong>数据口径：</strong>
        当前赛季复盘来自严格样本外验证，共 {len(validation)} 场；
        已发布预测账本已从 {ledger_snapshot_count}
        条快照去重为 {len(ledger)} 场比赛，
        其中自动结算 {len(settled)} 场、等待赛果 {len(waiting)} 场。
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🧪 模型治理与实验室")
governance_tab, squad_tab, comparison_tab, advisor_tab, health_tab = st.tabs(
    [
        "🏆 Champion / Challenger",
        "🧬 阵容特征实验",
        "📊 模型横向比较",
        "🤖 DeepSeek实验顾问",
        "🩺 系统健康",
    ]
)

with governance_tab:
    if CHALLENGER_DECISION_FILE.exists():
        decision = json.loads(
            CHALLENGER_DECISION_FILE.read_text(encoding="utf-8")
        )
        status = str(decision.get("status") or "UNKNOWN")
        recommendation = str(
            decision.get("recommendation")
            or decision.get("decision")
            or "暂无晋升建议"
        )
        columns = st.columns(4)
        columns[0].metric("治理状态", status)
        columns[1].metric(
            "当前Champion",
            str(
                decision.get("current_champion")
                or decision.get("champion_model")
                or "production"
            ),
        )
        columns[2].metric(
            "候选Challenger",
            str(
                decision.get("challenger_model")
                or decision.get("candidate_model")
                or "—"
            ),
        )
        columns[3].metric(
            "自动晋升",
            "禁止" if not decision.get("production_changed", False) else "已发生",
        )
        st.info(recommendation)
    else:
        st.info(
            "尚未生成Champion/Challenger判定。运行每日更新并开启验证后会自动出现。"
        )

    st.markdown("### 🧬 球员与阵容特征历史覆盖")
    if SQUAD_READINESS_FILE.exists():
        readiness = json.loads(
            SQUAD_READINESS_FILE.read_text(encoding="utf-8")
        )
        readiness_columns = st.columns(4)
        readiness_columns[0].metric(
            "快照状态", str(readiness.get("status") or "unknown")
        )
        readiness_columns[1].metric(
            "快照日期",
            f"{int(readiness.get('unique_snapshot_dates', 0))} 天",
        )
        readiness_columns[2].metric(
            "赛程观察",
            f"{int(readiness.get('fixture_observations', 0))} 条",
        )
        readiness_columns[3].metric("生产模型", "未修改")
        st.caption(str(readiness.get("note") or "正在积累历史。"))
    else:
        st.info(
            "尚未归档首个赛前球员快照。运行每日更新或单独执行 "
            "archive_squad_snapshot.py 后开始积累。"
        )

with squad_tab:
    st.markdown("### 🧬 阵容与替补深度 Shadow Challenger")
    st.caption(
        "使用开球前冻结的球员/阵容快照，按时间顺序训练和测试；"
        "不会读取赛后信息，也不会自动替换生产模型。"
    )
    dataset_audit = {}
    if SQUAD_DATASET_AUDIT_FILE.exists():
        dataset_audit = json.loads(
            SQUAD_DATASET_AUDIT_FILE.read_text(encoding="utf-8")
        )
    decision = {}
    if SQUAD_CHALLENGER_DECISION_FILE.exists():
        decision = json.loads(
            SQUAD_CHALLENGER_DECISION_FILE.read_text(encoding="utf-8")
        )
    columns = st.columns(4)
    columns[0].metric("数据状态", str(dataset_audit.get("status") or "待生成"))
    columns[1].metric("训练样本", int(dataset_audit.get("training_rows", 0)))
    columns[2].metric(
        "独立快照日期", int(dataset_audit.get("unique_snapshot_dates", 0))
    )
    columns[3].metric("生产概率", "未修改")
    if decision:
        status = str(decision.get("status") or "unknown")
        message = str(
            decision.get("recommendation")
            or decision.get("reason")
            or "暂无说明"
        )
        if status == "ELIGIBLE_FOR_MANUAL_REVIEW":
            st.success(f"{status}：{message}")
        else:
            st.info(f"{status}：{message}")
    else:
        st.info("首次运行每日更新后会生成阵容候选模型判定。")
    if SQUAD_CHALLENGER_METRICS_FILE.exists():
        st.dataframe(
            pd.read_csv(SQUAD_CHALLENGER_METRICS_FILE),
            hide_index=True,
            width="stretch",
        )
    if LLM_SQUAD_ADVICE_FILE.exists():
        squad_advice = json.loads(
            LLM_SQUAD_ADVICE_FILE.read_text(encoding="utf-8")
        )
        if squad_advice.get("status") == "success":
            st.markdown("#### 🤖 DeepSeek阵容实验建议")
            st.write(str(squad_advice.get("summary_zh") or "建议已生成。"))
            for experiment in squad_advice.get("recommended_experiments", []):
                if isinstance(experiment, dict):
                    with st.expander(str(experiment.get("title") or "未命名实验")):
                        st.write(f"**假设：** {experiment.get('hypothesis', '—')}")
                        st.write(f"**指标：** {experiment.get('primary_metric', 'Log Loss')}")
                        st.write(f"**防泄漏：** {experiment.get('leakage_guard', '—')}")
                        st.write(f"**风险：** {experiment.get('risk', '—')}")
    st.warning(
        "只有时间测试集 Log Loss 改善且Bootstrap 95%下界仍大于0，"
        "才会进入人工复核；任何情况下都不会自动晋升。"
    )

with comparison_tab:
    if CHALLENGER_METRICS_FILE.exists():
        challenger_metrics = pd.read_csv(CHALLENGER_METRICS_FILE)
        st.dataframe(
            challenger_metrics,
            hide_index=True,
            width="stretch",
            height=min(430, 80 + len(challenger_metrics) * 35),
        )
        st.caption(
            "优先比较Log Loss、Brier Score和RPS；准确率仅作辅助。"
        )
    else:
        st.info("暂无Champion/Challenger指标文件。")

with advisor_tab:
    if LLM_ADVICE_FILE.exists():
        advice = json.loads(LLM_ADVICE_FILE.read_text(encoding="utf-8"))
        advice_status = str(advice.get("status") or "unknown")
        if advice_status in {"success", "cache_hit"}:
            st.success(str(advice.get("summary_zh") or "实验建议已生成。"))
            experiments = advice.get("recommended_experiments", [])
            for index, experiment in enumerate(experiments, start=1):
                if not isinstance(experiment, dict):
                    continue
                priority = str(experiment.get("priority") or "medium").upper()
                with st.expander(
                    f"{index}. {experiment.get('title', '未命名实验')} · {priority}",
                    expanded=index == 1,
                ):
                    st.write(f"**假设：** {experiment.get('hypothesis', '—')}")
                    st.write(f"**主要指标：** {experiment.get('primary_metric', 'Log Loss')}")
                    st.write(f"**风险：** {experiment.get('risk', '—')}")
            st.warning(
                "DeepSeek只提出实验；所有候选特征仍须通过时间顺序回测、"
                "留出集和Champion/Challenger门槛，不能直接改生产概率。"
            )
        else:
            st.warning(f"DeepSeek实验顾问状态：{advice_status}")
            st.caption(str(advice.get("reason") or "请检查API配置。"))
    else:
        st.info("尚未生成DeepSeek模型实验建议。")

with health_tab:
    st.markdown("### 🩺 数据、页面与接口健康检查")
    if SYSTEM_HEALTH_FILE.exists():
        health = json.loads(SYSTEM_HEALTH_FILE.read_text(encoding="utf-8"))
        health_status = str(health.get("status") or "unknown")
        if health_status == "healthy":
            st.success("核心文件检查通过。")
        else:
            st.warning("部分核心文件仍需关注，请查看下表。")
        checks = pd.DataFrame(health.get("file_checks") or [])
        if not checks.empty:
            st.dataframe(
                checks[["name", "status", "rows", "age_hours"]],
                hide_index=True,
                width="stretch",
            )
        environment = health.get("environment") or {}
        api_columns = st.columns(3)
        api_columns[0].metric(
            "DeepSeek", "已配置" if environment.get("deepseek_api_key_configured") else "未配置"
        )
        api_columns[1].metric(
            "football-data", "已配置" if environment.get("football_data_api_key_configured") else "未配置"
        )
        api_columns[2].metric(
            "API-Football", "已配置" if environment.get("api_football_key_configured") else "未配置"
        )
        st.caption("这里只显示是否配置，绝不会显示或写出密钥内容。")
    else:
        st.info("运行 system_health_check.py 或每日更新后生成健康报告。")

with st.sidebar:
    st.markdown("## 🎛️ 模型中心筛选")

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

filtered_settled = settled.copy()
if not filtered_settled.empty:
    if "season" in filtered_settled.columns:
        filtered_settled = filtered_settled[
            filtered_settled["season"].astype(str).isin(selected_seasons)
        ]
    if selected_team != "全部球队":
        filtered_settled = filtered_settled[
            filtered_settled["home_team"].eq(selected_team)
            | filtered_settled["away_team"].eq(selected_team)
        ]
    if outcome_filter == "预测正确":
        filtered_settled = filtered_settled[
            filtered_settled["outcome_correct"]
        ]
    elif outcome_filter == "预测错误":
        filtered_settled = filtered_settled[
            ~filtered_settled["outcome_correct"]
        ]
    filtered_settled = filtered_settled[
        filtered_settled["maximum_probability"].ge(minimum_confidence)
    ].copy()

st.markdown("## ✅ 已发布预测 · 自动结算")
if settled.empty:
    st.info(
        "当前还没有已结算的正式赛前预测。比赛完赛并运行每日更新后，"
        "结果会自动出现在这里。"
    )
elif filtered_settled.empty:
    st.info("当前筛选条件下没有已结算的正式预测。")
else:
    settled_metric_columns = st.columns(6)
    settled_metric_columns[0].metric(
        "已结算比赛",
        f"{len(filtered_settled)} 场",
    )
    settled_metric_columns[1].metric(
        "胜平负准确率",
        f"{metric_value(filtered_settled, 'outcome_correct'):.1%}",
    )
    settled_metric_columns[2].metric(
        "精确比分命中率",
        f"{metric_value(filtered_settled, 'exact_score_correct'):.1%}",
    )
    settled_metric_columns[3].metric(
        "Log Loss",
        f"{metric_value(filtered_settled, 'log_loss'):.4f}",
    )
    settled_metric_columns[4].metric(
        "Brier Score",
        f"{metric_value(filtered_settled, 'brier_score'):.4f}",
    )
    settled_metric_columns[5].metric(
        "RPS",
        f"{metric_value(filtered_settled, 'ranked_probability_score'):.4f}",
    )

    settled_display = filtered_settled.copy()
    settled_display["状态"] = np.where(
        settled_display["outcome_correct"],
        "✅ 正确",
        "❌ 错误",
    )
    settled_display["比赛时间"] = settled_display[
        "actual_kickoff_utc"
    ].dt.strftime("%Y-%m-%d %H:%M UTC")
    settled_display["主胜"] = settled_display[
        "settled_home_probability"
    ].map(format_probability)
    settled_display["平局"] = settled_display[
        "settled_draw_probability"
    ].map(format_probability)
    settled_display["客胜"] = settled_display[
        "settled_away_probability"
    ].map(format_probability)
    settled_display["提前预测"] = settled_display[
        "prediction_lead_hours"
    ].map(
        lambda value: (
            "—" if pd.isna(value) else f"{float(value):.1f}小时"
        )
    )
    settled_display = settled_display.rename(
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
        settled_display[
            [
                "状态",
                "比赛时间",
                "主队",
                "实际比分",
                "客队",
                "预测赛果",
                "实际赛果",
                "预测比分",
                "主胜",
                "平局",
                "客胜",
                "提前预测",
            ]
        ],
        hide_index=True,
        width="stretch",
        height=min(460, 82 + 35 * len(settled_display)),
    )
    st.caption(
        "这里只统计开球前最后一次有效预测；开球后生成的记录不会进入指标。"
    )

st.markdown("## 🧪 当前赛季样本外验证")

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

st.markdown("## 📈 当前赛季表现趋势")
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
        calibration["实际准确率"] = (
            pd.to_numeric(
                calibration["实际准确率"],
                errors="coerce",
            )
            * 100.0
        )
        st.dataframe(
            calibration,
            hide_index=True,
            width="stretch",
            column_config={
                "实际准确率": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption("用于观察模型的高置信度预测是否真的更可靠。")

st.markdown("## ⚖️ 样本外各模型横向比较")
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
metric_display["准确率"] = (
    pd.to_numeric(
        metric_display["准确率"],
        errors="coerce",
    )
    * 100.0
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

st.markdown("## 🔍 当前赛季逐场复盘")
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

st.markdown("## ⏳ 已发布预测 · 等待赛果")
if waiting.empty:
    if ledger.empty:
        st.info("预测账本目前为空。")
    else:
        st.success("当前账本中的已发布预测均已完成结算。")
else:
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

