from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="系统运维 · EPL Predictor", page_icon="🛡️", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def value(payload: dict, key: str, default="—"):
    result = payload.get(key, default)
    return default if result in {None, ""} else result


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background:#070c16;color:#e8f0ff;}
    [data-testid="stHeader"] {background:rgba(7,12,22,.86);}
    [data-testid="stSidebar"] {background:#101827;border-right:1px solid #273550;}
    [data-testid="stMain"] h1,[data-testid="stMain"] h2,[data-testid="stMain"] h3{color:#fff!important;}
    div[data-testid="stMetric"]{padding:18px;border:1px solid #263754;border-radius:16px;background:linear-gradient(145deg,#111b2d,#0d1524);}
    .ops-hero{padding:28px 32px;border:1px solid #29415f;border-radius:20px;background:linear-gradient(120deg,#183a59,#0b5d50);margin-bottom:22px;}
    .ops-hero h1{margin:0 0 8px;color:white}.ops-hero p{margin:0;color:#c9e7e1;}
    .safe-note{padding:14px 16px;border-left:4px solid #00e89d;border-radius:8px;background:#111a2a;color:#cdd9ec;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<section class="ops-hero"><h1>🛡️ 系统运维与实验门禁</h1>'
    '<p>汇总数据契约、球员库质量、概率校准与长期阵容实验；所有候选模型默认处于影子模式。</p></section>',
    unsafe_allow_html=True,
)

readiness = read_json("operational_readiness.json")
health = read_json("system_health.json")
monitor = read_json("model_monitor.json")
registry = read_json("player_registry_quality.json")
contract = read_json("pipeline_contract_audit.json")
calibration = read_json("calibration_challenger_decision.json")
snapshots = read_json("squad_snapshot_readiness.json")

columns = st.columns(5)
columns[0].metric("服务状态", value(readiness, "serving_status"))
columns[1].metric("系统健康", value(health, "status"))
columns[2].metric("数据契约", value(contract, "status"))
columns[3].metric("模型监控", value(monitor, "status"))
columns[4].metric("球员库质量", value(registry, "status"))

if readiness.get("safe_to_serve_existing_predictions"):
    st.markdown(
        '<div class="safe-note">✅ 现有冻结概率可继续展示；本页不会自动替换生产模型。</div>',
        unsafe_allow_html=True,
    )
else:
    st.error("存在阻断项，请查看下方运维门禁。")

status_tab, calibration_tab, registry_tab, automation_tab = st.tabs(
    ["🚦 运维门禁", "🎯 概率校准", "🧑‍💻 球员库质量", "⏱️ 自动化"]
)

with status_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("阻断项")
        blockers = readiness.get("blockers") or []
        if blockers:
            for item in blockers:
                st.error(item)
        else:
            st.success("无阻断项")
    with right:
        st.subheader("警告")
        warnings = readiness.get("warnings") or []
        if warnings:
            for item in warnings:
                st.warning(item)
        else:
            st.success("无警告")
    if contract.get("issues"):
        st.subheader("数据契约明细")
        st.dataframe(pd.DataFrame(contract["issues"]), use_container_width=True, hide_index=True)

with calibration_tab:
    columns = st.columns(4)
    columns[0].metric("候选状态", value(calibration, "status"))
    columns[1].metric("观察数", int(calibration.get("observations", 0)))
    columns[2].metric("校准温度", f"{float(calibration.get('temperature', 0)):.3f}" if calibration.get("temperature") else "—")
    columns[3].metric("Log Loss改善", f"{float(calibration.get('log_loss_improvement', 0)):.4f}" if calibration else "—")
    st.caption("通过门禁仅代表可继续影子观察，不代表已进入生产。")
    metrics_path = REPORTS / "calibration_challenger_metrics.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)

with registry_tab:
    columns = st.columns(5)
    columns[0].metric("记录", int(registry.get("records", 0)))
    columns[1].metric("唯一姓名", int(registry.get("unique_player_names", 0)))
    columns[2].metric("球队", int(registry.get("teams", 0)))
    columns[3].metric("当前归属冲突", int(registry.get("current_source_team_conflicts", 0)))
    columns[4].metric("无效号码", int(registry.get("invalid_shirt_numbers", 0)))
    team_path = REPORTS / "player_registry_team_quality.csv"
    if team_path.exists():
        st.dataframe(pd.read_csv(team_path), use_container_width=True, hide_index=True)

with automation_tab:
    st.subheader("手动执行")
    st.code("python src\\pipeline\\run_operational_checks.py", language="powershell")
    st.subheader("比赛日近实时刷新")
    st.code("python src\\pipeline\\run_matchday_refresh.py", language="powershell")
    st.caption("只有比赛进入开球前3.5小时至开球后15分钟的窗口时才会刷新上下文。")
    st.subheader("历史积累")
    st.write(
        f"阵容快照：{int(snapshots.get('snapshot_count', 0))} 份；"
        f"独立日期：{int(snapshots.get('unique_snapshot_dates', 0))} 天；"
        f"状态：{snapshots.get('status', '—')}。"
    )
