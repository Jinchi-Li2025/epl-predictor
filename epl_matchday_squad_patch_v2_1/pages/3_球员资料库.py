from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="球员资料库 · EPL Predictor",
    page_icon="🧬",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REGISTRY_FILE = PROJECT_ROOT / "data" / "assets" / "player_registry.parquet"
OFFICIAL_SQUADS_URL = (
    "https://www.premierleague.com/en/news/4706139/"
    "see-all-the-202627-premier-league-squad-lists"
)
OFFICIAL_NUMBERS_URL = "https://www.premierleague.com/en/news/4725682"


def clean_text(value: object, fallback: str = "—") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.casefold() not in {"nan", "none"} else fallback


def safe_bool(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y"}
    return bool(value)


def player_age(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "—"
    today = date.today()
    age = today.year - parsed.year - (
        (today.month, today.day) < (parsed.month, parsed.day)
    )
    return str(age)


def appearance_text(row: pd.Series) -> str:
    total = pd.to_numeric(row.get("appearances"), errors="coerce")
    starts = pd.to_numeric(row.get("starts"), errors="coerce")
    subs = pd.to_numeric(row.get("substitute_appearances"), errors="coerce")
    if pd.isna(total):
        total = starts
    if pd.isna(total):
        return "—"
    return f"{int(total)} ({0 if pd.isna(subs) else int(subs)})"


def deepseek_player_report(player: pd.Series) -> dict:
    from src.llm.deepseek_client import DeepSeekClient

    client = DeepSeekClient()
    if not client.available:
        raise RuntimeError(client.unavailable_reason())
    facts = {
        key: (None if pd.isna(value) else value)
        for key, value in player.to_dict().items()
        if key not in {"news"}
    }
    facts["news"] = clean_text(player.get("news"), "")[:600]
    result = client.json_completion(
        system_prompt=(
            "你是英超球员资料分析员。只根据输入事实生成简体中文JSON。"
            "不得补造国籍、伤病、合同或比赛数据；缺失内容写明待核验。"
            "不得修改模型概率。返回字段：summary、role_notes、data_quality、"
            "candidate_features、cautions；除summary外均为字符串数组。"
        ),
        user_payload={"player": facts},
        temperature=0.1,
        max_tokens=1800,
    )
    return result

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 92% 4%, rgba(0,229,155,.09), transparent 28%),
            #070b13;
    }
    [data-testid="stHeader"] {
        background: rgba(7,11,19,.94);
        border-bottom: 1px solid #253651;
    }
    [data-testid="stSidebar"] {
        background: #0c1320;
        border-right: 1px solid #253651;
    }
    [data-testid="stSidebar"] * { color: #dce7f7 !important; }
    .registry-hero {
        padding: 30px 34px;
        margin-bottom: 22px;
        border: 1px solid #2b405f;
        border-radius: 22px;
        background:
            radial-gradient(circle at 90% 5%, rgba(0,229,155,.17), transparent 32%),
            linear-gradient(125deg, #15233a, #0d1727 62%, #102d2c);
        box-shadow: 0 24px 60px rgba(0,0,0,.27);
    }
    .registry-kicker {
        color: #00e59b;
        font-size: .72rem;
        font-weight: 850;
        letter-spacing: .14em;
    }
    .registry-hero h1 {
        margin: 8px 0 8px;
        color: white;
        font-size: clamp(2rem, 4vw, 3.2rem);
        letter-spacing: -.04em;
    }
    .registry-hero p {
        max-width: 900px;
        margin: 0;
        color: #afc0d8;
        line-height: 1.7;
    }
    div[data-testid="stMetric"] {
        min-height: 118px;
        padding: 17px 19px;
        border: 1px solid #273a57;
        border-radius: 15px;
        background: linear-gradient(145deg, #111c2d, #0c1422);
    }
    [data-testid="stMetricLabel"] p { color: #91a7c7 !important; }
    [data-testid="stMetricValue"] { color: #f5f8ff !important; }
    .player-profile {
        padding: 22px;
        border: 1px solid #2a3c59;
        border-radius: 18px;
        background: linear-gradient(145deg, #111c2e, #0b1320);
    }
    .player-profile-name {
        color: #fff;
        font-size: 1.5rem;
        font-weight: 820;
        letter-spacing: -.03em;
    }
    .player-profile-meta {
        margin-top: 4px;
        color: #94a8c5;
        font-size: .83rem;
    }
    .verified-pill, .unverified-pill {
        display: inline-block;
        margin-top: 12px;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: .68rem;
        font-weight: 800;
    }
    .verified-pill {
        color: #bfffe8;
        background: rgba(0,229,155,.12);
        border: 1px solid rgba(0,229,155,.28);
    }
    .unverified-pill {
        color: #ffe1a3;
        background: rgba(255,183,77,.10);
        border: 1px solid rgba(255,183,77,.27);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #283a57;
        border-radius: 15px;
        overflow: hidden;
    }
    h1, h2, h3 { color: #f5f8ff !important; }
    p, label { color: #b8c6da; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_registry(file_mtime_ns: int) -> pd.DataFrame:
    del file_mtime_ns
    data = pd.read_parquet(REGISTRY_FILE).copy()
    for column in [
        "minutes",
        "starts",
        "appearances",
        "substitute_appearances",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "form",
        "total_points",
        "clean_sheets",
        "saves",
        "yellow_cards",
        "red_cards",
        "now_cost",
        "selected_by_percent",
        "market_value",
    ]:
        if column not in data.columns:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    text_defaults = {
        "nationality": "",
        "citizenships": "",
        "multiple_nationality_status": "待核验（主源仅提供代表国籍）",
        "date_of_birth": "",
        "position_detail": "",
        "contract_start": "",
        "contract_until": "",
        "profile_source_url": "",
        "appearance_source_note": "",
    }
    for column, default in text_defaults.items():
        if column not in data.columns:
            data[column] = default
        data[column] = data[column].fillna(default).astype(str)

    if "squad_number" not in data.columns:
        data["squad_number"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    else:
        numbers = pd.to_numeric(data["squad_number"], errors="coerce")
        numbers = numbers.where(numbers.mod(1).eq(0) & numbers.between(1, 999))
        data["squad_number"] = numbers.astype("Int64")
    return data


st.markdown(
    """
    <section class="registry-hero">
        <div class="registry-kicker">🧬 SQUAD INTELLIGENCE · 2026/27</div>
        <h1>英超球员情报中心</h1>
        <p>
            汇总注册资格、号码、代表国籍、年龄、位置、可用性、合同与赛季表现。
            出场采用“总出场（替补出场）”口径；多重国籍无法由主源确认时会明确标记待核验。
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not REGISTRY_FILE.exists():
    st.error("尚未生成球员资料库。")
    st.code("python src\\assets\\build_player_registry.py", language="powershell")
    st.stop()

registry = load_registry(REGISTRY_FILE.stat().st_mtime_ns)

metric_columns = st.columns(6)
metric_columns[0].metric("覆盖球队", f"{registry['team'].nunique()} 支")
metric_columns[1].metric("资料库球员", f"{len(registry)} 人")
metric_columns[2].metric(
    "官方注册验证",
    f"{int(registry.get('registration_verified', False).fillna(False).astype(bool).sum())} 人",
)
metric_columns[3].metric(
    "Senior 25",
    f"{int(registry.get('registration_category', '').eq('senior_25').sum())} 人",
)
metric_columns[4].metric(
    "注册 U21",
    f"{int(registry.get('registration_category', '').eq('registered_u21').sum())} 人",
)
metric_columns[5].metric(
    "🌍 国籍资料",
    f"{int(registry['nationality'].str.len().gt(0).sum())} 人",
)

st.markdown("## 🔎 球员检索")
filter_columns = st.columns([1.2, 1, 1, 1, 1.4])
team_filter = filter_columns[0].selectbox(
    "球队", ["全部球队", *sorted(registry["team"].dropna().unique())]
)
position_filter = filter_columns[1].selectbox(
    "位置", ["全部位置", "G", "D", "M", "F"]
)
category_filter = filter_columns[2].selectbox(
    "注册类别",
    ["全部类别", "senior_25", "registered_u21", "fpl_unverified"],
)
availability_filter = filter_columns[3].selectbox(
    "可用性",
    ["全部状态", *sorted(registry["availability_status"].dropna().unique())],
)
search_text = filter_columns[4].text_input(
    "搜索球员", placeholder="姓名或球队"
).strip()

filtered = registry.copy()
if team_filter != "全部球队":
    filtered = filtered[filtered["team"].eq(team_filter)]
if position_filter != "全部位置":
    filtered = filtered[filtered["position_code"].eq(position_filter)]
if category_filter != "全部类别":
    filtered = filtered[filtered["registration_category"].eq(category_filter)]
if availability_filter != "全部状态":
    filtered = filtered[filtered["availability_status"].eq(availability_filter)]
if search_text:
    mask = (
        filtered["display_name"].astype(str).str.contains(search_text, case=False, regex=False)
        | filtered["registered_name"].astype(str).str.contains(search_text, case=False, regex=False)
        | filtered["team"].astype(str).str.contains(search_text, case=False, regex=False)
    )
    filtered = filtered[mask]

st.caption(f"当前筛选：{len(filtered)} 名球员")
table = filtered[
    [
        "squad_number",
        "display_name",
        "team",
        "position_code",
        "nationality",
        "date_of_birth",
        "registration_category",
        "availability_status",
        "appearances",
        "substitute_appearances",
        "starts",
        "minutes",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
    ]
].rename(
    columns={
        "squad_number": "号码",
        "display_name": "球员",
        "team": "球队",
        "position_code": "位置",
        "nationality": "代表国籍",
        "date_of_birth": "出生日期",
        "registration_category": "注册类别",
        "availability_status": "状态",
        "appearances": "总出场",
        "substitute_appearances": "替补出场",
        "starts": "首发",
        "minutes": "分钟",
        "goals_scored": "进球",
        "assists": "助攻",
        "expected_goals": "xG",
        "expected_assists": "xA",
    }
)
table["出场（替补）"] = filtered.apply(appearance_text, axis=1).values
table = table.drop(columns=["总出场", "替补出场"])
st.dataframe(table, hide_index=True, use_container_width=True, height=430)

if not filtered.empty:
    st.markdown("## 🪪 球员完整档案")
    player_options = filtered["player_id"].tolist()
    label_map = {
        row["player_id"]: f"{row['display_name']} · {row['team']}"
        for _, row in filtered.iterrows()
    }
    selected_id = st.selectbox(
        "选择球员", player_options, format_func=lambda value: label_map[value]
    )
    player = filtered.loc[filtered["player_id"].eq(selected_id)].iloc[0]
    player_number = (
        str(int(player["squad_number"]))
        if pd.notna(player.get("squad_number"))
        else "—"
    )
    profile_column, performance_column = st.columns([1, 2.2])
    verified = bool(player.get("registration_verified", False))
    category = str(player.get("registration_category") or "未分类")
    with profile_column:
        st.markdown(
            f"""
            <section class="player-profile">
                <div class="player-profile-name">{player['display_name']}</div>
                <div class="player-profile-meta">
                    {player['team']} · #{player_number} ·
                    {player.get('position_name', '位置待补充')}
                </div>
                <div class="{'verified-pill' if verified else 'unverified-pill'}">
                    {'英超官方注册名单已核对' if verified else 'FPL球员池；注册状态待核对'}
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"注册类别：{category}")
        st.write(f"🌍 代表国籍：{clean_text(player.get('nationality'), '待核验')}")
        st.write(
            "🛂 多重国籍："
            + clean_text(
                player.get("multiple_nationality_status"),
                "待核验（主源仅提供代表国籍）",
            )
        )
        st.write(
            f"🎂 出生日期：{clean_text(player.get('date_of_birth'))}"
            f" · {player_age(player.get('date_of_birth'))}岁"
        )
        st.write(f"📍 详细位置：{clean_text(player.get('position_detail'), clean_text(player.get('position_name')))}")
        st.write(f"🩺 当前状态：{player.get('availability_status', '状态未知')}")
        news = str(player.get("news") or "").strip()
        if news:
            st.caption(news)
    with performance_column:
        columns = st.columns(6)
        columns[0].metric("📊 出场（替补）", appearance_text(player))
        columns[1].metric("⏱️ 分钟", f"{int(player['minutes'])}")
        columns[2].metric("⚽ 进球", f"{int(player['goals_scored'])}")
        columns[3].metric("🎯 助攻", f"{int(player['assists'])}")
        columns[4].metric("xG", f"{float(player['expected_goals']):.2f}")
        columns[5].metric("xA", f"{float(player['expected_assists']):.2f}")

        overview_tab, contract_tab, discipline_tab, ai_tab = st.tabs(
            ["📈 赛季表现", "📝 注册与合同", "🟨 纪律与可用性", "🤖 AI球探摘要"]
        )
        with overview_tab:
            detail_columns = st.columns(4)
            detail_columns[0].metric("首发", f"{int(player['starts'])}")
            detail_columns[1].metric("零封", f"{int(player.get('clean_sheets', 0))}")
            detail_columns[2].metric("FPL积分", f"{int(player.get('total_points', 0))}")
            detail_columns[3].metric("状态指数", f"{float(player.get('form', 0)):.1f}")
            st.caption(clean_text(player.get("appearance_source_note"), "逐场出场履历尚未刷新。"))
        with contract_tab:
            st.write(f"注册类别：{category}")
            st.write(f"本土培养：{'是' if safe_bool(player.get('home_grown')) else '否/待核验'}")
            st.write(f"租借球员：{'是' if safe_bool(player.get('on_loan')) else '否'}")
            st.write(
                f"合同期：{clean_text(player.get('contract_start'))} — "
                f"{clean_text(player.get('contract_until'))}"
            )
            market_value = pd.to_numeric(
                player.get("market_value"), errors="coerce"
            )
            st.write(
                "公开估值："
                + ("—" if pd.isna(market_value) else f"€{float(market_value):,.0f}")
            )
            st.write(f"FPL身价：£{float(player.get('now_cost', 0)):.1f}m")
            st.write(
                f"FPL选择率：{float(player.get('selected_by_percent', 0)):.1f}%"
            )
        with discipline_tab:
            discipline = st.columns(3)
            discipline[0].metric("黄牌", f"{int(player.get('yellow_cards', 0))}")
            discipline[1].metric("红牌", f"{int(player.get('red_cards', 0))}")
            discipline[2].metric(
                "下轮出场概率",
                (
                    "—"
                    if pd.isna(player.get("chance_of_playing_next_round"))
                    else f"{float(player.get('chance_of_playing_next_round')):.0f}%"
                ),
            )
            news = clean_text(player.get("news"), "")
            if news:
                st.info(news)
        with ai_tab:
            st.caption(
                "DeepSeek只总结本页结构化事实并提出候选特征，不会补造资料，"
                "也不会直接修改生产模型或比赛概率。"
            )
            if st.button("🤖 生成 / 刷新球员分析", key=f"ai-{selected_id}"):
                try:
                    with st.spinner("正在生成球员分析……"):
                        report = deepseek_player_report(player)
                    st.session_state[f"player-report-{selected_id}"] = report
                except Exception as error:
                    st.error(f"DeepSeek暂不可用：{error}")
            report = st.session_state.get(f"player-report-{selected_id}")
            if report:
                st.success(clean_text(report.get("summary"), "分析已生成。"))
                for heading, key in [
                    ("角色与战术观察", "role_notes"),
                    ("可研究的模型特征", "candidate_features"),
                    ("数据质量", "data_quality"),
                    ("注意事项", "cautions"),
                ]:
                    items = report.get(key, [])
                    if isinstance(items, list) and items:
                        st.markdown(f"**{heading}**")
                        for item in items:
                            st.markdown(f"- {item}")
        st.info(
            "这些赛季字段可以进入候选特征层，但必须在时间顺序回测中证明能改善 "
            "Log Loss / Brier Score 后，才允许接入生产模型。"
        )

st.markdown("### 🔗 数据口径与可信度")
st.markdown(
    f"- [英超官方 2026/27 注册名单]({OFFICIAL_SQUADS_URL})\n"
    f"- [英超官方 2026/27 球衣号码]({OFFICIAL_NUMBERS_URL})\n"
    "- 球员位置、状态与赛季表现：Fantasy Premier League bootstrap-static 与逐球员历史接口。\n"
    "- 代表国籍、出生日期与合同：football-data.org 球队阵容资源（需配置 API Key）。\n"
    "- 多重国籍不是FPL/football-data主字段；无法由可核验来源确认时统一显示‘待核验’，不会猜测。"
)
