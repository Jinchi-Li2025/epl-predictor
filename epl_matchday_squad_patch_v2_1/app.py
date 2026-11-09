from __future__ import annotations

import base64
import html
import json
import mimetypes
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
from src.assets.kit_ui import render_match_kits


PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prematch_report.parquet"
)

TEAM_ASSETS_FILE = (
    PROJECT_ROOT
    / "data"
    / "assets"
    / "team_assets.parquet"
)

STADIUM_ASSETS_FILE = (
    PROJECT_ROOT
    / "data"
    / "assets"
    / "stadium_assets.parquet"
)

PLAYER_REGISTRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "assets"
    / "player_registry.parquet"
)


st.set_page_config(
    page_title="EPL Predictor",
    page_icon="⚽",
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
                rgba(0, 255, 135, 0.08),
                transparent 30%
            ),
            #080b14;
    }

    [data-testid="stSidebar"] {
        background: #101522;
        border-right: 1px solid #252c3c;
    }

    .main-header {
        padding: 28px 32px;
        border-radius: 22px;
        color: white;
        background:
            linear-gradient(
                135deg,
                #35105c 0%,
                #5d178c 55%,
                #00a86b 140%
            );
        box-shadow:
            0 18px 50px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }

    .main-header h1 {
        margin: 0;
        font-size: 42px;
        line-height: 1.05;
        letter-spacing: -1.5px;
    }

    .main-header p {
        margin: 10px 0 0 0;
        color: rgba(255, 255, 255, 0.80);
        font-size: 15px;
    }

.fixture-card {
    padding: 34px 28px 22px 28px;
    border-radius: 20px;
    background:
        linear-gradient(
            100deg,
            rgba(6, 10, 20, 0.96),
            rgba(10, 16, 29, 0.78),
            rgba(6, 10, 20, 0.94)
        ),
        linear-gradient(
            145deg,
            #141a28,
            #101521
        );
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border: 1px solid #35405a;
    margin: 14px 0 20px 0;
    min-height: 210px;
    box-shadow:
        0 18px 42px rgba(0, 0, 0, 0.34);
    overflow: hidden;
}

.fixture-card .fixture-meta,
.fixture-card .fixture-teams,
.fixture-card .fixture-venue,
.fixture-card .fixture-photo-credit {
    position: relative;
    z-index: 1;
}

.fixture-card .fixture-meta {
    color: #c4d2e8;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.90);
}

.fixture-card .team-home,
.fixture-card .team-away {
    text-shadow:
        0 3px 12px rgba(0, 0, 0, 0.95);
}

.fixture-card .fixture-venue {
    color: #d8e5f7;
    text-shadow:
        0 2px 8px rgba(0, 0, 0, 0.95);
}

.fixture-photo-credit {
    margin-top: 12px;
    color: rgba(224, 232, 245, 0.70);
    font-size: 0.68rem;
    text-align: center;
}

.fixture-photo-credit a {
    color: rgba(201, 220, 249, 0.86);
    text-decoration: none;
}

.fixture-photo-credit a:hover {
    color: #ffffff;
    text-decoration: underline;
}

    .fixture-meta {
        color: #8d99af;
        font-size: 13px;
        text-align: center;
        margin-bottom: 15px;
    }

    .fixture-teams {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 22px;
        align-items: center;
    }

    .team-home {
        text-align: right;
        font-size: 27px;
        font-weight: 750;
        color: #ffffff;
    }

    .team-away {
        text-align: left;
        font-size: 27px;
        font-weight: 750;
        color: #ffffff;
    }

    .versus {
        height: 52px;
        width: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #00ff87;
        color: #06120c;
        font-size: 13px;
        font-weight: 900;
    }

    .probability-card {
        padding: 18px;
        border-radius: 16px;
        background: #121827;
        border: 1px solid #293249;
        text-align: center;
        min-height: 128px;
    }

    .probability-label {
        color: #8e9ab0;
        font-size: 13px;
        margin-bottom: 5px;
    }

    .probability-number {
        color: white;
        font-size: 33px;
        font-weight: 800;
    }

    .probability-winner {
        border: 1px solid #00ff87;
        background:
            linear-gradient(
                145deg,
                rgba(0, 255, 135, 0.16),
                #121827
            );
    }

    .probability-bar-label {
        display: flex;
        justify-content: space-between;
        margin: 12px 2px 5px 2px;
        color: #d8dfeb;
        font-size: 14px;
    }

    .probability-track {
        width: 100%;
        height: 10px;
        background: #242b3a;
        border-radius: 999px;
        overflow: hidden;
    }

    .probability-fill {
        height: 100%;
        border-radius: 999px;
    }

    .model-panel {
        padding: 20px;
        border-radius: 18px;
        background: #111725;
        border: 1px solid #293249;
        height: 100%;
    }

    .model-panel-title {
        color: #ffffff;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 12px;
    }

    .score-display {
        color: #00ff87;
        font-size: 47px;
        line-height: 1;
        font-weight: 850;
        margin: 12px 0;
    }

    .small-muted {
        color: #8e9ab0;
        font-size: 13px;
    }

    .status-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        margin-right: 6px;
        background: rgba(0, 255, 135, 0.12);
        color: #00ff87;
        border: 1px solid rgba(0, 255, 135, 0.30);
        font-size: 12px;
        font-weight: 700;
    }

    .warning-box {
        border-left: 4px solid #ffb84d;
        padding: 13px 15px;
        border-radius: 8px;
        background: rgba(255, 184, 77, 0.09);
        color: #dfd4c4;
        font-size: 13px;
    }

    .footer-note {
        color: #778399;
        font-size: 12px;
        border-top: 1px solid #252c3c;
        padding-top: 16px;
        margin-top: 28px;
    }

    h1, h2, h3 {
        color: white;
    }

    [data-testid="stMetricValue"] {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* 主页面标题和正文 */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {
        color: #f4f7fb !important;
        letter-spacing: -0.02em;
    }

    [data-testid="stMain"]
    [data-testid="stMarkdownContainer"] p {
        color: #b9c5d8;
        line-height: 1.65;
    }

    /* 主页面指标卡片 */
    [data-testid="stMain"]
    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(20, 28, 45, 0.96),
                rgba(13, 19, 32, 0.96)
            );
        border: 1px solid #28344b;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 112px;
        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stMain"]
    [data-testid="stMetricLabel"] p {
        color: #8ea2c2 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMain"]
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 750 !important;
    }

    [data-testid="stMain"]
    [data-testid="stMetricDelta"] {
        color: #00e58b !important;
    }

    /* 标签页 */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 10px;
        border-bottom: 1px solid #2a3448;
    }

    [data-testid="stTabs"] button[role="tab"] {
        color: #8fa2c0 !important;
        background: transparent !important;
        border-radius: 8px 8px 0 0;
        padding-left: 14px;
        padding-right: 14px;
    }

    [data-testid="stTabs"]
    button[role="tab"][aria-selected="true"] {
        color: #00e58b !important;
        font-weight: 700 !important;
    }

    /* 浅色提示框必须使用深色文字 */
    [data-testid="stAlert"]
    [data-testid="stMarkdownContainer"] p {
        color: #182235 !important;
        line-height: 1.55;
    }

    /* 侧边栏可读性 */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f2f6fc !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stMarkdownContainer"] p {
        color: #aebbd0;
    }

    [data-testid="stSidebar"] hr {
        border-color: #2b3447;
    }

    /* 分隔线 */
    [data-testid="stMain"] hr {
        border-color: #263148;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* 比分候选卡片 */
    .scoreline-card {
        background: linear-gradient(
            145deg,
            #141c2d,
            #0e1523
        );
        border: 1px solid #29344a;
        border-radius: 13px;
        padding: 14px 12px;
        text-align: center;
        min-height: 112px;
        transition:
            transform 0.18s ease,
            border-color 0.18s ease;
    }

    .scoreline-card:hover {
        transform: translateY(-2px);
        border-color: #52617d;
    }

    .scoreline-primary {
        border-color: #00e58b;
        background: linear-gradient(
            145deg,
            rgba(0, 229, 139, 0.15),
            #101a25
        );
    }

    .scoreline-rank {
        color: #7f91ae;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .scoreline-score {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 7px 0 4px;
    }

    .scoreline-probability {
        color: #00e58b;
        font-size: 0.92rem;
        font-weight: 700;
    }

    /* 比赛概览卡片 */
    .overview-fixture-card {
        background: linear-gradient(
            145deg,
            #141c2d,
            #0d1421
        );
        border: 1px solid #29354a;
        border-radius: 15px;
        padding: 16px 18px;
        margin-bottom: 13px;
        min-height: 190px;
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.16);
        transition:
            transform 0.18s ease,
            border-color 0.18s ease;
    }

    .overview-fixture-card:hover {
        transform: translateY(-2px);
        border-color: #465675;
    }

    .overview-fixture-meta {
        color: #7f92af;
        font-size: 0.74rem;
        margin-bottom: 9px;
    }

    .overview-fixture-teams {
    display: grid;
    grid-template-columns:
        minmax(0, 1fr)
        auto
        minmax(0, 1fr);
    align-items: center;
    gap: 9px;
    color: #f8fafc;
    font-size: 1.03rem;
    font-weight: 750;
    margin-bottom: 12px;
}

.overview-team-side {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}

.overview-team-home {
    justify-content: flex-end;
    text-align: right;
}

.overview-team-away {
    justify-content: flex-start;
    text-align: left;
}

.overview-team-name {
    line-height: 1.25;
    overflow-wrap: anywhere;
}

.overview-fixture-card .team-crest,
.overview-fixture-card
.team-crest-placeholder {
    width: 38px;
    height: 38px;
    flex: 0 0 38px;
}

.overview-fixture-card
.team-crest-placeholder {
    font-size: 17px;
}

    .overview-versus {
        color: #00e58b;
        margin: 0 7px;
        font-size: 0.78rem;
    }

    .overview-probabilities {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 7px;
        margin: 11px 0;
    }

    .overview-probability {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid #283349;
        border-radius: 8px;
        padding: 7px 5px;
        text-align: center;
    }

    .overview-probability-label {
        color: #7f91ae;
        font-size: 0.66rem;
    }

    .overview-probability-value {
        color: #f5f7fb;
        font-size: 0.9rem;
        font-weight: 750;
    }

    .overview-card-footer {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 10px;
    }

    .overview-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 0.68rem;
        font-weight: 700;
        background: #202a3d;
        color: #bdc8d9;
    }

    .outcome-home {
        background: rgba(0, 229, 139, 0.15);
        color: #00e58b;
    }

    .outcome-draw {
        background: rgba(154, 120, 255, 0.17);
        color: #b6a0ff;
    }

    .outcome-away {
        background: rgba(57, 168, 255, 0.16);
        color: #62baff;
    }

    .agreement-good {
        color: #00e58b;
    }

    .agreement-warning {
        color: #ffbf5b;
    }
    .team-identity {
    display: flex;
    align-items: center;
    flex: 1;
    gap: 18px;
    min-width: 0;
}

.home-identity {
    justify-content: flex-end;
}

.away-identity {
    justify-content: flex-start;
}

.team-crest {
    width: 72px;
    height: 72px;
    object-fit: contain;
    flex: 0 0 72px;
    filter:
        drop-shadow(
            0 8px 14px
            rgba(0, 0, 0, 0.34)
        );
}

.team-crest-placeholder {
    width: 72px;
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 72px;
    border: 1px solid #2d374c;
    border-radius: 50%;
    background: #151c2c;
    font-size: 28px;
}

.fixture-venue {
    margin-top: 16px;
    color: #8fa1bd;
    font-size: 0.86rem;
    text-align: center;
    letter-spacing: 0.02em;
}

@media (max-width: 760px) {
    .fixture-teams {
        gap: 10px;
    }

    .team-identity {
        gap: 8px;
    }

    .team-crest,
    .team-crest-placeholder {
        width: 48px;
        height: 48px;
        flex-basis: 48px;
    }

    .team-home,
    .team-away {
        font-size: 0.95rem;
    }

    .versus {
        font-size: 0.72rem;
    }
}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root {
        --surface-0: #070b13;
        --surface-1: #0d1421;
        --surface-2: #121c2d;
        --surface-3: #172338;
        --line: #263753;
        --text: #f4f8ff;
        --muted: #9db0ca;
        --accent: #00e59b;
    }
    [data-testid="stHeader"] {
        background: rgba(7, 11, 19, .94) !important;
        border-bottom: 1px solid rgba(38, 55, 83, .75);
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--surface-0) !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] {
        background: #0c1320 !important;
    }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarNav"] span,
    [data-testid="stSidebarNav"] a {
        color: #dce7f7 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #151f30 !important;
        border-color: #32435e !important;
        color: #f4f8ff !important;
    }
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: #121c2d !important;
        color: #f4f8ff !important;
    }
    [role="option"] { color: #e8eff9 !important; }
    [role="option"]:hover { background: #1b2a41 !important; }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
        background: var(--surface-1) !important;
    }
    [data-testid="stDataFrame"] iframe {
        color-scheme: dark;
    }
    [data-testid="stVegaLiteChart"],
    [data-testid="stPlotlyChart"] {
        padding: 8px;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--surface-1) !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #91a5c2 !important;
    }
    [data-testid="stExpander"] {
        border-color: var(--line) !important;
        background: #0d1421 !important;
    }
    .lineup-team-shell {
        padding: 18px;
        border: 1px solid #263753;
        border-radius: 20px;
        background:
            radial-gradient(circle at 90% 0, rgba(0,229,155,.08), transparent 34%),
            linear-gradient(145deg, #101a2a, #0b111c);
        box-shadow: 0 18px 42px rgba(0,0,0,.23);
        margin-bottom: 14px;
    }
    .lineup-team-heading {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .lineup-team-heading h3 {
        margin: 0;
        font-size: 1.25rem;
    }
    .lineup-status-pill {
        display: inline-flex;
        align-items: center;
        min-height: 27px;
        padding: 4px 10px;
        border-radius: 999px;
        color: #bbffe7;
        background: rgba(0,229,155,.12);
        border: 1px solid rgba(0,229,155,.26);
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .03em;
        white-space: nowrap;
    }
    .lineup-status-pill.predicted {
        color: #d8e6ff;
        background: rgba(77,145,255,.12);
        border-color: rgba(77,145,255,.28);
    }
    .lineup-meta {
        color: #91a5c2;
        font-size: .78rem;
        margin-bottom: 14px;
    }
    .tactical-layout {
        display: grid;
        grid-template-columns: minmax(320px, 1.75fr) minmax(155px, .75fr);
        gap: 14px;
        align-items: stretch;
    }
    .tactical-pitch {
        position: relative;
        min-height: 520px;
        overflow: hidden;
        border: 3px solid rgba(239,255,245,.74);
        border-radius: 16px;
        background:
            repeating-linear-gradient(
                90deg,
                rgba(255,255,255,.026) 0,
                rgba(255,255,255,.026) 12.5%,
                rgba(0,0,0,.035) 12.5%,
                rgba(0,0,0,.035) 25%
            ),
            linear-gradient(180deg, #118149 0%, #0d6b3c 100%);
        box-shadow: inset 0 0 45px rgba(0,0,0,.18);
    }
    .pitch-halfway {
        position: absolute;
        left: 0; right: 0; top: 50%;
        border-top: 2px solid rgba(255,255,255,.62);
    }
    .pitch-centre-circle {
        position: absolute;
        left: 50%; top: 50%;
        width: 112px; height: 112px;
        border: 2px solid rgba(255,255,255,.62);
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }
    .pitch-centre-dot {
        position: absolute;
        left: 50%; top: 50%;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: rgba(255,255,255,.75);
        transform: translate(-50%, -50%);
    }
    .pitch-box {
        position: absolute;
        left: 18%; width: 64%; height: 16%;
        border: 2px solid rgba(255,255,255,.62);
    }
    .pitch-box.top { top: -2px; border-top: 0; }
    .pitch-box.bottom { bottom: -2px; border-bottom: 0; }
    .pitch-goal-box {
        position: absolute;
        left: 35%; width: 30%; height: 7%;
        border: 2px solid rgba(255,255,255,.62);
    }
    .pitch-goal-box.top { top: -2px; border-top: 0; }
    .pitch-goal-box.bottom { bottom: -2px; border-bottom: 0; }
    .pitch-player {
        position: absolute;
        width: 86px;
        transform: translate(-50%, -50%);
        text-align: center;
        z-index: 4;
    }
    .pitch-player-marker {
        display: flex;
        width: 40px; height: 40px;
        margin: 0 auto 4px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        color: #07150f;
        background: #f4fff9;
        border: 3px solid #00e59b;
        font-size: .66rem;
        font-weight: 900;
        box-shadow: 0 7px 14px rgba(0,0,0,.32);
    }
    .pitch-player-name {
        display: inline-block;
        max-width: 84px;
        padding: 2px 5px;
        overflow: hidden;
        border-radius: 5px;
        color: white;
        background: rgba(2,12,8,.68);
        font-size: .66rem;
        font-weight: 750;
        line-height: 1.15;
        text-overflow: ellipsis;
        white-space: nowrap;
        text-shadow: 0 1px 4px #000;
    }
    .bench-panel {
        min-height: 520px;
        padding: 13px;
        border: 1px solid #30425f;
        border-radius: 14px;
        background: rgba(7,13,23,.76);
    }
    .bench-title {
        margin: 0 0 10px;
        color: #eef5ff;
        font-size: .8rem;
        font-weight: 850;
        letter-spacing: .06em;
    }
    .bench-player {
        display: grid;
        grid-template-columns: 26px 1fr 30px;
        gap: 6px;
        align-items: center;
        min-height: 39px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(51,68,96,.55);
    }
    .bench-number {
        color: #7f96b6;
        font-size: .68rem;
        text-align: center;
    }
    .bench-name {
        overflow: hidden;
        color: #e7eef9;
        font-size: .72rem;
        font-weight: 680;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .bench-pos {
        padding: 3px 5px;
        border-radius: 6px;
        color: #00e59b;
        background: rgba(0,229,155,.10);
        font-size: .62rem;
        font-weight: 850;
        text-align: center;
    }
    .bench-empty {
        color: #7f93b2;
        font-size: .72rem;
        line-height: 1.55;
    }
    @media (max-width: 1100px) {
        .tactical-layout { grid-template-columns: 1fr; }
        .bench-panel { min-height: auto; }
        .tactical-pitch { min-height: 490px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"找不到预测文件：{PREDICTIONS_FILE}"
        )

    data = pd.read_parquet(PREDICTIONS_FILE)

    data["kickoff_utc"] = pd.to_datetime(
        data["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    data = data.sort_values(
        ["kickoff_utc", "match_id"],
        kind="stable",
    ).reset_index(drop=True)

    return data

@st.cache_data
def load_team_assets() -> pd.DataFrame:
    if not TEAM_ASSETS_FILE.exists():
        return pd.DataFrame()

    assets = pd.read_parquet(
        TEAM_ASSETS_FILE
    )

    required_columns = {
        "team_id",
        "team_name",
        "crest_path",
        "venue_name",
        "club_colors",
    }

    missing_columns = (
        required_columns
        - set(assets.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "球队资源文件缺少字段："
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return assets

@st.cache_data
def load_stadium_assets() -> pd.DataFrame:
    if not STADIUM_ASSETS_FILE.exists():
        return pd.DataFrame()

    assets = pd.read_parquet(
        STADIUM_ASSETS_FILE
    )

    required_columns = {
        "home_team_id",
        "home_team",
        "venue_name",
        "stadium_image_path",
        "commons_file_page",
        "artist",
        "license",
    }

    missing_columns = (
        required_columns
        - set(assets.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "球场资源文件缺少字段："
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return assets


@st.cache_data
def load_player_registry() -> pd.DataFrame:
    if not PLAYER_REGISTRY_FILE.exists():
        return pd.DataFrame()

    registry = pd.read_parquet(
        PLAYER_REGISTRY_FILE
    )

    if "team" not in registry.columns:
        return pd.DataFrame()

    return registry.copy()


@st.cache_data(
    show_spinner=False
)
def image_file_to_data_uri(
    relative_path: str,
) -> str:
    if not relative_path:
        return ""

    image_path = (
        PROJECT_ROOT
        / relative_path
    )

    if not image_path.exists():
        return ""

    mime_type, _ = (
        mimetypes.guess_type(
            image_path.name
        )
    )

    if not mime_type:
        mime_type = "image/png"

    encoded_image = (
        base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")
    )

    return (
        f"data:{mime_type};"
        f"base64,{encoded_image}"
    )


def clean_optional_text(
    value,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def find_team_asset(
    assets: pd.DataFrame,
    team_id,
    team_name,
) -> dict:
    if assets.empty:
        return {}

    clean_team_id = (
        clean_optional_text(team_id)
    )

    clean_team_name = (
        clean_optional_text(team_name)
    )

    if clean_team_id:
        id_matches = assets.loc[
            assets["team_id"].astype(str)
            == clean_team_id
        ]

        if not id_matches.empty:
            return (
                id_matches.iloc[0]
                .to_dict()
            )

    if clean_team_name:
        name_matches = assets.loc[
            assets["team_name"]
            .astype(str)
            .str.casefold()
            == clean_team_name.casefold()
        ]

        if not name_matches.empty:
            return (
                name_matches.iloc[0]
                .to_dict()
            )

    return {}


def build_crest_html(
    team_asset: dict,
    team_name: str,
) -> str:
    crest_path = clean_optional_text(
        team_asset.get("crest_path")
    )

    if not crest_path:
        return (
            '<div class="team-crest-placeholder">'
            "⚽"
            "</div>"
        )

    crest_data_uri = (
        image_file_to_data_uri(
            crest_path
        )
    )

    if not crest_data_uri:
        return (
            '<div class="team-crest-placeholder">'
            "⚽"
            "</div>"
        )

    safe_team_name = html.escape(
        str(team_name)
    )

    return (
        '<img class="team-crest" '
        f'src="{crest_data_uri}" '
        f'alt="{safe_team_name} crest">'
    )

def parse_scorelines(value) -> list[dict]:
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []

    return []


def parse_json_list(value) -> list[dict]:
    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, list):
            return [
                item
                for item in parsed
                if isinstance(item, dict)
            ]

    return []


def parse_json_object(value) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, dict):
            return parsed

    return {}


def parse_json_string_list(value) -> list[str]:
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if isinstance(item, str) and item.strip()
            ]

    return []


def render_news_sources(value) -> None:
    articles = parse_json_list(value)

    if not articles:
        st.caption("最近一次更新未取得可展示的公开新闻标题。")
        return

    with st.expander(
        f"查看公开新闻线索（{len(articles)}）",
        expanded=False,
    ):
        for article in articles:
            title = clean_optional_text(
                article.get("title")
            ) or "未命名报道"
            source = clean_optional_text(
                article.get("source")
            ) or "来源未标明"
            url = clean_optional_text(
                article.get("url")
            )
            published = clean_optional_text(
                article.get("published_at_utc")
            )

            safe_title = (
                title.replace("[", "\\[")
                .replace("]", "\\]")
            )
            safe_source = (
                source.replace("[", "\\[")
                .replace("]", "\\]")
            )

            if url.startswith(("https://", "http://")):
                st.markdown(
                    f"- [{safe_title}]({url}) · {safe_source}"
                )
            else:
                st.markdown(
                    f"- {safe_title} · {safe_source}"
                )

            if published:
                st.caption(published)

        st.caption(
            "这里展示的是公开标题与原始链接；"
            "重要结论请打开来源核对正文。"
        )


def squad_number_text(value) -> str:
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return "—"

    try:
        numeric = float(text)
        if numeric.is_integer() and 0 < numeric < 1000:
            return str(int(numeric))
        return "—"
    except ValueError:
        return text


def lineup_table(players: list[dict]) -> pd.DataFrame:
    rows = []

    for position, player in enumerate(players, start=1):
        number_text = squad_number_text(
            player.get("number")
        )

        appearances = pd.to_numeric(
            pd.Series([player.get("appearances")]), errors="coerce"
        ).iloc[0]
        starts = pd.to_numeric(
            pd.Series([player.get("starts")]), errors="coerce"
        ).iloc[0]
        substitutes = pd.to_numeric(
            pd.Series([player.get("substitute_appearances")]), errors="coerce"
        ).iloc[0]
        if pd.isna(appearances):
            appearances = starts
        appearances_text = (
            "—"
            if pd.isna(appearances)
            else f"{int(appearances)} ({0 if pd.isna(substitutes) else int(substitutes)})"
        )
        nationality = clean_optional_text(player.get("nationality")) or "待核验"
        multi_status = clean_optional_text(
            player.get("multiple_nationality_status")
        )
        nationality_text = nationality
        if multi_status and "待核验" not in multi_status:
            nationality_text += f" · {multi_status}"

        goals = pd.to_numeric(
            pd.Series([player.get("goals_scored")]), errors="coerce"
        ).fillna(0).iloc[0]
        assists = pd.to_numeric(
            pd.Series([player.get("assists")]), errors="coerce"
        ).fillna(0).iloc[0]

        rows.append(
            {
                "#": position,
                "名单身份": (
                    clean_optional_text(player.get("squad_role"))
                    or "—"
                ),
                "号码": number_text,
                "球员": (
                    player.get("display_name")
                    or player.get("name")
                    or "未知球员"
                ),
                "位置": player.get("pos") or "—",
                "🌍 国籍": nationality_text,
                "📊 出场": appearances_text,
                "⚽ 进球/助攻": f"{int(goals)} / {int(assists)}",
                "🩺 状态": (
                    clean_optional_text(player.get("availability_status"))
                    or "状态未知"
                ),
            }
        )

    return pd.DataFrame(rows)


def player_name_key(value) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value or ""),
    )
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def player_display_name(player: dict) -> str:
    return clean_optional_text(
        player.get("display_name")
        or player.get("name")
        or player.get("registered_name")
    ) or "未知球员"


def player_position(player: dict) -> str:
    raw = clean_optional_text(
        player.get("pos")
        or player.get("position_code")
        or player.get("position")
    ).upper()
    aliases = {
        "GK": "G",
        "GOALKEEPER": "G",
        "DEF": "D",
        "DEFENDER": "D",
        "MID": "M",
        "MIDFIELDER": "M",
        "FWD": "F",
        "FORWARD": "F",
    }
    return aliases.get(raw, raw[:1] or "—")


def enrich_lineup_players(
    team_name: str,
    players: list[dict],
    registry: pd.DataFrame,
) -> list[dict]:
    if not players or registry.empty:
        return [dict(player) for player in players]

    team_registry = registry.loc[
        registry["team"].astype(str).str.casefold()
        == str(team_name).casefold()
    ].copy()
    if team_registry.empty:
        return [dict(player) for player in players]

    enriched: list[dict] = []
    for player in players:
        output = dict(player)
        matched_row = None

        source_text = clean_optional_text(player.get("source")).casefold()
        player_id = player.get("id")
        if "fpl" in source_text and player_id is not None and "fpl_id" in team_registry:
            numeric_ids = pd.to_numeric(
                team_registry["fpl_id"], errors="coerce"
            )
            try:
                id_matches = team_registry.loc[
                    numeric_ids.eq(float(player_id))
                ]
            except (TypeError, ValueError):
                id_matches = pd.DataFrame()
            if not id_matches.empty:
                matched_row = id_matches.iloc[0]

        if matched_row is None:
            target_key = player_name_key(player_display_name(player))
            best_score = 0
            for _, candidate in team_registry.iterrows():
                candidate_keys = {
                    player_name_key(candidate.get("display_name")),
                    player_name_key(candidate.get("registered_name")),
                    player_name_key(candidate.get("web_name")),
                    player_name_key(candidate.get("fpl_full_name")),
                }
                candidate_keys.discard("")
                score = 0
                for candidate_key in candidate_keys:
                    if target_key == candidate_key:
                        score = max(score, 100)
                    elif (
                        len(target_key) >= 4
                        and len(candidate_key) >= 4
                        and (
                            target_key.endswith(candidate_key)
                            or candidate_key.endswith(target_key)
                        )
                    ):
                        score = max(score, 85)
                    elif (
                        target_key.split()
                        and candidate_key.split()
                        and target_key.split()[-1] == candidate_key.split()[-1]
                    ):
                        score = max(score, 70)
                if score > best_score:
                    best_score = score
                    matched_row = candidate

            if best_score < 70:
                matched_row = None

        if matched_row is not None:
            if squad_number_text(output.get("number")) == "—":
                registry_number = squad_number_text(
                    matched_row.get("squad_number")
                )
                if registry_number != "—":
                    output["number"] = registry_number
            if player_position(output) == "—":
                output["pos"] = matched_row.get("position_code")
            for column in [
                "nationality",
                "citizenships",
                "multiple_nationality_status",
                "date_of_birth",
                "appearances",
                "starts",
                "substitute_appearances",
                "goals_scored",
                "assists",
                "minutes",
                "expected_goals",
                "expected_assists",
                "availability_status",
                "news",
                "position_detail",
                "registration_category",
            ]:
                value = matched_row.get(column)
                if value is not None and not (
                    isinstance(value, float) and pd.isna(value)
                ):
                    output[column] = value

        enriched.append(output)

    return enriched


def formation_rows(formation_text: str) -> list[int]:
    values = [
        int(value)
        for value in re.findall(
            r"\d+",
            clean_optional_text(formation_text),
        )
    ]
    if values and sum(values) == 10:
        return [1, *values]
    return [1, 4, 4, 2]


def tactical_positions(
    players: list[dict],
    formation_text: str,
) -> list[tuple[dict, float, float]]:
    grid_players = []
    for player in players:
        grid = clean_optional_text(player.get("grid"))
        match = re.fullmatch(r"(\d+):(\d+)", grid)
        if match:
            grid_players.append(
                (player, int(match.group(1)), int(match.group(2)))
            )

    if len(grid_players) == len(players) and players:
        maximum_row = max(row for _, row, _ in grid_players)
        row_counts = {
            row: max(column for _, item_row, column in grid_players if item_row == row)
            for row in {item_row for _, item_row, _ in grid_players}
        }
        positioned = []
        for player, row, column in grid_players:
            count = row_counts[row]
            x_value = 100.0 * column / (count + 1)
            y_value = (
                88.0
                if maximum_row == 1
                else 88.0 - ((row - 1) * 72.0 / (maximum_row - 1))
            )
            positioned.append((player, x_value, y_value))
        return positioned

    position_order = {"G": 0, "D": 1, "M": 2, "F": 3, "—": 4}
    ordered = sorted(
        enumerate(players),
        key=lambda item: (
            position_order.get(player_position(item[1]), 4),
            item[0],
        ),
    )
    ordered_players = [player for _, player in ordered]
    rows = formation_rows(formation_text)
    positioned = []
    cursor = 0
    row_total = len(rows)

    for row_index, count in enumerate(rows):
        row_players = ordered_players[cursor : cursor + count]
        cursor += count
        y_value = (
            88.0
            if row_total == 1
            else 88.0 - (row_index * 72.0 / (row_total - 1))
        )
        for column_index, player in enumerate(row_players, start=1):
            x_value = 100.0 * column_index / (len(row_players) + 1)
            positioned.append((player, x_value, y_value))

    return positioned


def probable_bench_from_registry(
    team_name: str,
    starters: list[dict],
    registry: pd.DataFrame,
) -> list[dict]:
    if registry.empty:
        return []

    candidates = registry.loc[
        registry["team"].astype(str).str.casefold()
        == str(team_name).casefold()
    ].copy()
    if candidates.empty:
        return []

    starter_keys = {
        player_name_key(player_display_name(player))
        for player in starters
    }

    def is_starter(row: pd.Series) -> bool:
        names = {
            player_name_key(row.get("display_name")),
            player_name_key(row.get("registered_name")),
            player_name_key(row.get("web_name")),
        }
        names.discard("")
        for name in names:
            if name in starter_keys:
                return True
            if any(
                name.endswith(key) or key.endswith(name)
                for key in starter_keys
                if len(name) >= 4 and len(key) >= 4
            ):
                return True
        return False

    candidates = candidates.loc[
        ~candidates.apply(is_starter, axis=1)
    ].copy()

    for column in [
        "eligible_2026_27",
        "chance_of_playing_next_round",
        "starts",
        "minutes",
        "total_points",
    ]:
        if column not in candidates.columns:
            candidates[column] = 0

    candidates["eligible_2026_27"] = (
        candidates["eligible_2026_27"].fillna(False).astype(bool)
    )
    for column in [
        "chance_of_playing_next_round",
        "starts",
        "minutes",
        "total_points",
    ]:
        candidates[column] = pd.to_numeric(
            candidates[column], errors="coerce"
        ).fillna(0)

    candidates = candidates.sort_values(
        [
            "eligible_2026_27",
            "chance_of_playing_next_round",
            "starts",
            "minutes",
            "total_points",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
    )

    selected_indices: list[int] = []
    goalkeeper_rows = candidates.loc[
        candidates.get("position_code", pd.Series(index=candidates.index, dtype=str))
        .astype(str)
        .eq("G")
    ]
    if not goalkeeper_rows.empty:
        selected_indices.append(int(goalkeeper_rows.index[0]))

    for index in candidates.index:
        integer_index = int(index)
        if integer_index not in selected_indices:
            selected_indices.append(integer_index)
        if len(selected_indices) >= 9:
            break

    output = []
    for _, row in candidates.loc[selected_indices].iterrows():
        output.append(
            {
                "display_name": (
                    clean_optional_text(row.get("display_name"))
                    or clean_optional_text(row.get("registered_name"))
                ),
                "number": row.get("squad_number"),
                "pos": row.get("position_code"),
                "status": row.get("availability_status"),
                "source": "registered player pool estimate",
            }
        )
    return output


def tactical_board_html(
    players: list[dict],
    formation_text: str,
) -> str:
    nodes = []
    for player, x_value, y_value in tactical_positions(
        players, formation_text
    ):
        full_name = player_display_name(player)
        safe_name = html.escape(full_name)
        short_name = full_name.split()[-1] if full_name.split() else full_name
        number = squad_number_text(player.get("number"))
        marker = number if number != "—" else player_position(player)
        tooltip = html.escape(
            f"{full_name} · {player_position(player)}"
        )
        nodes.append(
            f'<div class="pitch-player" title="{tooltip}" '
            f'style="left:{x_value:.2f}%;top:{y_value:.2f}%">'
            f'<div class="pitch-player-marker">{html.escape(marker)}</div>'
            f'<div class="pitch-player-name">{html.escape(short_name)}</div>'
            "</div>"
        )
    return (
        '<div class="tactical-pitch">'
        '<div class="pitch-halfway"></div>'
        '<div class="pitch-centre-circle"></div>'
        '<div class="pitch-centre-dot"></div>'
        '<div class="pitch-box top"></div>'
        '<div class="pitch-box bottom"></div>'
        '<div class="pitch-goal-box top"></div>'
        '<div class="pitch-goal-box bottom"></div>'
        + "".join(nodes)
        + "</div>"
    )


def bench_html(players: list[dict], predicted: bool) -> str:
    title = "预计替补候选" if predicted else "官方替补席"
    if not players:
        rows = (
            '<div class="bench-empty">暂无替补名单。官方阵容发布后将自动更新。</div>'
        )
    else:
        entries = []
        for player in players:
            number = squad_number_text(player.get("number"))
            entries.append(
                '<div class="bench-player">'
                f'<div class="bench-number">{html.escape(number)}</div>'
                f'<div class="bench-name" title="{html.escape(player_display_name(player))}">'
                f'{html.escape(player_display_name(player))}</div>'
                f'<div class="bench-pos">{html.escape(player_position(player))}</div>'
                "</div>"
            )
        rows = "".join(entries)
    return (
        '<aside class="bench-panel">'
        f'<div class="bench-title">{title}</div>'
        f"{rows}</aside>"
    )


def render_team_lineup(
    team_name: str,
    confirmed: bool,
    official_players: list[dict],
    predicted_players: list[dict],
    substitutes: list[dict],
    formation,
    predicted_formation,
    coach,
    changes: dict,
    player_registry: pd.DataFrame,
) -> None:
    if confirmed and len(official_players) == 11:
        players = official_players
        formation_text = clean_optional_text(formation) or "未注明"
        status_text = "官方首发"
        predicted = False
    elif predicted_players:
        players = predicted_players
        formation_text = (
            clean_optional_text(predicted_formation)
            or "预计阵型未确定"
        )
        status_text = "预计阵容"
        predicted = True
    else:
        players = []
        formation_text = "—"
        status_text = "等待名单"
        predicted = True

    players = enrich_lineup_players(
        team_name,
        players,
        player_registry,
    )

    if predicted:
        bench_players = probable_bench_from_registry(
            team_name,
            players,
            player_registry,
        )
        bench_players = enrich_lineup_players(
            team_name,
            bench_players,
            player_registry,
        )
    else:
        bench_players = enrich_lineup_players(
            team_name,
            substitutes,
            player_registry,
        )

    # The details table represents the full matchday squad, not only the XI.
    # During the predicted stage, bench entries remain clearly marked as
    # candidates; after the official release they become official substitutes.
    starter_role = "预计首发" if predicted else "官方首发"
    bench_role = "预计替补候选" if predicted else "官方替补"
    matchday_players = [
        {**dict(player), "squad_role": starter_role}
        for player in players
    ]
    matchday_players.extend(
        {**dict(player), "squad_role": bench_role}
        for player in bench_players
    )

    coach_text = clean_optional_text(coach)
    source_text = (
        "API-Football 官方首发"
        if not predicted
        else "FPL 使用数据与注册球员池推定"
    )
    meta_parts = [f"阵型 {formation_text}", source_text]
    if coach_text and not predicted:
        meta_parts.append(f"主教练 {coach_text}")

    safe_team = html.escape(team_name)
    pill_class = "predicted" if predicted else ""
    board = (
        '<section class="lineup-team-shell">'
        '<div class="lineup-team-heading">'
        f"<h3>{safe_team}</h3>"
        f'<span class="lineup-status-pill {pill_class}">{status_text}</span>'
        "</div>"
        f'<div class="lineup-meta">{html.escape(" · ".join(meta_parts))}</div>'
        '<div class="tactical-layout">'
        + tactical_board_html(players, formation_text)
        + bench_html(bench_players, predicted)
        + "</div></section>"
    )
    st.markdown(board, unsafe_allow_html=True)

    if matchday_players:
        with st.expander(
            "📋 查看完整比赛名单、国籍与赛季表现明细",
            expanded=False,
        ):
            st.caption(
                (
                    "当前为预计比赛名单：替补部分是注册球员池中的高使用率候选，"
                    "不等同于官方比赛日名单。"
                    if predicted
                    else "当前为官方比赛名单：包含首发与官方替补。"
                )
            )
            st.dataframe(
                lineup_table(matchday_players),
                hide_index=True,
                use_container_width=True,
                height=min(720, 112 + 36 * len(matchday_players)),
            )

    if predicted and not player_registry.empty:
        st.caption(
            "预计替补仅表示当前注册球员池中的高使用率候选，"
            "不等同于比赛日官方替补名单。"
        )

    if confirmed and changes:
        predicted_not_starting = changes.get(
            "predicted_not_starting",
            [],
        )
        unexpected_starters = changes.get(
            "unexpected_starters",
            [],
        )

        if predicted_not_starting:
            st.caption(
                "预计首发中未进入官方首发："
                + "、".join(predicted_not_starting)
            )

        if unexpected_starters:
            st.caption(
                "相对预计阵容的新首发："
                + "、".join(unexpected_starters)
            )


def format_fixture_label(row: pd.Series) -> str:
    kickoff = row["kickoff_utc"]

    if pd.notna(kickoff):
        kickoff_text = kickoff.strftime(
            "%m-%d %H:%M UTC"
        )
    else:
        kickoff_text = "时间待定"

    return (
        f"{row['home_team']} vs "
        f"{row['away_team']} · {kickoff_text}"
    )


def render_probability_bar(
    label: str,
    probability: float,
    color: str,
) -> None:
    safe_label = html.escape(label)
    percentage = probability * 100

    st.markdown(
        f"""
        <div class="probability-bar-label">
            <span>{safe_label}</span>
            <strong>{percentage:.1f}%</strong>
        </div>
        <div class="probability-track">
            <div
                class="probability-fill"
                style="
                    width: {percentage:.2f}%;
                    background: {color};
                "
            ></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_description(row: pd.Series) -> str:
    probabilities = sorted(
        [
            float(row["home_probability"]),
            float(row["draw_probability"]),
            float(row["away_probability"]),
        ],
        reverse=True,
    )

    margin = probabilities[0] - probabilities[1]

    if margin >= 0.20:
        return "倾向较明确"

    if margin >= 0.08:
        return "中等倾向"

    return "非常接近"


def dixon_coles_outcome(row: pd.Series) -> str:
    probabilities = {
        "主胜": float(row["dc_home_probability"]),
        "平局": float(row["dc_draw_probability"]),
        "客胜": float(row["dc_away_probability"]),
    }

    return max(
        probabilities,
        key=probabilities.get,
    )


try:
    predictions = load_predictions()
    team_assets = load_team_assets()
    stadium_assets = load_stadium_assets()
    player_registry = load_player_registry()
except Exception as error:
    st.error(f"页面数据读取失败：{error}")
    st.stop()


stadium_assets_by_team = {}

if not stadium_assets.empty:
    for _, stadium_asset_row in (
        stadium_assets.iterrows()
    ):
        stadium_team_id = (
            clean_optional_text(
                stadium_asset_row.get(
                    "home_team_id"
                )
            )
        )

        if stadium_team_id:
            stadium_assets_by_team[
                stadium_team_id
            ] = stadium_asset_row.to_dict()


st.markdown(
    """
    <div class="main-header">
        <h1>EPL Match Intelligence</h1>
        <p>
            时间序列回测 · XGBoost胜平负概率 ·
            Dixon–Coles比分分布
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## ⚽ EPL Predictor")

    st.caption(
        "选择一场比赛查看完整预测"
    )

    matchweek_values = sorted(
        predictions["matchweek"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    round_options = {
        "全部比赛": None,
        **{
            f"第 {matchweek} 轮": matchweek
            for matchweek in matchweek_values
        },
    }

    selected_round_label = st.selectbox(
        "轮次",
        options=list(round_options.keys()),
    )

    selected_round = round_options[
        selected_round_label
    ]

    if selected_round is None:
        filtered_predictions = predictions
    else:
        filtered_predictions = predictions.loc[
            predictions["matchweek"]
            == selected_round
        ]

    match_options = (
        filtered_predictions.index.tolist()
    )

    selected_index = st.selectbox(
        "比赛",
        options=match_options,
        format_func=lambda index: (
            format_fixture_label(
                predictions.loc[index]
            )
        ),
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 当前模型")

    st.markdown(
        """
        <span class="status-chip">XGBoost</span>
        <span class="status-chip">Dixon–Coles</span>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "胜平负主概率来自XGBoost；"
        "预期进球和比分来自Dixon–Coles。"
    )

    st.markdown("### 验证状态")

    st.metric(
        "最终留出测试",
        "760 场",
        "优于联赛平均",
    )

    st.metric(
        "外部验证",
        "95 场",
        "Log Loss改善4.92%",
    )

    st.markdown(
        """
        <div class="warning-box">
            伤停、天气、裁判、新闻与阵容用于解释；
            未经回测前不会修改冻结模型概率。
        </div>
        """,
        unsafe_allow_html=True,
    )


row = predictions.loc[selected_index]

home_asset = find_team_asset(
    assets=team_assets,
    team_id=row.get(
        "home_team_id"
    ),
    team_name=row.get(
        "home_team"
    ),
)

away_asset = find_team_asset(
    assets=team_assets,
    team_id=row.get(
        "away_team_id"
    ),
    team_name=row.get(
        "away_team"
    ),
)

home_crest_html = build_crest_html(
    team_asset=home_asset,
    team_name=str(row["home_team"]),
)

away_crest_html = build_crest_html(
    team_asset=away_asset,
    team_name=str(row["away_team"]),
)

home_stadium_asset = (
    stadium_assets_by_team.get(
        clean_optional_text(
            row.get("home_team_id")
        ),
        {},
    )
)

stadium_image_path = (
    clean_optional_text(
        home_stadium_asset.get(
            "stadium_image_path"
        )
    )
)

stadium_image_data_uri = (
    image_file_to_data_uri(
        stadium_image_path
    )
)

fixture_card_style = ""

if stadium_image_data_uri:
    fixture_card_style = (
        "background-image:"
        "linear-gradient("
        "100deg,"
        "rgba(5,9,18,0.96),"
        "rgba(7,13,24,0.68),"
        "rgba(5,9,18,0.94)"
        "),"
        f"url('{stadium_image_data_uri}');"
        "background-size:cover;"
        "background-position:center;"
    )

fixture_card_style = html.escape(
    fixture_card_style,
    quote=True,
)

stadium_artist = html.escape(
    clean_optional_text(
        home_stadium_asset.get("artist")
    )
    or "作者未注明"
)

stadium_license = html.escape(
    clean_optional_text(
        home_stadium_asset.get("license")
    )
    or "许可未注明"
)

stadium_source_page = html.escape(
    clean_optional_text(
        home_stadium_asset.get(
            "commons_file_page"
        )
    ),
    quote=True,
)

stadium_credit_html = ""

if stadium_image_data_uri:
    if stadium_source_page:
        stadium_credit_html = (
            '<div class="fixture-photo-credit">'
            "球场图片："
            f'<a href="{stadium_source_page}" '
            'target="_blank" '
            'rel="noopener noreferrer">'
            f"{stadium_artist}"
            "</a>"
            f" · {stadium_license}"
            "</div>"
        )
    else:
        stadium_credit_html = (
            '<div class="fixture-photo-credit">'
            f"球场图片：{stadium_artist}"
            f" · {stadium_license}"
            "</div>"
        )

venue_name = clean_optional_text(
    row.get("venue_name")
)

if not venue_name:
    venue_name = clean_optional_text(
        home_asset.get("venue_name")
    )

if not venue_name:
    venue_name = "比赛场地待确认"

safe_venue_name = html.escape(
    venue_name
)

home_team = html.escape(str(row["home_team"]))
away_team = html.escape(str(row["away_team"]))

kickoff = row["kickoff_utc"]

if pd.notna(kickoff):
    london_kickoff = kickoff.tz_convert(
        "Europe/London"
    )

    kickoff_text = london_kickoff.strftime(
        "%Y-%m-%d %H:%M 英国时间"
    )
else:
    kickoff_text = "开球时间待定"

matchweek = row.get("matchweek")

if pd.notna(matchweek):
    matchweek_text = f"英超第 {int(matchweek)} 轮"
else:
    matchweek_text = "英超联赛"


st.html(
    f"""
    <div
        class="fixture-card"
        style="{fixture_card_style}"
    >
        <div class="fixture-meta">
            {html.escape(matchweek_text)}
            &nbsp;·&nbsp;
            {html.escape(kickoff_text)}
        </div>

        <div class="fixture-teams">
    <div class="team-identity home-identity">
        <div class="team-home">
            {home_team}
        </div>
        {home_crest_html}
    </div>

    <div class="versus">VS</div>

    <div class="team-identity away-identity">
        {away_crest_html}
        <div class="team-away">
            {away_team}
        </div>
    </div>
</div>

<div class="fixture-venue">
    🏟 {safe_venue_name}
</div>

{stadium_credit_html}
    </div>
    """
)


context_completed = int(
    row.get(
        "context_sections_completed",
        0,
    )
)

context_total = int(
    row.get(
        "context_sections_total",
        7,
    )
)

context_confidence = float(
    row.get(
        "context_confidence",
        0.0,
    )
)

context_status = str(
    row.get(
        "context_status",
        "信息有限",
    )
)

models_agree = bool(
    row.get(
        "models_agree",
        False,
    )
)

probability_separation = str(
    row.get(
        "probability_separation",
        "尚未计算",
    )
)


render_match_kits(
    str(row["home_team"]),
    str(row["away_team"]),
)

st.markdown("## 赛前情报状态")

status_columns = st.columns(4)

with status_columns[0]:
    st.metric(
        "上下文完整度",
        f"{context_completed}/{context_total}",
        f"{context_confidence:.0%}",
    )

with status_columns[1]:
    st.metric(
        "信息状态",
        context_status,
    )

with status_columns[2]:
    st.metric(
        "模型关系",
        (
            "方向一致"
            if models_agree
            else "存在分歧"
        ),
    )

with status_columns[3]:
    st.metric(
        "概率区分度",
        probability_separation,
    )


context_summary = str(
    row.get(
        "context_summary",
        "暂无上下文摘要。",
    )
)

if models_agree:
    st.info(context_summary)
else:
    st.warning(context_summary)


(
    summary_tab,
    lineup_tab,
    availability_tab,
    weather_tab,
    importance_tab,
) = st.tabs(
    [
        "🧠 综合提示",
        "🟩 战术板与首发",
        "🩺 伤停与球队消息",
        "🌦️ 天气与场地",
        "🏁 重要性与裁判",
    ]
)


with summary_tab:
    st.markdown("### 当前赛前判断")

    st.write(context_summary)

    llm_status = clean_optional_text(
        row.get("llm_status")
    )
    llm_brief = clean_optional_text(
        row.get("llm_prematch_brief")
    )
    llm_points = parse_json_string_list(
        row.get("llm_key_points_json")
    )

    if (
        llm_status in {"success", "cache_hit"}
        and llm_brief
    ):
        st.markdown("### 🤖 DeepSeek 赛前情报解读")
        st.write(llm_brief)

        for point in llm_points:
            st.markdown(f"- {point}")

        st.caption(
            "AI只整理页面列出的结构化数据和公开新闻标题；"
            "不会修改冻结模型概率。请通过来源链接核对重要事实。"
        )
    elif llm_status == "failed_fallback":
        st.caption(
            "DeepSeek本次请求失败，页面已自动使用确定性摘要，"
            "预测与其他功能不受影响。"
        )
    elif llm_status in {
        "missing_api_key",
        "disabled_by_config",
        "missing_model",
        "not_run",
    }:
        st.caption(
            "DeepSeek情报解读尚未启用；"
            "当前显示原有的结构化赛前信息。"
        )

    adjustment_note = str(
        row.get(
            "context_probability_adjustment_note",
            (
                "上下文仅用于解释，"
                "当前不修改模型概率。"
            ),
        )
    )

    st.caption(adjustment_note)

    lineups_confirmed = (
        str(
            row.get(
                "lineups_confirmed",
                False,
            )
        ).lower()
        == "true"
    )

    if lineups_confirmed:
        st.success("本场首发阵容已经确认。")
    else:
        st.warning(
            "本场首发阵容尚未确认；"
            "临近开赛时系统将每15分钟自动检查。"
        )


with lineup_tab:
    lineup_status = str(
        row.get("lineup_status", "waiting")
    )

    lineup_checked_at = pd.to_datetime(
        row.get("lineup_checked_at_utc"),
        utc=True,
        errors="coerce",
    )

    lineup_explanation = clean_optional_text(
        row.get("lineup_explanation")
    )

    if lineups_confirmed:
        st.success(
            "官方首发已确认。名单仅用于解释，"
            "不会修改页面中的冻结胜率。"
        )
    elif lineup_status == "partial":
        st.warning(
            "目前只获得部分官方名单，"
            "页面暂时继续展示预计阵容。"
        )
    else:
        st.info(
            "官方首发尚未发布。当前预计阵容来自"
            "FPL赛季出场与使用数据，不等同于官方名单。"
        )

    if lineup_explanation:
        st.caption(lineup_explanation)

    if pd.notna(lineup_checked_at):
        checked_london = lineup_checked_at.tz_convert(
            "Europe/London"
        )
        st.caption(
            "最近检查："
            + checked_london.strftime(
                "%Y-%m-%d %H:%M 英国时间"
            )
            + " · 来源：API-Football"
        )

    lineup_columns = st.columns(2)

    with lineup_columns[0]:
        render_team_lineup(
            team_name=str(row["home_team"]),
            confirmed=lineups_confirmed,
            official_players=parse_json_list(
                row.get("home_starting_xi_json")
            ),
            predicted_players=parse_json_list(
                row.get("home_predicted_xi_json")
            ),
            substitutes=parse_json_list(
                row.get("home_substitutes_json")
            ),
            formation=row.get("home_formation"),
            predicted_formation=row.get(
                "home_usual_formation"
            ),
            coach=row.get("home_coach"),
            changes=parse_json_object(
                row.get("home_lineup_changes_json")
            ),
            player_registry=player_registry,
        )

    with lineup_columns[1]:
        render_team_lineup(
            team_name=str(row["away_team"]),
            confirmed=lineups_confirmed,
            official_players=parse_json_list(
                row.get("away_starting_xi_json")
            ),
            predicted_players=parse_json_list(
                row.get("away_predicted_xi_json")
            ),
            substitutes=parse_json_list(
                row.get("away_substitutes_json")
            ),
            formation=row.get("away_formation"),
            predicted_formation=row.get(
                "away_usual_formation"
            ),
            coach=row.get("away_coach"),
            changes=parse_json_object(
                row.get("away_lineup_changes_json")
            ),
            player_registry=player_registry,
        )


with availability_tab:
    injury_columns = st.columns(2)

    home_injury_impact = row.get(
        "home_injury_impact"
    )

    away_injury_impact = row.get(
        "away_injury_impact"
    )

    home_news = str(
        row.get(
            "home_news_summary",
            "暂无结构化球队消息。",
        )
    )

    away_news = str(
        row.get(
            "away_news_summary",
            "暂无结构化球队消息。",
        )
    )

    with injury_columns[0]:
        st.markdown(f"### {home_team}")

        if pd.notna(home_injury_impact):
            st.metric(
                "伤停影响指数",
                f"{float(home_injury_impact):.3f}",
            )
        else:
            st.metric(
                "伤停影响指数",
                "未提供",
            )

        st.write(home_news)

        render_news_sources(
            row.get("home_news_items_json")
        )

    with injury_columns[1]:
        st.markdown(f"### {away_team}")

        if pd.notna(away_injury_impact):
            st.metric(
                "伤停影响指数",
                f"{float(away_injury_impact):.3f}",
            )
        else:
            st.metric(
                "伤停影响指数",
                "未提供",
            )

        st.write(away_news)

        render_news_sources(
            row.get("away_news_items_json")
        )

    st.caption(
        "伤停影响指数不是胜率，"
        "仅表示当前阵容可用性风险。"
    )


with weather_tab:
    weather_available = bool(
        row.get(
            "weather_available",
            False,
        )
    )

    if weather_available:
        weather_columns = st.columns(3)

        temperature = row.get(
            "temperature_c"
        )

        precipitation = row.get(
            "precipitation_mm"
        )

        wind_speed = row.get(
            "wind_speed_kph"
        )

        with weather_columns[0]:
            st.metric(
                "气温",
                (
                    f"{float(temperature):.1f} °C"
                    if pd.notna(temperature)
                    else "未提供"
                ),
            )

        with weather_columns[1]:
            st.metric(
                "降水量",
                (
                    f"{float(precipitation):.1f} mm"
                    if pd.notna(precipitation)
                    else "未提供"
                ),
            )

        with weather_columns[2]:
            st.metric(
                "风速",
                (
                    f"{float(wind_speed):.1f} km/h"
                    if pd.notna(wind_speed)
                    else "未提供"
                ),
            )

        venue_name = row.get(
            "venue_name"
        )

        weather_description = row.get(
            "weather_description"
        )

        st.write(
            f"**比赛场地：** "
            f"{venue_name or '未提供'}"
        )

        st.write(
            f"**天气描述：** "
            f"{weather_description or '未提供'}"
        )

        st.caption(
            f"天气来源："
            f"{row.get('weather_source') or '未提供'}"
        )
    else:
        st.warning(
            "当前尚未取得该场比赛的完整逐小时天气。"
        )


with importance_tab:
    importance_columns = st.columns(2)

    home_importance = row.get(
        "home_importance"
    )

    away_importance = row.get(
        "away_importance"
    )

    with importance_columns[0]:
        st.metric(
            f"{home_team} 重要性",
            (
                f"{float(home_importance):.3f}"
                if pd.notna(home_importance)
                else "未提供"
            ),
        )

    with importance_columns[1]:
        st.metric(
            f"{away_team} 重要性",
            (
                f"{float(away_importance):.3f}"
                if pd.notna(away_importance)
                else "未提供"
            ),
        )

    st.write(
        row.get(
            "importance_reason",
            "暂无比赛重要性说明。",
        )
    )

    referee_name = row.get(
        "referee_name"
    )

    if (
        pd.notna(referee_name)
        and str(referee_name).strip()
    ):
        st.success(
            f"已公布裁判：{referee_name}"
        )

        st.caption(
            f"裁判来源："
            f"{row.get('referee_source')}"
        )
    else:
        st.warning("本场裁判安排尚未公布。")

    st.caption(
        "重要性指数范围为0至1，"
        "它不是比赛结果概率。"
    )

st.markdown("## 胜平负概率")

probabilities = {
    "主胜": float(row["home_probability"]),
    "平局": float(row["draw_probability"]),
    "客胜": float(row["away_probability"]),
}

predicted_outcome = str(
    row["predicted_outcome"]
)

columns = st.columns(3)

for column, (label, probability) in zip(
    columns,
    probabilities.items(),
):
    winner_class = (
        " probability-winner"
        if label == predicted_outcome
        else ""
    )

    with column:
        st.markdown(
            f"""
            <div class="
                probability-card{winner_class}
            ">
                <div class="probability-label">
                    {html.escape(label)}
                </div>
                <div class="probability-number">
                    {probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


render_probability_bar(
    "主胜",
    float(row["home_probability"]),
    "#00ff87",
)

render_probability_bar(
    "平局",
    float(row["draw_probability"]),
    "#9a78ff",
)

render_probability_bar(
    "客胜",
    float(row["away_probability"]),
    "#39a8ff",
)


st.caption(
    f"模型结论：{predicted_outcome} · "
    f"{confidence_description(row)}"
)


left_column, right_column = st.columns(
    [1, 1.45],
    gap="large",
)


with left_column:
    st.markdown("## 比分模型")

    most_likely_score = html.escape(
        str(row["most_likely_score"])
    )

    st.html(
        f"""
        <div class="model-panel">
            <div class="model-panel-title">
                Dixon–Coles预测
            </div>

            <div class="small-muted">
                最可能比分
            </div>

            <div class="score-display">
                {most_likely_score}
            </div>

            <div class="small-muted">
                预期进球：
                {float(row['expected_home_goals']):.2f}
                &nbsp;–&nbsp;
                {float(row['expected_away_goals']):.2f}
            </div>
        </div>
        """
    )

    st.markdown("### 两个模型是否一致")

    dc_outcome = dixon_coles_outcome(row)

    if dc_outcome == predicted_outcome:
        st.success(
            f"一致：两个模型均倾向{predicted_outcome}"
        )
    else:
        st.warning(
            f"存在分歧：XGBoost倾向"
            f"{predicted_outcome}，"
            f"Dixon–Coles倾向{dc_outcome}"
        )


with right_column:
    st.markdown("## 最可能比分")

    scorelines = parse_scorelines(
        row["top_scorelines"]
    )

    if scorelines:
        visible_scorelines = scorelines[:5]

        scoreline_columns = st.columns(
            len(visible_scorelines)
        )

        for position, (
            column,
            item,
        ) in enumerate(
            zip(
                scoreline_columns,
                visible_scorelines,
            ),
            start=1,
        ):
            score = html.escape(
                str(
                    item.get(
                        "score",
                        "未知",
                    )
                )
            )

            probability = float(
                item.get(
                    "probability",
                    0.0,
                )
            )

            primary_class = (
                " scoreline-primary"
                if position == 1
                else ""
            )

            with column:
                st.html(
                    f"""
                    <div class="
                        scoreline-card{primary_class}
                    ">
                        <div class="scoreline-rank">
                            第 {position} 位
                        </div>

                        <div class="scoreline-score">
                            {score}
                        </div>

                        <div class="
                            scoreline-probability
                        ">
                            {probability:.1%}
                        </div>
                    </div>
                    """
                )
    else:
        st.info("没有读取到比分候选。")

    st.markdown("### Dixon–Coles胜平负参考")

    dc_columns = st.columns(3)

    dc_values = [
        (
            "主胜",
            float(
                row["dc_home_probability"]
            ),
        ),
        (
            "平局",
            float(
                row["dc_draw_probability"]
            ),
        ),
        (
            "客胜",
            float(
                row["dc_away_probability"]
            ),
        ),
    ]

    for column, (label, value) in zip(
        dc_columns,
        dc_values,
    ):
        column.metric(label, f"{value:.1%}")


overview_heading = (
    "全部比赛预测概览"
    if selected_round is None
    else f"英超第 {selected_round} 轮预测概览"
)

st.markdown(f"## {overview_heading}")

overview_data = (
    filtered_predictions
    .sort_values(
        [
            "kickoff_utc",
            "match_id",
        ],
        kind="stable",
    )
)

overview_column_count = (
    1
    if len(overview_data) == 1
    else 2
)

overview_columns = st.columns(
    overview_column_count,
    gap="medium",
)

for card_position, (
    _,
    fixture,
) in enumerate(
    overview_data.iterrows()
):
    column = overview_columns[
        card_position
        % overview_column_count
    ]

    fixture_home = html.escape(
        str(fixture["home_team"])
    )

    fixture_away = html.escape(
        str(fixture["away_team"])
    )

    fixture_home_asset = find_team_asset(
        assets=team_assets,
        team_id=fixture.get(
            "home_team_id"
        ),
        team_name=fixture.get(
            "home_team"
        ),
    )

    fixture_away_asset = find_team_asset(
        assets=team_assets,
        team_id=fixture.get(
            "away_team_id"
        ),
        team_name=fixture.get(
            "away_team"
        ),
    )

    fixture_home_crest_html = (
        build_crest_html(
            team_asset=fixture_home_asset,
            team_name=str(
                fixture["home_team"]
            ),
        )
    )

    fixture_away_crest_html = (
        build_crest_html(
            team_asset=fixture_away_asset,
            team_name=str(
                fixture["away_team"]
            ),
        )
    )

fixture_venue_name = (
    clean_optional_text(
        fixture.get("venue_name")
    )
)

if not fixture_venue_name:
    fixture_venue_name = (
        clean_optional_text(
            fixture_home_asset.get(
                "venue_name"
            )
        )
    )

    if not fixture_venue_name:
        fixture_venue_name = (
            "比赛场地待确认"
        )

    fixture_venue_name = html.escape(
        fixture_venue_name
    )

    fixture_home_colors = (
        clean_optional_text(
            fixture_home_asset.get(
                "club_colors"
            )
        )
    )

    fixture_away_colors = (
        clean_optional_text(
            fixture_away_asset.get(
                "club_colors"
            )
        )
    )

    fixture_home_colors = html.escape(
        fixture_home_colors
        or "暂无配色资料"
    )

    fixture_away_colors = html.escape(
        fixture_away_colors
        or "暂无配色资料"
    )

    fixture_outcome = str(
        fixture["predicted_outcome"]
    )

    outcome_class = {
        "主胜": "outcome-home",
        "平局": "outcome-draw",
        "客胜": "outcome-away",
    }.get(
        fixture_outcome,
        "",
    )

    fixture_kickoff = fixture.get(
        "kickoff_utc"
    )

    if pd.notna(fixture_kickoff):
        fixture_london_time = (
            fixture_kickoff.tz_convert(
                "Europe/London"
            )
        )

        fixture_time_text = (
            fixture_london_time.strftime(
                "%Y-%m-%d %H:%M 英国时间"
            )
        )
    else:
        fixture_time_text = (
            "开球时间待定"
        )

    agreement_value = fixture.get(
        "models_agree",
        False,
    )

    fixture_models_agree = (
        False
        if pd.isna(agreement_value)
        else bool(agreement_value)
    )

    agreement_text = (
        "模型一致"
        if fixture_models_agree
        else "模型分歧"
    )

    agreement_class = (
        "agreement-good"
        if fixture_models_agree
        else "agreement-warning"
    )

    fixture_context_status = html.escape(
        str(
            fixture.get(
                "context_status",
                "信息有限",
            )
        )
    )

    fixture_context_confidence = float(
        fixture.get(
            "context_confidence",
            0.0,
        )
    )

    fixture_context_confidence_text = (
        f"{fixture_context_confidence:.0%}"
    )

    fixture_score = html.escape(
        str(
            fixture[
                "most_likely_score"
            ]
        )
    )

    home_probability_text = (
        f"{float(fixture['home_probability']):.1%}"
    )

    draw_probability_text = (
        f"{float(fixture['draw_probability']):.1%}"
    )

    away_probability_text = (
        f"{float(fixture['away_probability']):.1%}"
    )

    with column:
        st.html(
            f"""
            <div class="overview-fixture-card">
                <div class="
                    overview-fixture-meta
                ">
                    {html.escape(fixture_time_text)}
                </div>

                <div class="
    overview-fixture-teams
">
    <div class="
        overview-team-side
        overview-team-home
    "
        title="俱乐部配色：{fixture_home_colors}"
    >
        <span class="
            overview-team-name
        ">
            {fixture_home}
        </span>
        {fixture_home_crest_html}
    </div>

    <span class="
        overview-versus
    ">
        VS
    </span>

    <div class="
        overview-team-side
        overview-team-away
    "
        title="俱乐部配色：{fixture_away_colors}"
    >
        {fixture_away_crest_html}
        <span class="
            overview-team-name
        ">
            {fixture_away}
        </span>
    </div>
</div>

                <div class="
                    overview-probabilities
                ">
                    <div class="
                        overview-probability
                    ">
                        <div class="
                            overview-probability-label
                        ">
                            主胜
                        </div>
                        <div class="
                            overview-probability-value
                        ">
                            {home_probability_text}
                        </div>
                    </div>

                    <div class="
                        overview-probability
                    ">
                        <div class="
                            overview-probability-label
                        ">
                            平局
                        </div>
                        <div class="
                            overview-probability-value
                        ">
                            {draw_probability_text}
                        </div>
                    </div>

                    <div class="
                        overview-probability
                    ">
                        <div class="
                            overview-probability-label
                        ">
                            客胜
                        </div>
                        <div class="
                            overview-probability-value
                        ">
                            {away_probability_text}
                        </div>
                    </div>
                </div>

                <div class="
                    overview-card-footer
                ">
                    <span class="
                        overview-badge
                        {outcome_class}
                    ">
                        主预测：{fixture_outcome}
                    </span>

                    <span class="
                        overview-badge
                    ">
                        比分：{fixture_score}
                    </span>

                    <span class="
                        overview-badge
                    ">
                        🏟 {fixture_venue_name}
                    </span>

                    <span class="
                        overview-badge
                        {agreement_class}
                    ">
                        {agreement_text}
                    </span>

                    <span class="
                        overview-badge
                    ">
                        {fixture_context_status}
                        ·
                        {fixture_context_confidence_text}
                    </span>
                </div>
            </div>
            """
        )


st.markdown(
    """
    <div class="warning-box">
        天气、伤停、裁判、新闻和比赛重要性
        已接入赛前报告；在完成历史回测前，
        这些信息仅用于解释，不修改冻结模型概率。
    </div>
    """,
    unsafe_allow_html=True,
)
