from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KIT_DATA_FILE = PROJECT_ROOT / "data" / "assets" / "kit_assets.parquet"

KIT_TYPE_LABELS = {
    "home": "主场球衣",
    "away": "客场球衣",
    "third": "第三球衣",
}


@st.cache_data(show_spinner=False)
def load_kit_assets(file_mtime_ns: int) -> pd.DataFrame:
    del file_mtime_ns
    if not KIT_DATA_FILE.exists():
        return pd.DataFrame()

    kits = pd.read_parquet(KIT_DATA_FILE)
    required_columns = {
        "team_name",
        "season",
        "kit_type",
        "kit_image_path",
    }
    if not required_columns.issubset(kits.columns):
        return pd.DataFrame()

    kits = kits.copy()
    for column in ["team_name", "season", "kit_type", "kit_image_path"]:
        kits[column] = kits[column].fillna("").astype(str).str.strip()

    kits["kit_type"] = kits["kit_type"].str.lower()
    kits = kits[kits["kit_image_path"].ne("")]
    return kits.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def image_to_data_uri(relative_path: str, file_mtime_ns: int) -> str:
    del file_mtime_ns
    image_path = PROJECT_ROOT / relative_path
    if not image_path.exists() or not image_path.is_file():
        return ""

    mime_type = mimetypes.guess_type(image_path.name)[0]
    if not mime_type:
        mime_type = {
            ".avif": "image/avif",
            ".webp": "image/webp",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(image_path.suffix.lower(), "application/octet-stream")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _load_assets() -> pd.DataFrame:
    if not KIT_DATA_FILE.exists():
        return pd.DataFrame()
    return load_kit_assets(KIT_DATA_FILE.stat().st_mtime_ns)


def _find_kit(
    kits: pd.DataFrame,
    team_name: str,
    preferred_types: tuple[str, ...],
) -> tuple[pd.Series | None, bool]:
    team_kits = kits[kits["team_name"].eq(team_name)]
    for position, kit_type in enumerate(preferred_types):
        candidates = team_kits[team_kits["kit_type"].eq(kit_type)]
        if not candidates.empty:
            return candidates.iloc[0], position > 0
    return None, False


def _kit_card(
    row: pd.Series | None,
    team_name: str,
    side_label: str,
    used_fallback: bool,
) -> str:
    safe_team = html.escape(team_name)
    safe_side = html.escape(side_label)

    if row is None:
        return (
            '<article class="match-kit-card match-kit-missing">'
            f'<div class="match-kit-side">{safe_side}</div>'
            '<div class="match-kit-empty">暂未读取到球衣图片</div>'
            f'<h3>{safe_team}</h3>'
            '</article>'
        )

    relative_path = str(row["kit_image_path"])
    image_path = PROJECT_ROOT / relative_path
    if not image_path.exists():
        image_uri = ""
    else:
        image_uri = image_to_data_uri(
            relative_path,
            image_path.stat().st_mtime_ns,
        )

    kit_type = str(row["kit_type"])
    kit_label = KIT_TYPE_LABELS.get(kit_type, kit_type)
    if used_fallback:
        kit_label += " · 自动替代"

    season = html.escape(str(row.get("season", "2026-27")))
    safe_kit_label = html.escape(kit_label)

    if image_uri:
        image_block = (
            '<div class="match-kit-image-wrap">'
            f'<img src="{image_uri}" alt="{safe_team} {safe_kit_label}">'
            "</div>"
        )
    else:
        image_block = '<div class="match-kit-empty">图片文件不存在</div>'

    return (
        '<article class="match-kit-card">'
        f'<div class="match-kit-side">{safe_side}</div>'
        f'{image_block}'
        '<div class="match-kit-card-copy">'
        f'<h3>{safe_team}</h3>'
        f'<p>{season} · {safe_kit_label}</p>'
        '</div>'
        '</article>'
    )


def _all_team_kits_html(kits: pd.DataFrame, team_name: str) -> str:
    cards: list[str] = []
    team_kits = kits[kits["team_name"].eq(team_name)].copy()
    sort_order = {"home": 0, "away": 1, "third": 2}
    team_kits["sort_order"] = team_kits["kit_type"].map(sort_order).fillna(99)
    team_kits = team_kits.sort_values("sort_order")

    for row in team_kits.itertuples(index=False):
        relative_path = str(row.kit_image_path)
        image_path = PROJECT_ROOT / relative_path
        if not image_path.exists():
            continue
        image_uri = image_to_data_uri(relative_path, image_path.stat().st_mtime_ns)
        label = KIT_TYPE_LABELS.get(str(row.kit_type), str(row.kit_type))
        cards.append(
            '<div class="kit-gallery-item">'
            f'<img src="{image_uri}" alt="{html.escape(team_name)} {html.escape(label)}">'
            f'<strong>{html.escape(label)}</strong>'
            "</div>"
        )

    if not cards:
        return ""

    return (
        '<section class="kit-gallery-team">'
        f'<h4>{html.escape(team_name)}</h4>'
        '<div class="kit-gallery-row">'
        + "".join(cards)
        + "</div></section>"
    )


def render_match_kits(home_team: str, away_team: str) -> None:
    kits = _load_assets()
    if kits.empty:
        st.info("尚未生成球衣资源；请先运行球衣下载脚本。")
        return

    home_row, home_fallback = _find_kit(
        kits,
        home_team,
        ("home", "third", "away"),
    )
    away_row, away_fallback = _find_kit(
        kits,
        away_team,
        ("away", "third", "home"),
    )

    st.markdown(
        """
        <style>
        .match-kit-section {
            margin: 18px 0 26px;
        }
        .match-kit-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 14px;
            margin-bottom: 12px;
        }
        .match-kit-heading h2 {
            margin: 0;
        }
        .match-kit-heading span {
            color: #8395b3;
            font-size: 12px;
        }
        .match-kit-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }
        .match-kit-card {
            position: relative;
            display: grid;
            grid-template-columns: minmax(130px, 36%) 1fr;
            align-items: center;
            min-height: 260px;
            overflow: hidden;
            border: 1px solid #263653;
            border-radius: 18px;
            background:
                radial-gradient(circle at 15% 10%, rgba(0, 232, 157, .12), transparent 42%),
                linear-gradient(145deg, #111b2d, #0c1423);
            box-shadow: 0 18px 40px rgba(0, 0, 0, .18);
        }
        .match-kit-side {
            position: absolute;
            top: 14px;
            left: 16px;
            z-index: 2;
            padding: 5px 9px;
            border-radius: 999px;
            background: rgba(0, 232, 157, .13);
            color: #32efb1;
            font-size: 11px;
            font-weight: 750;
            letter-spacing: .08em;
        }
        .match-kit-image-wrap {
            display: grid;
            place-items: center;
            height: 260px;
            padding: 28px 12px 14px;
            background: #f3f5f7;
        }
        .match-kit-image-wrap img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .match-kit-card-copy {
            padding: 28px;
        }
        .match-kit-card-copy h3 {
            margin: 0 0 8px;
            color: #ffffff;
            font-size: clamp(19px, 2vw, 29px);
        }
        .match-kit-card-copy p {
            margin: 0;
            color: #9bb0ce;
            font-size: 13px;
        }
        .match-kit-empty {
            display: grid;
            place-items: center;
            min-height: 220px;
            padding: 20px;
            color: #91a2bd;
        }
        .kit-gallery-team h4 {
            margin: 14px 0 8px;
        }
        .kit-gallery-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .kit-gallery-item {
            display: grid;
            gap: 8px;
            padding: 10px;
            text-align: center;
            border: 1px solid #263653;
            border-radius: 12px;
            background: #111a2b;
        }
        .kit-gallery-item img {
            width: 100%;
            height: 180px;
            object-fit: contain;
            border-radius: 8px;
            background: #f3f5f7;
        }
        .kit-gallery-item strong {
            color: #d9e5f7;
            font-size: 12px;
        }
        @media (max-width: 900px) {
            .match-kit-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 620px) {
            .match-kit-card {
                grid-template-columns: 42% 1fr;
                min-height: 210px;
            }
            .match-kit-image-wrap {
                height: 210px;
            }
            .match-kit-card-copy {
                padding: 18px;
            }
            .kit-gallery-row {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    section_html = (
        '<section class="match-kit-section">'
        '<div class="match-kit-heading">'
        '<h2>本赛季比赛球衣</h2>'
        '<span>2026/27 · 主队优先主场款，客队优先客场款</span>'
        "</div>"
        '<div class="match-kit-grid">'
        + _kit_card(home_row, home_team, "HOME TEAM", home_fallback)
        + _kit_card(away_row, away_team, "AWAY TEAM", away_fallback)
        + "</div></section>"
    )
    st.markdown(section_html, unsafe_allow_html=True)

    with st.expander("查看双方全部球衣", expanded=False):
        gallery_html = (
            _all_team_kits_html(kits, home_team)
            + _all_team_kits_html(kits, away_team)
        )
        if gallery_html:
            st.markdown(gallery_html, unsafe_allow_html=True)
        else:
            st.caption("暂未读取到双方的其他球衣。")

