from __future__ import annotations

import html
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEAM_ASSETS_FILE = (
    PROJECT_ROOT
    / "data"
    / "assets"
    / "team_assets.parquet"
)

KIT_ASSETS_FILE = (
    PROJECT_ROOT
    / "data"
    / "assets"
    / "kit_assets.parquet"
)

KIT_IMAGES_DIR = (
    PROJECT_ROOT
    / "assets"
    / "teams"
    / "kits"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "kit_assets_audit.csv"
)

GALLERY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "kit_asset_gallery.html"
)

KIT_ARCHIVE_URL = (
    "https://www.footballkitarchive.com/"
    "premier-league-kits-2026-27-l224/"
)

OFFICIAL_REFERENCE_URL = (
    "https://www.premierleague.com/en/news/"
    "4672981/premier-league-club-kits-"
    "for-202627-season"
)

SEASON = "2026-27"

KIT_TYPES = (
    "home",
    "away",
    "third",
)

KIT_TYPE_LABELS = {
    "home": "主场",
    "away": "客场",
    "third": "第三",
}


def normalize_team_name(value: object) -> str:
    text = str(value or "").casefold()

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"\b("
        r"association football club"
        r"|football club"
        r"|afc"
        r"|fc"
        r")\b",
        " ",
        text,
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def request_page(
    session: requests.Session,
    url: str,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = session.get(
                url,
                timeout=45,
            )

            response.raise_for_status()
            return response

        except requests.RequestException as error:
            last_error = error

            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        f"无法读取网页：{url}"
    )


def collect_kit_pages(
    session: requests.Session,
) -> dict[tuple[str, str], str]:
    response = request_page(
        session,
        KIT_ARCHIVE_URL,
    )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    kit_pages: dict[
        tuple[str, str],
        str,
    ] = {}

    pattern = re.compile(
        r"-2026-27-"
        r"(home|away|third)"
        r"-kit-\d+/?$",
        re.IGNORECASE,
    )

    for link in soup.select("a[href]"):
        href = urljoin(
            KIT_ARCHIVE_URL,
            str(link.get("href", "")),
        )

        match = pattern.search(href)

        if not match:
            continue

        kit_type = match.group(1).lower()

        link_text = link.get_text(
            " ",
            strip=True,
        )

        team_text = re.sub(
            r"\s+2026-27\s+"
            r"(Home|Away|Third).*$",
            "",
            link_text,
            flags=re.IGNORECASE,
        )

        team_key = normalize_team_name(
            team_text
        )

        if not team_key:
            continue

        kit_pages[
            (
                team_key,
                kit_type,
            )
        ] = href

    return kit_pages


def find_kit_page(
    kit_pages: dict[tuple[str, str], str],
    team_name: str,
    kit_type: str,
) -> str:
    target_key = normalize_team_name(
        team_name
    )

    exact_page = kit_pages.get(
        (
            target_key,
            kit_type,
        )
    )

    if exact_page:
        return exact_page

    candidates = []

    for (
        source_team_key,
        source_kit_type,
    ), page_url in kit_pages.items():
        if source_kit_type != kit_type:
            continue

        if (
            target_key in source_team_key
            or source_team_key in target_key
        ):
            candidates.append(page_url)

    if len(candidates) == 1:
        return candidates[0]

    return ""


def extract_image_url(
    session: requests.Session,
    page_url: str,
) -> str:
    response = request_page(
        session,
        page_url,
    )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    selectors = (
        'meta[property="og:image"]',
        'meta[property="og:image:secure_url"]',
        'meta[name="twitter:image"]',
    )

    for selector in selectors:
        element = soup.select_one(selector)

        if element is None:
            continue

        image_url = str(
            element.get("content", "")
        ).strip()

        if image_url:
            return urljoin(
                page_url,
                image_url,
            )

    page_title = soup.select_one("h1")

    title_text = (
        page_title.get_text(
            " ",
            strip=True,
        )
        if page_title is not None
        else ""
    )

    for image in soup.select("img[src]"):
        alt_text = str(
            image.get("alt", "")
        )

        if (
            title_text
            and title_text.casefold()
            in alt_text.casefold()
        ):
            return urljoin(
                page_url,
                str(image.get("src", "")),
            )

    return ""


def image_extension(
    response: requests.Response,
    image_url: str,
) -> str:
    content_type = (
        response.headers.get(
            "Content-Type",
            "",
        )
        .split(";")[0]
        .strip()
        .lower()
    )

    extension = (
        mimetypes.guess_extension(
            content_type
        )
        if content_type
        else None
    )

    if extension == ".jpe":
        extension = ".jpg"

    if extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        return extension

    url_suffix = Path(
        image_url.split("?")[0]
    ).suffix.lower()

    if url_suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        return url_suffix

    return ".jpg"


def download_kit_image(
    session: requests.Session,
    image_url: str,
    page_url: str,
    team_id: str,
    kit_type: str,
) -> str:
    response = session.get(
        image_url,
        headers={
            "Referer": page_url,
        },
        timeout=60,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    if not content_type.startswith(
        "image/"
    ):
        raise RuntimeError(
            "下载结果不是图片："
            f"{image_url}"
        )

    extension = image_extension(
        response,
        image_url,
    )

    file_name = (
        f"{team_id}-{kit_type}"
        f"{extension}"
    )

    output_file = (
        KIT_IMAGES_DIR
        / file_name
    )

    output_file.write_bytes(
        response.content
    )

    return output_file.relative_to(
        PROJECT_ROOT
    ).as_posix()


def clean_value(value: object) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def build_gallery(
    assets: pd.DataFrame,
) -> None:
    cards: list[str] = []

    for _, row in assets.iterrows():
        team_name = html.escape(
            clean_value(
                row.get("team_name")
            )
        )

        kit_type = clean_value(
            row.get("kit_type")
        )

        kit_label = html.escape(
            KIT_TYPE_LABELS.get(
                kit_type,
                kit_type,
            )
        )

        status = html.escape(
            clean_value(
                row.get("status")
            )
        )

        source_page = html.escape(
            clean_value(
                row.get("source_page")
            ),
            quote=True,
        )

        image_path = clean_value(
            row.get("kit_image_path")
        )

        if image_path:
            image_html = (
                '<img '
                f'src="../{html.escape(image_path, quote=True)}" '
                f'alt="{team_name} {kit_label}球衣">'
            )
        else:
            image_html = (
                '<div class="missing-image">'
                "尚未取得图片"
                "</div>"
            )

        if source_page:
            source_html = (
                f'<a href="{source_page}" '
                'target="_blank" '
                'rel="noopener noreferrer">'
                "查看来源"
                "</a>"
            )
        else:
            source_html = "无来源页面"

        cards.append(
            f"""
            <article class="kit-card">
                <div class="image-panel">
                    {image_html}
                </div>

                <h2>{team_name}</h2>

                <div class="kit-type">
                    {kit_label}球衣
                </div>

                <div class="status">
                    {status}
                </div>

                <div class="source">
                    {source_html}
                </div>
            </article>
            """
        )

    gallery_html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <title>2026/27球衣图片人工检查</title>

        <style>
            body {{
                margin: 0;
                padding: 28px;
                background: #070c16;
                color: #f8fafc;
                font-family:
                    Inter,
                    "Microsoft YaHei",
                    sans-serif;
            }}

            h1 {{
                margin: 0 0 8px 0;
            }}

            .description {{
                color: #9fb0c8;
                margin-bottom: 24px;
            }}

            .grid {{
                display: grid;
                grid-template-columns:
                    repeat(
                        auto-fit,
                        minmax(260px, 1fr)
                    );
                gap: 18px;
            }}

            .kit-card {{
                background: #111827;
                border: 1px solid #29364d;
                border-radius: 16px;
                padding: 14px;
            }}

            .image-panel {{
                height: 310px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 12px;
                background:
                    linear-gradient(
                        135deg,
                        #eef2f7,
                        #cbd5e1
                    );
                overflow: hidden;
            }}

            .image-panel img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}

            .missing-image {{
                color: #7c2d12;
                font-weight: 700;
            }}

            h2 {{
                margin: 14px 0 6px 0;
                font-size: 1.08rem;
            }}

            .kit-type {{
                color: #00e58b;
                font-weight: 750;
            }}

            .status {{
                margin-top: 8px;
                color: #9fb0c8;
                font-size: 0.82rem;
            }}

            .source {{
                margin-top: 8px;
            }}

            a {{
                color: #7dd3fc;
            }}
        </style>
    </head>

    <body>
        <h1>2026/27球衣图片人工检查</h1>

        <div class="description">
            共 {len(assets)} 条球衣记录。
            请重点核对球队、赛季和主客场类型。
        </div>

        <main class="grid">
            {''.join(cards)}
        </main>
    </body>
    </html>
    """

    GALLERY_FILE.write_text(
        gallery_html,
        encoding="utf-8",
    )


def main() -> None:
    if not TEAM_ASSETS_FILE.exists():
        raise FileNotFoundError(
            f"找不到球队资源：{TEAM_ASSETS_FILE}"
        )

    teams = pd.read_parquet(
        TEAM_ASSETS_FILE
    )

    required_columns = {
        "team_id",
        "team_name",
    }

    missing_columns = (
        required_columns
        - set(teams.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "球队资源缺少字段："
            + ", ".join(
                sorted(missing_columns)
            )
        )

    KIT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    KIT_ASSETS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "EPL-Predictor/"
                "1.0"
            ),
            "Accept-Language": (
                "en-GB,en;q=0.9"
            ),
        }
    )

    print(
        "开始读取2026/27英超球衣目录……"
    )

    kit_pages = collect_kit_pages(
        session
    )

    print(
        f"识别到球衣页面："
        f"{len(kit_pages)}"
    )

    rows: list[dict[str, object]] = []

    total = (
        len(teams)
        * len(KIT_TYPES)
    )

    position = 0

    for _, team in teams.iterrows():
        team_id = clean_value(
            team["team_id"]
        )

        team_name = clean_value(
            team["team_name"]
        )

        for kit_type in KIT_TYPES:
            position += 1

            print(
                f"{position}/{total} "
                f"{team_name} "
                f"{kit_type}"
            )

            page_url = find_kit_page(
                kit_pages,
                team_name,
                kit_type,
            )

            image_url = ""
            relative_path = ""
            status = "page_not_found"

            if page_url:
                try:
                    image_url = (
                        extract_image_url(
                            session,
                            page_url,
                        )
                    )

                    if image_url:
                        relative_path = (
                            download_kit_image(
                                session,
                                image_url,
                                page_url,
                                team_id,
                                kit_type,
                            )
                        )

                        status = (
                            "downloaded_"
                            "needs_review"
                        )
                    else:
                        status = (
                            "image_not_found"
                        )

                except Exception as error:
                    status = (
                        "download_failed: "
                        f"{error}"
                    )

            rows.append(
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "season": SEASON,
                    "kit_type": kit_type,
                    "kit_image_path": (
                        relative_path
                    ),
                    "source": (
                        "Football Kit Archive"
                    ),
                    "source_page": page_url,
                    "image_url": image_url,
                    "official_reference_page": (
                        OFFICIAL_REFERENCE_URL
                    ),
                    "checked_at_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "status": status,
                }
            )

            time.sleep(0.45)

    assets = pd.DataFrame(rows)

    kit_order = {
        "home": 0,
        "away": 1,
        "third": 2,
    }

    assets["_kit_order"] = assets[
        "kit_type"
    ].map(kit_order)

    assets = assets.sort_values(
        [
            "team_name",
            "_kit_order",
        ],
        kind="stable",
    ).drop(
        columns=["_kit_order"]
    ).reset_index(drop=True)

    assets.to_parquet(
        KIT_ASSETS_FILE,
        index=False,
    )

    assets.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    build_gallery(assets)

    downloaded_count = int(
        assets["kit_image_path"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    missing_count = (
        len(assets)
        - downloaded_count
    )

    print()
    print("球衣资源生成完成。")
    print(f"球队数量：{len(teams)}")
    print(f"球衣记录：{len(assets)}")
    print(
        f"成功下载："
        f"{downloaded_count}"
    )
    print(
        f"未能下载："
        f"{missing_count}"
    )
    print(
        f"资源文件："
        f"{KIT_ASSETS_FILE}"
    )
    print(
        f"检查报告："
        f"{AUDIT_FILE}"
    )
    print(
        f"检查页面："
        f"{GALLERY_FILE}"
    )
    print()
    print(
        "请打开检查页面，"
        "确认球队及主客场类型正确。"
    )


if __name__ == "__main__":
    main()