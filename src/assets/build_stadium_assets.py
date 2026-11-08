from __future__ import annotations

import html
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREMATCH_REPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prematch_report.parquet"
)

STADIUM_DIRECTORY = (
    PROJECT_ROOT
    / "assets"
    / "stadiums"
)

ASSET_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "assets"
)

STADIUM_ASSETS_FILE = (
    ASSET_DATA_DIRECTORY
    / "stadium_assets.parquet"
)

STADIUM_ASSETS_CSV = (
    ASSET_DATA_DIRECTORY
    / "stadium_assets.csv"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "stadium_assets_audit.csv"
)

GALLERY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "stadium_asset_gallery.html"
)

COMMONS_API = (
    "https://commons.wikimedia.org/"
    "w/api.php"
)

USER_AGENT = (
    "EPL-Predictor/1.0 "
    "(local educational project)"
)

VENUE_SEARCH_ALIASES = {
    "Gtech Community Stadium": [
        "Brentford Community Stadium",
        "Gtech Community Stadium Brentford",
    ],
    "MKM Stadium": [
        "MKM Stadium Hull",
        "KCOM Stadium Hull City",
        "KC Stadium Hull City",
    ],
    "The American Express Community Stadium": [
        "Falmer Stadium Brighton",
        "Amex Stadium Brighton",
    ],
    "Coventry Building Society Arena": [
        "Coventry Building Society Arena stadium",
        "Ricoh Arena Coventry stadium",
    ],
    "St James' Park": [
        "St James Park Newcastle stadium",
        "St James' Park Newcastle United",
    ],
    "Anfield": [
        "Anfield stadium Liverpool",
        "Anfield football ground",
    ],
}


def slugify(value: str) -> str:
    value = value.lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    return value.strip("-")


def strip_html(value: str | None) -> str:
    if not value:
        return ""

    value = re.sub(
        r"<[^>]+>",
        "",
        str(value),
    )

    return html.unescape(
        value
    ).strip()


def metadata_value(
    metadata: dict,
    key: str,
) -> str:
    value = metadata.get(
        key,
        {},
    ).get(
        "value",
        "",
    )

    return strip_html(value)


def search_commons(
    query: str,
) -> list[dict]:
    # 避免连续访问Wikimedia Commons触发临时限流。
    time.sleep(1.2)
    response = requests.get(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": 10,
            "prop": "imageinfo",
            "iiprop": (
                "url|size|mime|extmetadata"
            ),
            "iiurlwidth": 1600,
            "origin": "*",
        },
        headers={
            "User-Agent": USER_AGENT,
        },
        timeout=30,
    )

    response.raise_for_status()

    pages = (
        response.json()
        .get("query", {})
        .get("pages", {})
    )

    results = list(
        pages.values()
    )

    return sorted(
        results,
        key=lambda page: page.get(
            "index",
            999,
        ),
    )


def score_candidate(
    page: dict,
    venue_name: str,
) -> float:
    image_info_list = page.get(
        "imageinfo",
        [],
    )

    if not image_info_list:
        return -1000.0

    image_info = image_info_list[0]

    mime_type = str(
        image_info.get("mime", "")
    ).lower()

    if mime_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        return -1000.0

    title = str(
        page.get("title", "")
    ).lower()

    rejected_words = {
        "logo",
        "badge",
        "crest",
        "map",
        "plan",
        "diagram",
        "icon",
        "poster",
        "programme",
    }

    if any(
        word in title
        for word in rejected_words
    ):
        return -1000.0

    width = int(
        image_info.get("width", 0)
        or 0
    )

    height = int(
        image_info.get("height", 0)
        or 0
    )

    score = 0.0

    if width > height:
        score += 15.0
    else:
        score -= 8.0

    if width >= 1200:
        score += 8.0

    venue_tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            venue_name.lower(),
        )
        if len(token) >= 4
    ]

    for token in venue_tokens:
        if token in title:
            score += 5.0

    metadata = image_info.get(
        "extmetadata",
        {},
    )

    license_name = metadata_value(
        metadata,
        "LicenseShortName",
    )

    artist = metadata_value(
        metadata,
        "Artist",
    )

    if license_name:
        score += 3.0

    if artist:
        score += 2.0

    return score


def choose_stadium_image(
    venue_name: str,
    home_team: str,
) -> dict | None:
    aliases = VENUE_SEARCH_ALIASES.get(
        venue_name,
        [venue_name],
    )

    search_queries: list[str] = []

    for alias in aliases:
        search_queries.extend(
            [
                f'"{alias}"',
                f"{alias} stadium",
            ]
        )

    search_queries.append(
        f"{home_team} stadium"
    )

    candidates: list[dict] = []
    seen_titles: set[str] = set()

    for query in search_queries:
        try:
            results = search_commons(
                query
            )
        except requests.RequestException:
            continue

        for page in results:
            title = str(
                page.get("title", "")
            )

            if title in seen_titles:
                continue

            seen_titles.add(title)

            candidate_score = (
                score_candidate(
                    page,
                    venue_name,
                )
            )

            if candidate_score <= -100:
                continue

            page["_score"] = (
                candidate_score
            )

            candidates.append(page)

    if not candidates:
        return None

    candidates.sort(
        key=lambda page: page["_score"],
        reverse=True,
    )

    return candidates[0]


def download_stadium_image(
    page: dict,
    venue_slug: str,
) -> dict:
    image_info = page[
        "imageinfo"
    ][0]

    metadata = image_info.get(
        "extmetadata",
        {},
    )

    image_url = (
        image_info.get("thumburl")
        or image_info.get("url")
    )

    response = requests.get(
        image_url,
        headers={
            "User-Agent": USER_AGENT,
        },
        timeout=60,
    )

    response.raise_for_status()

    mime_type = str(
        image_info.get(
            "thumbmime",
            image_info.get(
                "mime",
                "image/jpeg",
            ),
        )
    ).lower()

    extension = {
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(
        mime_type,
        ".jpg",
    )

    output_file = (
        STADIUM_DIRECTORY
        / f"{venue_slug}{extension}"
    )

    output_file.write_bytes(
        response.content
    )

    relative_path = (
        output_file.relative_to(
            PROJECT_ROOT
        ).as_posix()
    )

    return {
        "stadium_image_path": (
            relative_path
        ),
        "commons_file_title": (
            page.get("title")
        ),
        "commons_file_page": (
            image_info.get(
                "descriptionurl"
            )
            or (
                "https://commons.wikimedia.org/"
                "wiki/"
                + quote(
                    str(page.get("title", "")),
                    safe=":",
                )
            )
        ),
        "image_url": image_url,
        "artist": (
            metadata_value(
                metadata,
                "Artist",
            )
            or metadata_value(
                metadata,
                "Credit",
            )
        ),
        "license": metadata_value(
            metadata,
            "LicenseShortName",
        ),
        "license_url": metadata_value(
            metadata,
            "LicenseUrl",
        ),
        "image_description": (
            metadata_value(
                metadata,
                "ImageDescription",
            )
        ),
        "selection_score": (
            page.get("_score")
        ),
    }


def build_gallery(
    assets: pd.DataFrame,
) -> None:
    cards: list[str] = []

    for _, row in assets.iterrows():
        image_path = str(
            row.get(
                "stadium_image_path",
                "",
            )
            or ""
        )

        image_source = (
            "../"
            + image_path
        )

        cards.append(
            f"""
            <article class="card">
                <img
                    src="{html.escape(image_source)}"
                    alt="{html.escape(str(row['venue_name']))}"
                >
                <h2>
                    {html.escape(str(row['venue_name']))}
                </h2>
                <p>
                    {html.escape(str(row['home_team']))}
                </p>
                <p>
                    图片：
                    {html.escape(str(row.get('commons_file_title', '')))}
                </p>
                <p>
                    作者：
                    {html.escape(str(row.get('artist', '')))}
                </p>
                <p>
                    许可：
                    {html.escape(str(row.get('license', '')))}
                </p>
            </article>
            """
        )

    gallery_html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <title>球场图片检查</title>
        <style>
            body {{
                margin: 0;
                padding: 28px;
                color: #edf3fc;
                background: #080d17;
                font-family: Arial, sans-serif;
            }}

            .grid {{
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(320px, 1fr));
                gap: 20px;
            }}

            .card {{
                padding: 14px;
                background: #111827;
                border: 1px solid #2a354c;
                border-radius: 14px;
            }}

            img {{
                width: 100%;
                height: 220px;
                object-fit: cover;
                border-radius: 10px;
            }}

            h2 {{
                margin: 14px 0 8px;
            }}

            p {{
                color: #aebbd0;
                margin: 5px 0;
            }}
        </style>
    </head>
    <body>
        <h1>球场图片人工检查</h1>
        <div class="grid">
            {''.join(cards)}
        </div>
    </body>
    </html>
    """

    GALLERY_FILE.write_text(
        gallery_html,
        encoding="utf-8",
    )


def main() -> None:
    if not PREMATCH_REPORT_FILE.exists():
        raise FileNotFoundError(
            "找不到赛前报告："
            f"{PREMATCH_REPORT_FILE}"
        )

    report = pd.read_parquet(
        PREMATCH_REPORT_FILE
    )

    required_columns = {
        "home_team_id",
        "home_team",
        "venue_name",
    }

    missing_columns = (
        required_columns
        - set(report.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "赛前报告缺少字段："
            + ", ".join(
                sorted(missing_columns)
            )
        )

    venues = report[
        [
            "home_team_id",
            "home_team",
            "venue_name",
        ]
    ].dropna(
        subset=["venue_name"]
    ).drop_duplicates(
        subset=["venue_name"],
        keep="last",
    )

    STADIUM_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ASSET_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []

    print(
        "开始从Wikimedia Commons"
        "搜索球场照片……"
    )

    for position, (
        _,
        venue,
    ) in enumerate(
        venues.iterrows(),
        start=1,
    ):
        venue_name = str(
            venue["venue_name"]
        )

        home_team = str(
            venue["home_team"]
        )

        print(
            f"{position}/{len(venues)} "
            f"{venue_name}"
        )

        selected_page = (
            choose_stadium_image(
                venue_name=venue_name,
                home_team=home_team,
            )
        )

        base_row = {
            "home_team_id": (
                venue["home_team_id"]
            ),
            "home_team": home_team,
            "venue_name": venue_name,
            "source": (
                "Wikimedia Commons"
            ),
            "checked_at_utc": (
                pd.Timestamp.now(
                    tz="UTC"
                ).isoformat()
            ),
        }

        if selected_page is None:
            base_row.update(
                {
                    "stadium_image_path": (
                        None
                    ),
                    "status": "not_found",
                }
            )

            rows.append(base_row)
            continue

        try:
            image_data = (
                download_stadium_image(
                    selected_page,
                    slugify(venue_name),
                )
            )

            base_row.update(
                image_data
            )

            base_row["status"] = (
                "downloaded_needs_review"
            )

        except requests.RequestException as error:
            base_row.update(
                {
                    "stadium_image_path": (
                        None
                    ),
                    "status": (
                        f"download_error: "
                        f"{error}"
                    ),
                }
            )

        rows.append(base_row)

    assets = pd.DataFrame(rows)

    assets.to_parquet(
        STADIUM_ASSETS_FILE,
        index=False,
    )

    assets.to_csv(
        STADIUM_ASSETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    assets.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    build_gallery(assets)

    downloaded_count = int(
        assets["status"]
        .eq("downloaded_needs_review")
        .sum()
    )

    print()
    print("球场照片候选生成完成。")
    print(f"球场数量：{len(assets)}")
    print(
        f"成功下载：{downloaded_count}"
    )
    print(
        f"需要人工检查："
        f"{downloaded_count}"
    )
    print(
        f"图片目录：{STADIUM_DIRECTORY}"
    )
    print(
        f"检查页面：{GALLERY_FILE}"
    )
    print()
    print(
        "请打开检查页面，确认每张图片"
        "确实对应正确球场。"
    )


if __name__ == "__main__":
    main()