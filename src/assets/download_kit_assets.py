from __future__ import annotations

import hashlib
import html
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = PROJECT_ROOT / "reports" / "footylogos_2026_27_kit_urls.csv"
KIT_DIR = PROJECT_ROOT / "assets" / "teams" / "kits" / "2026-27"
ASSET_DIR = PROJECT_ROOT / "data" / "assets"
PARQUET_FILE = ASSET_DIR / "kit_assets.parquet"
AUDIT_FILE = PROJECT_ROOT / "reports" / "kit_assets_audit.csv"
GALLERY_FILE = PROJECT_ROOT / "reports" / "kit_asset_gallery.html"

SEASON = "2026-27"
SOURCE_NAME = "FootyLogos"
REQUEST_DELAY_SECONDS = 0.15

TEAM_NAME_MAP = {
    "AFC Bournemouth": "AFC Bournemouth",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry City",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull City",
    "Ipswich Town": "Ipswich Town",
    "Leeds United": "Leeds United",
    "Liverpool FC": "Liverpool",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester United": "Manchester United",
    "Newcastle United": "Newcastle United",
    "Nottingham Forest": "Nottingham Forest",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham Hotspur",
}


def slugify(value: str) -> str:
    value = value.strip().lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def extension_from_response(response: requests.Response, url: str) -> str:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    known_types = {
        "image/avif": ".avif",
        "image/webp": ".webp",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }
    if content_type in known_types:
        return known_types[content_type]

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".avif", ".webp", ".png", ".jpg", ".jpeg"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    guessed = mimetypes.guess_extension(content_type)
    return guessed or ".img"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_source_records() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            "找不到球衣链接清单：\n"
            f"{SOURCE_CSV}\n"
            "请先运行上一条球衣链接提取命令。"
        )

    source = pd.read_csv(SOURCE_CSV, encoding="utf-8-sig")
    required = {"team_name", "kit_type", "image_url"}
    missing = required - set(source.columns)
    if missing:
        raise RuntimeError(f"球衣链接清单缺少字段：{sorted(missing)}")

    source = source.copy()
    source["team_name"] = source["team_name"].astype(str).str.strip()
    source["kit_type"] = source["kit_type"].astype(str).str.strip().str.lower()
    source["image_url"] = source["image_url"].astype(str).str.strip()
    source = source[source["kit_type"].isin(["home", "away", "third"])]
    source = source[source["image_url"].str.startswith(("https://", "http://"))]
    source = source.drop_duplicates(subset=["team_name", "kit_type"], keep="first")

    source["canonical_team_name"] = source["team_name"].map(TEAM_NAME_MAP)
    unknown = sorted(source.loc[source["canonical_team_name"].isna(), "team_name"].unique())
    if unknown:
        raise RuntimeError(f"存在未识别的球队名称：{unknown}")

    return source.sort_values(["canonical_team_name", "kit_type"]).reset_index(drop=True)


def existing_file(team_name: str, kit_type: str) -> Path | None:
    prefix = f"{slugify(team_name)}-{kit_type}"
    matches = [path for path in KIT_DIR.glob(f"{prefix}.*") if path.is_file()]
    return matches[0] if matches else None


def download_one(
    session: requests.Session,
    team_name: str,
    kit_type: str,
    image_url: str,
) -> dict[str, object]:
    downloaded_at = datetime.now(timezone.utc).isoformat()
    old_file = existing_file(team_name, kit_type)
    if old_file is not None and old_file.stat().st_size > 2_000:
        relative_path = old_file.relative_to(PROJECT_ROOT).as_posix()
        return {
            "team_name": team_name,
            "season": SEASON,
            "kit_type": kit_type,
            "kit_image_path": relative_path,
            "source_name": SOURCE_NAME,
            "source_url": image_url,
            "status": "existing",
            "http_status": 200,
            "file_size_bytes": old_file.stat().st_size,
            "sha256": sha256_bytes(old_file.read_bytes()),
            "downloaded_at_utc": downloaded_at,
            "error_message": "",
        }

    try:
        response = session.get(image_url, timeout=45)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            raise RuntimeError(f"响应不是图片：{content_type or 'unknown'}")
        if len(response.content) <= 2_000:
            raise RuntimeError(f"图片文件过小：{len(response.content)} bytes")

        extension = extension_from_response(response, image_url)
        file_name = f"{slugify(team_name)}-{kit_type}{extension}"
        destination = KIT_DIR / file_name
        destination.write_bytes(response.content)

        return {
            "team_name": team_name,
            "season": SEASON,
            "kit_type": kit_type,
            "kit_image_path": destination.relative_to(PROJECT_ROOT).as_posix(),
            "source_name": SOURCE_NAME,
            "source_url": image_url,
            "status": "downloaded",
            "http_status": response.status_code,
            "file_size_bytes": len(response.content),
            "sha256": sha256_bytes(response.content),
            "downloaded_at_utc": downloaded_at,
            "error_message": "",
        }
    except Exception as error:  # noqa: BLE001 - audit every failed asset
        status_code = getattr(getattr(error, "response", None), "status_code", "")
        return {
            "team_name": team_name,
            "season": SEASON,
            "kit_type": kit_type,
            "kit_image_path": "",
            "source_name": SOURCE_NAME,
            "source_url": image_url,
            "status": "failed",
            "http_status": status_code,
            "file_size_bytes": 0,
            "sha256": "",
            "downloaded_at_utc": downloaded_at,
            "error_message": str(error),
        }


def build_gallery(assets: pd.DataFrame) -> None:
    cards: list[str] = []
    for row in assets.itertuples(index=False):
        local_path = str(row.kit_image_path)
        if local_path:
            image_src = "../" + local_path
            image_html = (
                f'<img src="{html.escape(image_src)}" '
                f'alt="{html.escape(row.team_name)} {html.escape(row.kit_type)}">'
            )
        else:
            image_html = '<div class="missing">下载失败</div>'

        cards.append(
            "<article class=\"card\">"
            f"{image_html}"
            f"<h2>{html.escape(row.team_name)}</h2>"
            f"<p class=\"type\">{html.escape(row.kit_type.upper())}</p>"
            f"<p>状态：{html.escape(row.status)}</p>"
            f"<p class=\"source\">来源：{html.escape(row.source_name)}</p>"
            "</article>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2026/27 球衣图片人工检查</title>
  <style>
    body {{ margin: 0; padding: 28px; background: #070d18; color: #edf4ff; font-family: Arial, sans-serif; }}
    h1 {{ margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 18px; }}
    .card {{ padding: 14px; background: #111a2b; border: 1px solid #263653; border-radius: 14px; }}
    img, .missing {{ width: 100%; height: 260px; object-fit: contain; background: #f4f6f8; border-radius: 10px; }}
    .missing {{ display: grid; place-items: center; color: #b42318; }}
    h2 {{ margin: 12px 0 4px; font-size: 19px; }}
    p {{ margin: 5px 0; color: #a9b9d2; }}
    .type {{ color: #00e89d; font-weight: 700; }}
    .source {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>2026/27 球衣图片人工检查</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    GALLERY_FILE.write_text(document, encoding="utf-8")


def main() -> None:
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    source = load_source_records()
    print(f"读取到 {len(source)} 条球衣记录。")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/157.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.footylogos.com/football-kits/premier-league-2026-27",
        }
    )

    results: list[dict[str, object]] = []
    for position, row in enumerate(source.itertuples(index=False), start=1):
        print(
            f"{position}/{len(source)} "
            f"{row.canonical_team_name} {row.kit_type}"
        )
        result = download_one(
            session,
            row.canonical_team_name,
            row.kit_type,
            row.image_url,
        )
        results.append(result)
        if result["status"] == "failed":
            print(f"  失败：{result['error_message']}")
        time.sleep(REQUEST_DELAY_SECONDS)

    assets = pd.DataFrame(results)
    assets.to_parquet(PARQUET_FILE, index=False)
    assets.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")
    build_gallery(assets)

    successful = int(assets["status"].isin(["downloaded", "existing"]).sum())
    failed = int((assets["status"] == "failed").sum())
    team_count = int(assets.loc[assets["kit_image_path"].ne(""), "team_name"].nunique())

    print("\n球衣资源处理完成。")
    print(f"记录数量：{len(assets)}")
    print(f"覆盖球队：{team_count}")
    print(f"成功图片：{successful}")
    print(f"失败图片：{failed}")
    print(f"图片目录：{KIT_DIR}")
    print(f"Parquet：{PARQUET_FILE}")
    print(f"审计报告：{AUDIT_FILE}")
    print(f"检查图库：{GALLERY_FILE}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
