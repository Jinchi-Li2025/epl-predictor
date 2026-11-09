from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FILE = (
    PROJECT_ROOT / "data" / "context" / "prematch_context.parquet"
)
CONFIG_FILE = PROJECT_ROOT / "configs" / "llm_intelligence.json"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "team_news" / "team_news.json"
)
AUDIT_FILE = PROJECT_ROOT / "reports" / "team_news_fetch_audit.csv"
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search?q={query}"
    "&hl=en-GB&gl=GB&ceid=GB:en"
)
USER_AGENT = (
    "Mozilla/5.0 (compatible; EPLPredictor/1.0; "
    "+local-research-pipeline)"
)


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        value = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def parse_published(value: str) -> datetime | None:
    try:
        published = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return published.astimezone(timezone.utc)


def fetch_team_articles(
    team_name: str,
    *,
    lookback_days: int,
    max_articles: int,
) -> tuple[list[dict], str]:
    query = quote_plus(
        f'"{team_name}" football Premier League when:{lookback_days}d'
    )
    url = GOOGLE_NEWS_RSS.format(query=query)
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=25,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        published_text = clean_text(item.findtext("pubDate"))
        source_element = item.find("source")
        source = clean_text(
            source_element.text if source_element is not None else ""
        )
        published = parse_published(published_text)

        if not title or not link:
            continue
        if published is not None and published < cutoff:
            continue

        identity = (title.casefold(), link)
        if identity in seen:
            continue
        seen.add(identity)
        articles.append(
            {
                "title": title,
                "source": source or "Google News RSS",
                "url": link,
                "published_at_utc": (
                    published.isoformat() if published is not None else None
                ),
                "retrieval_note": (
                    "Google News RSS headline; open the linked source "
                    "before treating details as confirmed."
                ),
            }
        )
        if len(articles) >= max_articles:
            break

    return articles, url


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到赛前上下文文件：{CONTEXT_FILE}"
        )

    context = pd.read_parquet(CONTEXT_FILE)
    required = {"home_team", "away_team"}
    missing = required - set(context.columns)
    if missing:
        raise RuntimeError(f"上下文缺少字段：{sorted(missing)}")

    config = load_config()
    lookback_days = int(config.get("news_lookback_days", 7))
    max_articles = int(config.get("max_articles_per_team", 8))
    teams = sorted(
        set(context["home_team"].dropna().astype(str))
        | set(context["away_team"].dropna().astype(str))
    )
    checked_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, object] = {
        "generated_at_utc": checked_at,
        "lookback_days": lookback_days,
        "teams": {},
    }
    audit_rows: list[dict] = []

    print("开始抓取公开球队新闻标题……")
    for index, team_name in enumerate(teams, start=1):
        print(f"{index}/{len(teams)} {team_name}")
        status = "success"
        error_text = ""
        source_url = ""
        try:
            articles, source_url = fetch_team_articles(
                team_name,
                lookback_days=lookback_days,
                max_articles=max_articles,
            )
        except (
            requests.RequestException,
            ET.ParseError,
            ValueError,
        ) as error:
            articles = []
            status = "failed"
            error_text = f"{type(error).__name__}: {error}"

        payload["teams"][team_name] = articles  # type: ignore[index]
        audit_rows.append(
            {
                "team_name": team_name,
                "article_count": len(articles),
                "status": status,
                "source_url": source_url,
                "error": error_text,
                "checked_at_utc": checked_at,
            }
        )
        time.sleep(0.15)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(audit_rows).to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    success_count = sum(
        1 for row in audit_rows if row["status"] == "success"
    )
    article_count = sum(row["article_count"] for row in audit_rows)
    print("\n公开球队新闻抓取完成。")
    print(f"球队数量：{len(teams)}")
    print(f"成功球队：{success_count}")
    print(f"标题数量：{article_count}")
    print(f"原始数据：{OUTPUT_FILE}")
    print(f"抓取审计：{AUDIT_FILE}")
    print(
        "说明：新闻标题仅作为带链接的公开线索；"
        "系统不会据此修改冻结模型概率。"
    )


if __name__ == "__main__":
    main()
