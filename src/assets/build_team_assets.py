from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

FOOTBALL_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "football_data_epl_current.parquet"
)

CURRENT_COMPLETED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_completed_matches.parquet"
)

CURRENT_FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "current_fixtures.parquet"
)

ASSET_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "assets"
)

CREST_DIRECTORY = (
    PROJECT_ROOT
    / "assets"
    / "teams"
    / "crests"
)

TEAM_ASSETS_PARQUET = (
    ASSET_DATA_DIRECTORY
    / "team_assets.parquet"
)

TEAM_ASSETS_CSV = (
    ASSET_DATA_DIRECTORY
    / "team_assets.csv"
)

TEAM_ASSETS_AUDIT = (
    PROJECT_ROOT
    / "reports"
    / "team_assets_audit.csv"
)

API_URL = (
    "https://api.football-data.org/"
    "v4/competitions/PL/teams"
)


def load_local_environment() -> None:
    """读取项目.env，但不在终端输出任何密钥。"""

    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split("=", 1)

        key = key.strip()
        value = value.strip().strip(
            "\"'"
        )

        if key and key not in os.environ:
            os.environ[key] = value


def find_api_token() -> str:
    preferred_names = [
        "FOOTBALL_DATA_API_KEY",
        "FOOTBALL_DATA_TOKEN",
        "FOOTBALL_DATA_ORG_TOKEN",
        "FOOTBALL_DATA_API_TOKEN",
    ]

    for variable_name in preferred_names:
        value = os.getenv(variable_name)

        if value:
            return value.strip()

    raise RuntimeError(
        "没有找到football-data.org密钥。"
        "请确认.env中存在"
        "FOOTBALL_DATA_API_KEY。"
    )


def slugify(value: str) -> str:
    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "-",
        text,
    )

    return text.strip("-")


def load_project_team_ids() -> dict[str, str]:
    frames: list[pd.DataFrame] = []

    for path in [
        CURRENT_COMPLETED_FILE,
        CURRENT_FIXTURES_FILE,
    ]:
        if path.exists():
            frames.append(
                pd.read_parquet(path)
            )

    if not frames:
        return {}

    matches = pd.concat(
        frames,
        ignore_index=True,
    )

    home_teams = matches[
        [
            "home_team",
            "home_team_id",
        ]
    ].rename(
        columns={
            "home_team": "team_name",
            "home_team_id": "team_id",
        }
    )

    away_teams = matches[
        [
            "away_team",
            "away_team_id",
        ]
    ].rename(
        columns={
            "away_team": "team_name",
            "away_team_id": "team_id",
        }
    )

    teams = pd.concat(
        [
            home_teams,
            away_teams,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=["team_name"],
        keep="last",
    )

    return dict(
        zip(
            teams["team_name"],
            teams["team_id"],
        )
    )


def load_provider_team_names() -> dict[int, str]:
    if not FOOTBALL_DATA_FILE.exists():
        return {}

    matches = pd.read_parquet(
        FOOTBALL_DATA_FILE
    )

    home_teams = matches[
        [
            "provider_home_team_id",
            "home_team",
        ]
    ].rename(
        columns={
            "provider_home_team_id": (
                "provider_team_id"
            ),
            "home_team": "team_name",
        }
    )

    away_teams = matches[
        [
            "provider_away_team_id",
            "away_team",
        ]
    ].rename(
        columns={
            "provider_away_team_id": (
                "provider_team_id"
            ),
            "away_team": "team_name",
        }
    )

    teams = pd.concat(
        [
            home_teams,
            away_teams,
        ],
        ignore_index=True,
    ).dropna(
        subset=["provider_team_id"]
    ).drop_duplicates(
        subset=["provider_team_id"],
        keep="last",
    )

    return {
        int(row["provider_team_id"]): str(
            row["team_name"]
        )
        for _, row in teams.iterrows()
    }


def request_teams(
    api_token: str,
) -> list[dict]:
    response = requests.get(
        API_URL,
        headers={
            "X-Auth-Token": api_token,
            "User-Agent": (
                "EPL-Predictor/"
                "team-assets"
            ),
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "球队资料请求失败："
            f"HTTP {response.status_code}；"
            f"{response.text[:300]}"
        )

    payload = response.json()
    teams = payload.get("teams", [])

    if not teams:
        raise RuntimeError(
            "API没有返回任何英超球队。"
        )

    return teams


def determine_extension(
    url: str,
    content_type: str,
) -> str:
    suffix = Path(
        urlparse(url).path
    ).suffix.lower()

    if suffix in {
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        return suffix

    content_type = content_type.lower()

    if "svg" in content_type:
        return ".svg"

    if "png" in content_type:
        return ".png"

    if "webp" in content_type:
        return ".webp"

    return ".jpg"


def download_crest(
    crest_url: str | None,
    team_id: str,
) -> tuple[str | None, str]:
    if not crest_url:
        return None, "missing_url"

    try:
        response = requests.get(
            crest_url,
            headers={
                "User-Agent": (
                    "EPL-Predictor/"
                    "team-assets"
                ),
            },
            timeout=30,
        )

        response.raise_for_status()

        extension = determine_extension(
            crest_url,
            response.headers.get(
                "Content-Type",
                "",
            ),
        )

        output_file = (
            CREST_DIRECTORY
            / f"{team_id}{extension}"
        )

        output_file.write_bytes(
            response.content
        )

        relative_path = output_file.relative_to(
            PROJECT_ROOT
        )

        return (
            relative_path.as_posix(),
            "downloaded",
        )

    except requests.RequestException as error:
        return None, f"error: {error}"


def main() -> None:
    load_local_environment()

    api_token = find_api_token()

    ASSET_DATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CREST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEAM_ASSETS_AUDIT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    project_team_ids = (
        load_project_team_ids()
    )

    provider_names = (
        load_provider_team_names()
    )

    print(
        "开始同步英超球队资料和队徽……"
    )

    api_teams = request_teams(
        api_token
    )

    rows: list[dict] = []

    for position, team in enumerate(
        api_teams,
        start=1,
    ):
        provider_team_id = int(
            team["id"]
        )

        project_team_name = (
            provider_names.get(
                provider_team_id
            )
            or team.get("shortName")
            or team.get("name")
        )

        project_team_name = str(
            project_team_name
        )

        team_id = project_team_ids.get(
            project_team_name,
            slugify(project_team_name),
        )

        crest_url = team.get("crest")

        crest_path, crest_status = (
            download_crest(
                crest_url=crest_url,
                team_id=team_id,
            )
        )

        print(
            f"{position}/{len(api_teams)} "
            f"{project_team_name}: "
            f"{crest_status}"
        )

        rows.append(
            {
                "team_id": team_id,
                "team_name": (
                    project_team_name
                ),
                "provider_team_id": (
                    provider_team_id
                ),
                "provider_name": (
                    team.get("name")
                ),
                "short_name": (
                    team.get("shortName")
                ),
                "tla": team.get("tla"),
                "crest_url": crest_url,
                "crest_path": crest_path,
                "crest_status": (
                    crest_status
                ),
                "club_colors": (
                    team.get("clubColors")
                ),
                "venue_name": (
                    team.get("venue")
                ),
                "website": (
                    team.get("website")
                ),
                "founded": (
                    team.get("founded")
                ),
                "last_updated_utc": (
                    pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat()
                ),
                "source": (
                    "football-data.org v4"
                ),
            }
        )

    assets = pd.DataFrame(rows)

    assets = assets.sort_values(
        [
            "team_name",
            "provider_team_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    if assets["team_id"].duplicated().any():
        duplicates = assets.loc[
            assets["team_id"].duplicated(
                keep=False
            ),
            [
                "team_id",
                "team_name",
            ],
        ]

        raise RuntimeError(
            "球队资源中存在重复team_id：\n"
            + duplicates.to_string(
                index=False
            )
        )

    assets.to_parquet(
        TEAM_ASSETS_PARQUET,
        index=False,
    )

    assets.to_csv(
        TEAM_ASSETS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    assets[
        [
            "team_id",
            "team_name",
            "provider_team_id",
            "crest_path",
            "crest_status",
            "venue_name",
            "club_colors",
        ]
    ].to_csv(
        TEAM_ASSETS_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    downloaded_count = int(
        assets["crest_status"]
        .eq("downloaded")
        .sum()
    )

    print()
    print("球队资源同步完成。")
    print(f"球队数量：{len(assets)}")
    print(
        f"成功下载队徽："
        f"{downloaded_count}"
    )
    print(
        f"未成功下载："
        f"{len(assets) - downloaded_count}"
    )
    print(
        f"资源清单：{TEAM_ASSETS_PARQUET}"
    )
    print(
        f"可查看CSV：{TEAM_ASSETS_CSV}"
    )
    print(
        f"队徽目录：{CREST_DIRECTORY}"
    )

    print()
    print("球队资源摘要：")
    print(
        assets[
            [
                "team_name",
                "tla",
                "venue_name",
                "club_colors",
                "crest_status",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()