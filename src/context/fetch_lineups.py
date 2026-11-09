from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "context"
    / "prematch_context.parquet"
)

CONTEXT_CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "context"
    / "prematch_context.csv"
)

API_FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "api_football_epl_2026_27.parquet"
)

RAW_FPL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fpl"
    / "bootstrap_static.json"
)

RAW_LINEUPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "api_football"
    / "lineups"
)

AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "lineup_context_audit.csv"
)

API_URL = (
    "https://v3.football.api-sports.io/"
    "fixtures/lineups"
)

# 官方阵容通常在开赛前约20至40分钟出现。
# 从开赛前75分钟开始轮询，开赛30分钟后停止，
# 既能及时获取名单，也避免无意义消耗API额度。
POLL_START_MINUTES = 75
POLL_END_MINUTES = 30

TEAM_ALIASES = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Brighton and Hove Albion": (
        "Brighton & Hove Albion"
    ),
    "Leeds": "Leeds United",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
}

POSITION_NAMES = {
    1: "G",
    2: "D",
    3: "M",
    4: "F",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "获取预计阵容，并在临近开赛时查询官方首发。"
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "忽略开赛时间窗口，查询所有尚未确认的比赛。"
        ),
    )

    return parser.parse_args()


def canonical_team_name(value: Any) -> str:
    name = re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()

    name = re.sub(
        r"\s+(?:FC|AFC)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return TEAM_ALIASES.get(name, name)


def text_key(value: Any) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value or ""),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def is_roster_exit_news(value: Any) -> bool:
    text = str(value or "").strip().lower()

    phrases = (
        "on loan",
        "loaned to",
        "loan move",
        "has left the club",
        "left the club",
        "departed the club",
        "released by the club",
        "contract terminated",
        "transferred to",
    )

    return any(phrase in text for phrase in phrases)


def player_priority(player: dict[str, Any]) -> tuple:
    status = str(player.get("status") or "a").lower()
    chance = player.get("chance_of_playing_next_round")

    if chance is None:
        chance_value = 100 if status == "a" else 40
    else:
        chance_value = safe_int(chance)

    availability_penalty = {
        "a": 0,
        "d": 1,
        "i": 2,
        "n": 3,
        "s": 3,
        "u": 3,
    }.get(status, 1)

    return (
        -availability_penalty,
        chance_value,
        safe_int(player.get("starts")),
        safe_int(player.get("minutes")),
        safe_int(player.get("total_points")),
    )


def projected_player(player: dict[str, Any]) -> dict:
    first_name = str(
        player.get("first_name") or ""
    ).strip()

    second_name = str(
        player.get("second_name") or ""
    ).strip()

    full_name = f"{first_name} {second_name}".strip()
    display_name = str(
        player.get("web_name") or full_name or "Unknown"
    ).strip()

    return {
        "id": player.get("id"),
        "name": full_name or display_name,
        "display_name": display_name,
        "number": None,
        "pos": POSITION_NAMES.get(
            safe_int(player.get("element_type")),
            "",
        ),
        "grid": None,
        "source": "FPL usage estimate",
    }


def build_projected_xi(
    players: list[dict[str, Any]],
) -> tuple[list[dict], str | None]:
    active_players = [
        player
        for player in players
        if not is_roster_exit_news(player.get("news"))
    ]

    by_position: dict[int, list[dict[str, Any]]] = {
        position: []
        for position in (1, 2, 3, 4)
    }

    for player in active_players:
        position = safe_int(player.get("element_type"))
        if position in by_position:
            by_position[position].append(player)

    for position_players in by_position.values():
        position_players.sort(
            key=player_priority,
            reverse=True,
        )

    # FPL位置分类无法区分边锋与中场，因此采用保守的
    # 4-4-2阵型来生成“预计阵容”，并在页面明确标注
    # 这不是官方名单。
    selection_plan = {
        1: 1,
        2: 4,
        3: 4,
        4: 2,
    }

    selected: list[dict[str, Any]] = []

    for position, count in selection_plan.items():
        selected.extend(
            by_position[position][:count]
        )

    selected_ids = {
        player.get("id")
        for player in selected
    }

    if len(selected) < 11:
        remaining = sorted(
            [
                player
                for player in active_players
                if player.get("id") not in selected_ids
            ],
            key=player_priority,
            reverse=True,
        )

        selected.extend(
            remaining[: 11 - len(selected)]
        )

    projected = [
        projected_player(player)
        for player in selected[:11]
    ]

    formation = "4-4-2" if len(projected) == 11 else None

    return projected, formation


def load_fpl_projections() -> dict[str, dict]:
    if not RAW_FPL_FILE.exists():
        return {}

    try:
        payload = json.loads(
            RAW_FPL_FILE.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}

    teams = {
        safe_int(team.get("id")): canonical_team_name(
            team.get("name")
        )
        for team in payload.get("teams", [])
    }

    players_by_team: dict[str, list[dict[str, Any]]] = {}

    for player in payload.get("elements", []):
        team_name = teams.get(
            safe_int(player.get("team"))
        )

        if team_name:
            players_by_team.setdefault(
                team_name,
                [],
            ).append(player)

    projections = {}

    for team_name, players in players_by_team.items():
        lineup, formation = build_projected_xi(players)
        projections[team_name] = {
            "players": lineup,
            "formation": formation,
        }

    return projections


def request_lineup(
    api_key: str,
    fixture_id: int,
) -> tuple[dict, dict]:
    response = requests.get(
        API_URL,
        headers={
            "x-apisports-key": api_key,
        },
        params={
            "fixture": fixture_id,
        },
        timeout=30,
    )

    quota = {
        "daily_limit": response.headers.get(
            "x-ratelimit-requests-limit"
        ),
        "daily_remaining": response.headers.get(
            "x-ratelimit-requests-remaining"
        ),
    }

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(
            f"API返回错误：{payload['errors']}"
        )

    if not isinstance(payload.get("response"), list):
        raise RuntimeError(
            "API响应中没有有效的response列表。"
        )

    return payload, quota


def normalise_api_players(
    items: Any,
    role: str,
) -> list[dict]:
    players = []

    if not isinstance(items, list):
        return players

    for item in items:
        player = (
            item.get("player", {})
            if isinstance(item, dict)
            else {}
        )

        if not player.get("name"):
            continue

        players.append(
            {
                "id": player.get("id"),
                "name": str(player.get("name")),
                "display_name": str(player.get("name")),
                "number": player.get("number"),
                "pos": player.get("pos"),
                "grid": player.get("grid"),
                "role": role,
                "source": "API-Football",
            }
        )

    return players


def select_team_lineup(
    response_items: list[dict],
    provider_team_id: Any,
    team_name: str,
) -> dict | None:
    expected_id = safe_int(
        provider_team_id,
        default=-1,
    )

    expected_name = canonical_team_name(team_name)

    for item in response_items:
        team = item.get("team") or {}
        if safe_int(team.get("id"), -2) == expected_id:
            return item

    for item in response_items:
        team = item.get("team") or {}
        if canonical_team_name(team.get("name")) == expected_name:
            return item

    return None


def player_names_match(
    first: dict,
    second: dict,
) -> bool:
    first_names = {
        text_key(first.get("name")),
        text_key(first.get("display_name")),
    }

    second_names = {
        text_key(second.get("name")),
        text_key(second.get("display_name")),
    }

    first_names.discard("")
    second_names.discard("")

    if first_names & second_names:
        return True

    for first_name in first_names:
        for second_name in second_names:
            first_tokens = first_name.split()
            second_tokens = second_name.split()

            if (
                first_tokens
                and second_tokens
                and first_tokens[-1] == second_tokens[-1]
            ):
                return True

    return False


def compare_lineups(
    projected: list[dict],
    official: list[dict],
) -> dict:
    predicted_not_starting = [
        player.get("display_name") or player.get("name")
        for player in projected
        if not any(
            player_names_match(player, official_player)
            for official_player in official
        )
    ]

    unexpected_starters = [
        player.get("display_name") or player.get("name")
        for player in official
        if not any(
            player_names_match(player, projected_player_item)
            for projected_player_item in projected
        )
    ]

    return {
        "predicted_not_starting": predicted_not_starting,
        "unexpected_starters": unexpected_starters,
        "change_count": len(unexpected_starters),
    }


def find_fixture(
    row: pd.Series,
    fixtures: pd.DataFrame,
) -> pd.Series | None:
    direct = fixtures.loc[
        fixtures["match_id"].astype(str)
        == str(row["match_id"])
    ]

    if not direct.empty:
        return direct.iloc[0]

    home_team = canonical_team_name(row["home_team"])
    away_team = canonical_team_name(row["away_team"])
    kickoff = pd.to_datetime(
        row["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    candidates = fixtures.loc[
        fixtures["home_team"].map(canonical_team_name)
        .eq(home_team)
        & fixtures["away_team"].map(canonical_team_name)
        .eq(away_team)
    ].copy()

    if candidates.empty:
        return None

    candidate_kickoffs = pd.to_datetime(
        candidates["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    time_difference = (
        candidate_kickoffs - kickoff
    ).abs()

    candidates = candidates.loc[
        time_difference <= pd.Timedelta(hours=36)
    ].copy()

    if candidates.empty:
        return None

    candidates["time_difference"] = (
        pd.to_datetime(
            candidates["kickoff_utc"],
            utc=True,
            errors="coerce",
        )
        - kickoff
    ).abs()

    return candidates.sort_values(
        "time_difference"
    ).iloc[0]


def is_query_due(
    kickoff: pd.Timestamp,
    now: pd.Timestamp,
) -> bool:
    minutes_to_kickoff = (
        kickoff - now
    ).total_seconds() / 60.0

    return (
        -POLL_END_MINUTES
        <= minutes_to_kickoff
        <= POLL_START_MINUTES
    )


def apply_projection(
    context: pd.DataFrame,
    index: int,
    side: str,
    team_name: str,
    projections: dict[str, dict],
) -> None:
    projection = projections.get(
        canonical_team_name(team_name)
    )

    if not projection:
        return

    context.at[
        index,
        f"{side}_predicted_xi_json",
    ] = json_text(projection["players"])

    context.at[
        index,
        f"{side}_usual_formation",
    ] = projection["formation"]


def ensure_columns(context: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "lineups_confirmed": False,
        "lineup_status": "waiting",
        "lineup_checked_at_utc": pd.NA,
        "lineup_source": pd.NA,
        "home_predicted_xi_json": "[]",
        "away_predicted_xi_json": "[]",
        "home_starting_xi_json": "[]",
        "away_starting_xi_json": "[]",
        "home_substitutes_json": "[]",
        "away_substitutes_json": "[]",
        "home_lineup_changes_json": "{}",
        "away_lineup_changes_json": "{}",
        "home_coach": pd.NA,
        "away_coach": pd.NA,
        "lineup_explanation": (
            "官方首发尚未公布；临近开赛将自动刷新。"
        ),
    }

    for column, default in defaults.items():
        if column not in context:
            context[column] = default

    return context


def save_context(context: pd.DataFrame) -> None:
    context["lineups_confirmed"] = context[
        "lineups_confirmed"
    ].astype("boolean")

    context["updated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    temporary_file = CONTEXT_FILE.with_suffix(
        ".tmp.parquet"
    )

    context.to_parquet(
        temporary_file,
        index=False,
    )

    os.replace(temporary_file, CONTEXT_FILE)

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    arguments = parse_arguments()

    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到上下文文件：{CONTEXT_FILE}"
        )

    context = pd.read_parquet(CONTEXT_FILE).copy()

    api_fixture_data_available = (
        API_FIXTURES_FILE.exists()
    )

    if api_fixture_data_available:
        fixtures = pd.read_parquet(
            API_FIXTURES_FILE
        ).copy()
    else:
        fixtures = pd.DataFrame(
            columns=[
                "match_id",
                "home_team",
                "away_team",
                "kickoff_utc",
                "provider_fixture_id",
                "home_provider_team_id",
                "away_provider_team_id",
            ]
        )

        print(
            "提示：当前没有API-Football 2026赛季赛程文件。"
        )
        print(
            "将继续生成预计阵容；"
            "官方首发功能暂时停用，不会中断流水线。"
        )

    if context.empty:
        raise RuntimeError("赛前上下文为空。")

    context = ensure_columns(context)

    context["kickoff_utc"] = pd.to_datetime(
        context["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    fixtures["kickoff_utc"] = pd.to_datetime(
        fixtures["kickoff_utc"],
        utc=True,
        errors="coerce",
    )

    projections = load_fpl_projections()

    for index, row in context.iterrows():
        apply_projection(
            context,
            index,
            "home",
            str(row["home_team"]),
            projections,
        )

        apply_projection(
            context,
            index,
            "away",
            str(row["away_team"]),
            projections,
        )

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(
        "API_FOOTBALL_KEY",
        "",
    ).strip()

    now = pd.Timestamp.now(tz="UTC")
    audit_rows = []
    request_count = 0
    confirmed_count = 0

    print("正在检查未来比赛首发阵容……")

    for index, row in context.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        kickoff = row["kickoff_utc"]

        existing_confirmed = bool(
            row.get("lineups_confirmed", False)
        )

        fixture = find_fixture(row, fixtures)
        status = "waiting"
        detail = "等待进入官方首发查询时间窗口。"
        fixture_id = None
        quota_remaining = None

        if existing_confirmed:
            status = "confirmed"
            detail = "官方首发已经确认，无需继续轮询。"
            confirmed_count += 1
        elif not api_fixture_data_available:
            status = "provider_unavailable"
            detail = (
                "API-Football免费方案不提供2026赛季数据；"
                "当前展示FPL使用率预计阵容。"
            )
        elif fixture is None:
            status = "fixture_not_found"
            detail = "未在API-Football赛程中匹配到比赛。"
        elif pd.isna(kickoff):
            status = "kickoff_unknown"
            detail = "开球时间未知，暂不查询。"
        elif not arguments.force and not is_query_due(
            kickoff,
            now,
        ):
            minutes_to_kickoff = (
                kickoff - now
            ).total_seconds() / 60.0

            if minutes_to_kickoff > POLL_START_MINUTES:
                status = "scheduled"
                detail = (
                    "官方首发尚未到公布时间；"
                    "当前显示FPL使用率预计阵容。"
                )
            else:
                status = "not_available"
                detail = (
                    "本次查询窗口内没有获得完整官方首发。"
                )
        elif len(api_key) <= 10:
            status = "api_key_missing"
            detail = "未读取到有效的API_FOOTBALL_KEY。"
        else:
            fixture_id = safe_int(
                fixture.get("provider_fixture_id"),
                default=-1,
            )

            try:
                payload, quota = request_lineup(
                    api_key,
                    fixture_id,
                )

                request_count += 1
                quota_remaining = quota.get(
                    "daily_remaining"
                )

                RAW_LINEUPS_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                raw_file = (
                    RAW_LINEUPS_DIR
                    / f"fixture_{fixture_id}.json"
                )

                raw_file.write_text(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                response_items = payload["response"]

                home_item = select_team_lineup(
                    response_items,
                    fixture.get("home_provider_team_id"),
                    home_team,
                )

                away_item = select_team_lineup(
                    response_items,
                    fixture.get("away_provider_team_id"),
                    away_team,
                )

                home_starters = normalise_api_players(
                    (home_item or {}).get("startXI"),
                    "starter",
                )

                away_starters = normalise_api_players(
                    (away_item or {}).get("startXI"),
                    "starter",
                )

                checked_at = datetime.now(
                    timezone.utc
                ).isoformat()

                context.at[
                    index,
                    "lineup_checked_at_utc",
                ] = checked_at

                context.at[
                    index,
                    "lineup_source",
                ] = "API-Football"

                if home_starters:
                    context.at[
                        index,
                        "home_starting_xi_json",
                    ] = json_text(home_starters)

                    context.at[
                        index,
                        "home_substitutes_json",
                    ] = json_text(
                        normalise_api_players(
                            (home_item or {}).get(
                                "substitutes"
                            ),
                            "substitute",
                        )
                    )

                    context.at[
                        index,
                        "home_formation",
                    ] = (home_item or {}).get(
                        "formation"
                    )

                    context.at[
                        index,
                        "home_coach",
                    ] = (
                        (home_item or {}).get("coach")
                        or {}
                    ).get("name")

                if away_starters:
                    context.at[
                        index,
                        "away_starting_xi_json",
                    ] = json_text(away_starters)

                    context.at[
                        index,
                        "away_substitutes_json",
                    ] = json_text(
                        normalise_api_players(
                            (away_item or {}).get(
                                "substitutes"
                            ),
                            "substitute",
                        )
                    )

                    context.at[
                        index,
                        "away_formation",
                    ] = (away_item or {}).get(
                        "formation"
                    )

                    context.at[
                        index,
                        "away_coach",
                    ] = (
                        (away_item or {}).get("coach")
                        or {}
                    ).get("name")

                if (
                    len(home_starters) == 11
                    and len(away_starters) == 11
                ):
                    status = "confirmed"
                    detail = (
                        "双方官方首发已经确认；"
                        "名单仅用于赛前解释，不修改冻结概率。"
                    )

                    context.at[
                        index,
                        "lineups_confirmed",
                    ] = True

                    confirmed_count += 1

                    for side, official in (
                        ("home", home_starters),
                        ("away", away_starters),
                    ):
                        projected_raw = context.at[
                            index,
                            f"{side}_predicted_xi_json",
                        ]

                        try:
                            projected = json.loads(
                                projected_raw
                            )
                        except (
                            TypeError,
                            json.JSONDecodeError,
                        ):
                            projected = []

                        changes = compare_lineups(
                            projected,
                            official,
                        )

                        context.at[
                            index,
                            f"{side}_lineup_changes_json",
                        ] = json_text(changes)

                        context.at[
                            index,
                            f"{side}_rotation_index",
                        ] = round(
                            changes["change_count"] / 11,
                            4,
                        )

                        official_formation = context.at[
                            index,
                            f"{side}_formation",
                        ]

                        usual_formation = context.at[
                            index,
                            f"{side}_usual_formation",
                        ]

                        context.at[
                            index,
                            f"{side}_formation_changed",
                        ] = bool(
                            pd.notna(official_formation)
                            and pd.notna(usual_formation)
                            and str(official_formation)
                            != str(usual_formation)
                        )
                elif home_starters or away_starters:
                    status = "partial"
                    detail = (
                        "只获得一方或不完整名单，将继续刷新。"
                    )
                else:
                    status = "waiting"
                    detail = (
                        "API尚未发布官方首发，将继续自动刷新。"
                    )

            except Exception as error:
                status = "request_error"
                detail = str(error)

        context.at[index, "lineup_status"] = status
        context.at[index, "lineup_explanation"] = detail

        audit_rows.append(
            {
                "match_id": row["match_id"],
                "kickoff_utc": kickoff,
                "home_team": home_team,
                "away_team": away_team,
                "provider_fixture_id": fixture_id,
                "lineup_status": status,
                "lineups_confirmed": bool(
                    context.at[index, "lineups_confirmed"]
                ),
                "checked_at_utc": context.at[
                    index,
                    "lineup_checked_at_utc",
                ],
                "api_daily_remaining": quota_remaining,
                "detail": detail,
            }
        )

        print(
            f"{index + 1}/{len(context)} "
            f"{home_team} vs {away_team}: {status}"
        )

    save_context(context)

    AUDIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(audit_rows).to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n首发阵容上下文更新完成。")
    print(f"API请求次数：{request_count}")
    print(f"已确认比赛：{confirmed_count}")
    print(f"阵容审计：{AUDIT_FILE}")


if __name__ == "__main__":
    main()
