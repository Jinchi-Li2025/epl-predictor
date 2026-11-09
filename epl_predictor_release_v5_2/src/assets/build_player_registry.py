from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "assets" / "player_registry.parquet"
OUTPUT_CSV = PROJECT_ROOT / "data" / "assets" / "player_registry.csv"
AUDIT_FILE = PROJECT_ROOT / "reports" / "player_registry_audit.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "premier_league"
FPL_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "fpl" / "bootstrap_static.json"
FPL_HISTORY_DIR = PROJECT_ROOT / "data" / "raw" / "fpl" / "player_history"
FOOTBALL_DATA_RAW_FILE = (
    PROJECT_ROOT / "data" / "raw" / "football_data" / "pl_teams_squads.json"
)
FOOTBALL_DATA_TEAM_DIR = (
    PROJECT_ROOT / "data" / "raw" / "football_data" / "teams"
)
PROFILE_OVERRIDE_FILE = (
    PROJECT_ROOT / "data" / "manual" / "player_profile_overrides.csv"
)
PROFILE_OVERRIDE_AUDIT_FILE = (
    PROJECT_ROOT / "reports" / "player_profile_override_audit.csv"
)

OFFICIAL_SQUADS_URL = (
    "https://www.premierleague.com/en/news/4706139/"
    "see-all-the-202627-premier-league-squad-lists"
)
OFFICIAL_NUMBERS_URL = "https://www.premierleague.com/en/news/4725682"
FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_HISTORY_URL = (
    "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
)
FOOTBALL_DATA_TEAMS_URL = (
    "https://api.football-data.org/v4/competitions/PL/teams"
)

CURRENT_TEAMS = [
    "AFC Bournemouth",
    "Arsenal",
    "Aston Villa",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
]

TEAM_ALIASES = {
    "Bournemouth": "AFC Bournemouth",
    "Bournemouth AFC": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion",
    "Brighton Hove": "Brighton & Hove Albion",
    "Brighton Hove Albion": "Brighton & Hove Albion",
    "Brighton and Hove Albion": "Brighton & Hove Albion",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Nottm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
}

POSITION_CODES = {1: "G", 2: "D", 3: "M", 4: "F"}
POSITION_NAMES = {
    "G": "门将",
    "D": "后卫",
    "M": "中场",
    "F": "前锋",
    "": "待补充",
}
STATUS_NAMES = {
    "a": "可出场",
    "d": "出场存疑",
    "i": "伤缺",
    "n": "无法出场",
    "s": "停赛",
    "u": "状态未知",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="建立2026/27英超注册球员资料库。"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="仅使用已经保存的原始页面与FPL文件。",
    )
    parser.add_argument(
        "--refresh-profiles",
        action="store_true",
        help="强制刷新football-data逐队详细球员资料。",
    )
    parser.add_argument(
        "--skip-player-history",
        action="store_true",
        help="跳过FPL逐球员赛季履历刷新；已有缓存仍会被读取。",
    )
    return parser.parse_args()


def load_dotenv_if_present() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def canonical_team_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\s+(?:FC|AFC)$", "", text, flags=re.IGNORECASE)
    return TEAM_ALIASES.get(text, text)


def text_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fetch_page(
    session: requests.Session,
    url: str,
    destination: Path,
    offline: bool,
) -> str:
    if offline:
        if not destination.exists():
            return ""
        return destination.read_text(encoding="utf-8", errors="replace")

    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(response.text, encoding="utf-8")
        return response.text
    except requests.RequestException as error:
        if destination.exists():
            print(f"警告：官方页面更新失败，使用本地快照：{error}")
            return destination.read_text(encoding="utf-8", errors="replace")
        print(f"警告：官方页面暂时不可用：{error}")
        return ""


def visible_lines(page_html: str) -> list[str]:
    if not page_html:
        return []
    soup = BeautifulSoup(page_html, "html.parser")
    for item in soup(["script", "style", "noscript", "svg"]):
        item.decompose()
    text = html_lib.unescape(soup.get_text("\n"))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def registered_display_name(registered_name: str) -> str:
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", registered_name).strip(" *")
    if "," not in cleaned:
        return cleaned.title() if cleaned.isupper() else cleaned
    surname, given_names = [part.strip() for part in cleaned.split(",", 1)]
    surname = surname.title() if surname.isupper() else surname
    given_names = given_names.title() if given_names.isupper() else given_names
    return f"{given_names} {surname}".strip()


def parse_registered_squads(page_html: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_team = ""
    current_section = ""
    team_set = set(CURRENT_TEAMS)

    for line in visible_lines(page_html):
        canonical = canonical_team_name(line)
        if canonical in team_set:
            current_team = canonical
            current_section = ""
            continue

        lower = line.casefold()
        if "25 squad players" in lower:
            current_section = "senior_25"
            continue
        if lower.startswith("u21 players") or "under-21 players" in lower:
            current_section = "registered_u21"
            continue

        if not current_team or not current_section or "," not in line:
            continue
        if len(line) > 180 or line.count(",") > 3:
            continue
        surname = line.split(",", 1)[0].strip(" *")
        if not surname or not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", surname):
            continue

        home_grown = "*" in line
        on_loan = "loan" in lower
        cleaned = re.sub(r"\s*\([^)]*loan[^)]*\)", "", line, flags=re.I)
        cleaned = re.sub(r"\s*\*+\s*", " ", cleaned).strip()
        records.append(
            {
                "team": current_team,
                "registered_name": cleaned,
                "display_name": registered_display_name(cleaned),
                "registration_category": current_section,
                "home_grown": home_grown,
                "on_loan": on_loan,
                "registration_verified": True,
                "eligible_2026_27": True,
            }
        )

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["team"], text_key(record["registered_name"]))
        unique[key] = record
    return list(unique.values())


def parse_squad_numbers(page_html: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_team = ""
    pending_number: int | None = None
    team_set = set(CURRENT_TEAMS)

    for line in visible_lines(page_html):
        canonical = canonical_team_name(line)
        if canonical in team_set:
            current_team = canonical
            pending_number = None
            continue
        if not current_team:
            continue

        combined = re.fullmatch(r"(\d{1,3})\s*(?:[-–—.:]|\s)\s*(.+)", line)
        if combined:
            name = combined.group(2).strip()
            if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", name):
                records.append(
                    {
                        "team": current_team,
                        "squad_number": int(combined.group(1)),
                        "number_name": name,
                    }
                )
            pending_number = None
            continue

        if re.fullmatch(r"\d{1,3}", line):
            pending_number = int(line)
            continue

        if pending_number is not None:
            if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", line) and len(line) <= 80:
                records.append(
                    {
                        "team": current_team,
                        "squad_number": pending_number,
                        "number_name": line,
                    }
                )
            pending_number = None

    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        unique[(record["team"], record["squad_number"])] = record
    return list(unique.values())


def load_fpl_payload(session: requests.Session, offline: bool) -> dict[str, Any]:
    if not offline:
        try:
            response = session.get(FPL_URL, timeout=45)
            response.raise_for_status()
            payload = response.json()
            FPL_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
            FPL_RAW_FILE.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            return payload
        except (requests.RequestException, ValueError) as error:
            if FPL_RAW_FILE.exists():
                print(f"警告：FPL更新失败，使用本地快照：{error}")
            else:
                print(f"警告：FPL数据暂时不可用：{error}")
    if not FPL_RAW_FILE.exists():
        return {}
    return json.loads(FPL_RAW_FILE.read_text(encoding="utf-8"))


def fetch_fpl_player_history(
    player_id: int,
    offline: bool,
    refresh: bool,
) -> dict[str, Any]:
    cache_file = FPL_HISTORY_DIR / f"{player_id}.json"
    if cache_file.exists() and (offline or not refresh):
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if offline:
        return {}
    try:
        response = requests.get(
            FPL_HISTORY_URL.format(player_id=player_id),
            timeout=35,
            headers={"User-Agent": "EPL-Predictor/2.0"},
        )
        response.raise_for_status()
        payload = response.json()
        FPL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload
    except (requests.RequestException, ValueError, OSError):
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {}


def load_fpl_histories(
    player_ids: list[int],
    offline: bool,
    skip_refresh: bool,
) -> dict[int, dict[str, int]]:
    today = datetime.now(timezone.utc).date()

    def needs_refresh(player_id: int) -> bool:
        cache_file = FPL_HISTORY_DIR / f"{player_id}.json"
        if skip_refresh or offline or not cache_file.exists():
            return not (skip_refresh or offline)
        modified = datetime.fromtimestamp(
            cache_file.stat().st_mtime,
            tz=timezone.utc,
        ).date()
        return modified != today

    output: dict[int, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                fetch_fpl_player_history,
                player_id,
                offline,
                needs_refresh(player_id),
            ): player_id
            for player_id in player_ids
        }
        for future in as_completed(futures):
            player_id = futures[future]
            try:
                payload = future.result()
            except Exception:
                payload = {}
            appearances = sum(
                1
                for item in payload.get("history", [])
                if safe_number(item.get("minutes")) > 0
            )
            output[player_id] = {
                "appearances": appearances,
            }
    return output


def load_football_data_squads(
    session: requests.Session,
    offline: bool,
    refresh_profiles: bool = False,
) -> list[dict[str, Any]]:
    api_key = (
        os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
        or os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
    )
    if offline and FOOTBALL_DATA_RAW_FILE.exists():
        payload = json.loads(
            FOOTBALL_DATA_RAW_FILE.read_text(encoding="utf-8")
        )
    elif offline:
        return []
    else:
        if not api_key:
            if FOOTBALL_DATA_RAW_FILE.exists():
                payload = json.loads(
                    FOOTBALL_DATA_RAW_FILE.read_text(encoding="utf-8")
                )
            else:
                print(
                    "提示：未配置FOOTBALL_DATA_API_KEY，"
                    "国籍、出生日期与合同字段暂显示待核验。"
                )
                return []
        else:
            try:
                response = session.get(
                    FOOTBALL_DATA_TEAMS_URL,
                    headers={"X-Auth-Token": api_key},
                    timeout=45,
                )
                response.raise_for_status()
                payload = response.json()
                FOOTBALL_DATA_RAW_FILE.parent.mkdir(
                    parents=True, exist_ok=True
                )
                FOOTBALL_DATA_RAW_FILE.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            except (requests.RequestException, ValueError) as error:
                if not FOOTBALL_DATA_RAW_FILE.exists():
                    print(f"警告：football-data球员资料获取失败：{error}")
                    return []
                print(f"警告：使用football-data本地快照：{error}")
                payload = json.loads(
                    FOOTBALL_DATA_RAW_FILE.read_text(encoding="utf-8")
                )

    records: list[dict[str, Any]] = []
    FOOTBALL_DATA_TEAM_DIR.mkdir(parents=True, exist_ok=True)
    detail_requests_allowed = bool(api_key and not offline)
    remote_detail_requests = 0
    for team in payload.get("teams", []):
        team_id = team.get("id")
        detail_url = (
            f"https://api.football-data.org/v4/teams/{team_id}"
            if team_id is not None
            else ""
        )
        detail_file = FOOTBALL_DATA_TEAM_DIR / f"{team_id}.json"
        detail_payload: dict[str, Any] = {}
        profile_retrieved_at = ""
        cache_fresh = False
        if detail_file.exists():
            modified = datetime.fromtimestamp(
                detail_file.stat().st_mtime, tz=timezone.utc
            )
            profile_retrieved_at = modified.isoformat()
            cache_fresh = (
                datetime.now(timezone.utc) - modified
            ).total_seconds() < 7 * 24 * 3600
            try:
                detail_payload = json.loads(
                    detail_file.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                detail_payload = {}

        should_fetch_detail = (
            detail_requests_allowed
            and team_id is not None
            and (refresh_profiles or not cache_fresh)
        )
        if should_fetch_detail:
            if remote_detail_requests:
                # football-data free accounts allow 10 requests per minute.
                time.sleep(6.2)
            try:
                detail_response = session.get(
                    detail_url,
                    headers={"X-Auth-Token": api_key},
                    timeout=45,
                )
                if detail_response.status_code == 429:
                    wait_seconds = int(
                        detail_response.headers.get("X-RequestCounter-Reset", "7")
                    ) + 1
                    time.sleep(max(wait_seconds, 7))
                    detail_response = session.get(
                        detail_url,
                        headers={"X-Auth-Token": api_key},
                        timeout=45,
                    )
                detail_response.raise_for_status()
                candidate_payload = detail_response.json()
                if candidate_payload.get("squad"):
                    detail_payload = candidate_payload
                    profile_retrieved_at = datetime.now(timezone.utc).isoformat()
                    detail_file.write_text(
                        json.dumps(candidate_payload, ensure_ascii=False),
                        encoding="utf-8",
                    )
                remote_detail_requests += 1
            except requests.HTTPError as error:
                status_code = getattr(error.response, "status_code", None)
                print(
                    f"警告：football-data球队详情获取失败 "
                    f"({team.get('name')}, HTTP {status_code})。"
                )
                if status_code == 403:
                    detail_requests_allowed = False
            except (requests.RequestException, ValueError, OSError) as error:
                print(
                    f"警告：football-data球队详情获取失败 "
                    f"({team.get('name')}): {error}"
                )

        source_team = detail_payload if detail_payload.get("squad") else team
        if not profile_retrieved_at and FOOTBALL_DATA_RAW_FILE.exists():
            profile_retrieved_at = datetime.fromtimestamp(
                FOOTBALL_DATA_RAW_FILE.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()
        team_name = canonical_team_name(
            source_team.get("shortName") or source_team.get("name")
        )
        squad = source_team.get("squad", []) or []
        for player in squad:
            contract = player.get("contract") or {}
            records.append(
                {
                    "team": team_name,
                    "football_data_player_id": player.get("id"),
                    "profile_name": player.get("name") or "",
                    "first_name": player.get("firstName") or "",
                    "last_name": player.get("lastName") or "",
                    "date_of_birth": player.get("dateOfBirth") or "",
                    "nationality": player.get("nationality") or "",
                    "position_detail": player.get("position") or "",
                    "profile_shirt_number": player.get("shirtNumber"),
                    "market_value": player.get("marketValue"),
                    "contract_start": contract.get("start") or "",
                    "contract_until": contract.get("until") or "",
                    "profile_source_url": detail_url or FOOTBALL_DATA_TEAMS_URL,
                    "profile_retrieved_at_utc": profile_retrieved_at,
                }
            )
    return records


def player_match_score(left: Any, right: Any) -> float:
    left_key = text_key(left)
    right_key = text_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_tokens = left_key.split()
    right_tokens = right_key.split()
    ratio = SequenceMatcher(None, left_key, right_key).ratio()
    overlap = len(set(left_tokens) & set(right_tokens)) / max(
        1, min(len(left_tokens), len(right_tokens))
    )
    surname_bonus = 0.18 if left_tokens[-1] == right_tokens[-1] else 0.0
    return min(1.0, 0.58 * ratio + 0.32 * overlap + surname_bonus)


def best_match(
    name: str,
    candidates: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_score = 0.0
    for index, candidate in enumerate(candidates):
        score = max(
            (player_match_score(name, candidate.get(field)) for field in fields),
            default=0.0,
        )
        if score > best_score:
            best_index = index
            best_score = score
    return best_index, best_score


def fpl_records(
    payload: dict[str, Any],
    histories: dict[int, dict[str, int]],
) -> list[dict[str, Any]]:
    teams = {
        int(team["id"]): canonical_team_name(team.get("name"))
        for team in payload.get("teams", [])
        if team.get("id") is not None
    }
    output = []
    for player in payload.get("elements", []):
        team = teams.get(int(player.get("team") or 0), "")
        if team not in CURRENT_TEAMS:
            continue
        full_name = " ".join(
            part
            for part in [player.get("first_name"), player.get("second_name")]
            if part
        ).strip()
        position_code = POSITION_CODES.get(int(player.get("element_type") or 0), "")
        status = str(player.get("status") or "u").lower()
        fpl_id = int(player.get("id") or 0)
        starts = int(round(safe_number(player.get("starts"))))
        appearances = max(
            starts,
            int(histories.get(fpl_id, {}).get("appearances", starts)),
        )
        substitute_appearances = max(0, appearances - starts)
        output.append(
            {
                "fpl_id": fpl_id,
                "team": team,
                "fpl_full_name": full_name,
                "web_name": player.get("web_name") or full_name,
                "position_code": position_code,
                "position_name": POSITION_NAMES[position_code],
                "availability_code": status,
                "availability_status": STATUS_NAMES.get(status, "状态未知"),
                "chance_of_playing_next_round": player.get(
                    "chance_of_playing_next_round"
                ),
                "news": player.get("news") or "",
                "minutes": safe_number(player.get("minutes")),
                "starts": starts,
                "appearances": appearances,
                "substitute_appearances": substitute_appearances,
                "goals_scored": safe_number(player.get("goals_scored")),
                "assists": safe_number(player.get("assists")),
                "clean_sheets": safe_number(player.get("clean_sheets")),
                "saves": safe_number(player.get("saves")),
                "yellow_cards": safe_number(player.get("yellow_cards")),
                "red_cards": safe_number(player.get("red_cards")),
                "expected_goals": safe_number(player.get("expected_goals")),
                "expected_assists": safe_number(player.get("expected_assists")),
                "expected_goal_involvements": safe_number(
                    player.get("expected_goal_involvements")
                ),
                "form": safe_number(player.get("form")),
                "total_points": safe_number(player.get("total_points")),
                "now_cost": safe_number(player.get("now_cost")) / 10.0,
                "selected_by_percent": safe_number(player.get("selected_by_percent")),
                "appearance_source_note": (
                    "FPL逐场minutes>0计总出场；替补出场=总出场-首发"
                    if fpl_id in histories
                    else "FPL逐场履历未刷新；总出场暂按首发下限显示"
                ),
            }
        )
    return output


def merge_profile_fields(
    frame: pd.DataFrame,
    profiles: list[dict[str, Any]],
) -> pd.DataFrame:
    output = frame.copy()
    defaults = {
        "football_data_player_id": pd.NA,
        "date_of_birth": "",
        "nationality": "",
        "citizenships": "",
        "multiple_nationality_status": "待核验（主源仅提供代表国籍）",
        "position_detail": "",
        "market_value": pd.NA,
        "contract_start": "",
        "contract_until": "",
        "profile_source_url": "",
        "nationality_source_url": "",
        "contract_source_url": "",
        "market_value_source_url": "",
        "profile_retrieved_at_utc": "",
        "height_cm": pd.NA,
        "birth_place": "",
        "profile_match_score": pd.NA,
    }
    for column, default in defaults.items():
        output[column] = default

    profiles_by_team = {
        team: [row for row in profiles if row.get("team") == team]
        for team in CURRENT_TEAMS
    }
    for index, row in output.iterrows():
        candidates = profiles_by_team.get(row.get("team"), [])
        best_index: int | None = None
        best_score = 0.0
        for source_name in {
            row.get("registered_name"),
            row.get("display_name"),
            row.get("fpl_full_name"),
        }:
            candidate_index, score = best_match(
                str(source_name or ""),
                candidates,
                ("profile_name", "first_name", "last_name"),
            )
            if score > best_score:
                best_index = candidate_index
                best_score = score
        if best_index is None or best_score < 0.62:
            continue
        profile = candidates[best_index]
        for column in [
            "football_data_player_id",
            "date_of_birth",
            "nationality",
            "position_detail",
            "market_value",
            "contract_start",
            "contract_until",
        ]:
            output.at[index, column] = profile.get(column)
        nationality = str(profile.get("nationality") or "").strip()
        output.at[index, "citizenships"] = nationality
        source_url = str(
            profile.get("profile_source_url") or FOOTBALL_DATA_TEAMS_URL
        )
        output.at[index, "profile_source_url"] = source_url
        output.at[index, "nationality_source_url"] = source_url
        output.at[index, "profile_retrieved_at_utc"] = profile.get(
            "profile_retrieved_at_utc", ""
        )
        if str(profile.get("contract_start") or "").strip() or str(
            profile.get("contract_until") or ""
        ).strip():
            output.at[index, "contract_source_url"] = source_url
        if profile.get("market_value") is not None and not pd.isna(
            profile.get("market_value")
        ):
            output.at[index, "market_value_source_url"] = source_url
        output.at[index, "profile_match_score"] = round(best_score, 4)
        shirt_number = profile.get("profile_shirt_number")
        if pd.isna(row.get("squad_number")) and shirt_number is not None:
            try:
                numeric_number = int(float(shirt_number))
                if 0 < numeric_number < 1000:
                    output.at[index, "squad_number"] = numeric_number
            except (TypeError, ValueError):
                pass
    return output


def nonempty(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().casefold() not in {"", "nan", "none", "<na>"}


def apply_profile_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    if not PROFILE_OVERRIDE_FILE.exists():
        return frame
    overrides = pd.read_csv(PROFILE_OVERRIDE_FILE, encoding="utf-8-sig")
    if not {"player_name", "team"}.issubset(overrides.columns):
        raise RuntimeError(
            "player_profile_overrides.csv must contain player_name and team."
        )
    output = frame.copy()
    audit_rows: list[dict[str, Any]] = []
    field_map = {
        "representative_nationality": "nationality",
        "citizenships": "citizenships",
        "date_of_birth": "date_of_birth",
        "birth_place": "birth_place",
        "height_cm": "height_cm",
        "position_detail": "position_detail",
        "contract_start": "contract_start",
        "contract_until": "contract_until",
        "market_value_eur": "market_value",
    }
    for _, override in overrides.iterrows():
        team = canonical_team_name(override.get("team"))
        player_name = str(override.get("player_name") or "").strip()
        candidates = output.loc[output["team"].eq(team)]
        best_index = None
        best_score = 0.0
        for index, candidate in candidates.iterrows():
            score = max(
                player_match_score(player_name, candidate.get(column))
                for column in [
                    "display_name",
                    "registered_name",
                    "fpl_full_name",
                    "web_name",
                ]
            )
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None or best_score < 0.75:
            audit_rows.append(
                {
                    "player_name": player_name,
                    "team": team,
                    "status": "unmatched",
                    "match_score": best_score,
                }
            )
            continue
        for source_column, target_column in field_map.items():
            value = override.get(source_column)
            if not nonempty(value):
                continue
            if target_column in {"market_value", "height_cm"}:
                value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
            output.at[best_index, target_column] = value

        citizenships = str(override.get("citizenships") or "").strip()
        if citizenships:
            values = [item.strip() for item in citizenships.split("|") if item.strip()]
            output.at[best_index, "multiple_nationality_status"] = (
                "已核验：" + " / ".join(values)
                if len(values) > 1
                else "已核验：单一已知公民身份"
            )
        source_url = str(override.get("source_url") or "").strip()
        contract_source = str(
            override.get("contract_source_url") or source_url
        ).strip()
        valuation_source = str(
            override.get("valuation_source_url") or source_url
        ).strip()
        if source_url:
            output.at[best_index, "profile_source_url"] = source_url
            output.at[best_index, "nationality_source_url"] = source_url
        if contract_source and (
            nonempty(override.get("contract_start"))
            or nonempty(override.get("contract_until"))
        ):
            output.at[best_index, "contract_source_url"] = contract_source
        if valuation_source and nonempty(override.get("market_value_eur")):
            output.at[best_index, "market_value_source_url"] = valuation_source
        output.at[best_index, "profile_retrieved_at_utc"] = str(
            override.get("verified_at_utc")
            or datetime.now(timezone.utc).isoformat()
        )
        audit_rows.append(
            {
                "player_name": player_name,
                "team": team,
                "status": "applied",
                "match_score": round(best_score, 4),
                "matched_player_id": output.at[best_index, "player_id"],
            }
        )

    PROFILE_OVERRIDE_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        PROFILE_OVERRIDE_AUDIT_FILE, index=False, encoding="utf-8-sig"
    )
    return output


def add_derived_performance_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Build transparent rate features without guessing missing statistics."""
    output = frame.copy()
    minutes = pd.to_numeric(output.get("minutes", 0), errors="coerce").fillna(0.0)
    appearances = pd.to_numeric(
        output.get("appearances", 0), errors="coerce"
    ).fillna(0.0)
    starts = pd.to_numeric(output.get("starts", 0), errors="coerce").fillna(0.0)
    substitutes = pd.to_numeric(
        output.get("substitute_appearances", 0), errors="coerce"
    ).fillna(0.0)
    denominator = minutes.where(minutes.gt(0), np.nan) / 90.0
    for source, target in [
        ("goals_scored", "goals_per90"),
        ("assists", "assists_per90"),
        ("expected_goals", "xg_per90"),
        ("expected_assists", "xa_per90"),
        ("expected_goal_involvements", "xgi_per90"),
    ]:
        values = pd.to_numeric(output.get(source, 0), errors="coerce").fillna(0.0)
        output[target] = (values / denominator).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    output["goal_involvements_per90"] = (
        output["goals_per90"] + output["assists_per90"]
    )
    appearance_denominator = appearances.where(appearances.gt(0), np.nan)
    output["start_rate"] = (starts / appearance_denominator).fillna(0.0)
    output["bench_appearance_rate"] = (
        substitutes / appearance_denominator
    ).fillna(0.0)
    return output


def merge_sources(
    official: list[dict[str, Any]],
    numbers: list[dict[str, Any]],
    fpl: list[dict[str, Any]],
) -> pd.DataFrame:
    fpl_by_team = {
        team: [row for row in fpl if row["team"] == team]
        for team in CURRENT_TEAMS
    }
    used_fpl_ids: set[Any] = set()
    merged: list[dict[str, Any]] = []

    for row in official:
        candidates = [
            item
            for item in fpl_by_team.get(row["team"], [])
            if item.get("fpl_id") not in used_fpl_ids
        ]
        index, score = best_match(
            row["display_name"],
            candidates,
            ("fpl_full_name", "web_name"),
        )
        combined = dict(row)
        combined["fpl_match_score"] = round(score, 4)
        if index is not None and score >= 0.62:
            fpl_row = candidates[index]
            used_fpl_ids.add(fpl_row.get("fpl_id"))
            combined.update(fpl_row)
            combined["display_name"] = fpl_row.get("web_name") or row["display_name"]
        merged.append(combined)

    for row in fpl:
        if row.get("fpl_id") in used_fpl_ids:
            continue
        merged.append(
            {
                **row,
                "registered_name": row.get("fpl_full_name"),
                "display_name": row.get("web_name"),
                "registration_category": "fpl_unverified",
                "home_grown": pd.NA,
                "on_loan": False,
                "registration_verified": False,
                "eligible_2026_27": row.get("availability_code") != "u",
                "fpl_match_score": pd.NA,
            }
        )

    number_by_team = {
        team: [row for row in numbers if row["team"] == team]
        for team in CURRENT_TEAMS
    }
    for row in merged:
        candidates = number_by_team.get(row["team"], [])
        index: int | None = None
        score = 0.0
        source_names = {
            str(row.get("registered_name") or "").strip(),
            str(row.get("display_name") or "").strip(),
            str(row.get("web_name") or "").strip(),
            str(row.get("fpl_full_name") or "").strip(),
        }
        for source_name in source_names:
            if not source_name:
                continue
            candidate_index, candidate_score = best_match(
                source_name,
                candidates,
                ("number_name",),
            )
            if candidate_score > score:
                index = candidate_index
                score = candidate_score
        row["squad_number"] = (
            candidates[index]["squad_number"]
            if index is not None and score >= 0.62
            else pd.NA
        )
        row["number_match_score"] = round(score, 4) if index is not None else pd.NA

    frame = pd.DataFrame(merged)
    required_defaults = {
        "fpl_id": pd.NA,
        "fpl_full_name": "",
        "web_name": "",
        "position_code": "",
        "position_name": "待补充",
        "availability_code": "u",
        "availability_status": "状态未知",
        "chance_of_playing_next_round": pd.NA,
        "news": "",
        "minutes": 0.0,
        "starts": 0.0,
        "appearances": 0.0,
        "substitute_appearances": 0.0,
        "goals_scored": 0.0,
        "assists": 0.0,
        "clean_sheets": 0.0,
        "saves": 0.0,
        "yellow_cards": 0.0,
        "red_cards": 0.0,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "expected_goal_involvements": 0.0,
        "form": 0.0,
        "total_points": 0.0,
        "now_cost": 0.0,
        "selected_by_percent": 0.0,
        "appearance_source_note": "",
    }
    for column, default in required_defaults.items():
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)

    frame["player_id"] = [
        f"{text_key(team).replace(' ', '-')}-{text_key(name).replace(' ', '-')}"
        for team, name in zip(frame["team"], frame["registered_name"])
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    frame["generated_at_utc"] = generated_at
    frame["registration_source_url"] = OFFICIAL_SQUADS_URL
    frame["squad_number_source_url"] = OFFICIAL_NUMBERS_URL
    frame["performance_source_url"] = FPL_URL
    frame = frame.sort_values(
        ["team", "registration_verified", "squad_number", "display_name"],
        ascending=[True, False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    return frame


def write_outputs(frame: pd.DataFrame) -> None:
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT_PARQUET, index=False)
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    audit = (
        frame.groupby("team", as_index=False)
        .agg(
            player_count=("player_id", "count"),
            officially_verified=("registration_verified", "sum"),
            senior_25=(
                "registration_category",
                lambda values: int((values == "senior_25").sum()),
            ),
            registered_u21=(
                "registration_category",
                lambda values: int((values == "registered_u21").sum()),
            ),
            fpl_matched=("fpl_id", lambda values: int(values.notna().sum())),
            numbered=("squad_number", lambda values: int(values.notna().sum())),
            appearance_history=(
                "appearances",
                lambda values: int(pd.to_numeric(values, errors="coerce").gt(0).sum()),
            ),
            nationality_profile=(
                "nationality",
                lambda values: int(values.fillna("").astype(str).str.len().gt(0).sum()),
            ),
        )
    )
    audit["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    arguments = parse_arguments()
    load_dotenv_if_present()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131 Safari/537.36"
            )
        }
    )

    squads_html = fetch_page(
        session,
        OFFICIAL_SQUADS_URL,
        RAW_DIR / "official_2026_27_squads.html",
        arguments.offline,
    )
    numbers_html = fetch_page(
        session,
        OFFICIAL_NUMBERS_URL,
        RAW_DIR / "official_2026_27_squad_numbers.html",
        arguments.offline,
    )
    fpl_payload = load_fpl_payload(session, arguments.offline)
    fpl_player_ids = [
        int(player["id"])
        for player in fpl_payload.get("elements", [])
        if player.get("id") is not None
    ]
    print(f"开始读取FPL逐场履历：{len(fpl_player_ids)}名球员……")
    histories = load_fpl_histories(
        fpl_player_ids,
        offline=arguments.offline,
        skip_refresh=arguments.skip_player_history,
    )
    profiles = load_football_data_squads(
        session,
        arguments.offline,
        refresh_profiles=arguments.refresh_profiles,
    )

    official = parse_registered_squads(squads_html)
    numbers = parse_squad_numbers(numbers_html)
    fpl = fpl_records(fpl_payload, histories)

    if not official and not fpl:
        raise RuntimeError(
            "没有取得任何球员数据。请确认网络连接后重新运行，"
            "或先保存FPL bootstrap-static.json。"
        )

    registry = merge_sources(official, numbers, fpl)
    registry = merge_profile_fields(registry, profiles)
    registry = apply_profile_overrides(registry)
    registry = add_derived_performance_fields(registry)
    write_outputs(registry)

    print("\n英超注册球员资料库生成完成。")
    print(f"球员总数：{len(registry)}")
    print(f"官方注册验证：{int(registry['registration_verified'].sum())}")
    print(f"FPL状态与统计匹配：{int(registry['fpl_id'].notna().sum())}")
    print(f"逐场出场履历覆盖：{int(registry['appearances'].gt(0).sum())}")
    print(f"国籍/出生日期资料覆盖：{int(registry['nationality'].ne('').sum())}")
    print(f"Parquet：{OUTPUT_PARQUET}")
    print(f"CSV：{OUTPUT_CSV}")
    print(f"审计报告：{AUDIT_FILE}")
    if not official:
        print("警告：英超官方注册名单页面未解析成功，当前仅保留FPL球员池。")


if __name__ == "__main__":
    main()
