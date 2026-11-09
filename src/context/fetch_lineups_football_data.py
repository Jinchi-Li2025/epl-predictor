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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FILE = PROJECT_ROOT / "data" / "context" / "prematch_context.parquet"
CONTEXT_CSV_FILE = PROJECT_ROOT / "data" / "context" / "prematch_context.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "football_data" / "lineups"
AUDIT_FILE = PROJECT_ROOT / "reports" / "football_data_lineup_audit.csv"
MATCHES_URL = "https://api.football-data.org/v4/competitions/PL/matches"
MATCH_URL = "https://api.football-data.org/v4/matches/{match_id}"
POLL_START_MINUTES = 150
POLL_END_MINUTES = -360

TEAM_ALIASES = {
    "bournemouth": "afc bournemouth",
    "brighton hove": "brighton hove albion",
    "brighton and hove albion": "brighton hove albion",
    "man city": "manchester city",
    "man united": "manchester united",
    "man utd": "manchester united",
    "newcastle": "newcastle united",
    "nott m forest": "nottingham forest",
    "spurs": "tottenham hotspur",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use football-data.org as an official-lineup fallback."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Query every context fixture, ignoring the normal kickoff window.",
    )
    return parser.parse_args()


def load_dotenv() -> None:
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(), value.strip().strip('"').strip("'")
        )


def team_key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    text = re.sub(r"\b(?:fc|afc)\b", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return TEAM_ALIASES.get(text, text)


def position_code(value: Any) -> str:
    text = str(value or "").casefold()
    if "goal" in text:
        return "G"
    if any(word in text for word in ["back", "defence", "defender"]):
        return "D"
    if any(word in text for word in ["midfield", "winger"]):
        return "M"
    if any(word in text for word in ["forward", "striker"]):
        return "F"
    return "—"


def normalise_players(items: list[dict[str, Any]], role: str) -> list[dict]:
    output = []
    for item in items or []:
        output.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or "Unknown player",
                "display_name": item.get("name") or "Unknown player",
                "number": item.get("shirtNumber"),
                "pos": position_code(item.get("position")),
                "position_detail": item.get("position") or "",
                "role": role,
                "source": "football-data.org",
            }
        )
    return output


def find_match(row: pd.Series, matches: list[dict]) -> dict | None:
    home_key = team_key(row.get("home_team"))
    away_key = team_key(row.get("away_team"))
    kickoff = pd.to_datetime(row.get("kickoff_utc"), utc=True, errors="coerce")
    best: tuple[float, dict] | None = None
    for match in matches:
        provider_home = team_key((match.get("homeTeam") or {}).get("name"))
        provider_away = team_key((match.get("awayTeam") or {}).get("name"))
        if provider_home != home_key or provider_away != away_key:
            continue
        provider_kickoff = pd.to_datetime(
            match.get("utcDate"), utc=True, errors="coerce"
        )
        difference = (
            abs((provider_kickoff - kickoff).total_seconds())
            if pd.notna(provider_kickoff) and pd.notna(kickoff)
            else 0.0
        )
        if difference <= 12 * 3600 and (best is None or difference < best[0]):
            best = (difference, match)
    return None if best is None else best[1]


def lineup_changes(projected_text: Any, official: list[dict]) -> dict:
    try:
        projected = json.loads(str(projected_text or "[]"))
    except (TypeError, json.JSONDecodeError):
        projected = []
    projected_names = {team_key(item.get("name")) for item in projected}
    official_names = {team_key(item.get("name")) for item in official}
    return {
        "predicted_not_starting": sorted(projected_names - official_names),
        "unexpected_starters": sorted(official_names - projected_names),
        "change_count": len(projected_names - official_names),
    }


def save_context(context: pd.DataFrame) -> None:
    context["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = CONTEXT_FILE.with_suffix(".tmp.parquet")
    context.to_parquet(temporary, index=False)
    os.replace(temporary, CONTEXT_FILE)
    context.to_csv(CONTEXT_CSV_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    arguments = parse_arguments()
    load_dotenv()
    api_key = (
        os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
        or os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
    )
    if not api_key:
        print("football-data lineup fallback skipped: API key is missing.")
        return
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(f"Missing context file: {CONTEXT_FILE}")

    context = pd.read_parquet(CONTEXT_FILE).copy()
    context["kickoff_utc"] = pd.to_datetime(
        context["kickoff_utc"], utc=True, errors="coerce"
    )
    now = pd.Timestamp.now(tz="UTC")
    due_mask = context["kickoff_utc"].map(
        lambda kickoff: (
            False
            if pd.isna(kickoff)
            else POLL_END_MINUTES
            <= (kickoff - now).total_seconds() / 60.0
            <= POLL_START_MINUTES
        )
    )
    due = context if arguments.force else context.loc[due_mask]
    if due.empty:
        print("football-data lineup fallback: no fixture is in the query window.")
        return

    date_from = due["kickoff_utc"].min().date().isoformat()
    date_to = due["kickoff_utc"].max().date().isoformat()
    session = requests.Session()
    session.headers.update({"X-Auth-Token": api_key})
    response = session.get(
        MATCHES_URL,
        params={"dateFrom": date_from, "dateTo": date_to},
        timeout=45,
    )
    response.raise_for_status()
    matches = response.json().get("matches", [])
    audit_rows = []
    confirmed_count = 0

    for index, row in due.iterrows():
        matched = find_match(row, matches)
        status = "match_not_found"
        provider_id = None
        if matched is not None:
            provider_id = matched.get("id")
            detail_response = session.get(
                MATCH_URL.format(match_id=provider_id), timeout=45
            )
            detail_response.raise_for_status()
            payload = detail_response.json()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / f"match_{provider_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            home = payload.get("homeTeam") or {}
            away = payload.get("awayTeam") or {}
            home_xi = normalise_players(home.get("lineup", []), "starter")
            away_xi = normalise_players(away.get("lineup", []), "starter")
            home_bench = normalise_players(home.get("bench", []), "substitute")
            away_bench = normalise_players(away.get("bench", []), "substitute")

            context.at[index, "lineup_checked_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            context.at[index, "lineup_source"] = "football-data.org"
            if home_xi:
                context.at[index, "home_starting_xi_json"] = json.dumps(
                    home_xi, ensure_ascii=False
                )
                context.at[index, "home_substitutes_json"] = json.dumps(
                    home_bench, ensure_ascii=False
                )
                context.at[index, "home_formation"] = home.get("formation")
                context.at[index, "home_coach"] = (home.get("coach") or {}).get(
                    "name"
                )
            if away_xi:
                context.at[index, "away_starting_xi_json"] = json.dumps(
                    away_xi, ensure_ascii=False
                )
                context.at[index, "away_substitutes_json"] = json.dumps(
                    away_bench, ensure_ascii=False
                )
                context.at[index, "away_formation"] = away.get("formation")
                context.at[index, "away_coach"] = (away.get("coach") or {}).get(
                    "name"
                )

            if len(home_xi) == 11 and len(away_xi) == 11:
                status = "confirmed"
                context.at[index, "lineups_confirmed"] = True
                context.at[index, "lineup_status"] = "confirmed"
                context.at[index, "lineup_explanation"] = (
                    "football-data.org official starting XI and bench confirmed; "
                    "the frozen probabilities remain unchanged."
                )
                for side, official in [("home", home_xi), ("away", away_xi)]:
                    changes = lineup_changes(
                        context.at[index, f"{side}_predicted_xi_json"], official
                    )
                    context.at[index, f"{side}_lineup_changes_json"] = json.dumps(
                        changes, ensure_ascii=False
                    )
                    context.at[index, f"{side}_rotation_index"] = round(
                        changes["change_count"] / 11.0, 4
                    )
                confirmed_count += 1
            elif home_xi or away_xi:
                status = "partial"
            else:
                status = "not_published"

        audit_rows.append(
            {
                "match_id": row.get("match_id"),
                "kickoff_utc": row.get("kickoff_utc"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "football_data_match_id": provider_id,
                "status": status,
            }
        )

    save_context(context)
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        AUDIT_FILE, index=False, encoding="utf-8-sig"
    )
    print("football-data lineup fallback completed.")
    print(f"Fixtures checked: {len(due)}")
    print(f"Confirmed lineups: {confirmed_count}")
    print("Published probabilities were not modified.")


if __name__ == "__main__":
    main()
