from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_FILE = PROJECT_ROOT / "data" / "assets" / "player_registry.parquet"
SUMMARY_FILE = PROJECT_ROOT / "reports" / "player_registry_quality.json"
TEAM_FILE = PROJECT_ROOT / "reports" / "player_registry_team_quality.csv"
CONFLICT_FILE = PROJECT_ROOT / "reports" / "player_registry_conflicts.csv"


def normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.casefold().isin({"true", "1", "yes"})


def valid_identifier(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("") & series.astype(str).str.lower().ne("nan")


def source_priority(row: pd.Series) -> int:
    if pd.notna(row.get("fpl_id")) and str(row.get("fpl_id")).strip().lower() != "nan":
        return 30
    if str(row.get("registration_verified", False)).casefold() in {"true", "1", "yes"}:
        return 20
    if pd.notna(row.get("football_data_player_id")):
        return 10
    return 1


def first_text(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def shirt_number_valid(value: Any) -> bool:
    if value is None or pd.isna(value) or str(value).strip() in {"", "—", "-"}:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(number) and number.is_integer() and 1 <= number <= 99


def main() -> None:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Player registry is missing: {REGISTRY_FILE}")
    registry = pd.read_parquet(REGISTRY_FILE).copy()
    if registry.empty:
        raise RuntimeError("Player registry is empty.")

    for column in ["team", "display_name", "registered_name", "fpl_full_name", "web_name"]:
        if column not in registry:
            registry[column] = ""
    registry["_name_key"] = registry.apply(
        lambda row: normalise(
            first_text(
                row,
                ["fpl_full_name", "registered_name", "display_name", "web_name"],
            )
        ),
        axis=1,
    )
    registry["_priority"] = registry.apply(source_priority, axis=1)
    registry["_fpl"] = valid_identifier(
        registry.get("fpl_id", pd.Series(index=registry.index, dtype=object))
    )
    registry["_registered"] = truthy(
        registry.get("registration_verified", pd.Series(False, index=registry.index))
    )
    registry["_nationality"] = valid_identifier(
        registry.get("nationality", pd.Series(index=registry.index, dtype=object))
    )
    registry["_birth_date"] = valid_identifier(
        registry.get("date_of_birth", pd.Series(index=registry.index, dtype=object))
    )
    shirt_column = next(
        (column for column in ["shirt_number", "squad_number", "number"] if column in registry),
        None,
    )
    registry["_shirt_valid"] = (
        registry[shirt_column].map(shirt_number_valid) if shirt_column else True
    )

    conflicts: list[dict] = []
    for name, group in registry.loc[registry["_name_key"].str.len().ge(4)].groupby("_name_key"):
        maximum = int(group["_priority"].max())
        current = group.loc[group["_priority"].eq(maximum)]
        teams = sorted(set(current["team"].dropna().astype(str)) - {""})
        if len(teams) > 1:
            conflicts.append(
                {
                    "player_key": name,
                    "maximum_source_priority": maximum,
                    "teams": " | ".join(teams),
                    "records": int(len(group)),
                }
            )

    team_rows = []
    for team, group in registry.groupby("team", dropna=False, sort=True):
        team_rows.append(
            {
                "team": str(team or ""),
                "records": int(len(group)),
                "unique_names": int(group["_name_key"].replace("", np.nan).nunique()),
                "fpl_matched": int(group["_fpl"].sum()),
                "officially_registered": int(group["_registered"].sum()),
                "nationality_known": int(group["_nationality"].sum()),
                "birth_date_known": int(group["_birth_date"].sum()),
                "invalid_shirt_numbers": int((~group["_shirt_valid"]).sum()),
            }
        )
    team_quality = pd.DataFrame(team_rows)
    conflict_frame = pd.DataFrame(
        conflicts,
        columns=["player_key", "maximum_source_priority", "teams", "records"],
    )
    high_priority_conflicts = int(
        conflict_frame["maximum_source_priority"].ge(30).sum()
        if not conflict_frame.empty
        else 0
    )
    invalid_numbers = int((~registry["_shirt_valid"]).sum())
    current_teams = int(registry["team"].replace("", np.nan).nunique())
    status = (
        "healthy"
        if current_teams == 20 and high_priority_conflicts == 0 and invalid_numbers == 0
        else "attention_required"
    )
    payload = {
        "status": status,
        "records": int(len(registry)),
        "unique_player_names": int(registry["_name_key"].replace("", np.nan).nunique()),
        "teams": current_teams,
        "fpl_matched": int(registry["_fpl"].sum()),
        "officially_registered": int(registry["_registered"].sum()),
        "nationality_known": int(registry["_nationality"].sum()),
        "birth_date_known": int(registry["_birth_date"].sum()),
        "invalid_shirt_numbers": invalid_numbers,
        "current_source_team_conflicts": high_priority_conflicts,
        "all_name_conflicts": int(len(conflict_frame)),
        "rules": {
            "current_fpl_priority": 30,
            "official_registration_priority": 20,
            "football_data_profile_priority": 10,
            "fallback_priority": 1,
        },
        "production_model_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    team_quality.to_csv(TEAM_FILE, index=False, encoding="utf-8-sig")
    conflict_frame.to_csv(CONFLICT_FILE, index=False, encoding="utf-8-sig")
    print("Player registry quality audit completed.")
    print(f"Status: {status}")
    print(f"Teams: {current_teams}; records: {len(registry)}")
    print(f"Current-source team conflicts: {high_priority_conflicts}")
    print(f"Invalid shirt numbers: {invalid_numbers}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
