from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_FILE = PROJECT_ROOT / "data" / "context" / "prematch_context.parquet"
CONTEXT_CSV = PROJECT_ROOT / "data" / "context" / "prematch_context.csv"
OVERRIDE_FILE = PROJECT_ROOT / "data" / "manual" / "lineup_overrides.csv"
AUDIT_FILE = PROJECT_ROOT / "reports" / "manual_lineup_override_audit.csv"


def player_records(frame: pd.DataFrame, role: str) -> list[dict]:
    records = []
    for _, row in frame.iterrows():
        number = pd.to_numeric(row.get("squad_number"), errors="coerce")
        records.append(
            {
                "name": str(row.get("player_name") or "").strip(),
                "display_name": str(row.get("player_name") or "").strip(),
                "number": None if pd.isna(number) else int(number),
                "pos": str(row.get("position") or "—").strip().upper(),
                "role": role,
                "source": "manual_verified_override",
            }
        )
    return records


def main() -> None:
    if not OVERRIDE_FILE.exists():
        print(f"Manual lineup override skipped: {OVERRIDE_FILE} does not exist.")
        return
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(f"Missing context file: {CONTEXT_FILE}")
    overrides = pd.read_csv(OVERRIDE_FILE, encoding="utf-8-sig")
    required = {"match_id", "team", "role", "player_name", "position"}
    missing = required - set(overrides.columns)
    if missing:
        raise RuntimeError(f"Manual lineup file is missing columns: {sorted(missing)}")
    overrides = overrides.loc[
        overrides["player_name"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    context = pd.read_parquet(CONTEXT_FILE).copy()
    audit = []
    for match_id, match_rows in overrides.groupby("match_id", sort=False):
        target = context.index[context["match_id"].astype(str).eq(str(match_id))]
        if len(target) != 1:
            audit.append({"match_id": match_id, "status": "match_not_found"})
            continue
        index = target[0]
        home_team = str(context.at[index, "home_team"])
        away_team = str(context.at[index, "away_team"])
        valid = True
        for side, team in [("home", home_team), ("away", away_team)]:
            team_rows = match_rows.loc[
                match_rows["team"].astype(str).str.casefold().eq(team.casefold())
            ]
            starters = team_rows.loc[
                team_rows["role"].astype(str).str.casefold().eq("starter")
            ]
            substitutes = team_rows.loc[
                team_rows["role"].astype(str).str.casefold().isin(["substitute", "bench"])
            ]
            if len(starters) != 11:
                valid = False
                audit.append(
                    {"match_id": match_id, "team": team, "status": "requires_11_starters", "count": len(starters)}
                )
                continue
            context.at[index, f"{side}_starting_xi_json"] = json.dumps(
                player_records(starters, "starter"), ensure_ascii=False
            )
            context.at[index, f"{side}_substitutes_json"] = json.dumps(
                player_records(substitutes, "substitute"), ensure_ascii=False
            )
            if "formation" in team_rows and team_rows["formation"].notna().any():
                context.at[index, f"{side}_formation"] = str(
                    team_rows["formation"].dropna().iloc[0]
                )
        if valid:
            context.at[index, "lineups_confirmed"] = True
            context.at[index, "lineup_status"] = "confirmed_manual_override"
            context.at[index, "lineup_source"] = "manual_verified_override"
            context.at[index, "lineup_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
            audit.append({"match_id": match_id, "status": "applied"})
    temporary = CONTEXT_FILE.with_suffix(".tmp.parquet")
    context.to_parquet(temporary, index=False)
    os.replace(temporary, CONTEXT_FILE)
    context.to_csv(CONTEXT_CSV, index=False, encoding="utf-8-sig")
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")
    print(f"Manual lineup overrides processed. Audit: {AUDIT_FILE}")
    print("Production probabilities were not changed.")


if __name__ == "__main__":
    main()
