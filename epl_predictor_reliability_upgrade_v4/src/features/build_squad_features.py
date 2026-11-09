from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_FILE = PROJECT_ROOT / "data" / "assets" / "player_registry.parquet"
PREMATCH_FILE = PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet"
TEAM_OUTPUT = PROJECT_ROOT / "data" / "processed" / "team_squad_features.parquet"
MATCH_OUTPUT = PROJECT_ROOT / "data" / "processed" / "upcoming_squad_features.parquet"
AUDIT_OUTPUT = PROJECT_ROOT / "reports" / "squad_feature_audit.csv"


NUMERIC_COLUMNS = [
    "minutes",
    "starts",
    "appearances",
    "substitute_appearances",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "form",
    "total_points",
]


def safe_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column not in output.columns:
            output[column] = 0.0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    if "chance_of_playing_next_round" not in output.columns:
        output["chance_of_playing_next_round"] = np.nan
    output["chance_of_playing_next_round"] = pd.to_numeric(
        output["chance_of_playing_next_round"], errors="coerce"
    )
    return output


def availability_weight(row: pd.Series) -> float:
    status = str(row.get("availability_code") or "u").lower()
    chance = row.get("chance_of_playing_next_round")
    if pd.notna(chance):
        return float(np.clip(float(chance) / 100.0, 0.0, 1.0))
    return {
        "a": 1.0,
        "d": 0.55,
        "i": 0.0,
        "n": 0.0,
        "s": 0.0,
        "u": 0.35,
    }.get(status, 0.35)


def aggregate_team(team_players: pd.DataFrame) -> dict:
    players = safe_numeric(team_players)
    players["availability_weight"] = players.apply(availability_weight, axis=1)
    players["weighted_form"] = players["form"] * players["availability_weight"]
    players["weighted_xgi"] = (
        players["expected_goal_involvements"] * players["availability_weight"]
    )
    players["weighted_points"] = (
        players["total_points"] * players["availability_weight"]
    )

    regulars = players.sort_values(
        ["starts", "minutes", "total_points"],
        ascending=False,
        kind="stable",
    ).head(11)
    ordered_players = players.sort_values(
        ["starts", "minutes", "total_points"],
        ascending=False,
        kind="stable",
    )
    bench = ordered_players.iloc[11:20]
    status = players.get(
        "availability_code", pd.Series("u", index=players.index)
    ).astype(str)
    category = players.get(
        "registration_category", pd.Series("", index=players.index)
    ).astype(str)
    position = players.get(
        "position_code", pd.Series("", index=players.index)
    ).astype(str)

    return {
        "team": str(players.iloc[0]["team"]),
        "registered_player_count": int(len(players)),
        "verified_registered_count": int(
            players.get(
                "registration_verified", pd.Series(False, index=players.index)
            ).fillna(False).astype(bool).sum()
        ),
        "senior_registered_count": int(category.eq("senior_25").sum()),
        "u21_registered_count": int(category.eq("registered_u21").sum()),
        "available_player_count": int(status.eq("a").sum()),
        "doubtful_player_count": int(status.eq("d").sum()),
        "unavailable_player_count": int(status.isin(["i", "n", "s"]).sum()),
        "goalkeeper_depth": int(position.eq("G").sum()),
        "defender_depth": int(position.eq("D").sum()),
        "midfielder_depth": int(position.eq("M").sum()),
        "forward_depth": int(position.eq("F").sum()),
        "squad_minutes": float(players["minutes"].sum()),
        "squad_starts": float(players["starts"].sum()),
        "squad_appearances": float(players["appearances"].sum()),
        "squad_substitute_appearances": float(
            players["substitute_appearances"].sum()
        ),
        "squad_goals": float(players["goals_scored"].sum()),
        "squad_assists": float(players["assists"].sum()),
        "squad_expected_goals": float(players["expected_goals"].sum()),
        "squad_expected_assists": float(players["expected_assists"].sum()),
        "availability_weighted_form": float(players["weighted_form"].sum()),
        "availability_weighted_xgi": float(players["weighted_xgi"].sum()),
        "availability_weighted_points": float(players["weighted_points"].sum()),
        "probable_xi_minutes": float(regulars["minutes"].sum()),
        "probable_xi_starts": float(regulars["starts"].sum()),
        "probable_xi_form": float(regulars["weighted_form"].sum()),
        "probable_xi_xgi": float(regulars["weighted_xgi"].sum()),
        "probable_xi_availability": float(
            regulars["availability_weight"].mean() if len(regulars) else 0.0
        ),
        "bench_player_count": int(len(bench)),
        "bench_minutes": float(bench["minutes"].sum()),
        "bench_form": float(bench["weighted_form"].sum()),
        "bench_xgi": float(bench["weighted_xgi"].sum()),
        "bench_points": float(bench["weighted_points"].sum()),
        "bench_availability": float(
            bench["availability_weight"].mean() if len(bench) else 0.0
        ),
        "squad_absence_burden": float(
            (1.0 - players["availability_weight"]).sum()
        ),
        "selection_stability": float(
            regulars["starts"].sum() / max(players["starts"].sum(), 1.0)
        ),
    }


def build_match_features(team_features: pd.DataFrame) -> pd.DataFrame:
    if not PREMATCH_FILE.exists():
        return pd.DataFrame()
    fixtures = pd.read_parquet(PREMATCH_FILE)
    required = [column for column in ["match_id", "kickoff_utc", "home_team", "away_team"] if column in fixtures.columns]
    fixtures = fixtures[required].drop_duplicates("match_id" if "match_id" in required else None)

    feature_columns = [column for column in team_features.columns if column != "team"]
    home = team_features.rename(
        columns={"team": "home_team", **{column: f"home_{column}" for column in feature_columns}}
    )
    away = team_features.rename(
        columns={"team": "away_team", **{column: f"away_{column}" for column in feature_columns}}
    )
    output = fixtures.merge(home, on="home_team", how="left").merge(
        away, on="away_team", how="left"
    )
    for column in feature_columns:
        home_column = f"home_{column}"
        away_column = f"away_{column}"
        if home_column in output.columns and away_column in output.columns:
            output[f"squad_diff_{column}"] = (
                pd.to_numeric(output[home_column], errors="coerce")
                - pd.to_numeric(output[away_column], errors="coerce")
            )
    output["production_model_input"] = False
    output["feature_status"] = "candidate_requires_backtest"
    return output


def main() -> None:
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"找不到球员资料库：{REGISTRY_FILE}\n"
            "请先运行 src/assets/build_player_registry.py。"
        )

    registry = pd.read_parquet(REGISTRY_FILE)
    if registry.empty:
        raise RuntimeError("球员资料库为空。")

    team_rows = [
        aggregate_team(group)
        for _, group in registry.groupby("team", sort=True)
        if not group.empty
    ]
    team_features = pd.DataFrame(team_rows)
    match_features = build_match_features(team_features)

    TEAM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    team_features.to_parquet(TEAM_OUTPUT, index=False)
    team_features.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8-sig")
    if not match_features.empty:
        match_features.to_parquet(MATCH_OUTPUT, index=False)

    print("\n球队阵容候选特征生成完成。")
    print(f"球队数量：{len(team_features)}")
    print(f"球队特征：{TEAM_OUTPUT}")
    if not match_features.empty:
        print(f"未来比赛候选特征：{MATCH_OUTPUT}")
    print("安全边界：这些字段尚未接入生产概率，必须先通过时间序列回测。")


if __name__ == "__main__":
    main()
