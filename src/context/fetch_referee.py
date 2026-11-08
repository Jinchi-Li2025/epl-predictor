from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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

LIVE_FIXTURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "football_data_epl_current.parquet"
)

REFEREE_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "referee_context_audit.csv"
)

CONTEXT_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_context_audit.csv"
)


def clean_optional_text(
    value: Any,
) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.lower() in {
        "none",
        "nan",
        "<na>",
    }:
        return None

    return text


def fixture_key(
    home_team: str,
    away_team: str,
) -> tuple[str, str]:
    return (
        str(home_team).strip(),
        str(away_team).strip(),
    )


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到上下文文件：{CONTEXT_FILE}"
        )

    if not LIVE_FIXTURES_FILE.exists():
        raise FileNotFoundError(
            f"找不到实时赛程文件："
            f"{LIVE_FIXTURES_FILE}"
        )

    context = pd.read_parquet(
        CONTEXT_FILE
    )

    live_fixtures = pd.read_parquet(
        LIVE_FIXTURES_FILE
    )

    required_columns = {
        "home_team",
        "away_team",
        "referee",
        "provider_status",
    }

    missing_columns = (
        required_columns
        - set(live_fixtures.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "实时赛程文件缺少字段："
            f"{sorted(missing_columns)}"
        )

    if live_fixtures.duplicated(
        ["home_team", "away_team"]
    ).any():
        raise RuntimeError(
            "实时赛程中存在重复主客队组合。"
        )

    referee_lookup = {}

    for _, fixture in live_fixtures.iterrows():
        key = fixture_key(
            fixture["home_team"],
            fixture["away_team"],
        )

        referee_lookup[key] = {
            "referee": clean_optional_text(
                fixture["referee"]
            ),
            "provider_status": (
                clean_optional_text(
                    fixture["provider_status"]
                )
            ),
        }

    checked_at = pd.Timestamp.now(
        tz="UTC"
    )

    audit_rows: list[dict[str, Any]] = []

    print("开始同步已公布的裁判安排……")

    for index, row in context.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        print(
            f"{index + 1}/{len(context)} "
            f"{home_team} vs {away_team}"
        )

        key = fixture_key(
            home_team,
            away_team,
        )

        fixture_information = (
            referee_lookup.get(key)
        )

        audit_row = {
            "match_id": row["match_id"],
            "home_team": home_team,
            "away_team": away_team,
            "kickoff_utc": row["kickoff_utc"],
            "provider_status": None,
            "referee_name": None,
            "assignment_status": None,
            "cards_per_match_available": False,
            "penalties_per_match_available": False,
            "checked_at_utc": (
                checked_at.isoformat()
            ),
        }

        if fixture_information is None:
            audit_row["assignment_status"] = (
                "match_not_found"
            )
            audit_rows.append(audit_row)
            continue

        referee_name = fixture_information[
            "referee"
        ]

        audit_row["provider_status"] = (
            fixture_information[
                "provider_status"
            ]
        )

        if referee_name is None:
            audit_row["assignment_status"] = (
                "not_announced"
            )
            audit_rows.append(audit_row)
            continue

        context.at[
            index,
            "referee_name",
        ] = referee_name

        # 免费数据源没有可靠的裁判场均牌数
        # 和场均点球数，因此保持缺失值。
        context.at[
            index,
            "referee_cards_per_match",
        ] = pd.NA

        context.at[
            index,
            "referee_penalties_per_match",
        ] = pd.NA

        context.at[
            index,
            "referee_source",
        ] = (
            "football-data.org v4 "
            "(assignment only)"
        )

        context.at[
            index,
            "referee_checked_at_utc",
        ] = checked_at.isoformat()

        context.at[
            index,
            "updated_at_utc",
        ] = checked_at.isoformat()

        audit_row["referee_name"] = (
            referee_name
        )
        audit_row["assignment_status"] = (
            "assigned"
        )

        audit_rows.append(audit_row)

    weather_complete = (
        context["temperature_c"].notna()
        & context["wind_speed_kph"].notna()
        & context["weather_source"].notna()
    )

    referee_complete = (
        context["referee_name"].notna()
        & context["referee_source"].notna()
    )

    completed_sections = (
        weather_complete.astype(int)
        + referee_complete.astype(int)
    )

    context["context_confidence"] = (
        completed_sections / 7
    )

    context.loc[
        completed_sections > 0,
        "context_status",
    ] = "partial"

    context.loc[
        completed_sections == 0,
        "context_status",
    ] = "pending"

    context.to_parquet(
        CONTEXT_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    referee_audit = pd.DataFrame(
        audit_rows
    )

    referee_audit.to_csv(
        REFEREE_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    context_audit = context[
        [
            "home_team",
            "away_team",
        ]
    ].copy()

    context_audit["completed_sections"] = (
        completed_sections
    )
    context_audit["total_sections"] = 7
    context_audit["completeness"] = (
        completed_sections / 7
    )

    context_audit.to_csv(
        CONTEXT_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    assigned_count = int(
        referee_audit[
            "assignment_status"
        ]
        .eq("assigned")
        .sum()
    )

    unannounced_count = int(
        referee_audit[
            "assignment_status"
        ]
        .eq("not_announced")
        .sum()
    )

    missing_count = int(
        referee_audit[
            "assignment_status"
        ]
        .eq("match_not_found")
        .sum()
    )

    print("\n裁判上下文更新完成。")
    print(f"比赛数量：{len(context)}")
    print(f"已公布裁判：{assigned_count}")
    print(f"尚未公布裁判：{unannounced_count}")
    print(f"未匹配比赛：{missing_count}")
    print(
        "说明：场均牌数和场均点球数"
        "因免费源不可用而保持空值。"
    )
    print(f"上下文文件：{CONTEXT_FILE}")
    print(
        f"裁判检查报告："
        f"{REFEREE_AUDIT_FILE}"
    )

    print("\n裁判安排：")
    print(
        context[
            [
                "home_team",
                "away_team",
                "referee_name",
                "referee_source",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()