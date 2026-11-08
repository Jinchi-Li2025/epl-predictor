from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


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

WEATHER_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "weather_context_audit.csv"
)

CONTEXT_AUDIT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prematch_context_audit.csv"
)

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

MAX_FORECAST_DAYS = 16
class ForecastUnavailableError(RuntimeError):
    """天气接口尚未提供该远期时段的完整数值。"""


# 球场坐标仅用于天气网格定位。
STADIUMS = {
    "Arsenal": {
        "venue": "Emirates Stadium",
        "latitude": 51.5549,
        "longitude": -0.1084,
    },
    "Aston Villa": {
        "venue": "Villa Park",
        "latitude": 52.5092,
        "longitude": -1.8848,
    },
    "AFC Bournemouth": {
        "venue": "Vitality Stadium",
        "latitude": 50.7352,
        "longitude": -1.8383,
    },
    "Brentford": {
        "venue": "Gtech Community Stadium",
        "latitude": 51.4908,
        "longitude": -0.2887,
    },
    "Brighton & Hove Albion": {
        "venue": "American Express Stadium",
        "latitude": 50.8616,
        "longitude": -0.0837,
    },
    "Chelsea": {
        "venue": "Stamford Bridge",
        "latitude": 51.4817,
        "longitude": -0.1910,
    },
    "Coventry City": {
        "venue": "Coventry Building Society Arena",
        "latitude": 52.4481,
        "longitude": -1.4956,
    },
    "Crystal Palace": {
        "venue": "Selhurst Park",
        "latitude": 51.3983,
        "longitude": -0.0856,
    },
    "Everton": {
        "venue": "Hill Dickinson Stadium",
        "latitude": 53.4252,
        "longitude": -3.0030,
    },
    "Fulham": {
        "venue": "Craven Cottage",
        "latitude": 51.4749,
        "longitude": -0.2217,
    },
    "Hull City": {
        "venue": "MKM Stadium",
        "latitude": 53.7461,
        "longitude": -0.3676,
    },
    "Ipswich Town": {
        "venue": "Portman Road",
        "latitude": 52.0549,
        "longitude": 1.1455,
    },
    "Leeds United": {
        "venue": "Elland Road",
        "latitude": 53.7778,
        "longitude": -1.5721,
    },
    "Liverpool": {
        "venue": "Anfield",
        "latitude": 53.4308,
        "longitude": -2.9608,
    },
    "Manchester City": {
        "venue": "Etihad Stadium",
        "latitude": 53.4831,
        "longitude": -2.2004,
    },
    "Manchester United": {
        "venue": "Old Trafford",
        "latitude": 53.4631,
        "longitude": -2.2913,
    },
    "Newcastle United": {
        "venue": "St James' Park",
        "latitude": 54.9752,
        "longitude": -1.6224,
    },
    "Nottingham Forest": {
        "venue": "City Ground",
        "latitude": 52.9399,
        "longitude": -1.1326,
    },
    "Sunderland": {
        "venue": "Stadium of Light",
        "latitude": 54.9146,
        "longitude": -1.3884,
    },
    "Tottenham Hotspur": {
        "venue": "Tottenham Hotspur Stadium",
        "latitude": 51.6043,
        "longitude": -0.0664,
    },
}


WEATHER_DESCRIPTIONS = {
    0: "晴朗",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "中等毛毛雨",
    55: "强毛毛雨",
    56: "轻微冻雨",
    57: "强冻雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "轻微冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "强阵雨",
    85: "小阵雪",
    86: "强阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴强冰雹",
}


def numeric_value(
    values: list[Any] | None,
    position: int,
) -> float | None:
    if values is None:
        return None

    if position >= len(values):
        return None

    value = values[position]

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def request_forecast(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": (
                    "temperature_2m,"
                    "precipitation_probability,"
                    "precipitation,"
                    "wind_speed_10m,"
                    "wind_gusts_10m,"
                    "weather_code"
                ),
                "timezone": "UTC",
                "forecast_days": MAX_FORECAST_DAYS,
                "temperature_unit": "celsius",
                "precipitation_unit": "mm",
                "wind_speed_unit": "kmh",
            },
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"天气接口连接失败：{error}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            "天气接口没有返回有效JSON。"
        ) from error

    if not response.ok:
        raise RuntimeError(
            f"天气接口返回HTTP {response.status_code}："
            f"{payload}"
        )

    if payload.get("error"):
        raise RuntimeError(
            f"天气接口返回错误：{payload}"
        )

    return payload


def extract_kickoff_weather(
    payload: dict[str, Any],
    kickoff_utc: pd.Timestamp,
) -> dict[str, Any]:
    hourly = payload.get("hourly") or {}
    time_values = hourly.get("time") or []

    if not time_values:
        raise RuntimeError(
            "天气接口返回结果中没有逐小时时间。"
        )

    forecast_times = pd.to_datetime(
        time_values,
        utc=True,
        errors="coerce",
    )

    valid_positions = [
        position
        for position, value in enumerate(
            forecast_times
        )
        if not pd.isna(value)
    ]

    if not valid_positions:
        raise RuntimeError(
            "天气接口返回的时间无法解析。"
        )

    nearest_position = min(
        valid_positions,
        key=lambda position: abs(
            forecast_times[position]
            - kickoff_utc
        ),
    )

    matched_time = forecast_times[
        nearest_position
    ]

    time_difference_minutes = abs(
        (
            matched_time - kickoff_utc
        ).total_seconds()
    ) / 60

    if time_difference_minutes > 90:
        raise RuntimeError(
            "没有找到接近开球时间的逐小时天气。"
        )

    temperature = numeric_value(
        hourly.get("temperature_2m"),
        nearest_position,
    )

    precipitation_probability = numeric_value(
        hourly.get("precipitation_probability"),
        nearest_position,
    )

    precipitation = numeric_value(
        hourly.get("precipitation"),
        nearest_position,
    )

    wind_speed = numeric_value(
        hourly.get("wind_speed_10m"),
        nearest_position,
    )

    wind_gust = numeric_value(
        hourly.get("wind_gusts_10m"),
        nearest_position,
    )

    weather_code_value = numeric_value(
        hourly.get("weather_code"),
        nearest_position,
    )

    required_values = [
        temperature,
        precipitation,
        wind_speed,
        weather_code_value,
    ]

    if any(
        value is None
        for value in required_values
    ):
        raise ForecastUnavailableError(
            "该比赛距离开球时间较远，"
            "天气模型尚未提供完整预报数值。"
        )

    weather_code = int(weather_code_value)
    base_description = (
        WEATHER_DESCRIPTIONS.get(
            weather_code,
            f"天气代码{weather_code}",
        )
    )

    description_parts = [base_description]

    if precipitation_probability is not None:
        description_parts.append(
            "降水概率"
            f"{precipitation_probability:.0f}%"
        )

    if wind_gust is not None:
        description_parts.append(
            f"阵风{wind_gust:.1f}km/h"
        )

    return {
        "matched_forecast_time": matched_time,
        "temperature_c": temperature,
        "precipitation_probability": (
            precipitation_probability
        ),
        "precipitation_mm": precipitation,
        "wind_speed_kph": wind_speed,
        "wind_gust_kph": wind_gust,
        "weather_code": weather_code,
        "weather_description": "；".join(
            description_parts
        ),
    }


def safe_confidence(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"找不到上下文文件：{CONTEXT_FILE}"
        )

    context = pd.read_parquet(CONTEXT_FILE)

    if context.empty:
        raise RuntimeError(
            "上下文文件为空。"
        )

    context["kickoff_utc"] = pd.to_datetime(
        context["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    checked_at = pd.Timestamp.now(tz="UTC")
    maximum_forecast_time = (
        checked_at
        + pd.Timedelta(
            days=MAX_FORECAST_DAYS
        )
    )

    audit_rows: list[dict[str, Any]] = []

    print("开始获取比赛地逐小时天气……")

    for index, row in context.iterrows():
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])
        kickoff_utc = row["kickoff_utc"]

        print(
            f"{index + 1}/{len(context)} "
            f"{home_team} vs {away_team}"
        )

        stadium = STADIUMS.get(home_team)

        audit_row: dict[str, Any] = {
            "match_id": row["match_id"],
            "home_team": home_team,
            "away_team": away_team,
            "kickoff_utc": kickoff_utc,
            "hours_to_kickoff": (
                kickoff_utc - checked_at
            ).total_seconds() / 3600,
            "status": None,
            "venue_name": None,
            "matched_forecast_time": None,
            "temperature_c": None,
            "precipitation_probability": None,
            "precipitation_mm": None,
            "wind_speed_kph": None,
            "wind_gust_kph": None,
            "error": None,
        }

        if stadium is None:
            audit_row["status"] = (
                "unknown_home_team"
            )
            audit_row["error"] = (
                "没有找到主队球场坐标"
            )
            audit_rows.append(audit_row)
            continue

        context.at[
            index,
            "venue_name",
        ] = stadium["venue"]

        audit_row["venue_name"] = (
            stadium["venue"]
        )

        if kickoff_utc <= checked_at:
            audit_row["status"] = (
                "already_started"
            )
            audit_rows.append(audit_row)
            continue

        if kickoff_utc > maximum_forecast_time:
            audit_row["status"] = (
                "outside_forecast_range"
            )
            audit_rows.append(audit_row)
            continue

        try:
            payload = request_forecast(
                latitude=stadium["latitude"],
                longitude=stadium["longitude"],
            )

            weather = extract_kickoff_weather(
                payload=payload,
                kickoff_utc=kickoff_utc,
            )
        except ForecastUnavailableError as error:
            audit_row["status"] = (
                "forecast_values_unavailable"
            )
            audit_row["error"] = str(error)
            audit_rows.append(audit_row)
            continue
        except RuntimeError as error:
            audit_row["status"] = "api_error"
            audit_row["error"] = str(error)
            audit_rows.append(audit_row)
            continue

        context.at[
            index,
            "temperature_c",
        ] = weather["temperature_c"]

        context.at[
            index,
            "precipitation_mm",
        ] = weather["precipitation_mm"]

        context.at[
            index,
            "wind_speed_kph",
        ] = weather["wind_speed_kph"]

        context.at[
            index,
            "weather_code",
        ] = weather["weather_code"]

        context.at[
            index,
            "weather_description",
        ] = weather["weather_description"]

        context.at[
            index,
            "weather_source",
        ] = "Open-Meteo Forecast API"

        context.at[
            index,
            "weather_checked_at_utc",
        ] = checked_at.isoformat()

        context.at[
            index,
            "context_status",
        ] = "partial"

        existing_confidence = safe_confidence(
            context.at[
                index,
                "context_confidence",
            ]
        )

        context.at[
            index,
            "context_confidence",
        ] = max(
            existing_confidence,
            1 / 7,
        )

        context.at[
            index,
            "updated_at_utc",
        ] = checked_at.isoformat()

        audit_row.update(weather)
        audit_row["status"] = "available"
        audit_rows.append(audit_row)

        time.sleep(0.2)

    context.to_parquet(
        CONTEXT_FILE,
        index=False,
    )

    context.to_csv(
        CONTEXT_CSV_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    weather_audit = pd.DataFrame(
        audit_rows
    )

    weather_audit.to_csv(
        WEATHER_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    weather_complete = (
        context["temperature_c"].notna()
        & context["wind_speed_kph"].notna()
        & context["weather_source"].notna()
    )

    context_audit = context[
        [
            "home_team",
            "away_team",
        ]
    ].copy()

    context_audit["completed_sections"] = (
        weather_complete.astype(int)
    )
    context_audit["total_sections"] = 7
    context_audit["completeness"] = (
        context_audit["completed_sections"]
        / context_audit["total_sections"]
    )

    context_audit.to_csv(
        CONTEXT_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    available_count = int(
        weather_audit["status"]
        .eq("available")
        .sum()
    )

    outside_count = int(
        weather_audit["status"]
        .eq("outside_forecast_range")
        .sum()
    )

    unavailable_count = int(
        weather_audit["status"]
        .eq("forecast_values_unavailable")
        .sum()
    )

    error_count = int(
        weather_audit["status"]
        .isin(
            [
                "api_error",
                "unknown_home_team",
            ]
        )
        .sum()
    )

    print("\n天气上下文更新完成。")
    print(f"比赛数量：{len(context)}")
    print(f"成功获得天气：{available_count}")
    print(f"超出预报范围：{outside_count}")
    print(
        "尚未提供完整预报："
        f"{unavailable_count}"
    )
    print(f"错误数量：{error_count}")
    print(f"上下文文件：{CONTEXT_FILE}")
    print(f"天气检查报告：{WEATHER_AUDIT_FILE}")

    display_columns = [
        "home_team",
        "away_team",
        "venue_name",
        "temperature_c",
        "precipitation_mm",
        "wind_speed_kph",
        "weather_description",
    ]

    print("\n比赛天气：")
    print(
        context[
            display_columns
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()