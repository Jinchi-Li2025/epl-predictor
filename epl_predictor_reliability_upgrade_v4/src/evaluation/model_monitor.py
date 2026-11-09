from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTLED_FILE = PROJECT_ROOT / "data" / "history" / "settled_predictions.parquet"
VALIDATION_FILE = PROJECT_ROOT / "reports" / "current_external_validation_predictions.parquet"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "model_monitor.json"
CALIBRATION_FILE = PROJECT_ROOT / "reports" / "model_monitor_calibration.csv"
ALERTS_FILE = PROJECT_ROOT / "reports" / "model_monitor_alerts.csv"


def load_observations() -> tuple[pd.DataFrame, str]:
    if SETTLED_FILE.exists():
        data = pd.read_parquet(SETTLED_FILE).copy()
        mapping = {
            "settled_home_probability": "p_home",
            "settled_draw_probability": "p_draw",
            "settled_away_probability": "p_away",
            "actual_result": "actual_result",
            "actual_kickoff_utc": "kickoff_utc",
        }
        if not data.empty and all(column in data for column in mapping):
            return data.rename(columns=mapping), "published_settled_predictions"
    if VALIDATION_FILE.exists():
        return pd.read_parquet(VALIDATION_FILE).copy(), "external_validation"
    return pd.DataFrame(), "none"


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    targets = np.eye(3)[labels]
    return {
        "log_loss": float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
        "brier_score": float(np.square(clipped - targets).sum(axis=1).mean()),
        "accuracy": float((clipped.argmax(axis=1) == labels).mean()),
    }


def main() -> None:
    data, source = load_observations()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if data.empty:
        payload = {
            "status": "insufficient_data",
            "reason": "No settled or external-validation observations are available.",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "production_model_changed": False,
        }
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(payload["reason"])
        return

    required = ["p_home", "p_draw", "p_away", "actual_result"]
    if "kickoff_utc" in data:
        data["kickoff_utc"] = pd.to_datetime(
            data["kickoff_utc"], utc=True, errors="coerce"
        )
        data = data.sort_values("kickoff_utc", na_position="first")
    data = data.dropna(subset=[column for column in required if column in data]).copy()
    labels = data["actual_result"].astype(str).str.upper().map({"H": 0, "D": 1, "A": 2})
    data = data.loc[labels.notna()].copy()
    labels = labels.loc[labels.notna()].astype(int).to_numpy()
    probabilities = data[["p_home", "p_draw", "p_away"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(probabilities).all(axis=1) & (probabilities.sum(axis=1) > 0)
    data = data.loc[valid].copy()
    probabilities = probabilities[valid]
    labels = labels[valid]
    if len(data) == 0:
        raise RuntimeError("No valid probability rows remain after validation.")

    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bins = np.linspace(0.0, 1.0, 11)
    calibration_rows = []
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & (
            confidence <= upper if upper == 1.0 else confidence < upper
        )
        if not mask.any():
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(mean_confidence - accuracy)
        calibration_rows.append(
            {
                "confidence_bin": f"{lower:.1f}-{upper:.1f}",
                "matches": int(mask.sum()),
                "mean_confidence": mean_confidence,
                "actual_accuracy": accuracy,
                "calibration_gap": mean_confidence - accuracy,
            }
        )

    overall = probability_metrics(probabilities, labels)
    recent_count = min(25, len(data) // 2)
    degradation = None
    if recent_count >= 10:
        previous = probability_metrics(
            probabilities[-2 * recent_count : -recent_count],
            labels[-2 * recent_count : -recent_count],
        )
        recent = probability_metrics(probabilities[-recent_count:], labels[-recent_count:])
        degradation = recent["log_loss"] - previous["log_loss"]
    else:
        previous = recent = {}

    alerts = []
    if len(data) < 30:
        alerts.append({"severity": "info", "code": "SMALL_SAMPLE", "message": "Fewer than 30 observations; monitoring is descriptive only."})
    if len(data) >= 30 and ece > 0.08:
        alerts.append({"severity": "warning", "code": "CALIBRATION_GAP", "message": f"Expected calibration error is {ece:.3f}."})
    if degradation is not None and degradation > 0.10:
        alerts.append({"severity": "warning", "code": "RECENT_LOG_LOSS_DEGRADATION", "message": f"Recent Log Loss worsened by {degradation:.3f}."})
    status = "attention_required" if any(item["severity"] == "warning" for item in alerts) else "healthy"
    payload = {
        "status": status,
        "source": source,
        "observations": int(len(data)),
        "expected_calibration_error": ece,
        "overall_metrics": overall,
        "recent_window": recent_count,
        "previous_window_metrics": previous,
        "recent_window_metrics": recent,
        "recent_log_loss_degradation": degradation,
        "alerts": alerts,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_model_changed": False,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(calibration_rows).to_csv(CALIBRATION_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(alerts, columns=["severity", "code", "message"]).to_csv(
        ALERTS_FILE, index=False, encoding="utf-8-sig"
    )
    print("Model monitoring completed.")
    print(f"Status: {status}; observations: {len(data)}; ECE: {ece:.4f}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
