from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTLED_FILE = PROJECT_ROOT / "data" / "history" / "settled_predictions.parquet"
VALIDATION_FILE = (
    PROJECT_ROOT / "reports" / "current_external_validation_predictions.parquet"
)
DECISION_FILE = PROJECT_ROOT / "reports" / "calibration_challenger_decision.json"
METRICS_FILE = PROJECT_ROOT / "reports" / "calibration_challenger_metrics.csv"
ARTIFACT_FILE = PROJECT_ROOT / "artifacts" / "calibration_challenger.json"


def load_data() -> tuple[pd.DataFrame, str]:
    if SETTLED_FILE.exists():
        data = pd.read_parquet(SETTLED_FILE).copy()
        rename = {
            "settled_home_probability": "p_home",
            "settled_draw_probability": "p_draw",
            "settled_away_probability": "p_away",
            "actual_kickoff_utc": "kickoff_utc",
        }
        data = data.rename(columns=rename)
        if not data.empty:
            return data, "published_settled_predictions"
    if VALIDATION_FILE.exists():
        return pd.read_parquet(VALIDATION_FILE).copy(), "external_validation"
    return pd.DataFrame(), "none"


def prepare(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    required = ["p_home", "p_draw", "p_away", "actual_result"]
    missing = [column for column in required if column not in data]
    if missing:
        raise RuntimeError("Calibration input is missing: " + ", ".join(missing))
    output = data.copy()
    if "kickoff_utc" in output:
        output["kickoff_utc"] = pd.to_datetime(
            output["kickoff_utc"], utc=True, errors="coerce"
        )
        output = output.sort_values("kickoff_utc", na_position="first")
    labels = output["actual_result"].astype(str).str.upper().map(
        {"H": 0, "D": 1, "A": 2}
    )
    probabilities = output[["p_home", "p_draw", "p_away"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = (
        labels.notna().to_numpy()
        & np.isfinite(probabilities).all(axis=1)
        & (probabilities >= 0.0).all(axis=1)
        & (probabilities.sum(axis=1) > 0.0)
    )
    output = output.loc[valid].reset_index(drop=True)
    labels_array = labels.loc[valid].astype(int).to_numpy()
    probabilities = probabilities[valid]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, labels_array, output


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    targets = np.eye(3)[labels]
    return {
        "log_loss": float(-np.log(clipped[np.arange(len(labels)), labels]).mean()),
        "brier_score": float(np.square(clipped - targets).sum(axis=1).mean()),
        "accuracy": float((clipped.argmax(axis=1) == labels).mean()),
    }


def choose_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.60, 2.50, 191)
    losses = [
        metrics(temperature_scale(probabilities, value), labels)["log_loss"]
        for value in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def paired_bootstrap(
    baseline: np.ndarray,
    challenger: np.ndarray,
    labels: np.ndarray,
    iterations: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(20261110)
    baseline_loss = -np.log(
        np.clip(baseline[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    challenger_loss = -np.log(
        np.clip(challenger[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    improvements = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(labels), size=len(labels))
        improvements[index] = float(
            baseline_loss[sample].mean() - challenger_loss[sample].mean()
        )
    return tuple(float(value) for value in np.quantile(improvements, [0.025, 0.975]))


def write_decision(payload: dict) -> None:
    DECISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    DECISION_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    data, source = load_data()
    generated = datetime.now(timezone.utc).isoformat()
    if data.empty:
        payload = {
            "status": "insufficient_data",
            "reason": "No settled or external-validation probabilities are available.",
            "source": source,
            "production_model_changed": False,
            "generated_at_utc": generated,
        }
        write_decision(payload)
        print(payload["reason"])
        return

    probabilities, labels, ordered = prepare(data)
    if len(labels) < 60:
        payload = {
            "status": "insufficient_data",
            "reason": "At least 60 chronological observations are required.",
            "source": source,
            "observations": int(len(labels)),
            "production_model_changed": False,
            "generated_at_utc": generated,
        }
        write_decision(payload)
        print(payload["reason"])
        return

    train_end = max(int(len(labels) * 0.70), 40)
    train_end = min(train_end, len(labels) - 20)
    train_probabilities = probabilities[:train_end]
    train_labels = labels[:train_end]
    test_probabilities = probabilities[train_end:]
    test_labels = labels[train_end:]

    temperature = choose_temperature(train_probabilities, train_labels)
    challenger = temperature_scale(test_probabilities, temperature)
    baseline_metrics = metrics(test_probabilities, test_labels)
    challenger_metrics = metrics(challenger, test_labels)
    improvement = baseline_metrics["log_loss"] - challenger_metrics["log_loss"]
    lower, upper = paired_bootstrap(
        test_probabilities, challenger, test_labels
    )
    brier_change = (
        challenger_metrics["brier_score"] - baseline_metrics["brier_score"]
    )
    promising = (
        len(test_labels) >= 30
        and improvement >= 0.005
        and lower > 0.0
        and brier_change <= 0.002
    )

    rows = [
        {"model": "production_baseline", "test_rows": len(test_labels), **baseline_metrics},
        {"model": "temperature_challenger", "test_rows": len(test_labels), **challenger_metrics},
    ]
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_FILE.write_text(
        json.dumps(
            {
                "kind": "multiclass_temperature_scaling",
                "temperature": temperature,
                "source": source,
                "training_rows": train_end,
                "created_at_utc": generated,
                "production_artifact": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = {
        "status": "promising_shadow_candidate" if promising else "keep_production",
        "source": source,
        "observations": int(len(labels)),
        "training_rows": int(train_end),
        "test_rows": int(len(test_labels)),
        "temperature": temperature,
        "baseline_metrics": baseline_metrics,
        "challenger_metrics": challenger_metrics,
        "log_loss_improvement": improvement,
        "bootstrap_95_ci": [lower, upper],
        "brier_change": brier_change,
        "gates": {
            "minimum_test_rows": 30,
            "minimum_log_loss_improvement": 0.005,
            "bootstrap_lower_bound_above_zero": True,
            "maximum_brier_worsening": 0.002,
        },
        "recommendation": (
            "Candidate passed the first shadow gate; continue out-of-sample observation."
            if promising
            else "Keep the current production probabilities."
        ),
        "production_model_changed": False,
        "generated_at_utc": generated,
    }
    write_decision(payload)
    print("Calibration Shadow Challenger completed.")
    print(f"Status: {payload['status']}")
    print(f"Temperature: {temperature:.3f}")
    print(f"Test Log Loss improvement: {improvement:.6f}")
    print(f"Bootstrap 95% CI: [{lower:.6f}, {upper:.6f}]")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
