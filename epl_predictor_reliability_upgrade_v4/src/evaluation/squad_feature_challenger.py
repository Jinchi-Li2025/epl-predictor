from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = (
    PROJECT_ROOT / "data" / "processed" / "squad_challenger_training.parquet"
)
AUDIT_FILE = PROJECT_ROOT / "reports" / "squad_training_dataset_audit.json"
METRICS_FILE = PROJECT_ROOT / "reports" / "squad_challenger_metrics.csv"
DECISION_FILE = PROJECT_ROOT / "reports" / "squad_challenger_decision.json"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "squad_feature_challenger.json"

MINIMUM_ROWS = 80
MINIMUM_SNAPSHOT_DATES = 8
MINIMUM_TEST_ROWS = 20
RANDOM_SEED = 20261109


def write_decision(status: str, reason: str, **extra: object) -> None:
    DECISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "reason": reason,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_model_changed": False,
        "promotion_mode": "manual_only",
        **extra,
    }
    DECISION_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(reason)
    print("Production model was not modified.")


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    targets = np.eye(3)[labels]
    log_loss = float(-np.log(clipped[np.arange(len(labels)), labels]).mean())
    brier = float(np.square(clipped - targets).sum(axis=1).mean())
    cumulative_probability = np.cumsum(clipped, axis=1)[:, :-1]
    cumulative_target = np.cumsum(targets, axis=1)[:, :-1]
    rps = float(
        np.square(cumulative_probability - cumulative_target).sum(axis=1).mean()
        / 2.0
    )
    accuracy = float((clipped.argmax(axis=1) == labels).mean())
    return {
        "log_loss": log_loss,
        "brier_score": brier,
        "ranked_probability_score": rps,
        "accuracy": accuracy,
    }


def bootstrap_improvement(
    baseline: np.ndarray,
    challenger: np.ndarray,
    labels: np.ndarray,
    samples: int = 1000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        base_loss = probability_metrics(baseline[indices], labels[indices])["log_loss"]
        challenge_loss = probability_metrics(
            challenger[indices], labels[indices]
        )["log_loss"]
        values.append(base_loss - challenge_loss)
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return float(lower), float(median), float(upper)


def main() -> None:
    if not DATASET_FILE.exists():
        write_decision(
            "INSUFFICIENT_HISTORY",
            "Squad training dataset is missing. Build it after snapshots settle.",
        )
        return
    data = pd.read_parquet(DATASET_FILE).copy()
    if AUDIT_FILE.exists():
        audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    else:
        audit = {}
    unique_dates = int(audit.get("unique_snapshot_dates", 0))
    if len(data) < MINIMUM_ROWS or unique_dates < MINIMUM_SNAPSHOT_DATES:
        write_decision(
            "INSUFFICIENT_HISTORY",
            (
                f"Need at least {MINIMUM_ROWS} settled pre-kickoff rows and "
                f"{MINIMUM_SNAPSHOT_DATES} snapshot dates; currently "
                f"{len(data)} rows and {unique_dates} dates."
            ),
            training_rows=int(len(data)),
            unique_snapshot_dates=unique_dates,
        )
        return

    try:
        import xgboost as xgb
    except ImportError:
        write_decision(
            "DEPENDENCY_MISSING",
            "xgboost is not installed in the active virtual environment.",
        )
        return

    data["kickoff_utc"] = pd.to_datetime(
        data["kickoff_utc"], utc=True, errors="coerce"
    )
    data = data.dropna(subset=["kickoff_utc"]).sort_values("kickoff_utc")
    label_map = {"H": 0, "D": 1, "A": 2}
    labels = data["actual_result"].map(label_map)
    data = data.loc[labels.notna()].copy()
    labels = labels.loc[labels.notna()].astype(int).to_numpy()

    feature_columns = [
        "baseline_home_probability",
        "baseline_draw_probability",
        "baseline_away_probability",
        *sorted(
            column for column in data.columns if column.startswith("squad_diff_")
        ),
    ]
    features = data[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    test_rows = max(MINIMUM_TEST_ROWS, int(round(len(data) * 0.20)))
    test_rows = min(test_rows, len(data) - 40)
    train_end = len(data) - test_rows
    validation_rows = max(12, int(round(train_end * 0.20)))
    fit_end = train_end - validation_rows
    if fit_end < 30:
        write_decision(
            "INSUFFICIENT_HISTORY",
            "Not enough chronological training rows after validation/test split.",
        )
        return

    x_fit = features.iloc[:fit_end]
    y_fit = labels[:fit_end]
    x_validation = features.iloc[fit_end:train_end]
    y_validation = labels[fit_end:train_end]
    x_test = features.iloc[train_end:]
    y_test = labels[train_end:]

    d_fit = xgb.DMatrix(x_fit, label=y_fit, feature_names=feature_columns)
    d_validation = xgb.DMatrix(
        x_validation, label=y_validation, feature_names=feature_columns
    )
    d_test = xgb.DMatrix(x_test, feature_names=feature_columns)
    model = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": 3,
            "eta": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 5,
            "lambda": 3.0,
            "seed": RANDOM_SEED,
            "eval_metric": "mlogloss",
            "nthread": 4,
        },
        d_fit,
        num_boost_round=500,
        evals=[(d_validation, "validation")],
        early_stopping_rounds=35,
        verbose_eval=False,
    )
    challenger_probabilities = model.predict(
        d_test, iteration_range=(0, model.best_iteration + 1)
    )
    baseline_probabilities = data.iloc[train_end:][
        [
            "baseline_home_probability",
            "baseline_draw_probability",
            "baseline_away_probability",
        ]
    ].to_numpy(dtype=float)

    baseline_metrics = probability_metrics(baseline_probabilities, y_test)
    challenger_metrics = probability_metrics(challenger_probabilities, y_test)
    lower, median, upper = bootstrap_improvement(
        baseline_probabilities, challenger_probabilities, y_test
    )
    metrics = pd.DataFrame(
        [
            {"model": "production_baseline", "test_rows": len(y_test), **baseline_metrics},
            {"model": "squad_feature_challenger", "test_rows": len(y_test), **challenger_metrics},
        ]
    )
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_FILE)

    observed_improvement = (
        baseline_metrics["log_loss"] - challenger_metrics["log_loss"]
    )
    eligible = observed_improvement > 0 and lower > 0
    status = "ELIGIBLE_FOR_MANUAL_REVIEW" if eligible else "SHADOW_ONLY"
    write_decision(
        status,
        (
            "Candidate passed the statistical gate and may receive manual review."
            if eligible
            else "Candidate has not passed the conservative promotion gate."
        ),
        training_rows=int(train_end),
        test_rows=int(len(y_test)),
        feature_columns=feature_columns,
        baseline_metrics=baseline_metrics,
        challenger_metrics=challenger_metrics,
        log_loss_improvement=float(observed_improvement),
        bootstrap_95_interval=[lower, upper],
        bootstrap_median=median,
        model_file=str(MODEL_FILE),
    )


if __name__ == "__main__":
    main()
