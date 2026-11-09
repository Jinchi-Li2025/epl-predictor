from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = (
    PROJECT_ROOT
    / "configs"
    / "champion_challenger_policy.json"
)
PRODUCTION_CONFIG_FILE = (
    PROJECT_ROOT / "configs" / "production_model.json"
)
FINAL_PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "final_holdout_predictions.parquet"
)
CURRENT_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "current_external_validation_predictions.parquet"
)
METRICS_FILE = (
    PROJECT_ROOT / "reports" / "champion_challenger_metrics.csv"
)
DECISION_FILE = (
    PROJECT_ROOT / "reports" / "champion_challenger_decision.json"
)

CLASS_ORDER = ["H", "D", "A"]
RESULT_TO_INDEX = {
    result: index
    for index, result in enumerate(CLASS_ORDER)
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def required_probability_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_p_home",
        f"{prefix}_p_draw",
        f"{prefix}_p_away",
    ]


def validate_predictions(
    data: pd.DataFrame,
    dataset_name: str,
    model_names: list[str],
) -> pd.DataFrame:
    required_columns = {
        "match_id",
        "actual_result",
    }

    for model_name in model_names:
        required_columns.update(
            required_probability_columns(model_name)
        )

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise RuntimeError(
            f"{dataset_name}缺少字段：{sorted(missing_columns)}"
        )

    if data.empty:
        raise RuntimeError(f"{dataset_name}为空。")

    if data["match_id"].isna().any():
        raise RuntimeError(f"{dataset_name}存在空比赛ID。")

    if data["match_id"].duplicated().any():
        raise RuntimeError(f"{dataset_name}存在重复比赛ID。")

    valid_results = data["actual_result"].isin(CLASS_ORDER)
    if not valid_results.all():
        invalid_results = sorted(
            data.loc[~valid_results, "actual_result"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            f"{dataset_name}存在无效赛果：{invalid_results}"
        )

    validated = data.copy()
    for model_name in model_names:
        columns = required_probability_columns(model_name)
        probabilities = (
            validated[columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0.0).any()
            or (probabilities.sum(axis=1) <= 0.0).any()
        ):
            raise RuntimeError(
                f"{dataset_name}中的{model_name}概率无效。"
            )

        probabilities = (
            probabilities
            / probabilities.sum(axis=1, keepdims=True)
        )
        validated[columns] = probabilities

    return validated


def probability_matrix(
    data: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    return data[
        required_probability_columns(model_name)
    ].to_numpy(dtype=float)


def actual_indices(data: pd.DataFrame) -> np.ndarray:
    return (
        data["actual_result"]
        .map(RESULT_TO_INDEX)
        .to_numpy(dtype=int)
    )


def per_match_log_loss(
    probabilities: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    selected = probabilities[
        np.arange(len(indices)),
        indices,
    ]
    return -np.log(np.clip(selected, 1e-15, 1.0))


def calculate_metrics(
    probabilities: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    targets = np.eye(3, dtype=float)[indices]
    log_losses = per_match_log_loss(probabilities, indices)
    brier = np.sum((probabilities - targets) ** 2, axis=1)
    cumulative_probabilities = np.cumsum(
        probabilities,
        axis=1,
    )[:, :2]
    cumulative_targets = np.cumsum(
        targets,
        axis=1,
    )[:, :2]
    rps = np.mean(
        (cumulative_probabilities - cumulative_targets) ** 2,
        axis=1,
    )

    return {
        "matches": int(len(indices)),
        "log_loss": float(np.mean(log_losses)),
        "brier_score": float(np.mean(brier)),
        "ranked_probability_score": float(np.mean(rps)),
        "accuracy": float(
            np.mean(np.argmax(probabilities, axis=1) == indices)
        ),
    }


def metric_rows_for_dataset(
    data: pd.DataFrame,
    dataset_name: str,
    model_names: list[str],
) -> list[dict]:
    indices = actual_indices(data)
    rows: list[dict] = []

    for model_name in model_names:
        metrics = calculate_metrics(
            probability_matrix(data, model_name),
            indices,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                **metrics,
            }
        )

    return rows


def bootstrap_improvement_interval(
    champion_losses: np.ndarray,
    challenger_losses: np.ndarray,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    improvement = champion_losses - challenger_losses
    rng = np.random.default_rng(seed)
    sample_means = np.empty(samples, dtype=float)
    number_of_matches = len(improvement)

    for index in range(samples):
        sample_indices = rng.integers(
            0,
            number_of_matches,
            size=number_of_matches,
        )
        sample_means[index] = float(
            improvement[sample_indices].mean()
        )

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        sample_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )

    return {
        "mean_absolute_improvement": float(improvement.mean()),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "confidence_level": float(confidence_level),
    }


def chronological_block_summary(
    data: pd.DataFrame,
    champion: str,
    challenger: str,
    block_size: int,
) -> dict[str, object]:
    ordered = data.copy()
    if "kickoff_utc" in ordered.columns:
        ordered["kickoff_utc"] = pd.to_datetime(
            ordered["kickoff_utc"],
            errors="coerce",
            utc=True,
        )
        ordered = ordered.sort_values(
            ["kickoff_utc", "match_id"],
            kind="stable",
        )
    else:
        ordered = ordered.sort_values("match_id", kind="stable")

    indices = actual_indices(ordered)
    champion_losses = per_match_log_loss(
        probability_matrix(ordered, champion),
        indices,
    )
    challenger_losses = per_match_log_loss(
        probability_matrix(ordered, challenger),
        indices,
    )

    blocks: list[dict] = []
    for start in range(0, len(ordered), block_size):
        stop = min(start + block_size, len(ordered))
        improvement = float(
            champion_losses[start:stop].mean()
            - challenger_losses[start:stop].mean()
        )
        blocks.append(
            {
                "start_row": start + 1,
                "end_row": stop,
                "matches": stop - start,
                "log_loss_improvement": improvement,
                "challenger_won": bool(improvement > 0.0),
            }
        )

    wins = sum(block["challenger_won"] for block in blocks)
    return {
        "block_size": int(block_size),
        "block_count": int(len(blocks)),
        "challenger_block_wins": int(wins),
        "block_win_rate": float(wins / len(blocks)),
        "blocks": blocks,
    }


def find_metrics(
    metrics: pd.DataFrame,
    dataset: str,
    model: str,
) -> pd.Series:
    rows = metrics.loc[
        metrics["dataset"].eq(dataset)
        & metrics["model"].eq(model)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"无法唯一定位指标：{dataset}/{model}"
        )
    return rows.iloc[0]


def build_decision(
    final_data: pd.DataFrame,
    current_data: pd.DataFrame,
    metrics: pd.DataFrame,
    policy: dict,
) -> dict:
    champion = str(policy["champion"])
    challenger = str(policy["challenger"])
    final_champion = find_metrics(metrics, "final_holdout", champion)
    final_challenger = find_metrics(metrics, "final_holdout", challenger)
    current_champion = find_metrics(
        metrics,
        "current_external_validation",
        champion,
    )
    current_challenger = find_metrics(
        metrics,
        "current_external_validation",
        challenger,
    )

    current_indices = actual_indices(current_data)
    champion_losses = per_match_log_loss(
        probability_matrix(current_data, champion),
        current_indices,
    )
    challenger_losses = per_match_log_loss(
        probability_matrix(current_data, challenger),
        current_indices,
    )
    bootstrap = bootstrap_improvement_interval(
        champion_losses,
        challenger_losses,
        samples=int(policy["bootstrap_samples"]),
        confidence_level=float(policy["bootstrap_confidence_level"]),
        seed=int(policy["bootstrap_seed"]),
    )
    blocks = chronological_block_summary(
        current_data,
        champion,
        challenger,
        int(policy["chronological_block_size"]),
    )

    final_regression = (
        float(final_challenger["log_loss"])
        - float(final_champion["log_loss"])
    ) / float(final_champion["log_loss"])
    current_relative_improvement = (
        float(current_champion["log_loss"])
        - float(current_challenger["log_loss"])
    ) / float(current_champion["log_loss"])
    brier_regression = (
        float(current_challenger["brier_score"])
        - float(current_champion["brier_score"])
    )
    rps_regression = (
        float(current_challenger["ranked_probability_score"])
        - float(current_champion["ranked_probability_score"])
    )

    gates = {
        "minimum_sample_size": bool(
            len(current_data)
            >= int(policy["minimum_current_season_matches"])
        ),
        "current_log_loss_improvement": bool(
            current_relative_improvement
            >= float(policy["minimum_relative_log_loss_improvement"])
        ),
        "final_holdout_stability": bool(
            final_regression
            <= float(policy["maximum_final_holdout_relative_regression"])
        ),
        "brier_not_worse": bool(
            brier_regression
            <= float(policy["maximum_brier_regression"])
        ),
        "rps_not_worse": bool(
            rps_regression
            <= float(policy["maximum_rps_regression"])
        ),
        "positive_bootstrap_lower_bound": bool(
            bootstrap["lower_bound"] > 0.0
            if policy["require_positive_ci_lower_bound"]
            else True
        ),
        "enough_chronological_blocks": bool(
            blocks["block_count"]
            >= int(policy["minimum_block_count"])
        ),
        "chronological_block_consistency": bool(
            blocks["block_win_rate"]
            >= float(policy["minimum_block_win_rate"])
        ),
    }

    eligible = all(gates.values())
    status = "PROMOTION_REVIEW_REQUIRED" if eligible else "SHADOW_ONLY"
    failed_gates = [name for name, passed in gates.items() if not passed]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "recommended_action": (
            "REVIEW_CHALLENGER_FOR_MANUAL_PROMOTION"
            if eligible
            else "KEEP_CURRENT_CHAMPION"
        ),
        "production_changed": False,
        "champion": champion,
        "challenger": challenger,
        "current_validation_matches": int(len(current_data)),
        "final_holdout_matches": int(len(final_data)),
        "current_relative_log_loss_improvement": float(
            current_relative_improvement
        ),
        "final_holdout_relative_regression": float(final_regression),
        "current_brier_regression": float(brier_regression),
        "current_rps_regression": float(rps_regression),
        "bootstrap_log_loss_improvement": bootstrap,
        "chronological_blocks": blocks,
        "promotion_gates": gates,
        "failed_gates": failed_gates,
        "promotion_mode": policy["promotion_mode"],
    }


def print_summary(
    metrics: pd.DataFrame,
    decision: dict,
) -> None:
    display = metrics[
        [
            "dataset",
            "model",
            "matches",
            "log_loss",
            "brier_score",
            "ranked_probability_score",
            "accuracy",
        ]
    ]
    print("\nChampion / Challenger 指标：")
    print(display.to_string(index=False))
    print(f"\n状态：{decision['status']}")
    print(f"建议：{decision['recommended_action']}")
    print(
        "当前赛季Log Loss相对改善："
        f"{decision['current_relative_log_loss_improvement']:.2%}"
    )
    print(
        "Bootstrap改善区间："
        f"[{decision['bootstrap_log_loss_improvement']['lower_bound']:.6f}, "
        f"{decision['bootstrap_log_loss_improvement']['upper_bound']:.6f}]"
    )
    if decision["failed_gates"]:
        print("尚未通过：" + ", ".join(decision["failed_gates"]))
    print("生产模型未被修改。")


def main() -> None:
    required_files = [
        POLICY_FILE,
        PRODUCTION_CONFIG_FILE,
        FINAL_PREDICTIONS_FILE,
        CURRENT_PREDICTIONS_FILE,
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "缺少Champion/Challenger评估文件：\n"
            + "\n".join(str(path) for path in missing_files)
        )

    policy = load_json(POLICY_FILE)
    production_config = load_json(PRODUCTION_CONFIG_FILE)
    champion = str(policy["champion"])
    challenger = str(policy["challenger"])

    if production_config.get("primary_outcome_model") != champion:
        raise RuntimeError(
            "生产配置中的主模型与Champion策略不一致："
            f"{production_config.get('primary_outcome_model')} != {champion}"
        )

    model_names = [champion, challenger]
    final_data = validate_predictions(
        pd.read_parquet(FINAL_PREDICTIONS_FILE),
        "最终留出集",
        model_names,
    )
    current_data = validate_predictions(
        pd.read_parquet(CURRENT_PREDICTIONS_FILE),
        "当前赛季外部验证",
        model_names,
    )

    metric_rows = metric_rows_for_dataset(
        final_data,
        "final_holdout",
        model_names,
    )
    metric_rows.extend(
        metric_rows_for_dataset(
            current_data,
            "current_external_validation",
            model_names,
        )
    )
    metrics = pd.DataFrame(metric_rows)
    decision = build_decision(
        final_data,
        current_data,
        metrics,
        policy,
    )

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")
    DECISION_FILE.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(metrics, decision)
    print(f"\n指标文件：{METRICS_FILE}")
    print(f"判断文件：{DECISION_FILE}")


if __name__ == "__main__":
    main()
