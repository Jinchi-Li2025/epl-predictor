from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORY_DIR = DATA_DIR / "history"
BACKUP_DIR = DATA_DIR / "backups"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs" / "pipeline"

COMPLETED_CANDIDATE_FILE = (
    PROCESSED_DIR
    / "current_completed_matches_candidate.parquet"
)

FIXTURES_CANDIDATE_FILE = (
    PROCESSED_DIR
    / "current_fixtures_candidate.parquet"
)

CURRENT_COMPLETED_FILE = (
    PROCESSED_DIR
    / "current_completed_matches.parquet"
)

CURRENT_FIXTURES_FILE = (
    PROCESSED_DIR
    / "current_fixtures.parquet"
)

PREMATCH_REPORT_FILE = (
    PROCESSED_DIR
    / "prematch_report.parquet"
)

PREDICTION_LEDGER_FILE = (
    HISTORY_DIR
    / "prediction_ledger.parquet"
)

PREDICTION_LEDGER_CSV = (
    REPORTS_DIR
    / "prediction_ledger.csv"
)


def write_log(
    log_file: TextIO,
    message: str,
) -> None:
    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    line = f"[{timestamp}] {message}"

    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def run_script(
    log_file: TextIO,
    label: str,
    relative_path: str,
) -> None:
    script_path = (
        PROJECT_ROOT / relative_path
    )

    if not script_path.exists():
        raise FileNotFoundError(
            f"找不到流水线脚本：{script_path}"
        )

    write_log(
        log_file,
        f"开始步骤：{label}",
    )

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError(
            f"无法读取步骤输出：{label}"
        )

    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
        log_file.flush()

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"步骤失败：{label}，"
            f"退出代码：{return_code}"
        )

    write_log(
        log_file,
        f"步骤完成：{label}",
    )


def validate_candidate_files() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not COMPLETED_CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            "找不到候选已完赛文件："
            f"{COMPLETED_CANDIDATE_FILE}"
        )

    if not FIXTURES_CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            "找不到候选未来赛程文件："
            f"{FIXTURES_CANDIDATE_FILE}"
        )

    completed = pd.read_parquet(
        COMPLETED_CANDIDATE_FILE
    )

    fixtures = pd.read_parquet(
        FIXTURES_CANDIDATE_FILE
    )

    required_columns = {
        "match_id",
        "season",
        "kickoff_utc",
        "home_team_id",
        "home_team",
        "away_team_id",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "status",
    }

    for name, data in (
        ("已完赛候选", completed),
        ("未来赛程候选", fixtures),
    ):
        missing_columns = (
            required_columns - set(data.columns)
        )

        if missing_columns:
            raise RuntimeError(
                f"{name}缺少字段："
                f"{sorted(missing_columns)}"
            )

        if data["match_id"].isna().any():
            raise RuntimeError(
                f"{name}存在空比赛ID。"
            )

        if data["match_id"].duplicated().any():
            raise RuntimeError(
                f"{name}存在重复比赛ID。"
            )

        if data["kickoff_utc"].isna().any():
            raise RuntimeError(
                f"{name}存在空开球时间。"
            )

    combined = pd.concat(
        [
            completed,
            fixtures,
        ],
        ignore_index=True,
    )

    if len(combined) != 380:
        raise RuntimeError(
            "候选赛季应有380场，"
            f"实际为{len(combined)}场。"
        )

    if combined["match_id"].duplicated().any():
        raise RuntimeError(
            "已完赛与未来赛程之间存在重复比赛ID。"
        )

    seasons = (
        combined["season"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(seasons) != 1:
        raise RuntimeError(
            "候选数据中存在多个赛季："
            f"{seasons}"
        )

    team_ids = set(
        combined["home_team_id"]
        .dropna()
        .astype(str)
    ) | set(
        combined["away_team_id"]
        .dropna()
        .astype(str)
    )

    if len(team_ids) != 20:
        raise RuntimeError(
            "候选数据应包含20支球队，"
            f"实际为{len(team_ids)}支。"
        )

    if not completed.empty:
        completed_missing_result = (
            completed["home_goals"].isna()
            | completed["away_goals"].isna()
            | completed["result"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        if completed_missing_result.any():
            raise RuntimeError(
                "候选已完赛比赛存在空比分或空赛果。"
            )

    if (
        fixtures["home_goals"].notna().any()
        or fixtures["away_goals"].notna().any()
    ):
        raise RuntimeError(
            "候选未来赛程中存在已经填写的比分。"
        )

    return completed, fixtures


def datasets_equal(
    candidate_file: Path,
    current_file: Path,
) -> bool:
    if not current_file.exists():
        return False

    candidate = pd.read_parquet(
        candidate_file
    )

    current = pd.read_parquet(
        current_file
    )

    if set(candidate.columns) != set(
        current.columns
    ):
        return False

    ordered_columns = sorted(
        candidate.columns.tolist()
    )

    candidate = (
        candidate[ordered_columns]
        .sort_values(
            "match_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    current = (
        current[ordered_columns]
        .sort_values(
            "match_id",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    try:
        pd.testing.assert_frame_equal(
            candidate,
            current,
            check_dtype=False,
            check_exact=False,
        )
    except AssertionError:
        return False

    return True


def atomic_copy(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = (
        destination.parent
        / f"{destination.name}.tmp"
    )

    shutil.copy2(
        source,
        temporary_file,
    )

    os.replace(
        temporary_file,
        destination,
    )


def promote_candidates(
    log_file: TextIO,
) -> None:
    completed, fixtures = (
        validate_candidate_files()
    )

    completed_unchanged = datasets_equal(
        COMPLETED_CANDIDATE_FILE,
        CURRENT_COMPLETED_FILE,
    )

    fixtures_unchanged = datasets_equal(
        FIXTURES_CANDIDATE_FILE,
        CURRENT_FIXTURES_FILE,
    )

    if (
        completed_unchanged
        and fixtures_unchanged
    ):
        write_log(
            log_file,
            "候选数据与当前正式数据一致，"
            "无需替换。",
        )
        return

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup_folder = (
        BACKUP_DIR
        / f"current_season_{timestamp}"
    )

    backup_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for current_file in (
        CURRENT_COMPLETED_FILE,
        CURRENT_FIXTURES_FILE,
    ):
        if current_file.exists():
            shutil.copy2(
                current_file,
                backup_folder
                / current_file.name,
            )

    atomic_copy(
        COMPLETED_CANDIDATE_FILE,
        CURRENT_COMPLETED_FILE,
    )

    atomic_copy(
        FIXTURES_CANDIDATE_FILE,
        CURRENT_FIXTURES_FILE,
    )

    write_log(
        log_file,
        "候选数据已通过检查并切换为正式数据。"
        f"已完赛：{len(completed)}；"
        f"未来赛程：{len(fixtures)}；"
        f"备份：{backup_folder}",
    )


def archive_predictions(
    log_file: TextIO,
) -> None:
    if not PREMATCH_REPORT_FILE.exists():
        raise FileNotFoundError(
            "找不到统一赛前报告："
            f"{PREMATCH_REPORT_FILE}"
        )

    current_report = pd.read_parquet(
        PREMATCH_REPORT_FILE
    ).copy()

    archived_at = datetime.now(
        timezone.utc
    ).isoformat()

    current_report[
        "archived_at_utc"
    ] = archived_at

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_name = (
        "prematch_report_"
        + datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        + ".parquet"
    )

    snapshot_file = (
        HISTORY_DIR / snapshot_name
    )

    current_report.to_parquet(
        snapshot_file,
        index=False,
    )

    if PREDICTION_LEDGER_FILE.exists():
        history = pd.read_parquet(
            PREDICTION_LEDGER_FILE
        )

        ledger = pd.concat(
            [
                history,
                current_report,
            ],
            ignore_index=True,
            sort=False,
        )
    else:
        ledger = current_report

    duplicate_columns = [
        "match_id",
        "generated_at_utc",
    ]

    ledger = ledger.drop_duplicates(
        subset=duplicate_columns,
        keep="last",
    )

    ledger = ledger.sort_values(
        [
            "kickoff_utc",
            "match_id",
            "generated_at_utc",
        ],
        kind="stable",
    ).reset_index(drop=True)

    temporary_ledger = (
        HISTORY_DIR
        / "prediction_ledger.tmp.parquet"
    )

    ledger.to_parquet(
        temporary_ledger,
        index=False,
    )

    os.replace(
        temporary_ledger,
        PREDICTION_LEDGER_FILE,
    )

    ledger.to_csv(
        PREDICTION_LEDGER_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    write_log(
        log_file,
        "预测历史已保存。"
        f"本次：{len(current_report)}场；"
        f"历史版本总行数：{len(ledger)}；"
        f"快照：{snapshot_file}",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行EPL预测生产更新流水线。"
        )
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help=(
            "跳过远程数据下载和正式数据切换，"
            "用于测试后续流水线。"
        ),
    )

    parser.add_argument(
        "--no-promote",
        action="store_true",
        help=(
            "生成并检查候选数据，"
            "但不替换正式赛季文件。"
        ),
    )

    parser.add_argument(
        "--with-validation",
        action="store_true",
        help=(
            "额外运行当前赛季外部验证。"
        ),
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        LOG_DIR
        / (
            "update_"
            + datetime.now(
                timezone.utc
            ).strftime("%Y%m%dT%H%M%SZ")
            + ".log"
        )
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        try:
            write_log(
                log_file,
                "EPL生产更新流水线开始。",
            )

            if PREMATCH_REPORT_FILE.exists():
                write_log(
                    log_file,
                    "开始保存刷新前的赛前预测。",
                )

                archive_predictions(
                    log_file
                )

            if not arguments.skip_fetch:
                run_script(
                    log_file,
                    "下载football-data当前赛季数据",
                    "src/data/fetch_football_data.py",
                )

                run_script(
                    log_file,
                    "刷新API-Football比赛ID与赛程",
                    "src/data/fetch_api_football.py",
                )

                run_script(
                    log_file,
                    "生成并核对当前赛季候选文件",
                    "src/data/build_live_current_season.py",
                )

                validate_candidate_files()

                if arguments.no_promote:
                    write_log(
                        log_file,
                        "已按参数要求跳过正式数据切换。",
                    )
                else:
                    promote_candidates(
                        log_file
                    )

            if arguments.with_validation:
                run_script(
                    log_file,
                    "运行当前赛季外部验证",
                    (
                        "src/evaluation/"
                        "current_season_validation.py"
                    ),
                )

                run_script(
                    log_file,
                    "评估Champion与Shadow Challenger",
                    (
                        "src/evaluation/"
                        "champion_challenger.py"
                    ),
                )

                run_script(
                    log_file,
                    "生成DeepSeek模型实验建议",
                    (
                        "src/evaluation/"
                        "llm_model_advisor.py"
                    ),
                )

            run_script(
                log_file,
                "结算已经完赛的历史预测",
                (
                    "src/evaluation/"
                    "settle_prediction_ledger.py"
                ),
            )

            production_steps = [
                (
                    "更新英超注册球员资料库",
                    (
                        "src/assets/"
                        "build_player_registry.py"
                    ),
                ),
                (
                    "生成未来比赛赛前特征",
                    (
                        "src/features/"
                        "build_upcoming_features.py"
                    ),
                ),
                (
                    "训练生产模型并生成预测",
                    (
                        "src/prediction/"
                        "predict_upcoming.py"
                    ),
                ),
                (
                    "生成赛前上下文模板",
                    (
                        "src/context/"
                        "build_context_template.py"
                    ),
                ),
                (
                    "更新逐小时天气",
                    "src/context/fetch_weather.py",
                ),
                (
                    "更新裁判安排",
                    "src/context/fetch_referee.py",
                ),
                (
                    "更新比赛重要性",
                    (
                        "src/context/"
                        "build_importance_context.py"
                    ),
                ),
                (
                    "更新伤停可用性",
                    (
                        "src/context/"
                        "fetch_availability.py"
                    ),
                ),
                (
                    "更新预计阵容与官方首发",
                    (
                        "src/context/"
                        "fetch_lineups.py"
                    ),
                ),
                (
                    "使用football-data补充官方首发与替补",
                    (
                        "src/context/"
                        "fetch_lineups_football_data.py"
                    ),
                ),
                (
                    "应用经过人工核验的官方名单覆盖",
                    (
                        "src/context/"
                        "apply_lineup_overrides.py"
                    ),
                ),
                (
                    "抓取公开球队新闻标题",
                    (
                        "src/context/"
                        "fetch_team_news.py"
                    ),
                ),
                (
                    "生成结构化球队新闻与DeepSeek摘要",
                    (
                        "src/context/"
                        "build_team_news_context.py"
                    ),
                ),
                (
                    "生成统一赛前报告",
                    (
                        "src/context/"
                        "build_prematch_report.py"
                    ),
                ),
                (
                    "生成阵容深度候选特征（不修改生产概率）",
                    (
                        "src/features/"
                        "build_squad_features.py"
                    ),
                ),
                (
                    "归档赛前球员与阵容历史快照",
                    (
                        "src/evaluation/"
                        "archive_squad_snapshot.py"
                    ),
                ),
                (
                    "构建无数据泄漏的阵容训练集",
                    (
                        "src/features/"
                        "build_squad_training_dataset.py"
                    ),
                ),
                (
                    "评估阵容特征Shadow Challenger",
                    (
                        "src/evaluation/"
                        "squad_feature_challenger.py"
                    ),
                ),
                (
                    "生成DeepSeek阵容特征实验建议",
                    (
                        "src/evaluation/"
                        "llm_squad_advisor.py"
                    ),
                ),
                (
                    "生成系统健康报告",
                    (
                        "src/pipeline/"
                        "system_health_check.py"
                    ),
                ),
            ]

            for label, script in production_steps:
                run_script(
                    log_file,
                    label,
                    script,
                )

            archive_predictions(
                log_file
            )

            write_log(
                log_file,
                "EPL生产更新流水线全部完成。",
            )

            print(
                "\n更新成功。"
                f"\n日志文件：{log_path}"
                "\n重新打开或刷新Streamlit页面即可。"
            )

            return 0

        except Exception as error:
            write_log(
                log_file,
                f"流水线失败：{error}",
            )

            error_details = traceback.format_exc()

            print(error_details)
            log_file.write(
                error_details + "\n"
            )

            print(
                "\n更新失败。"
                f"\n请检查日志：{log_path}"
            )

            return 1


if __name__ == "__main__":
    raise SystemExit(main())
