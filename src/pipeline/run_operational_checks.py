from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs" / "operations"
STEPS = [
    ("player registry quality", "src/evaluation/registry_quality_audit.py"),
    ("pipeline contracts", "src/pipeline/pipeline_contract_audit.py"),
    ("model monitor", "src/evaluation/model_monitor.py"),
    ("calibration challenger", "src/evaluation/calibration_challenger.py"),
    ("squad training dataset", "src/features/build_squad_training_dataset.py"),
    ("squad feature challenger", "src/evaluation/squad_feature_challenger.py"),
    ("system health", "src/pipeline/system_health_check.py"),
    ("operational readiness", "src/pipeline/operational_readiness.py"),
]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        "checks_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log"
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        for label, relative_path in STEPS:
            path = PROJECT_ROOT / relative_path
            if not path.exists():
                message = f"SKIPPED missing optional step: {relative_path}"
                print(message)
                log.write(message + "\n")
                continue
            print(f"START {label}")
            log.write(f"START {label}\n")
            process = subprocess.Popen(
                [sys.executable, str(path)],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Operational step failed: {label} ({return_code})")
            log.write(f"DONE {label}\n")
    print("All operational checks completed.")
    print(f"Log: {log_path}")
    print("Production model was not modified.")


if __name__ == "__main__":
    main()
