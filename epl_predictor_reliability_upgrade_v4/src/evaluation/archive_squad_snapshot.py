from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYER_REGISTRY = PROJECT_ROOT / "data" / "assets" / "player_registry.parquet"
SQUAD_FEATURES = (
    PROJECT_ROOT / "data" / "processed" / "upcoming_squad_features.parquet"
)
PREMATCH_REPORT = PROJECT_ROOT / "data" / "processed" / "prematch_report.parquet"
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "history" / "squad_snapshots"
MANIFEST_FILE = SNAPSHOT_ROOT / "manifest.csv"
READINESS_FILE = PROJECT_ROOT / "reports" / "squad_snapshot_readiness.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(MANIFEST_FILE)


def readiness_payload(manifest: pd.DataFrame) -> dict:
    if manifest.empty:
        snapshot_count = 0
        unique_dates = 0
        fixture_observations = 0
        earliest = None
        latest = None
    else:
        created = pd.to_datetime(
            manifest["created_at_utc"], errors="coerce", utc=True
        )
        snapshot_count = int(len(manifest))
        unique_dates = int(created.dt.date.nunique())
        fixture_observations = int(
            pd.to_numeric(
                manifest.get("fixture_rows", 0), errors="coerce"
            ).fillna(0).sum()
        )
        earliest = created.min().isoformat() if created.notna().any() else None
        latest = created.max().isoformat() if created.notna().any() else None

    gates = {
        "minimum_unique_snapshot_dates": 8,
        "minimum_fixture_observations": 80,
        "chronological_validation_required": True,
        "production_auto_promotion_allowed": False,
    }
    ready = (
        unique_dates >= gates["minimum_unique_snapshot_dates"]
        and fixture_observations >= gates["minimum_fixture_observations"]
    )
    return {
        "status": "candidate_backtest_ready" if ready else "collecting_history",
        "snapshot_count": snapshot_count,
        "unique_snapshot_dates": unique_dates,
        "fixture_observations": fixture_observations,
        "earliest_snapshot_utc": earliest,
        "latest_snapshot_utc": latest,
        "gates": gates,
        "production_model_changed": False,
        "note": (
            "历史覆盖达到最低实验门槛；仍须执行时间顺序回测和留出集验证。"
            if ready
            else "正在积累赛前球员与阵容历史；当前不会接入生产概率。"
        ),
    }


def main() -> None:
    inputs = {
        "player_registry": PLAYER_REGISTRY,
        "squad_features": SQUAD_FEATURES,
        "prematch_report": PREMATCH_REPORT,
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "无法归档阵容快照，缺少文件：\n" + "\n".join(missing)
        )

    hashes = {name: sha256(path) for name, path in inputs.items()}
    manifest = load_manifest()
    if not manifest.empty:
        duplicate = pd.Series(True, index=manifest.index)
        for name, digest in hashes.items():
            column = f"{name}_sha256"
            duplicate &= manifest.get(
                column, pd.Series("", index=manifest.index)
            ).astype(str).eq(digest)
        if duplicate.any():
            payload = readiness_payload(manifest)
            READINESS_FILE.parent.mkdir(parents=True, exist_ok=True)
            READINESS_FILE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print("阵容快照内容未变化，本次不重复归档。")
            print(f"历史快照：{int(len(manifest))}")
            print(f"准备状态：{payload['status']}")
            return

    now = datetime.now(timezone.utc)
    snapshot_id = now.strftime("%Y%m%dT%H%M%SZ")
    destination = SNAPSHOT_ROOT / snapshot_id
    destination.mkdir(parents=True, exist_ok=False)

    for name, source in inputs.items():
        shutil.copy2(source, destination / f"{name}.parquet")

    record = {
        "snapshot_id": snapshot_id,
        "created_at_utc": now.isoformat(),
        "player_rows": parquet_rows(PLAYER_REGISTRY),
        "fixture_rows": parquet_rows(SQUAD_FEATURES),
        "report_rows": parquet_rows(PREMATCH_REPORT),
        **{f"{name}_sha256": digest for name, digest in hashes.items()},
    }
    manifest = pd.concat(
        [manifest, pd.DataFrame([record])], ignore_index=True
    )
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_FILE, index=False, encoding="utf-8-sig")

    payload = readiness_payload(manifest)
    READINESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    READINESS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("赛前球员与阵容历史快照已归档。")
    print(f"快照ID：{snapshot_id}")
    print(f"历史快照：{len(manifest)}")
    print(f"赛程观察数：{payload['fixture_observations']}")
    print(f"准备状态：{payload['status']}")
    print("生产模型未被修改。")


if __name__ == "__main__":
    main()
