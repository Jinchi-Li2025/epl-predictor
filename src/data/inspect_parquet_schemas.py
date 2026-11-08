from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILES = [
    PROJECT_ROOT / "data" / "processed" / "matches.parquet",
    PROJECT_ROOT / "data" / "processed" / "current_completed_matches.parquet",
    PROJECT_ROOT / "data" / "processed" / "current_fixtures.parquet",
    PROJECT_ROOT / "data" / "processed" / "football_data_epl_current.parquet",
]


def main() -> None:
    for path in FILES:
        print(f"\n=== {path.name} ===")

        if not path.exists():
            print("文件不存在")
            continue

        data = pd.read_parquet(path)

        print(f"行数：{len(data)}")
        print(f"列数：{len(data.columns)}")
        print(f"列名：{data.columns.tolist()}")
        print(f"字段类型：{data.dtypes.astype(str).to_dict()}")

        if len(data) > 0:
            print("第一行：")
            print(data.head(1).to_string(index=False))


if __name__ == "__main__":
    main()