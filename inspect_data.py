
from pathlib import Path
import pandas as pd
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def inspect_csv(file_path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    # Detect comma, tab, or semicolon separators
    try:
        df = pd.read_csv(
            file_path,
            sep=None,
            engine="python",
            low_memory=False
        )
    except Exception as error:
        print(f"Automatic delimiter detection failed: {error}")
        print("Trying tab-separated format...")
        df = pd.read_csv(
            file_path,
            sep="\t",
            low_memory=False
        )

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string() if not missing.empty else "None found")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nPossible date columns:",
          [c for c in df.columns if "date" in c.lower()])
    print("Possible sales columns:",
          [c for c in df.columns if "sales" in c.lower()])


def main():
    files = [
        DATA_DIR / "rossmann" / "train.csv",
        DATA_DIR / "rossmann" / "store.csv",
        DATA_DIR / "walmart" / "train.csv",
        DATA_DIR / "walmart" / "features.csv",
        DATA_DIR / "walmart" / "stores.csv",
    ]

    superstore_dir = DATA_DIR / "superstore"
    if superstore_dir.exists():
        files.extend(superstore_dir.glob("*.csv"))

    for file_path in files:
        if file_path.exists():
            try:
                inspect_csv(file_path)
            except Exception as error:
                print(f"\nCould not inspect {file_path.name}: {error}")
        else:
            print(f"\nFILE NOT FOUND: {file_path}")

    print("\nInspection finished.")


if __name__ == "__main__":
    main()
