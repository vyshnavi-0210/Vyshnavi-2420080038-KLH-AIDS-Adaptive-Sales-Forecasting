
from pathlib import Path
import csv
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Read tab-separated Rossmann and Walmart files
# --------------------------------------------------

def read_tab_file(file_path):
    return pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False
    )


# --------------------------------------------------
# 3. Read Superstore without modifying its source file
# --------------------------------------------------

def read_superstore(file_path):
    # First try normal comma-separated CSV parsing.
    try:
        df = pd.read_csv(
            file_path,
            sep=",",
            encoding="utf-8-sig",
            low_memory=False
        )

        if "Order Date" in df.columns and "Sales" in df.columns:
            print("Superstore loaded using standard CSV parsing.")
            return df

    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        print("Standard CSV parsing failed:", error)

    # Try a different parser configuration without changing the file.
    try:
        df = pd.read_csv(
            file_path,
            sep=",",
            encoding="utf-8-sig",
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="error"
        )

        if "Order Date" in df.columns and "Sales" in df.columns:
            print("Superstore loaded using fallback parsing.")
            return df

        raise ValueError(
            "Superstore columns were not parsed correctly. "
            f"Columns found: {df.columns.tolist()}"
        )

    except Exception as error:
        raise RuntimeError(
            "Could not parse Superstore safely. "
            "The original file has not been changed. "
            f"Parser details: {error}"
        ) from error


# --------------------------------------------------
# 4. Date conversion
# --------------------------------------------------

def convert_date(df, column, dayfirst=False):
    df[column] = pd.to_datetime(
        df[column],
        dayfirst=dayfirst,
        errors="coerce"
    )
    return df


# --------------------------------------------------
# 5. Save processed dataset
# --------------------------------------------------

def save_data(df, filename):
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


# --------------------------------------------------
# 6. Process Rossmann
# --------------------------------------------------

def process_rossmann():
    path = DATA_DIR / "rossmann" / "train.csv"

    if not path.exists():
        print("\nRossmann train.csv not found.")
        return

    print("\n" + "=" * 60)
    print("PROCESSING ROSSMANN")
    print("=" * 60)

    df = read_tab_file(path)

    df = convert_date(df, "Date", dayfirst=True)

    # Remove exact duplicate rows and rows missing
    # essential forecasting fields.
    df = df.drop_duplicates()
    df = df.dropna(subset=["Date", "Sales"])

    save_data(df, "rossmann_train_clean.csv")

    store_path = DATA_DIR / "rossmann" / "store.csv"

    if not store_path.exists():
        print(
            "WARNING: Rossmann store.csv is missing. "
            "Training data was processed, but store metadata "
            "has not been joined."
        )


# --------------------------------------------------
# 7. Process Walmart
# --------------------------------------------------

def process_walmart():
    path = DATA_DIR / "walmart" / "train.csv"

    if not path.exists():
        print("\nWalmart train.csv not found.")
        return

    print("\n" + "=" * 60)
    print("PROCESSING WALMART")
    print("=" * 60)

    df = read_tab_file(path)

    df = convert_date(df, "Date", dayfirst=True)

    df = df.drop_duplicates()
    df = df.dropna(subset=["Date", "Weekly_Sales"])

    save_data(df, "walmart_train_clean.csv")

    # Inspect features separately. Do not drop rows just because
    # markdown or economic indicator values are missing.
    features_path = DATA_DIR / "walmart" / "features.csv"

    if features_path.exists():
        features = read_tab_file(features_path)
        features = convert_date(features, "Date", dayfirst=True)

        print("\nWalmart features loaded.")
        print("Feature rows:", f"{len(features):,}")
        print("Missing values by column:")
        print(features.isna().sum().to_string())


# --------------------------------------------------
# 8. Process Superstore
# --------------------------------------------------

def process_superstore():
    folder = DATA_DIR / "superstore"

    files = [
        p for p in folder.glob("*.csv")
        if p.name.lower() != "superstore.csv"
    ] if folder.exists() else []

    if not files:
        print("No clean Superstore CSV found.")
        return

    path = files[0]
    print(f"Using Superstore file: {path}")
    df = read_superstore(path)

# --------------------------------------------------
# 9. Main program
# --------------------------------------------------

def main():
    print("RETAIL SALES FORECASTING - PREPROCESSING")
    print("Original source files will not be modified.")

    process_rossmann()
    process_walmart()
    process_superstore()

    print("\n" + "=" * 60)
    print("PREPROCESSING RUN FINISHED")
    print(f"Processed files directory: {OUTPUT_DIR}")
    print("=" * 60)
    


if __name__ == "__main__":
    main()