from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"

for folder in [RESULTS, MODELS, FIGURES, REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)

def read_file(path):
    for sep in [None, "\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
            if len(df.columns) > 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            pass
    raise ValueError("Could not read " + str(path))

def find_dataset(name):
    if name == "Rossmann":
        paths = [
            DATA / "processed" / "rossmann_train_clean.csv",
            DATA / "rossmann" / "train.csv"
        ]
    elif name == "Walmart":
        paths = [
            DATA / "processed" / "walmart_train_clean.csv",
            DATA / "walmart" / "train.csv"
        ]
    else:
        paths = [
            DATA / "processed" / "superstore_clean.csv"
        ]
        folder = DATA / "superstore"
        if folder.exists():
            paths += [
                p for p in folder.glob("*.csv")
                if "backup" not in p.name.lower()
                and p.name.lower() != "superstore.csv"
            ]

    for path in paths:
        if path.exists():
            try:
                df = read_file(path)
                if any(c in df.columns for c in ["Sales", "Weekly_Sales"]):
                    return df, path
            except Exception as e:
                print("Could not read", path.name, ":", e)
    return None, None

def prepare(df):
    df = df.copy()
    target = "Weekly_Sales" if "Weekly_Sales" in df.columns else "Sales"
    date_col = next((c for c in ["Date", "Order Date"] if c in df.columns), None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["quarter"] = df[date_col].dt.quarter
        df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])

    if len(df) > 60000:
        df = df.tail(60000).copy()

    excluded = [target]
    if date_col:
        excluded.append(date_col)

    X = df.drop(columns=excluded, errors="ignore").copy()

    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64") // 10**9
        elif X[col].dtype == "object":
            X[col] = X[col].fillna("Missing").astype(str)
            X[col] = pd.factorize(X[col])[0]
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X = X.fillna(X.median()).fillna(0)

    y = df[target].astype(float)
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), target, date_col

def run_dataset(name):
    print("\n" + "=" * 55)
    print("PROCESSING:", name)
    print("=" * 55)

    df, source = find_dataset(name)
    if df is None:
        print("SKIPPED: Dataset not found or not readable.")
        return None

    df, X, y, target, date_col = prepare(df)
    if len(df) < 40 or X.shape[1] == 0:
        print("SKIPPED: Insufficient usable rows or features.")
        return None

    split = int(len(df) * 0.8)
    split = max(20, min(split, len(df) - 1))

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=14, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=50, max_depth=14, n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, random_state=42)
    }

    rows = []
    fitted = {}
    predictions = {}

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fitted[model_name] = model
            predictions[model_name] = pred
            rows.append({
                "Dataset": name,
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred)
            })
            print(model_name, "completed")
        except Exception as e:
            print(model_name, "failed:", e)

    if not rows:
        print("No model trained successfully.")
        return None

    metrics = pd.DataFrame(rows).sort_values("RMSE")
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = predictions[best_name]

    safe = name.lower()
    metrics.to_csv(RESULTS / f"{safe}_model_comparison.csv", index=False)

    prediction_df = pd.DataFrame({
        "Actual_Sales": y_test,
        "Predicted_Sales": best_pred
    })
    if date_col:
        prediction_df.insert(0, "Date", df[date_col].iloc[split:].values)
    prediction_df.to_csv(RESULTS / f"{safe}_test_predictions.csv", index=False)

    model_path = MODELS / f"{safe}_best_model.joblib"
    joblib.dump({
        "model": best_model,
        "features": list(X.columns),
        "target": target,
        "dataset": name,
        "model_name": best_name
    }, model_path)

    # Future forecast: use recent feature profile with future calendar features.
    # This is a baseline forecast, not a fully recursive forecasting model.
    horizon = 30
    future = X.tail(horizon).copy().reset_index(drop=True)

    if date_col:
        last_date = df[date_col].iloc[-1]
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        if "year" in future:
            future["year"] = dates.year
        if "month" in future:
            future["month"] = dates.month
        if "day" in future:
            future["day"] = dates.day
        if "day_of_week" in future:
            future["day_of_week"] = dates.dayofweek
        if "quarter" in future:
            future["quarter"] = dates.quarter
        if "week_of_year" in future:
            future["week_of_year"] = dates.isocalendar().week.astype(int).values
    else:
        dates = None

    forecast_values = best_model.predict(future)
    forecast_df = pd.DataFrame({"Forecast_Sales": forecast_values})
    if dates is not None:
        forecast_df.insert(0, "Forecast_Date", dates)
    forecast_df.to_csv(RESULTS / f"{safe}_30_period_forecast.csv", index=False)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics["Model"], metrics["RMSE"])
    ax.set_title(f"{name}: Model RMSE Comparison")
    ax.set_ylabel("RMSE (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{safe}_model_comparison.png", dpi=150)
    plt.close(fig)

    # Actual vs predicted chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(y_test)), y_test.values, label="Actual")
    ax.plot(np.arange(len(best_pred)), best_pred, label="Predicted")
    ax.set_title(f"{name}: Actual vs Predicted Sales")
    ax.set_xlabel("Test record")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{safe}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    # Forecast chart
    fig, ax = plt.subplots(figsize=(10, 5))
    if date_col:
        ax.plot(dates, forecast_values, marker="o")
        ax.set_xlabel("Forecast date")
    else:
        ax.plot(range(1, horizon + 1), forecast_values, marker="o")
        ax.set_xlabel("Forecast period")
    ax.set_title(f"{name}: Baseline 30-Period Forecast")
    ax.set_ylabel("Predicted sales")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{safe}_forecast.png", dpi=150)
    plt.close(fig)

    # Feature importance when supported
    if hasattr(best_model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        imp.to_csv(RESULTS / f"{safe}_feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        top = imp.head(15).sort_values("Importance")
        ax.barh(top["Feature"], top["Importance"])
        ax.set_title(f"{name}: Feature Importance")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{safe}_feature_importance.png", dpi=150)
        plt.close(fig)

    print("Source:", source)
    print("Rows:", len(df))
    print("Features:", X.shape[1])
    print("Best model:", best_name)
    print(metrics.to_string(index=False))
    print("Saved model, metrics, predictions, forecast and graphs.")

    return metrics

def main():
    all_metrics = []
    for name in ["Rossmann", "Walmart", "Superstore"]:
        try:
            result = run_dataset(name)
            if result is not None:
                all_metrics.append(result)
        except Exception as e:
            print("ERROR processing", name, ":", repr(e))

    if not all_metrics:
        print("\nNo datasets were successfully processed.")
        print("Check the data folders and dataset formats.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(RESULTS / "all_dataset_model_comparison.csv", index=False)

    lines = [
        "# Retail Sales Forecasting Project — Execution Report",
        "",
        "## Project title",
        "An Explainable and Time-Aware Comparative Framework for Retail Sales Forecasting Using Machine Learning Regression Models",
        "",
        "## Method",
        "Three retail datasets were checked. Available datasets were cleaned, sorted by date where a date column was available, and transformed into numeric model features.",
        "An ordered 80/20 train-test split was used. Regression models were compared using MAE, RMSE and R-squared.",
        "",
        "## Actual experiment results",
        combined.to_markdown(index=False),
        "",
        "## Generated outputs",
        "- Model comparison CSV files",
        "- Held-out test prediction CSV files",
        "- Baseline 30-period forecast CSV files",
        "- Saved best-performing model files",
        "- Model comparison, prediction and forecast plots",
        "- Feature importance files for models that support it",
        "",
        "## Important limitations",
        "The 30-period forecast is a baseline scenario generated using recent feature profiles with updated calendar features. It is not a validated recursive or multi-step time-series forecasting method.",
        "Categorical values are encoded for this prototype. Dataset-specific feature engineering and leakage audits should be performed before research publication.",
        "Feature importance is not the same as causal explanation. SHAP and LIME explanations should be generated and interpreted separately.",
        "Results apply only to datasets successfully processed in this execution.",
        "",
        "## Conclusion",
        "This execution provides a comparative machine learning prototype for retail sales forecasting. Model performance should be interpreted using the reported held-out test metrics."
    ]
    (REPORTS / "final_project_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Generate a simple PowerPoint report
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt

        prs = Presentation()

        def slide(title, body):
            s = prs.slides.add_slide(prs.slide_layouts[1])
            s.shapes.title.text = title
            s.placeholders[1].text = body
            return s

        s = prs.slides.add_slide(prs.slide_layouts[0])
        s.shapes.title.text = "Retail Sales Forecasting"
        s.placeholders[1].text = "An Explainable and Time-Aware Comparative Framework Using Machine Learning Regression Models"

        slide("Problem Statement",
              "Retail businesses need reliable sales estimates. This project compares regression models and explores historical sales patterns to support forecasting and interpretation.")

        slide("Objectives",
              "1. Prepare three retail datasets.\n2. Compare regression algorithms.\n3. Evaluate models using MAE, RMSE and R2.\n4. Generate baseline future forecasts.\n5. Explore explainability using SHAP and LIME.")

        slide("Methodology",
              "Dataset loading → preprocessing → feature engineering → time-ordered train/test split → model training → evaluation → forecasting → dashboard.")

        slide("Datasets",
              "Rossmann Store Sales\nWalmart Sales\nSuperstore Sales\nOnly datasets successfully processed are included in the experiment results.")

        slide("Model Comparison Results",
              combined.round(3).to_string(index=False)[:3500])

        slide("Limitations",
              "The baseline forecast is not a validated recursive time-series model. Further feature engineering, leakage checks and dataset-specific validation are needed.")

        slide("Conclusion",
              "The prototype compares regression models across available retail datasets and provides forecast and interpretability outputs for further evaluation.")

        prs.save(REPORTS / "retail_sales_forecasting_presentation.pptx")
        print("PowerPoint created.")
    except Exception as e:
        print("PowerPoint generation skipped:", e)

    print("\n" + "=" * 55)
    print("PROJECT BUILD FINISHED")
    print("=" * 55)
    print("Successful datasets:", combined["Dataset"].unique().tolist())
    print("Metrics:", RESULTS / "all_dataset_model_comparison.csv")
    print("Report:", REPORTS / "final_project_report.md")
    print("Models:", MODELS)
    print("Charts:", FIGURES)
    print("Presentation:", REPORTS / "retail_sales_forecasting_presentation.pptx")

if __name__ == "__main__":
    main()
