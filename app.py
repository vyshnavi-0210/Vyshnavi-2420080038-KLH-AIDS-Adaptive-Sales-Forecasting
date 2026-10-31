from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

st.set_page_config(page_title="Retail Sales Forecasting", page_icon="📊", layout="wide")

st.title("📊 Explainable and Time-Aware Retail Sales Forecasting")
st.caption("Comparative Machine Learning Framework | Rossmann • Walmart • Superstore")

@st.cache_data(show_spinner=False)
def load_dataset(name):
    folders = {
        "Rossmann": DATA / "rossmann",
        "Walmart": DATA / "walmart",
        "Superstore": DATA / "superstore"
    }
    folder = folders[name]

    if name == "Rossmann":
        candidates = [folder / "train.csv", DATA / "processed" / "rossmann_train_clean.csv"]
    elif name == "Walmart":
        candidates = [folder / "train.csv", DATA / "processed" / "walmart_train_clean.csv"]
    else:
        candidates = [
            p for p in folder.glob("*.csv")
            if "backup" not in p.name.lower() and "superstore.csv" != p.name.lower()
        ] if folder.exists() else []
        candidates += list((DATA / "processed").glob("superstore_clean.csv"))

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
            if len(df.columns) <= 1:
                df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            target = next((c for c in ["Sales", "Weekly_Sales"] if c in df.columns), None)
            if target and len(df) > 20:
                return df, target, str(path)
        except Exception:
            continue
    return None, None, None

def prepare(df, target):
    data = df.copy()
    date_col = next((c for c in ["Date", "Order Date"] if c in data.columns), None)

    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce", dayfirst=True)
        data = data.dropna(subset=[date_col])
        data["year"] = data[date_col].dt.year
        data["month"] = data[date_col].dt.month
        data["day"] = data[date_col].dt.day
        data["day_of_week"] = data[date_col].dt.dayofweek
        data["quarter"] = data[date_col].dt.quarter
        data = data.sort_values(date_col)

    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    data = data.replace([np.inf, -np.inf], np.nan)

    if len(data) > 50000:
        data = data.sample(50000, random_state=42).sort_index()

    excluded = [target]
    if date_col:
        excluded.append(date_col)

    X = data.drop(columns=excluded, errors="ignore")
    y = data[target]

    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64") // 10**9
        if X[col].dtype == "object":
            X[col] = X[col].fillna("Missing").astype(str)
            X[col] = pd.factorize(X[col])[0]
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X = X.fillna(X.median()).fillna(0)
    return data, X, y, date_col

@st.cache_resource(show_spinner="Training and comparing models...")
def train_models(name):
    df, target, path = load_dataset(name)
    if df is None:
        return None

    data, X, y, date_col = prepare(df, target)
    if len(X) < 30 or X.shape[1] == 0:
        return None

    split = int(len(X) * 0.8)
    split = max(1, min(split, len(X)-1))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=40, max_depth=12, n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=70, max_leaf_nodes=15, random_state=42)
    }

    results = []
    fitted = {}
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[model_name] = model
        predictions[model_name] = pred
        results.append({
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        })

    metrics = pd.DataFrame(results).sort_values("RMSE")
    best_name = metrics.iloc[0]["Model"]
    return {
        "data": data, "X": X, "y": y, "X_test": X_test, "y_test": y_test,
        "metrics": metrics, "models": fitted, "predictions": predictions,
        "best": best_name, "target": target, "path": path, "date_col": date_col
    }

with st.sidebar:
    st.header("Project Controls")
    dataset_name = st.selectbox("Select dataset", ["Rossmann", "Walmart", "Superstore"])
    st.info("Choose a dataset to view its analysis and model results.")
    st.caption("Evaluation uses an ordered 80/20 train-test split.")

result = train_models(dataset_name)

if result is None:
    st.error("Dataset could not be loaded or has insufficient valid data.")
    st.write("Check that the dataset is inside data/rossmann, data/walmart or data/superstore.")
    st.write("Rossmann and Walmart need train.csv. Superstore needs a clean CSV with Order Date and Sales.")
    st.stop()

df = result["data"]
metrics = result["metrics"]
best = result["best"]
best_pred = result["predictions"][best]
y_test = result["y_test"]

st.success(f"Loaded dataset: {dataset_name} | Source: {result['path']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records", f"{len(df):,}")
c2.metric("Features used", result["X"].shape[1])
c3.metric("Best model", best)
c4.metric("Best model RMSE", f"{metrics.iloc[0]['RMSE']:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sales Overview", "Model Comparison", "Forecast", "Explainability", "Dataset Details"
])

with tab1:
    st.subheader("Retail Sales Overview")
    st.metric("Total sales", f"{df[result['target']].sum():,.2f}")
    st.metric("Average sales", f"{df[result['target']].mean():,.2f}")
    if result["date_col"]:
        trend = df.groupby(result["date_col"])[result["target"]].sum().reset_index()
        st.plotly_chart(px.line(trend, x=result["date_col"], y=result["target"],
                                title="Sales Trend Over Time"), use_container_width=True)
    else:
        st.plotly_chart(px.histogram(df, x=result["target"], title="Sales Distribution"),
                        use_container_width=True)

with tab2:
    st.subheader("Regression Model Performance")
    st.dataframe(metrics.round(3), use_container_width=True)
    st.plotly_chart(px.bar(metrics, x="Model", y="RMSE", title="RMSE Comparison",
                           text_auto=".2s"), use_container_width=True)
    st.plotly_chart(px.bar(metrics, x="Model", y="R2", title="R² Score Comparison",
                           text_auto=".3f"), use_container_width=True)
    comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": best_pred})
    st.plotly_chart(px.scatter(comparison, x="Actual", y="Predicted",
                               title=f"Actual vs Predicted — {best}"), use_container_width=True)

with tab3:
    st.subheader("Sales Prediction")
    st.write("The model predicts sales for new input feature values.")
    model = result["models"][best]
    selected_row = st.slider("Select a test record", 0, len(result["X_test"])-1, 0)
    row = result["X_test"].iloc[[selected_row]]
    prediction = float(model.predict(row)[0])
    actual = float(y_test.iloc[selected_row])
    a, b = st.columns(2)
    a.metric("Predicted sales", f"{prediction:,.2f}")
    b.metric("Actual sales", f"{actual:,.2f}")
    st.caption("This is a held-out test-record prediction, not a claim of a validated future forecast.")

with tab4:
    st.subheader("Explainability — Feature Importance")
    model = result["models"]["Random Forest"]
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": result["X"].columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)
        st.plotly_chart(px.bar(importance.sort_values("Importance"),
                               x="Importance", y="Feature", orientation="h",
                               title="Random Forest Feature Importance"),
                        use_container_width=True)
        st.write("Features with larger importance values contributed more to the Random Forest's predictive splits.")
    else:
        st.info("Feature importance is not available for this model.")

with tab5:
    st.subheader("Dataset Information")
    st.write("Dataset dimensions:", df.shape)
    st.write("Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing Values"), use_container_width=True)
    st.write("Preview")
    st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.caption("Academic prototype. Model metrics are experimental and depend on dataset quality, feature selection and time ordering.")

# ==========================================
# EXPLAINABLE AI: SHAP AND LIME
# ==========================================

import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

st.divider()
st.header("Explainable AI: SHAP and LIME")
st.write("Understand which features influence the sales predictions.")

rf_model = result["models"]["Random Forest"]
feature_names = list(result["X"].columns)

explain_tab1, explain_tab2 = st.tabs(["SHAP Explanation", "LIME Explanation"])

with explain_tab1:
    st.subheader("SHAP Feature Importance")
    st.write("SHAP estimates how much each feature contributes to model predictions.")

    try:
        sample = result["X"].sample(
            min(100, len(result["X"])),
            random_state=42
        )

        shap_explainer = shap.TreeExplainer(rf_model)
        shap_values = shap_explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        if getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[:, :, 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            sample,
            plot_type="bar",
            show=False,
            max_display=15
        )
        st.pyplot(plt.gcf())
        plt.close("all")

        st.success("SHAP explanation generated successfully.")

    except Exception as e:
        st.error(f"SHAP explanation error: {e}")

with explain_tab2:
    st.subheader("LIME Individual Prediction Explanation")
    st.write("LIME explains why the selected record received its predicted sales value.")

    try:
        training_sample = result["X"].sample(
            min(500, len(result["X"])),
            random_state=42
        )

        lime_explainer = LimeTabularExplainer(
            training_data=training_sample.to_numpy(),
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=True
        )

        record_number = st.slider(
            "Choose a test record for LIME",
            min_value=0,
            max_value=len(result["X_test"]) - 1,
            value=0,
            key="lime_record"
        )

        selected_record = result["X_test"].iloc[record_number].to_numpy()

        explanation = lime_explainer.explain_instance(
            selected_record,
            lambda values: rf_model.predict(
                pd.DataFrame(values, columns=feature_names)
            ),
            num_features=min(10, len(feature_names))
        )

        st.metric(
            "Random Forest predicted sales",
            f"{rf_model.predict(result['X_test'].iloc[[record_number]])[0]:,.2f}"
        )

        lime_data = pd.DataFrame(
            explanation.as_list(),
            columns=["Feature condition", "Contribution"]
        )

        st.dataframe(lime_data, use_container_width=True)

        lime_fig = explanation.as_pyplot_figure()
        st.pyplot(lime_fig)
        plt.close(lime_fig)

        st.success("LIME explanation generated successfully.")

    except Exception as e:
        st.error(f"LIME explanation error: {e}")

