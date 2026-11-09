import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

MODEL_FILE = Path("student_model.pkl")
DATA_FILE = Path("student_data.csv")

@st.cache_resource
def prepare_artifacts():
    # Streamlit Cloud does not have locally generated model/data files initially.
    # Train once automatically if either file is missing.
    if not MODEL_FILE.exists() or not DATA_FILE.exists():
        with st.spinner("Preparing dataset and training the model for first use..."):
            subprocess.run(
                [sys.executable, "train_model.py"],
                check=True,
                capture_output=True,
                text=True
            )
    model = joblib.load(MODEL_FILE)
    data = pd.read_csv(DATA_FILE)
    return model, data

st.title("🎓 Student Performance Prediction")
st.caption(
    "Classroom prototype for academic risk exploration. "
    "Predictions are not diagnoses and must not be used as the sole basis for real student decisions."
)

try:
    model, data = prepare_artifacts()
except Exception as error:
    st.error("Unable to prepare the dataset/model. Please check the deployment logs.")
    st.exception(error)
    st.stop()

if "G3" not in data.columns:
    st.error("Dataset must contain the G3 final grade column.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data. Do not include G3.")
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])
if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        required_columns = list(model.feature_names_in_)
        missing = [column for column in required_columns if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif new_data.empty:
            st.warning("The uploaded CSV has no records.")
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "AT RISK" if int(value) == 1 else "NOT AT RISK"
                for value in predictions
            ]
            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted at-risk records", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "student_predictions.csv",
                "text/csv"
            )
    except Exception as error:
        st.error(f"Could not process this CSV: {error}")
