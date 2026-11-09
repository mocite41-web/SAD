# Student Performance Prediction System

## Problem
This classroom prototype explores whether historical student attributes and academic indicators can help flag students who may need academic support.

## Dataset
UCI Machine Learning Repository — Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction target
- `G3 < 10`: **AT RISK** (class 1)
- `G3 >= 10`: **NOT AT RISK** (class 0)

`G3` is the target and is excluded from the model's input features to prevent target leakage.

## Technologies
Python, pandas, scikit-learn, Random Forest, joblib, Streamlit, UCI repository interface.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
The app normally opens at http://localhost:8501.

## Files
- `train_model.py`: downloads data, preprocesses it, trains/evaluates the model, saves CSV and model.
- `app.py`: Streamlit dashboard, CSV upload predictions, and downloads.
- `requirements.txt`: Python dependencies.

## Limitations and responsible use
This is a classroom demonstration based on public historical data. It may not generalize to other schools or populations. Predictions are not diagnoses and must not be the sole basis for real student decisions. Faculty should review the context and provide appropriate support rather than treating the prediction as a final judgment.
