# app.py — Fraud Detection API
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any

# Paths
ARTIFACTS = Path("artifacts")

# Load artifacts at startup
with open(ARTIFACTS / "LightGBM_fraud_v1.pkl", "rb") as f:
    model = pickle.load(f)

with open(ARTIFACTS / "preprocessing_pipeline.pkl", "rb") as f:
    bundle = pickle.load(f)

with open(ARTIFACTS / "threshold.pkl", "rb") as f:
    th_obj = pickle.load(f)
THRESHOLD = th_obj["threshold"]

with open(ARTIFACTS / "model_metadata.json") as f:
    metadata = json.load(f)

META_COLS       = ["isFraud", "TransactionDT", "TransactionID"]
RAW_STRING_COLS = ["id_30", "id_31", "id_33", "id_34", "DeviceInfo", "day_bin"]

# Preprocessing
def preprocess(df, bundle):
    df = df.copy()
    existing_drop = [c for c in bundle["drop_cols"] if c in df.columns]
    df = df.drop(columns=existing_drop)
    for col in bundle["missing_flag_cols"]:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    for col in bundle["binary_cols"]:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0})
    for col in bundle["binary_device_cols"]:
        if col in df.columns:
            df[col] = df[col].map({"desktop": 0, "mobile": 1})
    for col, mapping in bundle["label_map"].items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    for col, freq_map in bundle["freq_maps"].items():
        if col in df.columns:
            df[col] = df[col].map(freq_map).fillna(0.0)
    return df

def get_feature_df(df):
    drop = [c for c in META_COLS + RAW_STRING_COLS if c in df.columns]
    return df.drop(columns=drop)

def predict(raw_dict: dict) -> dict:
    df      = pd.DataFrame([raw_dict])
    df_proc = preprocess(df, bundle)
    X       = bundle["preprocessor"].transform(get_feature_df(df_proc))
    proba   = float(model.predict_proba(X)[:, 1][0])
    pred    = int(proba >= THRESHOLD)
    return {"fraud_proba": proba, "fraud_pred": pred, "threshold": THRESHOLD}

# FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0.0")

class TransactionPayload(BaseModel):
    data: dict[str, Any]

@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "model_type": metadata["model_type"],
        "threshold" : THRESHOLD,
        "n_features": metadata["n_features"],
    }

@app.post("/predict")
def predict_endpoint(payload: TransactionPayload):
    try:
        result = predict(payload.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))