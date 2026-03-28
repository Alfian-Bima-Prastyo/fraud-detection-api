---
title: Fraud Detection API
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
---

# Fraud Detection API

REST API untuk deteksi fraud transaksi e-commerce menggunakan LightGBM (tuned).

## Model Performance
| Metric | Value |
|--------|-------|
| PR-AUC | 0.5951 |
| ROC-AUC | 0.9201 |
| Recall | 84% |
| Threshold | 0.0198 |
| Est. Cost Savings | 67% (247,995) |

## Endpoints

### `GET /health`
```json
{
  "status": "ok",
  "model_type": "LightGBM (tuned)",
  "threshold": 0.0198,
  "n_features": 396
}
```

### `POST /predict`
Request:
```json
{
  "data": {
    "TransactionAmt": 46.725,
    "ProductCD": "C",
    "card4": "mastercard",
    ...
  }
}
```
Response:
```json
{
  "fraud_proba": 0.3039,
  "fraud_pred": 1,
  "threshold": 0.0198
}
```

## Notes
- Input: 378 raw features (pre-transform) — pipeline handle sisanya
- `xgb_fraud_v1.pkl` berisi LightGBM tuned (naming dari training)
- Deploy: `uvicorn app:app --host 0.0.0.0 --port 7860`
```

---

### Struktur `fraud-detection-api/` sekarang
```
fraud-detection-api/
├── app.py               ✅ sudah ada (copy dari artifacts/)
├── requirements.txt     ← buat sekarang
├── README.md            ← buat sekarang
└── artifacts/
    ├── xgb_fraud_v1.pkl
    ├── preprocessing_pipeline.pkl
    ├── feature_columns.pkl
    ├── frequency_maps.pkl
    ├── outlier_caps.pkl
    ├── threshold.pkl
    ├── encoding_schema.json
    ├── eda_feature_decisions.json
    ├── model_metadata.json
    ├── inference_schema.json
    ├── sample_payload.json
    ├── monitoring_baseline.pkl
    └── drift_thresholds.json