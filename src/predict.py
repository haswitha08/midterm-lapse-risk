"""Prediction service for household lapse risk (FastAPI)."""
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
META = json.loads((MODEL_DIR / "meta.json").read_text())
FEATURES = META["features"]
THRESHOLD = META["threshold"]
HIGH = META["tiers"]["high"]

model = XGBClassifier()
model.load_model(MODEL_DIR / "model.json")


class Household(BaseModel):
    recency: int = Field(..., ge=0, description="Weeks since the last purchase (0 = bought this week)")
    tenure: int = Field(..., ge=1, description="Weeks since the first observed purchase")
    spend_4w: float = Field(..., ge=0, description="Spend in the last 4 weeks")
    spend_12w: float = Field(..., ge=0, description="Spend in the last 12 weeks")
    baskets_4w: int = Field(..., ge=0, description="Shopping trips in the last 4 weeks")
    baskets_12w: int = Field(..., ge=0, description="Shopping trips in the last 12 weeks")
    active_weeks_12w: int = Field(..., ge=0, le=12, description="Weeks with a purchase, out of the last 12")
    avg_basket_12w: float = Field(..., ge=0, description="Average trip value over the last 12 weeks")
    disc_share_12w: float = Field(..., ge=0, description="Share of the pre-discount value that was discounted (last 12 weeks)")
    redeem_12w: int = Field(..., ge=0, description="Coupon redemptions in the last 12 weeks")
    redeem_ever: int = Field(..., ge=0, description="Coupon redemptions to date")


app = FastAPI(title="Household lapse-risk service", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/predict")
def predict(h: Household):
    row = pd.DataFrame([h.model_dump()])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    tier = "high" if p >= HIGH else "medium" if p >= THRESHOLD else "low"
    return {
        "lapse_probability": round(p, 4),
        "risk_tier": tier,
        "contact": bool(p >= THRESHOLD),
        "threshold": THRESHOLD,
    }