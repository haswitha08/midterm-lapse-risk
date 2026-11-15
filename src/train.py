"""Train the household lapse-risk model (ML Zoomcamp 2026 midterm).

Usage (from the project root):
    python src/train.py --raw data/raw --out models
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

FEATURES = ["recency", "tenure", "spend_4w", "spend_12w", "baskets_4w", "baskets_12w",
            "active_weeks_12w", "avg_basket_12w", "disc_share_12w", "redeem_12w",
            "redeem_ever"]
PARAMS = dict(n_estimators=100, max_depth=4, learning_rate=0.02, min_child_weight=1,
              colsample_bytree=0.8, subsample=0.8, eval_metric="logloss",
              random_state=42, n_jobs=-1)
THRESHOLD = 0.30
HORIZON = 4
FIRST_CUTOFF, LAST_CUTOFF = 12, 98


def build_snapshots(raw: Path) -> pd.DataFrame:
    """One row per (household, cut-off week). Features use weeks <= cut-off only;
    the label looks at the next HORIZON weeks."""
    tx = pd.read_csv(raw / "transaction_data.csv",
                     usecols=["household_key", "BASKET_ID", "DAY", "SALES_VALUE",
                              "RETAIL_DISC", "COUPON_DISC", "WEEK_NO"])
    tx["disc"] = -(tx["RETAIL_DISC"] + tx["COUPON_DISC"])
    tx["gross"] = tx["SALES_VALUE"] + tx["disc"]
    weekly = (tx.groupby(["household_key", "WEEK_NO"])
                .agg(spend=("SALES_VALUE", "sum"), baskets=("BASKET_ID", "nunique"),
                     disc=("disc", "sum"), gross=("gross", "sum"))
                .reset_index())
    weeks = np.arange(1, int(tx["WEEK_NO"].max()) + 1)
    hh_ids = np.sort(weekly["household_key"].unique())

    def mat(df, col):
        return (df.pivot(index="household_key", columns="WEEK_NO", values=col)
                  .reindex(index=hh_ids, columns=weeks).fillna(0).to_numpy())

    S, B, D, G = (mat(weekly, c) for c in ["spend", "baskets", "disc", "gross"])
    A = B > 0

    # coupon redemptions are recorded by DAY: map each day to its WEEK_NO
    wk_end = tx.groupby("WEEK_NO")["DAY"].max().sort_index().to_numpy()
    red = pd.read_csv(raw / "coupon_redempt.csv")
    red["WEEK_NO"] = np.searchsorted(wk_end, red["DAY"]) + 1
    rw = red.groupby(["household_key", "WEEK_NO"]).size().rename("n").reset_index()
    R = mat(rw, "n")

    first_week = A.argmax(axis=1) + 1
    last_seen = np.maximum.accumulate(np.where(A, weeks, 0), axis=1)

    frames = []
    for t in range(FIRST_CUTOFF, LAST_CUTOFF + 1):
        c = t - 1                                   # column index of week t
        elig = first_week <= t                      # household already seen by week t
        sp12, bk12 = S[:, c-11:c+1].sum(axis=1), B[:, c-11:c+1].sum(axis=1)
        g12, d12 = G[:, c-11:c+1].sum(axis=1), D[:, c-11:c+1].sum(axis=1)
        frame = pd.DataFrame({
            "household_key": hh_ids, "cutoff_week": t,
            "recency": t - last_seen[:, c],
            "tenure": t - first_week + 1,
            "spend_4w": S[:, c-3:c+1].sum(axis=1), "spend_12w": sp12,
            "baskets_4w": B[:, c-3:c+1].sum(axis=1), "baskets_12w": bk12,
            "active_weeks_12w": A[:, c-11:c+1].sum(axis=1),
            "avg_basket_12w": np.where(bk12 > 0, sp12 / np.maximum(bk12, 1), 0.0),
            "disc_share_12w": np.where(g12 > 0, d12 / np.maximum(g12, 1e-9), 0.0),
            "redeem_12w": R[:, c-11:c+1].sum(axis=1),
            "redeem_ever": R[:, :c+1].sum(axis=1),
            "lapse": (A[:, c+1:c+1+HORIZON].sum(axis=1) == 0).astype(int),
        })
        frames.append(frame[elig])
    return pd.concat(frames, ignore_index=True)


def evaluate(model, d: pd.DataFrame) -> dict:
    p = model.predict_proba(d[FEATURES])[:, 1]
    y = d["lapse"].to_numpy()
    pred = p >= THRESHOLD
    return {
        "rows": int(len(d)),
        "lapse_rate": round(float(y.mean()), 3),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "contacts_share": round(float(pred.mean()), 3),
        "precision_at_threshold": round(float(y[pred].mean()), 3),
        "recall_at_threshold": round(float(y[pred].sum() / y.sum()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    snap = build_snapshots(raw)
    print("snapshots:", snap.shape, "| lapse rate:", round(snap["lapse"].mean(), 3))

    # time-based split with 4-week gaps (labels look 4 weeks ahead)
    train = snap[snap["cutoff_week"] <= 66]
    val = snap[snap["cutoff_week"].between(71, 82)]
    test = snap[snap["cutoff_week"] >= 87]

    model = XGBClassifier(**PARAMS).fit(train[FEATURES], train["lapse"])
    metrics = {"validation": evaluate(model, val), "test": evaluate(model, test)}
    print(json.dumps(metrics, indent=2))

    # final model for the API: train + validation (cut-offs <= 82)
    fit_part = snap[snap["cutoff_week"] <= 82]
    final = XGBClassifier(**PARAMS).fit(fit_part[FEATURES], fit_part["lapse"])
    final.save_model(out / "model.json")

    meta = {"features": FEATURES, "threshold": THRESHOLD, "horizon_weeks": HORIZON,
            "tiers": {"high": 0.50, "medium": THRESHOLD},
            "params": PARAMS, "metrics": metrics}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # sanity check: the saved file must reload and give the same predictions
    loaded = XGBClassifier()
    loaded.load_model(out / "model.json")
    sample = test[FEATURES].head(200)
    assert np.allclose(loaded.predict_proba(sample)[:, 1],
                       final.predict_proba(sample)[:, 1], atol=1e-6)
    print("saved model reloads and matches: OK")


if __name__ == "__main__":
    main()