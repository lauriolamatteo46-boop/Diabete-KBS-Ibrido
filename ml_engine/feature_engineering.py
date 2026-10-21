"""Discretizzazione centralizzata: mantiene identiche le soglie in training e GUI."""
import pandas as pd

def discretize(df: pd.DataFrame) -> pd.DataFrame:
    o = pd.DataFrame(index=df.index)
    num = lambda c: pd.to_numeric(
        df[c] if c in df.columns else pd.Series(float("nan"), index=df.index),
        errors="coerce"
    )

    o["age_group"] = pd.cut(
        num("age_years"), [-1, 44, 64, 200], labels=["young", "middle", "older"]
    ).astype(str).fillna("unknown").replace({"nan": "unknown", "<NA>": "unknown"})

    o["bmi_group"] = pd.cut(
        num("bmi"), [-1, 24.9, 29.9, 200], labels=["normal", "overweight", "obese"]
    ).astype(str).fillna("unknown").replace({"nan": "unknown", "<NA>": "unknown"})

    o["glucose_group"] = pd.cut(
        num("fasting_glucose_mg_dl"), [-1, 99, 125, 2000], labels=["normal", "borderline", "high"]
    ).astype(str).fillna("unknown").replace({"nan": "unknown", "<NA>": "unknown"})

    o["hba1c_group"] = pd.cut(
        num("hba1c_percent"), [-1, 5.6, 6.4, 100], labels=["normal", "borderline", "high"]
    ).astype(str).fillna("unknown").replace({"nan": "unknown", "<NA>": "unknown"})

    s, d = num("systolic_bp_mmhg"), num("diastolic_bp_mmhg")
    o["bp_group"] = "normal"
    o.loc[(s >= 140) | (d >= 90), "bp_group"] = "high"
    o.loc[s.isna() & d.isna(), "bp_group"] = "unknown"

    t = num("triglycerides_mg_dl")
    o["lipid_risk"] = "normal"
    o.loc[t >= 150, "lipid_risk"] = "high"
    o.loc[t.isna(), "lipid_risk"] = "unknown"

    return o

