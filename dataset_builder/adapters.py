"""Adattatori: convertono Mendeley, DiaBD e Pima nello schema canonico."""
import os, re, pandas as pd
from dataset_builder.canonical_schema import COLUMNS

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def find_column(df, aliases):
    lookup = {normalize(c): c for c in df.columns}
    return next((lookup[normalize(a)] for a in aliases if normalize(a) in lookup), None)

def numeric(df, aliases):
    c = find_column(df, aliases)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series(pd.NA, index=df.index)

def text(df, aliases):
    c = find_column(df, aliases)
    return df[c].astype(str).str.strip().str.lower() if c else pd.Series(pd.NA, index=df.index)

def standard_target(series):
    x = series.astype(str).str.strip().str.lower()
    out = pd.Series("normal", index=series.index)
    out[x.isin(["p", "prediabetes", "prediabetic"]) | x.str.contains("prediabet", na=False)] = "prediabetes"
    out[x.isin(["1", "1.0", "yes", "y", "true", "positive"]) | (x.str.contains("diabet", na=False) & ~x.str.contains("pre", na=False))] = "type2_diabetes"
    return out

def empty_frame(df, source):
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = source
    out["source_row"] = df.index
    for c in COLUMNS:
        if c not in out:
            out[c] = pd.NA
    return out

def adapt_mendeley(df):
    o = empty_frame(df, "mendeley")
    mapping = {
        "age_years": ["age"],
        "fasting_glucose_mg_dl": ["sugarlevelblood", "bloodsugar", "glucose"],
        "bmi": ["bmi"],
        "hba1c_percent": ["hba1c"],
        "creatinine_mg_dl": ["creatinineratio", "creatinine", "cr"],
        "urea_mg_dl": ["urea"],
        "total_cholesterol_mg_dl": ["cholesterol", "chol"],
        "hdl_mg_dl": ["hdl"],
        "ldl_mg_dl": ["ldl"],
        "vldl_mg_dl": ["vldl"],
        "triglycerides_mg_dl": ["triglycerides", "tg"],
    }
    for k, a in mapping.items():
        o[k] = numeric(df, a)
    
    # Conversione unità di misura da mmol/L e umol/L a mg/dL
    o["triglycerides_mg_dl"] = o["triglycerides_mg_dl"] * 88.57
    o["total_cholesterol_mg_dl"] = o["total_cholesterol_mg_dl"] * 38.67
    o["hdl_mg_dl"] = o["hdl_mg_dl"] * 38.67
    o["ldl_mg_dl"] = o["ldl_mg_dl"] * 38.67
    o["vldl_mg_dl"] = o["vldl_mg_dl"] * 38.67
    o["urea_mg_dl"] = o["urea_mg_dl"] * 6.0
    o["creatinine_mg_dl"] = o["creatinine_mg_dl"] / 88.4

    raw_sex = text(df, ["gender", "sex"])
    o["sex"] = raw_sex.map({"f": "female", "m": "male", "female": "female", "male": "male"}).fillna(raw_sex)
    c = find_column(df, ["class", "diagnosis", "diabetic", "outcome"])
    if c:
        o["diabetes_class"] = standard_target(df[c])
    return o

def adapt_diabd(df):
    o = empty_frame(df, "diabd")
    mapping = {
        "age_years": ["age"],
        "weight_kg": ["weight"],
        "height_cm": ["height"],
        "bmi": ["bmi"],
        "systolic_bp_mmhg": ["systolic_bp", "systolicbp"],
        "diastolic_bp_mmhg": ["diastolic_bp", "diastolicbp"],
        "heart_rate_bpm": ["pulse_rate", "pulserate"],
        "fasting_glucose_mg_dl": ["glucose"],
        "family_history_diabetes": ["family_diabetes", "family_history", "familyhistory"],
        "hypertension_history": ["hypertensive", "hypertension"],
        "heart_disease": ["cardiovascular_disease", "heartdisease"],
    }
    for k, a in mapping.items():
        o[k] = numeric(df, a)

    # Conversione altezza (metri -> cm) e glicemia (mmol/L -> mg/dL)
    o.loc[o["height_cm"] < 3, "height_cm"] = o.loc[o["height_cm"] < 3, "height_cm"] * 100
    o["fasting_glucose_mg_dl"] = o["fasting_glucose_mg_dl"] * 18.0182

    raw_sex = text(df, ["gender", "sex"])
    o["sex"] = raw_sex.map({"f": "female", "m": "male", "female": "female", "male": "male"}).fillna(raw_sex)
    c = find_column(df, ["diabetic", "diabetes", "outcome"])
    if c:
        o["diabetes_class"] = standard_target(df[c])
    return o

def adapt_pima(df):
    o = empty_frame(df, "pima")
    mapping = {
        "fasting_glucose_mg_dl": ["glucose"],
        "diastolic_bp_mmhg": ["bloodpressure"],
        "insulin_uiu_ml": ["insulin"],
        "bmi": ["bmi"],
        "age_years": ["age"],
    }
    for k, a in mapping.items():
        o[k] = numeric(df, a)
    o["sex"] = "female"  # Pima Indian Dataset include solo donne
    c = find_column(df, ["outcome", "diabetes"])
    if c:
        o["diabetes_class"] = standard_target(df[c])
    return o

def adapt(path, df):
    name = os.path.basename(path).lower()
    cols = " ".join(normalize(c) for c in df.columns)
    if "diabd" in name or ("systolic" in cols and "pulse" in cols):
        return adapt_diabd(df)
    if "pima" in name or ("pregnancies" in cols and "insulin" in cols):
        return adapt_pima(df)
    return adapt_mendeley(df)

