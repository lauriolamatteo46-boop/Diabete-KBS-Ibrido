"""Schema canonico condiviso da tutti gli adattatori di dataset."""
COLUMNS=["source_dataset","source_row","age_years","sex","weight_kg","height_cm","bmi","systolic_bp_mmhg","diastolic_bp_mmhg","heart_rate_bpm","fasting_glucose_mg_dl","hba1c_percent","insulin_uiu_ml","total_cholesterol_mg_dl","hdl_mg_dl","ldl_mg_dl","vldl_mg_dl","triglycerides_mg_dl","creatinine_mg_dl","urea_mg_dl","family_history_diabetes","hypertension_history","heart_disease","diabetes_class"]
NUMERIC=[c for c in COLUMNS if c not in {"source_dataset","source_row","sex","diabetes_class"}]
CLASSES=["normal","prediabetes","type2_diabetes"]
