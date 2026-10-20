"""Regole cliniche esplicite: trasformano misure numeriche in fattori simbolici."""
def derive(v):
 g=lambda k:float(v.get(k) or 0);sex=str(v.get("sex","female")).lower()
 return {"glicemia_diagnostica":g("fasting_glucose_mg_dl")>=126,"glicemia_borderline":100<=g("fasting_glucose_mg_dl")<126,"hba1c_diagnostica":g("hba1c_percent")>=6.5,"hba1c_borderline":5.7<=g("hba1c_percent")<6.5,"ipertensione":g("systolic_bp_mmhg")>=140 or g("diastolic_bp_mmhg")>=90,"obesita":g("bmi")>=30,"trigliceridi_alti":g("triglycerides_mg_dl")>=150,"hdl_basso":g("hdl_mg_dl")>0 and g("hdl_mg_dl")<(40 if sex.startswith("m") else 50)}
def diagnose_rules(f):
 score=.80 if f["hba1c_diagnostica"] or f["glicemia_diagnostica"] else .45 if f["hba1c_borderline"] or f["glicemia_borderline"] else 0;score=min(1,score+.07*sum(f[x] for x in ["ipertensione","obesita","trigliceridi_alti","hdl_basso"]));label="type2_diabetes" if score>=.70 else "prediabetes" if score>=.30 else "normal";return {"label":label,"score":score,"factors":[k for k,v in f.items() if v]}
