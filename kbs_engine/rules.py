"""
rules.py
========

Modulo di conoscenza simbolica clinica (stile Datalog).
Trasforma le misure numeriche in fatti simbolici ed esegue la diagnosi differenziale tra:
- Diabete di Tipo 1 (type1_diabetes)
- Diabete di Tipo 2 (type2_diabetes)
- Prediabete (prediabetes)
- Non diabete con fattori di rischio (normal)
- Stato in salute ottimale (healthy)
"""

from typing import Any


def _val(v: dict[str, Any], key: str) -> float | None:
    """Estrae un valore numerico in modo sicuro, restituendo None se assente."""
    val = v.get(key)
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def derive(v: dict[str, Any]) -> dict[str, bool]:
    """Deriva i fatti simbolici a partire dai valori clinici del paziente."""
    sex = str(v.get("sex", "female")).strip().lower()

    glucose = _val(v, "fasting_glucose_mg_dl")
    hba1c = _val(v, "hba1c_percent")
    systolic = _val(v, "systolic_bp_mmhg")
    diastolic = _val(v, "diastolic_bp_mmhg")
    bmi = _val(v, "bmi")
    tg = _val(v, "triglycerides_mg_dl")
    hdl = _val(v, "hdl_mg_dl")
    age = _val(v, "age_years")
    insulin = _val(v, "insulin_uiu_ml")

    return {
        # Marcatori glicemici diagnostici (Linee guida ADA / OMS)
        "glicemia_diagnostica": glucose is not None and glucose >= 126.0,
        "glicemia_borderline": glucose is not None and 100.0 <= glucose < 126.0,
        "glicemia_normale": glucose is not None and glucose < 100.0,

        "hba1c_diagnostica": hba1c is not None and hba1c >= 6.5,
        "hba1c_borderline": hba1c is not None and 5.7 <= hba1c < 6.5,
        "hba1c_normale": hba1c is not None and hba1c < 5.7,

        # Pressione arteriosa
        "ipertensione": (
            (systolic is not None and systolic >= 140.0)
            or (diastolic is not None and diastolic >= 90.0)
        ),
        "pressione_ottimale": (
            systolic is not None and diastolic is not None
            and systolic < 120.0 and diastolic < 80.0
        ),

        # Indice di Massa Corporea (BMI)
        "obesita": bmi is not None and bmi >= 30.0,
        "sovrappeso": bmi is not None and 25.0 <= bmi < 30.0,
        "normopeso_magrezza": bmi is not None and bmi < 25.0,

        # Profilo lipidico
        "trigliceridi_alti": tg is not None and tg >= 150.0,
        "trigliceridi_normali": tg is not None and tg < 150.0,
        "hdl_basso": hdl is not None and (
            hdl < (40.0 if sex.startswith("m") else 50.0)
        ),

        # Anagrafica ed eziologia insulinica per diagnosi differenziale Tipo 1 vs Tipo 2
        "eta_giovanile": age is not None and age < 35.0,
        "eta_matura": age is not None and age >= 35.0,

        # L'insulina a digiuno < 5 µIU/mL indica carenza assoluta (Tipo 1)
        # Un'insulina elevata (>= 15 µIU/mL) indica insulino-resistenza (Tipo 2)
        "carenza_insulinica": insulin is not None and 0.0 < insulin < 5.0,
        "iperinsulinemia": insulin is not None and insulin >= 15.0,
    }


def diagnose_rules(f: dict[str, bool]) -> dict[str, Any]:
    """
    Applica il ragionamento clinico differenziale sulle regole Datalog.

    Categorie diagnostiche distinte:
    1. type1_diabetes: Diabete di Tipo 1 (carenza insulinica / esordio giovanile o magro)
    2. type2_diabetes: Diabete di Tipo 2 (insulino-resistenza / età matura o obesità/lipidi)
    3. prediabetes: Prediabete (alterata glicemia a digiuno o HbA1c borderline)
    4. normal: Non diabete (glicemia normale ma fattori di rischio cardiovascolare/metabolico)
    5. healthy: Stato in salute ottimale (tutti i parametri nella norma)
    """
    ha_diabete = f["glicemia_diagnostica"] or f["hba1c_diagnostica"]
    ha_prediabete = (f["glicemia_borderline"] or f["hba1c_borderline"]) and not ha_diabete

    # -----------------------------------------------------------------------
    # CASO 1: QUADRO DIABETE CONCLAMATO (Differenziazione Tipo 1 vs Tipo 2)
    # -----------------------------------------------------------------------
    if ha_diabete:
        punti_tipo1 = 0
        punti_tipo2 = 0

        # Carenza insulinica: marcatore quasi patognomonico di Tipo 1
        if f["carenza_insulinica"]:
            punti_tipo1 += 3

        if f["eta_giovanile"]:
            punti_tipo1 += 1
        else:
            punti_tipo2 += 1

        if f["normopeso_magrezza"]:
            punti_tipo1 += 1

        if f["obesita"]:
            punti_tipo2 += 2
        elif f["sovrappeso"]:
            punti_tipo2 += 1

        if f["iperinsulinemia"]:
            punti_tipo2 += 2

        if f["ipertensione"]:
            punti_tipo2 += 1

        if f["trigliceridi_alti"] or f["hdl_basso"]:
            punti_tipo2 += 1

        # Decisione clinica differenziale
        if f["carenza_insulinica"] or (punti_tipo1 > punti_tipo2 and (f["eta_giovanile"] or f["normopeso_magrezza"])):
            label = "type1_diabetes"
            score = 0.95 if f["carenza_insulinica"] else 0.88
            reason = (
                "Presenza di criteri diagnostici di diabete con caratteristiche tipiche di Tipo 1 "
                "(carenza insulinica marcata, età giovanile o normopeso senza sindrome metabolica)."
            )
        else:
            label = "type2_diabetes"
            score = 0.90
            reason = (
                "Presenza di criteri diagnostici di diabete con caratteristiche di Tipo 2 "
                "(insulino-resistenza, sovrappeso/obesità, età adulta o comorbilità metaboliche)."
            )

    # -----------------------------------------------------------------------
    # CASO 2: PREDIABETE
    # -----------------------------------------------------------------------
    elif ha_prediabete:
        label = "prediabetes"
        score = 0.80
        reason = (
            "Glicemia a digiuno alterata (100-125 mg/dL) o HbA1c borderline (5.7-6.4%): "
            "rischio aumentato di progressione verso diabete di tipo 2."
        )

    # -----------------------------------------------------------------------
    # CASO 3: NON DIABETE / STATO IN SALUTE
    # -----------------------------------------------------------------------
    else:
        # Non ci sono alterazioni glicemiche. Verifichiamo altri fattori di rischio
        fattori_rischio = [
            f["ipertensione"],
            f["obesita"],
            f["sovrappeso"],
            f["trigliceridi_alti"],
            f["hdl_basso"],
        ]

        if any(fattori_rischio):
            label = "normal"
            score = 0.85
            reason = (
                "Assenza di diabete (glicemia e HbA1c nella norma), ma presenza di fattori di rischio "
                "cardiovascolare o metabolico (pressione, lipidi o peso) da monitorare."
            )
        else:
            label = "healthy"
            score = 0.95
            reason = (
                "Stato in salute ottimale: parametri glicemici, pressori, lipidici e ponderali "
                "nei limiti fisiologici di normalità."
            )

    return {
        "label": label,
        "score": score,
        "reason": reason,
        "factors": [k for k, v in f.items() if v],
    }
