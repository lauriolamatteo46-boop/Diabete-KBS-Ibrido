"""
test_full_project.py
====================

Suite di test globale e approfondita per il progetto Diabete KBS Ibrido.
Esegue una validazione completa su tutti i livelli architetturali:

1. Validazione Dati e Modelli Persistiti (CSV, BIF, PKL)
2. Feature Engineering e Discretizzazione (soglie cliniche, gestione None/NaN)
3. Conoscenza Simbolica e Regole Datalog (derivazione fatti e diagnosi differenziale)
4. Modelli di Machine Learning e Probabilistici (SVM + Rete Bayesiana)
5. Valutazione Clinica Differenziale Completa:
   - Diabete di Tipo 1 (carenza insulinica, esordio giovanile, normopeso)
   - Diabete di Tipo 2 (insulino-resistenza, età matura, obesità, sindrome metabolica)
   - Prediabete (alterata glicemia a digiuno / HbA1c borderline)
   - Non Diabete con fattori di rischio (glicemia normale con ipertensione o lipidi alti)
   - Stato in Salute Ottimale (tutti i parametri normali)
   - Stress test di robustezza (valori nulli, valori estremi)
6. Ontologia Clinica e Generatore di Referti PDF
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

# Assicura che la root del progetto sia accessibile per tutti gli import
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd

from dataset_builder.canonical_schema import COLUMNS
from diagnosis_engine import HybridDiagnosis, diagnose
from kbs_engine.datalog_engine import DatalogEngine
from kbs_engine.ontology_manager import OntologyManager
from kbs_engine.rules import derive, diagnose_rules
from ml_engine.bn_model import MODEL_PATH as BN_MODEL_PATH, BayesianModel
from ml_engine.feature_engineering import discretize
from ml_engine.svm_model import MODEL_PATH as SVM_MODEL_PATH, SVMModel
from report_generator import export_pdf


class TestRunner:
    """Gestisce l'esecuzione ordinata, il conteggio dei pass/fail e il report finale."""

    def __init__(self) -> None:
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.results: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, details: str = "") -> bool:
        self.total += 1
        start = time.perf_counter()
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name} -> {details}")

        self.results.append({
            "name": name,
            "status": status,
            "details": details,
        })
        return condition


def test_section_1_data_and_models(runner: TestRunner) -> None:
    print("\n" + "=" * 72)
    print("SEZIONE 1: VALIDAZIONE DATI E MODELLI PERSISTITI")
    print("=" * 72)

    csv_path = os.path.join(ROOT_DIR, "data", "processed", "diabetes_clinical_unified.csv")
    csv_exists = os.path.exists(csv_path)
    runner.check("Esistenza CSV unificato", csv_exists, f"Percorso non trovato: {csv_path}")

    if csv_exists:
        df = pd.read_csv(csv_path)
        runner.check(
            "Dimensioni CSV unificato (> 5000 righe)",
            len(df) >= 5000,
            f"Trovate solo {len(df)} righe",
        )
        runner.check(
            "Presenza colonne canoniche (24 colonne)",
            all(col in df.columns for col in COLUMNS),
            "Mancano alcune colonne canoniche",
        )
        classes_present = set(df["diabetes_class"].dropna().unique())
        runner.check(
            "Presenza classi target nel dataset (normal, prediabetes, type2_diabetes)",
            {"normal", "prediabetes", "type2_diabetes"}.issubset(classes_present),
            f"Classi presenti: {classes_present}",
        )

    bif_exists = os.path.exists(BN_MODEL_PATH)
    runner.check("Esistenza file Rete Bayesiana (.bif)", bif_exists, f"Manca: {BN_MODEL_PATH}")

    pkl_exists = os.path.exists(SVM_MODEL_PATH)
    runner.check("Esistenza modello SVM (.pkl)", pkl_exists, f"Manca: {SVM_MODEL_PATH}")


def test_section_2_feature_engineering(runner: TestRunner) -> None:
    print("\n" + "=" * 72)
    print("SEZIONE 2: FEATURE ENGINEERING & DISCRETIZZAZIONE")
    print("=" * 72)

    # 1. Test soglie trigliceridi (incluso regression check per lo stato 'high')
    df_tg = pd.DataFrame([
        {"triglycerides_mg_dl": 120},  # normal
        {"triglycerides_mg_dl": 150},  # high
        {"triglycerides_mg_dl": 600},  # high
        {"triglycerides_mg_dl": None}, # unknown
    ])
    d_tg = discretize(df_tg)
    runner.check(
        "Discretizzazione trigliceridi (normal / high / unknown)",
        d_tg["lipid_risk"].tolist() == ["normal", "high", "high", "unknown"],
        f"Ottenuto: {d_tg['lipid_risk'].tolist()}",
    )

    # 2. Test soglie glicemia
    df_glu = pd.DataFrame([
        {"fasting_glucose_mg_dl": 85},   # normal
        {"fasting_glucose_mg_dl": 110},  # borderline
        {"fasting_glucose_mg_dl": 140},  # high
        {"fasting_glucose_mg_dl": None}, # unknown
    ])
    d_glu = discretize(df_glu)
    runner.check(
        "Discretizzazione glicemia (normal / borderline / high / unknown)",
        d_glu["glucose_group"].tolist() == ["normal", "borderline", "high", "unknown"],
        f"Ottenuto: {d_glu['glucose_group'].tolist()}",
    )

    # 3. Test soglie HbA1c
    df_hba1c = pd.DataFrame([
        {"hba1c_percent": 5.0},  # normal
        {"hba1c_percent": 6.0},  # borderline
        {"hba1c_percent": 8.0},  # high
        {"hba1c_percent": None}, # unknown
    ])
    d_hba1c = discretize(df_hba1c)
    runner.check(
        "Discretizzazione HbA1c (normal / borderline / high / unknown)",
        d_hba1c["hba1c_group"].tolist() == ["normal", "borderline", "high", "unknown"],
        f"Ottenuto: {d_glu['glucose_group'].tolist()}",
    )

    # 4. Test robustezza su DataFrame vuoto o con colonne assenti
    df_empty = pd.DataFrame([{}])
    d_empty = discretize(df_empty)
    runner.check(
        "Robustezza discretizzazione su input vuoto (tutti 'unknown')",
        all(d_empty.iloc[0][col] == "unknown" for col in d_empty.columns),
        f"Ottenuto: {d_empty.iloc[0].to_dict()}",
    )


def test_section_3_symbolic_kbs(runner: TestRunner) -> None:
    print("\n" + "=" * 72)
    print("SEZIONE 3: CONOSCENZA SIMBOLICA & REGOLE CLINICHE DATALOG")
    print("=" * 72)

    # Test derivazione fatti
    paziente_t1 = {
        "fasting_glucose_mg_dl": 180,
        "hba1c_percent": 8.5,
        "insulin_uiu_ml": 2.5,
        "age_years": 20,
        "bmi": 19.0,
        "systolic_bp_mmhg": 115,
        "diastolic_bp_mmhg": 75,
    }
    f_t1 = derive(paziente_t1)
    runner.check("Fatto simbolico: glicemia_diagnostica", f_t1["glicemia_diagnostica"] is True)
    runner.check("Fatto simbolico: hba1c_diagnostica", f_t1["hba1c_diagnostica"] is True)
    runner.check("Fatto simbolico: carenza_insulinica", f_t1["carenza_insulinica"] is True)
    runner.check("Fatto simbolico: eta_giovanile", f_t1["eta_giovanile"] is True)
    runner.check("Fatto simbolico: normopeso_magrezza", f_t1["normopeso_magrezza"] is True)
    runner.check("Fatto simbolico: assenza ipertensione", f_t1["ipertensione"] is False)

    # Diagnosi regole Tipo 1
    diag_t1 = diagnose_rules(f_t1)
    runner.check(
        "Diagnosi Datalog: identificazione type1_diabetes",
        diag_t1["label"] == "type1_diabetes",
        f"Ottenuto: {diag_t1['label']}",
    )

    # Test paziente Tipo 2
    paziente_t2 = {
        "fasting_glucose_mg_dl": 160,
        "hba1c_percent": 8.0,
        "insulin_uiu_ml": 22.0,
        "age_years": 58,
        "bmi": 33.0,
        "systolic_bp_mmhg": 150,
        "diastolic_bp_mmhg": 95,
        "triglycerides_mg_dl": 220,
    }
    f_t2 = derive(paziente_t2)
    runner.check("Fatto simbolico: iperinsulinemia", f_t2["iperinsulinemia"] is True)
    runner.check("Fatto simbolico: obesita", f_t2["obesita"] is True)
    runner.check("Fatto simbolico: ipertensione", f_t2["ipertensione"] is True)
    runner.check("Fatto simbolico: trigliceridi_alti", f_t2["trigliceridi_alti"] is True)

    diag_t2 = diagnose_rules(f_t2)
    runner.check(
        "Diagnosi Datalog: identificazione type2_diabetes",
        diag_t2["label"] == "type2_diabetes",
        f"Ottenuto: {diag_t2['label']}",
    )

    # Test DatalogEngine Facade
    datalog_facade = DatalogEngine()
    facade_res = datalog_facade.diagnosi(paziente_t1)
    runner.check(
        "Funzionamento Facade DatalogEngine",
        facade_res["label"] == "type1_diabetes",
        f"Ottenuto: {facade_res['label']}",
    )


def test_section_4_models_inference(runner: TestRunner) -> None:
    print("\n" + "=" * 72)
    print("SEZIONE 4: MODELLI DI MACHINE LEARNING & PROBABILISTICI")
    print("=" * 72)

    # Test SVM
    svm = SVMModel()
    svm.load()
    p_test = {"age_years": 50, "bmi": 26.0, "fasting_glucose_mg_dl": 105, "sex": "female"}
    pred_svm = svm.predict(p_test)
    runner.check("SVM output contiene 'normal' e 'type2_diabetes'", set(pred_svm.keys()) == {"normal", "type2_diabetes"})
    sum_svm = pred_svm["normal"] + pred_svm["type2_diabetes"]
    runner.check("SVM somma probabilita ~= 1.0", abs(sum_svm - 1.0) < 1e-4, f"Somma: {sum_svm}")

    # Test Rete Bayesiana
    bn = BayesianModel()
    bn.load()
    pred_bn = bn.predict(p_test)
    runner.check(
        "BN output contiene 'normal', 'prediabetes', 'type2_diabetes'",
        {"normal", "prediabetes", "type2_diabetes"}.issubset(pred_bn.keys()),
    )
    sum_bn = sum(pred_bn.values())
    runner.check("BN somma probabilita ~= 1.0", abs(sum_bn - 1.0) < 1e-4, f"Somma: {sum_bn}")

    # Test BN con evidenze parziali e valori estremi (non deve andare in eccezione)
    try:
        pred_extreme = bn.predict({"triglycerides_mg_dl": 850, "fasting_glucose_mg_dl": 280})
        bn_stable = True
    except Exception as e:
        bn_stable = False
        print("Errore BN su valori estremi:", e)

    runner.check("BN stabilità su evidenze parziali ed estreme (lipid_risk 'high')", bn_stable)


def test_section_5_clinical_differential_diagnosis(runner: TestRunner) -> None:
    print("\n" + "=" * 72)
    print("SEZIONE 5: VALUTAZIONE CLINICA DIFFERENZIALE (PIPELINE IBRIDA)")
    print("=" * 72)

    # CASO 5.1: Diabete di Tipo 1 (Giovane con deficit insulinico marcato)
    c1 = {
        "age_years": 18,
        "sex": "male",
        "bmi": 19.5,
        "fasting_glucose_mg_dl": 240,
        "hba1c_percent": 9.2,
        "insulin_uiu_ml": 2.1,
        "systolic_bp_mmhg": 115,
        "diastolic_bp_mmhg": 75,
        "triglycerides_mg_dl": 95,
    }
    r1 = diagnose(c1)
    runner.check(
        "Caso 5.1 -> Diabete Tipo 1 (giovane, magro, insulina 2.1)",
        r1["label"] == "type1_diabetes",
        f"Atteso: type1_diabetes, Ottenuto: {r1['label']} ({r1['display_name']})",
    )

    # CASO 5.2: Diabete di Tipo 1 (Adulto giovane normopeso, carenza insulinica)
    c2 = {
        "age_years": 29,
        "sex": "female",
        "bmi": 20.8,
        "fasting_glucose_mg_dl": 195,
        "hba1c_percent": 8.3,
        "insulin_uiu_ml": 3.4,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 76,
        "triglycerides_mg_dl": 110,
    }
    r2 = diagnose(c2)
    runner.check(
        "Caso 5.2 -> Diabete Tipo 1 (29 anni, BMI 20.8, insulina 3.4)",
        r2["label"] == "type1_diabetes",
        f"Atteso: type1_diabetes, Ottenuto: {r2['label']} ({r2['display_name']})",
    )

    # CASO 5.3: Diabete di Tipo 2 (Adulto, obeso, sindrome metabolica, iperinsulino-resistenza)
    c3 = {
        "age_years": 58,
        "sex": "male",
        "bmi": 33.5,
        "fasting_glucose_mg_dl": 168,
        "hba1c_percent": 8.3,
        "insulin_uiu_ml": 22.0,
        "systolic_bp_mmhg": 150,
        "diastolic_bp_mmhg": 95,
        "triglycerides_mg_dl": 240,
        "hdl_mg_dl": 36,
    }
    r3 = diagnose(c3)
    runner.check(
        "Caso 5.3 -> Diabete Tipo 2 (58 anni, obeso, sindrome metabolica)",
        r3["label"] == "type2_diabetes",
        f"Atteso: type2_diabetes, Ottenuto: {r3['label']} ({r3['display_name']})",
    )

    # CASO 5.4: Diabete di Tipo 2 (Anziana con dislipidemia e sovrappeso)
    c4 = {
        "age_years": 67,
        "sex": "female",
        "bmi": 28.5,
        "fasting_glucose_mg_dl": 145,
        "hba1c_percent": 7.6,
        "systolic_bp_mmhg": 142,
        "diastolic_bp_mmhg": 88,
        "triglycerides_mg_dl": 210,
    }
    r4 = diagnose(c4)
    runner.check(
        "Caso 5.4 -> Diabete Tipo 2 (67 anni, sovrappeso, dislipidemia)",
        r4["label"] == "type2_diabetes",
        f"Atteso: type2_diabetes, Ottenuto: {r4['label']} ({r4['display_name']})",
    )

    # CASO 5.5: Prediabete (Alterata Glicemia a Digiuno - IFG)
    c5 = {
        "age_years": 50,
        "sex": "male",
        "bmi": 26.5,
        "fasting_glucose_mg_dl": 115,
        "hba1c_percent": 5.5,
        "systolic_bp_mmhg": 125,
        "diastolic_bp_mmhg": 80,
    }
    r5 = diagnose(c5)
    runner.check(
        "Caso 5.5 -> Prediabete (glicemia 115 mg/dL)",
        r5["label"] == "prediabetes",
        f"Atteso: prediabetes, Ottenuto: {r5['label']} ({r5['display_name']})",
    )

    # CASO 5.6: Prediabete (HbA1c Borderline 6.1%)
    c6 = {
        "age_years": 52,
        "sex": "female",
        "bmi": 27.0,
        "fasting_glucose_mg_dl": 96,
        "hba1c_percent": 6.1,
        "systolic_bp_mmhg": 120,
        "diastolic_bp_mmhg": 80,
    }
    r6 = diagnose(c6)
    runner.check(
        "Caso 5.6 -> Prediabete (HbA1c 6.1%)",
        r6["label"] == "prediabetes",
        f"Atteso: prediabetes, Ottenuto: {r6['label']} ({r6['display_name']})",
    )

    # CASO 5.7: Non Diabete con Ipertensione e Dislipidemia (Check salvaguardia clinica)
    c7 = {
        "age_years": 59,
        "sex": "male",
        "bmi": 28.0,
        "fasting_glucose_mg_dl": 92,
        "hba1c_percent": 5.1,
        "systolic_bp_mmhg": 155,
        "diastolic_bp_mmhg": 96,
        "triglycerides_mg_dl": 220,
        "hdl_mg_dl": 35,
    }
    r7 = diagnose(c7)
    runner.check(
        "Caso 5.7 -> Non Diabete con Ipertensione e Dislipidemia (glicemia 92)",
        r7["label"] == "normal",
        f"Atteso: normal, Ottenuto: {r7['label']} ({r7['display_name']})",
    )

    # CASO 5.8: Non Diabete con Obesità Isolata
    c8 = {
        "age_years": 44,
        "sex": "female",
        "bmi": 32.5,
        "fasting_glucose_mg_dl": 88,
        "hba1c_percent": 5.0,
        "systolic_bp_mmhg": 120,
        "diastolic_bp_mmhg": 80,
        "triglycerides_mg_dl": 120,
    }
    r8 = diagnose(c8)
    runner.check(
        "Caso 5.8 -> Non Diabete con Obesità (glicemia 88)",
        r8["label"] == "normal",
        f"Atteso: normal, Ottenuto: {r8['label']} ({r8['display_name']})",
    )

    # CASO 5.9: Stato in Salute Ottimale (Giovane)
    c9 = {
        "age_years": 24,
        "sex": "female",
        "bmi": 20.5,
        "fasting_glucose_mg_dl": 84,
        "hba1c_percent": 4.8,
        "systolic_bp_mmhg": 114,
        "diastolic_bp_mmhg": 74,
        "triglycerides_mg_dl": 85,
        "hdl_mg_dl": 65,
    }
    r9 = diagnose(c9)
    runner.check(
        "Caso 5.9 -> Stato in Salute Ottimale (24 anni, tutti parametri fisiologici)",
        r9["label"] == "healthy",
        f"Atteso: healthy, Ottenuto: {r9['label']} ({r9['display_name']})",
    )

    # CASO 5.10: Stato in Salute Ottimale (Adulto)
    c10 = {
        "age_years": 45,
        "sex": "male",
        "bmi": 23.0,
        "fasting_glucose_mg_dl": 90,
        "hba1c_percent": 5.0,
        "systolic_bp_mmhg": 118,
        "diastolic_bp_mmhg": 78,
        "triglycerides_mg_dl": 110,
        "hdl_mg_dl": 55,
    }
    r10 = diagnose(c10)
    runner.check(
        "Caso 5.10 -> Stato in Salute Ottimale (45 anni, normopeso, pressione 118/78)",
        r10["label"] == "healthy",
        f"Atteso: healthy, Ottenuto: {r10['label']} ({r10['display_name']})",
    )

    # CASO 5.11: Stress Test (Dizionario completamente vuoto)
    try:
        r11 = diagnose({})
        empty_safe = r11["label"] in ["healthy", "normal"]
    except Exception as e:
        empty_safe = False
        print("Errore su dizionario vuoto:", e)

    runner.check("Caso 5.11 -> Stress test su input completamente vuoto (senza crash)", empty_safe)

    # CASO 5.12: Stress Test Trigliceridi Elevatissimi (Vecchio bug trigger)
    try:
        r12 = diagnose({"triglycerides_mg_dl": 950, "fasting_glucose_mg_dl": 95})
        tg_safe = r12["label"] == "normal"
    except Exception as e:
        tg_safe = False
        print("Errore su trigliceridi elevati:", e)

    runner.check("Caso 5.12 -> Gestione trigliceridi estremi (950 mg/dL) senza crash", tg_safe)


def test_section_6_ontology_and_pdf(runner: TestRunner) -> None:
    print("\n" + "=" * 72)
    print("SEZIONE 6: ONTOLOGIA CLINICA & GENERAZIONE REPORT PDF")
    print("=" * 72)

    onto_mgr = OntologyManager()
    expected_labels = ["healthy", "normal", "prediabetes", "type1_diabetes", "type2_diabetes"]

    all_onto_valid = True
    for lab in expected_labels:
        info = onto_mgr.get_info(lab)
        if not (info.get("display_name") and info.get("severity") and info.get("recommendations")):
            all_onto_valid = False
            print(f"Mancano campi ontologia per: {lab}")

    runner.check(
        "Ontologia: completezza metadati per tutte le 5 classi cliniche",
        all_onto_valid,
    )

    # Test generazione PDF
    paziente_pdf = {
        "age_years": 22,
        "sex": "female",
        "bmi": 19.8,
        "fasting_glucose_mg_dl": 225,
        "hba1c_percent": 9.0,
        "insulin_uiu_ml": 2.2,
        "systolic_bp_mmhg": 115,
        "diastolic_bp_mmhg": 75,
    }
    res_pdf = diagnose(paziente_pdf)

    try:
        pdf_path = export_pdf(paziente_pdf, res_pdf)
        pdf_ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000
    except Exception as e:
        pdf_ok = False
        print("Errore esportazione PDF:", e)

    runner.check(
        "Generazione ed esportazione referto clinico PDF reale",
        pdf_ok,
        f"File generato non valido o vuoto: {pdf_path if 'pdf_path' in locals() else 'N/D'}",
    )


def main() -> None:
    runner = TestRunner()

    print("=" * 72)
    print(" SUITE GLOBALE DI COLLAUDO - DIABETE KBS IBRIDO")
    print(" Controllo di integrità, modelli, regole Datalog e diagnosi differenziale")
    print("=" * 72)

    try:
        test_section_1_data_and_models(runner)
        test_section_2_feature_engineering(runner)
        test_section_3_symbolic_kbs(runner)
        test_section_4_models_inference(runner)
        test_section_5_clinical_differential_diagnosis(runner)
        test_section_6_ontology_and_pdf(runner)
    except Exception as error:
        print("\n[!] ERRORE CRITICO NON GESTITO NELLA SUITE DI TEST:")
        print(f"    {type(error).__name__}: {error}")
        traceback.print_exc()

    # Report finale
    print("\n" + "=" * 72)
    print("RIASSUNTO FINALE DEL COLLAUDO DEL PROGETTO")
    print("=" * 72)
    print(f"Test totali eseguiti: {runner.total}")
    print(f"Test superati (PASS):  {runner.passed}")
    print(f"Test falliti  (FAIL):  {runner.failed}")
    success_rate = (runner.passed / runner.total * 100.0) if runner.total > 0 else 0.0
    print(f"Percentuale successo:  {success_rate:.1f}%")
    print("=" * 72)

    if runner.failed == 0:
        print("\n>>> ESITO GLOBALE: TUTTI I TEST SUPERATI CON SUCCESSO! <<<")
        print("Il programma differenzia correttamente Diabete Tipo 1, Tipo 2, Prediabete,")
        print("Non Diabete con fattori di rischio e Stato in salute ottimale.\n")
    else:
        print(f"\n>>> ATTENZIONE: {runner.failed} TEST NON SUPERATI. Verificare i dettagli sopra. <<<\n")


if __name__ == "__main__":
    main()
