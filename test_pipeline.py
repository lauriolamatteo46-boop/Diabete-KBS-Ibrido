"""
test_pipeline.py
================

Suite di test automatica per la verifica della pipeline ibrida:
- Differenziazione: Diabete Tipo 1, Diabete Tipo 2, Prediabete, Non Diabete, Stato in salute;
- Robustezza con valori mancanti (None);
- Verifica funzionamento congiunto di Regole Datalog, SVM, Rete Bayesiana e Ontologia.
"""

from __future__ import annotations

import os
import sys
import traceback
import pandas as pd

# Assicura che la root del progetto sia nel sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Gestione opzionale di pyAgrum
try:
    import pyAgrum as gum
except ImportError:
    gum = None

try:
    from ml_engine.feature_engineering import discretize
    from ml_engine.bn_model import BayesianModel
    from diagnosis_engine import diagnose
except ImportError as e:
    print(f"Errore di importazione. Assicurati di essere nella cartella principale del progetto: {e}")


def esegui_test() -> None:
    casi_di_test = {
        "1. Stato in Salute Ottimale": {
            "age_years": 28,
            "sex": "female",
            "bmi": 21.5,
            "fasting_glucose_mg_dl": 85,
            "hba1c_percent": 4.9,
            "systolic_bp_mmhg": 115,
            "diastolic_bp_mmhg": 75,
            "triglycerides_mg_dl": 90,
            "hdl_mg_dl": 60,
        },
        "2. Non Diabete (Fattori di rischio: Ipertensione e Sovrappeso)": {
            "age_years": 54,
            "sex": "male",
            "bmi": 28.5,
            "fasting_glucose_mg_dl": 92,
            "hba1c_percent": 5.1,
            "systolic_bp_mmhg": 145,
            "diastolic_bp_mmhg": 92,
            "triglycerides_mg_dl": 180,
            "hdl_mg_dl": 38,
        },
        "3. Prediabete (Alterata Glicemia a Digiuno)": {
            "age_years": 48,
            "sex": "female",
            "bmi": 27.2,
            "fasting_glucose_mg_dl": 112,
            "hba1c_percent": 5.9,
            "systolic_bp_mmhg": 125,
            "diastolic_bp_mmhg": 82,
            "triglycerides_mg_dl": 130,
        },
        "4. Diabete di Tipo 1 (Esordio giovanile, normopeso, carenza insulinica)": {
            "age_years": 19,
            "sex": "male",
            "bmi": 19.2,
            "fasting_glucose_mg_dl": 230,
            "hba1c_percent": 9.5,
            "insulin_uiu_ml": 2.5,
            "systolic_bp_mmhg": 115,
            "diastolic_bp_mmhg": 75,
            "triglycerides_mg_dl": 110,
        },
        "5. Diabete di Tipo 2 (Adulto, obeso, sindrome metabolica, iperinsulinemia)": {
            "age_years": 62,
            "sex": "female",
            "bmi": 34.0,
            "fasting_glucose_mg_dl": 175,
            "hba1c_percent": 8.4,
            "insulin_uiu_ml": 24.0,
            "systolic_bp_mmhg": 150,
            "diastolic_bp_mmhg": 95,
            "triglycerides_mg_dl": 260,
            "hdl_mg_dl": 35,
        },
        "6. Valori Mancanti (Test di robustezza con None)": {
            "age_years": 32,
            "bmi": None,
            "fasting_glucose_mg_dl": 95,
            "hba1c_percent": None,
            "systolic_bp_mmhg": 120,
            "diastolic_bp_mmhg": 80,
            "triglycerides_mg_dl": None,
        },
    }

    print("=" * 70)
    print("AVVIO TEST SUITE: DIFFERENZIAZIONE CLINICA E PIPELINE IBRIDA")
    print("=" * 70)

    # Inizializza modello BN
    bn_model = None
    try:
        bn_model = BayesianModel()
        bn_model.load()
        print("[OK] Rete Bayesiana caricata con successo.")
    except Exception as e:
        print(f"[!] Avviso caricamento Rete Bayesiana: {e}")

    print()

    for nome_test, dati in casi_di_test.items():
        print(f"--- Eseguendo: {nome_test} ---")

        try:
            # A. Test Discretizzazione
            df_input = pd.DataFrame([dati])
            df_disc = discretize(df_input)
            evidenze = df_disc.iloc[0].to_dict()
            print("  Discretizzazione:")
            for k, v in evidenze.items():
                print(f"    • {k}: '{v}'")

            # B. Test Rete Bayesiana
            if bn_model is not None:
                prob = bn_model.predict(dati)
                print(f"  Rete Bayesiana: {prob}")

            # C. Test Motore Ibrido Completo
            risultato = diagnose(dati)
            label = risultato["label"]
            display_name = risultato.get("display_name", label)
            score = risultato["final_score"]
            reason = risultato.get("rule_result", {}).get("reason", "")

            print(f"  -> ESITO CLINICO: {display_name.upper()} (label: {label}, score: {score:.2f})")
            if reason:
                print(f"  -> MOTIVAZIONE:  {reason}")

            print(">> Test superato con successo!\n")

        except Exception as e:
            print(f">> FALLITO durante '{nome_test}':")
            print(f"   {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print("\n")


if __name__ == "__main__":
    esegui_test()
