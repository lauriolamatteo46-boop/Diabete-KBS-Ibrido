"""
diagnosis_engine.py
===================

Motore centrale del Diabete KBS Ibrido.
Esegue l'integrazione multi-paradigma tra:
- Conoscenza simbolica / Datalog (soglie cliniche e diagnosi differenziale Tipo 1 vs Tipo 2 vs Sano);
- Modello Machine Learning SVM (stima statistica supervisionata);
- Rete Bayesiana (inferenza probabilistica condizionata);
- Ontologia clinica (tassonomia, gravità, spiegazione e raccomandazioni).
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Permette gli import quando main_gui.py o altri script vengono avviati direttamente.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kbs_engine.datalog_engine import DatalogEngine
from kbs_engine.ontology_manager import OntologyManager
from ml_engine.svm_model import SVMModel, MODEL_PATH as SVM_MODEL_PATH
from ml_engine.bn_model import BayesianModel, MODEL_PATH as BN_MODEL_PATH


class HybridDiagnosis:
    """
    Combina quattro componenti del progetto:

    - Datalog/regole (0.40): interpreta le soglie cliniche e differenzia Tipo 1, Tipo 2, Prediabete e Stato in salute;
    - SVM (0.25): stima statistica supervisionata sul rischio di diabete;
    - BN (0.35): probabilità condizionate sulle classi cliniche;
    - Ontologia: arricchisce l'esito finale con display_name, descrizione, severità e raccomandazioni.
    """

    RULE_WEIGHT = 0.40
    SVM_WEIGHT = 0.25
    BN_WEIGHT = 0.35

    def __init__(self) -> None:
        self.rule_engine = DatalogEngine()
        self.ontology = OntologyManager()
        self.svm: SVMModel | None = None
        self.bn: BayesianModel | None = None

        self._load_optional_models()

    def _load_optional_models(self) -> None:
        """
        Carica SVM e BN se i file addestrati sono disponibili.
        Se non sono presenti, il sistema degrada con grazia usando solo regole e ontologia.
        """
        if os.path.exists(SVM_MODEL_PATH):
            try:
                self.svm = SVMModel()
                self.svm.load()
            except Exception as error:
                print("Avviso: SVM non caricata:", error)
                self.svm = None

        if os.path.exists(BN_MODEL_PATH):
            try:
                self.bn = BayesianModel()
                self.bn.load()
            except Exception as error:
                print("Avviso: Rete Bayesiana non caricata:", error)
                self.bn = None

    def evaluate(self, patient_values: dict[str, Any]) -> dict[str, Any]:
        """Esegue una valutazione clinica completa integrando i quattro paradigmi."""

        # 1. Valutazione simbolica: soglie esplicite e diagnosi differenziale
        rule_result = self.rule_engine.diagnosi(patient_values)
        rule_label = rule_result["label"]
        rule_score = rule_result["score"]

        # Inizializziamo i punteggi per le 5 classi distinte
        combined_scores: dict[str, float] = {
            "healthy": 0.0,
            "normal": 0.0,
            "prediabetes": 0.0,
            "type1_diabetes": 0.0,
            "type2_diabetes": 0.0,
        }

        # Assegnazione del contributo del motore a regole
        combined_scores[rule_label] += self.RULE_WEIGHT * rule_score

        svm_probabilities = None
        bn_probabilities = None

        # 2. Contributo SVM (Machine Learning supervisionato)
        if self.svm is not None:
            svm_probabilities = self.svm.predict(patient_values)
            p_non_diab = float(svm_probabilities.get("normal", 0.0))
            p_diab = float(svm_probabilities.get("type2_diabetes", 0.0))

            # Instradamento della probabilità di non-diabete
            if rule_label == "healthy":
                combined_scores["healthy"] += self.SVM_WEIGHT * p_non_diab
            elif rule_label == "normal":
                combined_scores["normal"] += self.SVM_WEIGHT * p_non_diab
            elif rule_label == "prediabetes":
                combined_scores["prediabetes"] += self.SVM_WEIGHT * (0.6 * p_non_diab)
                combined_scores["normal"] += self.SVM_WEIGHT * (0.4 * p_non_diab)
            else:
                combined_scores["normal"] += self.SVM_WEIGHT * p_non_diab

            # Instradamento della probabilità di diabete secondo differenziazione clinica
            if rule_label == "type1_diabetes":
                combined_scores["type1_diabetes"] += self.SVM_WEIGHT * p_diab
            elif rule_label == "type2_diabetes":
                combined_scores["type2_diabetes"] += self.SVM_WEIGHT * p_diab
            elif rule_label == "prediabetes":
                combined_scores["prediabetes"] += self.SVM_WEIGHT * (0.4 * p_diab)
                combined_scores["type2_diabetes"] += self.SVM_WEIGHT * (0.6 * p_diab)
            else:
                # Per soggetti normali/sani, p_diab rappresenta rischio statistico di fondo
                combined_scores["type2_diabetes"] += self.SVM_WEIGHT * p_diab

        # 3. Contributo Rete Bayesiana (Modello grafico probabilistico)
        if self.bn is not None:
            bn_probabilities = self.bn.predict(patient_values)
            p_bn_norm = float(bn_probabilities.get("normal", 0.0))
            p_bn_pre = float(bn_probabilities.get("prediabetes", 0.0))
            p_bn_diab = float(bn_probabilities.get("type2_diabetes", 0.0))

            combined_scores["prediabetes"] += self.BN_WEIGHT * p_bn_pre

            if rule_label == "healthy":
                combined_scores["healthy"] += self.BN_WEIGHT * p_bn_norm
            else:
                combined_scores["normal"] += self.BN_WEIGHT * p_bn_norm

            if rule_label == "type1_diabetes":
                combined_scores["type1_diabetes"] += self.BN_WEIGHT * p_bn_diab
            else:
                combined_scores["type2_diabetes"] += self.BN_WEIGHT * p_bn_diab

        # Salvaguardia di consistenza clinica:
        # Se glicemia o HbA1c sono diagnostiche per diabete, la decisione finale
        # deve essere strettamente Diabete di Tipo 1 o Diabete di Tipo 2.
        # Se invece glicemia e HbA1c sono perfettamente normali, non può essere
        # diagnosticato un diabete conclamato.
        factors = set(rule_result.get("factors", []))
        ha_diabete_clinico = "glicemia_diagnostica" in factors or "hba1c_diagnostica" in factors
        ha_prediabete_clinico = (
            ("glicemia_borderline" in factors or "hba1c_borderline" in factors)
            and not ha_diabete_clinico
        )
        ha_glicemia_normale = not ha_diabete_clinico and not ha_prediabete_clinico

        eligible_classes = None
        if ha_diabete_clinico:
            eligible_classes = ["type1_diabetes", "type2_diabetes"]
        elif ha_prediabete_clinico:
            eligible_classes = ["prediabetes", "normal"]
        elif ha_glicemia_normale:
            eligible_classes = ["healthy", "normal"]

        if eligible_classes:
            final_label = max(
                eligible_classes,
                key=lambda c: combined_scores.get(c, 0.0),
            )
        else:
            final_label = max(
                combined_scores,
                key=combined_scores.get,
            )

        ontology_info = self.ontology.get_info(final_label)

        return {
            "label": final_label,
            "display_name": ontology_info.get("display_name", final_label),
            "final_score": combined_scores[final_label],
            "all_scores": combined_scores,
            "rule_result": rule_result,
            "svm": svm_probabilities,
            "bn": bn_probabilities,
            "ontology": ontology_info,
            "models_loaded": {
                "svm": self.svm is not None,
                "bayesian_network": self.bn is not None,
            },
        }


def diagnose(patient_values: dict[str, Any]) -> dict[str, Any]:
    """Funzione pubblica di diagnosi usata da GUI, test e report PDF."""
    return HybridDiagnosis().evaluate(patient_values)
