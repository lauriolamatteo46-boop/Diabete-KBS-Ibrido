"""
diagnosis_engine.py
===================

Motore centrale del Diabete KBS Ibrido.

Correzione di compatibilità:
Questo file usa i nomi DATASET_PATH e MODEL_PATH adottati dalle versioni
strutturate di svm_model.py e bn_model.py. La vecchia versione tentava di
importare una costante chiamata PATH, che non esiste più.
"""

import os
import sys
from typing import Any

# Permette gli import quando main_gui.py viene avviato direttamente.
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

    - Datalog/regole: interpreta direttamente le soglie cliniche;
    - SVM: stima statistica binaria normale/diabete tipo 2;
    - BN: probabilità sulle tre classi normale/prediabete/diabete tipo 2;
    - Ontologia: descrizione, gravità e raccomandazioni dell'esito.
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
        Carica SVM e BN solo se i file addestrati sono disponibili.

        Questo permette alla GUI di funzionare anche prima del training:
        in quel caso l'esito userà solamente le regole cliniche e l'ontologia.
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
        """Esegue una valutazione completa e restituisce un risultato unico."""

        # 1. Risultato simbolico: soglie e fattori espliciti.
        rule_result = self.rule_engine.diagnosi(patient_values)

        # Dizionario comune per i punteggi provenienti dai tre motori.
        combined_scores = {
            "normal": 0.0,
            "prediabetes": 0.0,
            "type2_diabetes": 0.0,
        }

        # Le regole forniscono una classe e uno score da 0 a 1.
        combined_scores[rule_result["label"]] += (
            self.RULE_WEIGHT * rule_result["score"]
        )

        svm_probabilities = None
        bn_probabilities = None

        # 2. La SVM ha solamente normal e type2_diabetes.
        if self.svm is not None:
            svm_probabilities = self.svm.predict(patient_values)

            combined_scores["normal"] += (
                self.SVM_WEIGHT * svm_probabilities["normal"]
            )
            combined_scores["type2_diabetes"] += (
                self.SVM_WEIGHT * svm_probabilities["type2_diabetes"]
            )

        # 3. La BN restituisce tutte e tre le classi cliniche.
        if self.bn is not None:
            bn_probabilities = self.bn.predict(patient_values)

            for clinical_class, probability in bn_probabilities.items():
                combined_scores.setdefault(clinical_class, 0.0)
                combined_scores[clinical_class] += (
                    self.BN_WEIGHT * probability
                )

        final_label = max(
            combined_scores,
            key=combined_scores.get,
        )

        return {
            "label": final_label,
            "final_score": combined_scores[final_label],
            "all_scores": combined_scores,
            "rule_result": rule_result,
            "svm": svm_probabilities,
            "bn": bn_probabilities,
            "ontology": self.ontology.get_info(final_label),
            "models_loaded": {
                "svm": self.svm is not None,
                "bayesian_network": self.bn is not None,
            },
        }


def diagnose(patient_values: dict[str, Any]) -> dict[str, Any]:
    """Funzione pubblica usata dalla GUI e dal generatore PDF."""
    return HybridDiagnosis().evaluate(patient_values)