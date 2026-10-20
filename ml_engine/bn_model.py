"""
bn_model.py
===========

Modulo di addestramento e utilizzo della Rete Bayesiana (Bayesian Network)
del progetto Diabete KBS Ibrido.

Il modello usa il dataset clinico unificato creato con:
    python dataset_builder/build_unified_dataset.py

e salva la rete addestrata in:
    data/processed/bayesian_diabetes_model.bif

Esecuzione dalla cartella principale del progetto:
    python ml_engine/bn_model.py

Nota didattica:
Una Rete Bayesiana discreta lavora su stati discreti e non direttamente su
valori continui. Per questo i valori clinici vengono prima trasformati in
fasce interpretabili: ad esempio glicemia normale, borderline o elevata.

La struttura della rete è volutamente esplicita e leggibile:

    age_group ────────┐
    bmi_group ────────┤
    glucose_group ────┤
    hba1c_group ─────┼──> diabetes_class
    bp_group ─────────┤
    lipid_risk ───────┘

In questo modo il nodo diabetes_class dipende dai principali fattori clinici.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# 1. Percorsi del progetto
# ---------------------------------------------------------------------------
# Il file può essere avviato direttamente con:
#     python ml_engine/bn_model.py
# Inseriamo la root del progetto nel percorso degli import per rendere
# disponibile il modulo ml_engine.feature_engineering.
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.readwrite import BIFReader, BIFWriter

from ml_engine.feature_engineering import discretize


# ---------------------------------------------------------------------------
# 2. Configurazione
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "diabetes_clinical_unified.csv",
)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "processed",
    "bayesian_diabetes_model.bif",
)

TARGET_COLUMN = "diabetes_class"

# Nodi osservati che diventano genitori della variabile target.
# Gli stessi nomi sono prodotti dalla funzione discretize().
EVIDENCE_NODES = [
    "age_group",
    "bmi_group",
    "glucose_group",
    "hba1c_group",
    "bp_group",
    "lipid_risk",
]

EXPECTED_CLASSES = [
    "normal",
    "prediabetes",
    "type2_diabetes",
]


class BayesianModel:
    """
    Gestisce ciclo di vita, inferenza e persistenza della Rete Bayesiana.

    Metodi principali:
    - train(): discretizza i dati, stima CPD bayesiane e salva la rete BIF.
    - load(): carica il file BIF e prepara il motore di inferenza.
    - predict(): restituisce P(normal), P(prediabetes), P(type2_diabetes).
    """

    def __init__(self) -> None:
        self.model: DiscreteBayesianNetwork | None = None
        self.inference: VariableElimination | None = None
        self.metadata: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Controlli e preparazione del dataset
    # -----------------------------------------------------------------------
    @staticmethod
    def _check_dataset_exists() -> None:
        """Verifica che il Dataset Builder abbia prodotto il CSV unificato."""
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                "Dataset unificato non trovato.\n\n"
                "Prima esegui:\n"
                "  python dataset_builder/build_unified_dataset.py"
            )

    @staticmethod
    def _load_training_data() -> pd.DataFrame:
        """
        Carica esclusivamente righe dotate di target diabetes_class.

        Le righe senza diagnosi possono essere usate in altri contesti, ma non
        possono contribuire all'apprendimento supervisionato delle CPD.
        """
        dataframe = pd.read_csv(DATASET_PATH)

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"La colonna target '{TARGET_COLUMN}' manca dal dataset."
            )

        dataframe = dataframe[
            dataframe[TARGET_COLUMN].notna()
        ].copy()

        if dataframe.empty:
            raise ValueError(
                "Il dataset non contiene righe etichettate per la Rete Bayesiana."
            )

        return dataframe

    @staticmethod
    def _prepare_discrete_data(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Trasforma le feature continue in stati discreti per la BN.

        La funzione discretize() produce:
        - age_group: young / middle / older / unknown
        - bmi_group: normal / overweight / obese / unknown
        - glucose_group: normal / borderline / high / unknown
        - hba1c_group: normal / borderline / high / unknown
        - bp_group: normal / high / unknown
        - lipid_risk: normal / high / unknown

        Gli stati 'unknown' sono importanti: rappresentano analisi non
        disponibili, senza eliminare inutilmente la riga dal dataset.
        """
        discrete = discretize(dataframe)
        discrete[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(str)

        # pgmpy richiede stati coerenti e discreti. Convertire tutto a stringa
        # evita differenze fra valori numerici, NaN e tipi pandas.
        for column in EVIDENCE_NODES + [TARGET_COLUMN]:
            discrete[column] = (
                discrete[column]
                .fillna("unknown")
                .astype(str)
            )

        return discrete[EVIDENCE_NODES + [TARGET_COLUMN]].copy()

    # -----------------------------------------------------------------------
    # Costruzione e stima della rete
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_network_structure() -> DiscreteBayesianNetwork:
        """
        Costruisce la struttura causale/dipendenziale dichiarata nel progetto.

        Tutti i fattori clinici sono genitori del nodo diabetes_class.
        Non viene appresa automaticamente la struttura: questa scelta rende
        il modello più spiegabile e più semplice da discutere all'esame.
        """
        edges = [
            (node, TARGET_COLUMN)
            for node in EVIDENCE_NODES
        ]

        return DiscreteBayesianNetwork(edges)

    def _estimate_cpds(self, discrete_data: pd.DataFrame) -> None:
        """
        Stima le probabilità condizionate (CPD) tramite BayesianEstimator.

        Usiamo prior_type='BDeu' e un piccolo equivalent_sample_size per
        evitare probabilità esattamente zero quando una combinazione di stati
        è rara o assente nel dataset. Le CPD vengono calcolate esplicitamente
        e aggiunte al modello: è più compatibile di model.fit(..., prior_type)
        con versioni diverse di pgmpy.
        """
        if self.model is None:
            raise RuntimeError("Rete non creata prima della stima delle CPD.")

        estimator = BayesianEstimator(
            self.model,
            discrete_data,
        )

        cpds = estimator.get_parameters(
            prior_type="BDeu",
            equivalent_sample_size=5,
        )

        self.model.add_cpds(*cpds)

        if not self.model.check_model():
            raise RuntimeError(
                "La rete Bayesiana non ha superato il controllo di validità."
            )

    # -----------------------------------------------------------------------
    # Addestramento e salvataggio
    # -----------------------------------------------------------------------
    def train(self) -> None:
        """Esegue discretizzazione, training delle CPD e salvataggio in BIF."""
        self._check_dataset_exists()

        print("=" * 68)
        print("ADDESTRAMENTO RETE BAYESIANA (BAYESIAN NETWORK)")
        print("=" * 68)
        print(f"Dataset: {DATASET_PATH}")

        raw_data = self._load_training_data()
        discrete_data = self._prepare_discrete_data(raw_data)

        print(f"Righe etichettate utilizzate: {len(discrete_data)}")
        print("Distribuzione delle classi:")
        print(discrete_data[TARGET_COLUMN].value_counts())

        print("\nStati discreti rilevati:")
        for node in EVIDENCE_NODES:
            states = sorted(discrete_data[node].unique().tolist())
            print(f"  - {node}: {states}")

        self.model = self._build_network_structure()

        print("\nStima delle tabelle di probabilità condizionata (CPD)...")
        self._estimate_cpds(discrete_data)

        # VariableElimination permette di interrogare P(diabetes_class | evidenze)
        # quando il modello è in memoria, utile per test immediati e GUI.
        self.inference = VariableElimination(self.model)

        self.metadata = {
            "algorithm": "DiscreteBayesianNetwork_BayesianEstimator_BDeu",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": DATASET_PATH,
            "rows_used": int(len(discrete_data)),
            "target": TARGET_COLUMN,
            "evidence_nodes": EVIDENCE_NODES,
            "structure_edges": [
                [node, TARGET_COLUMN]
                for node in EVIDENCE_NODES
            ],
            "prior_type": "BDeu",
            "equivalent_sample_size": 5,
        }

        self._save()

        print("\nAddestramento completato correttamente.")
        print(f"Rete Bayesiana salvata: {MODEL_PATH}")

    def _save(self) -> None:
        """Salva la rete nel formato BIF, portabile e leggibile."""
        if self.model is None:
            raise RuntimeError("Nessun modello da salvare.")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        BIFWriter(self.model).write_bif(MODEL_PATH)

    # -----------------------------------------------------------------------
    # Caricamento e inferenza su un nuovo paziente
    # -----------------------------------------------------------------------
    def load(self) -> None:
        """Carica il file BIF salvato e inizializza l'inferenza esatta."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Modello BN non trovato.\n\n"
                "Esegui prima:\n"
                "  python ml_engine/bn_model.py"
            )

        self.model = BIFReader(MODEL_PATH).get_model()
        self.inference = VariableElimination(self.model)

    def _build_evidence(
        self,
        patient_values: dict[str, Any],
    ) -> dict[str, str]:
        """
        Converte i valori della GUI negli stessi stati discreti del training.

        Per coerenza, la discretizzazione di un singolo paziente usa la stessa
        funzione centrale usata sul dataset. Questo evita soglie duplicate in
        file diversi e riduce il rischio di incoerenza tra training e GUI.
        """
        patient_dataframe = pd.DataFrame([patient_values])
        discrete_patient = discretize(patient_dataframe)

        evidence = {}
        for node in EVIDENCE_NODES:
            state = discrete_patient.iloc[0][node]

            if pd.isna(state):
                state = "unknown"

            evidence[node] = str(state)

        return evidence

    def predict(
        self,
        patient_values: dict[str, Any],
    ) -> dict[str, float]:
        """
        Restituisce le probabilità delle tre classi cliniche.

        Output previsto:
        {
            "normal": 0.20,
            "prediabetes": 0.55,
            "type2_diabetes": 0.25,
        }

        Se il training non ha osservato una delle tre classi, essa viene
        comunque aggiunta all'output con probabilità 0.0, così il motore di
        fusione riceve sempre una struttura coerente.
        """
        if self.inference is None:
            raise RuntimeError(
                "Rete Bayesiana non caricata. Chiama load() oppure train() prima di predict()."
            )

        evidence = self._build_evidence(patient_values)

        query = self.inference.query(
            variables=[TARGET_COLUMN],
            evidence=evidence,
            show_progress=False,
        )

        probabilities = {
            query.state_names[TARGET_COLUMN][index]: float(value)
            for index, value in enumerate(query.values)
        }

        # Garantisce sempre le tre chiavi attese dal diagnosis_engine.
        for clinical_class in EXPECTED_CLASSES:
            probabilities.setdefault(clinical_class, 0.0)

        return probabilities


if __name__ == "__main__":
    bayesian_model = BayesianModel()
    bayesian_model.train()