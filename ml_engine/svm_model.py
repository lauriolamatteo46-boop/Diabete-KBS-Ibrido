"""
svm_model.py
===========

Modulo di addestramento e utilizzo della Support Vector Machine (SVM)
del progetto Diabete KBS Ibrido.

Il modulo legge il dataset clinico unificato creato da:
    python dataset_builder/build_unified_dataset.py

e salva il modello addestrato in:
    data/processed/svm_diabetes_model.pkl

Esecuzione dalla cartella principale del progetto:
    python ml_engine/svm_model.py

Il file esportato contiene:
- la pipeline completa di preprocessing + SVM;
- l'elenco ordinato delle feature utilizzate;
- metadati utili per tracciabilità e riuso nella GUI.

Nota didattica:
La SVM è usata per stimare la probabilità della classe "type2_diabetes".
Le probabilità non sono native per una SVM standard: la classe SVC viene
calibrata con CalibratedClassifierCV, così il motore ibrido può combinarle
con regole e Rete Bayesiana.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# 1. Percorsi del progetto
# ---------------------------------------------------------------------------
# Quando il file è avviato direttamente con:
#     python ml_engine/svm_model.py
# Python mette ml_engine/ nel percorso degli import. Aggiungiamo quindi
# esplicitamente la root del progetto per poter importare moduli fratelli.
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# 2. File di input/output e configurazione
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
    "svm_diabetes_model.pkl",
)

# Feature candidate del modello. Alcune fonti possono non possedere tutti
# questi valori: il programma seleziona automaticamente solo quelle presenti
# e con almeno un dato valido.
CANDIDATE_FEATURES = [
    "age_years",
    "sex",
    "bmi",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "fasting_glucose_mg_dl",
    "hba1c_percent",
    "insulin_uiu_ml",
    "total_cholesterol_mg_dl",
    "hdl_mg_dl",
    "ldl_mg_dl",
    "triglycerides_mg_dl",
    "creatinine_mg_dl",
    "urea_mg_dl",
]

TARGET_COLUMN = "diabetes_class"
POSITIVE_CLASS = "type2_diabetes"
RANDOM_STATE = 42
TEST_SIZE = 0.25


class SVMModel:
    """
    Gestisce l'intero ciclo di vita del classificatore SVM.

    Metodi principali:
    - train(): legge i dati, prepara le feature, addestra e salva il modello.
    - load(): carica un modello già addestrato da disco.
    - predict(): restituisce probabilità normal/diabete per un nuovo paziente.
    """

    def __init__(self) -> None:
        self.model: Pipeline | None = None
        self.features: list[str] = []
        self.metadata: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Caricamento e preparazione dati
    # -----------------------------------------------------------------------
    @staticmethod
    def _check_dataset_exists() -> None:
        """Controlla che il CSV unificato sia stato creato prima del training."""
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                "Dataset unificato non trovato.\n\n"
                "Prima esegui:\n"
                "  python dataset_builder/build_unified_dataset.py"
            )

    @staticmethod
    def _select_available_features(dataframe: pd.DataFrame) -> list[str]:
        """
        Restituisce le feature candidate presenti e non completamente vuote.

        Questo passaggio rende il progetto robusto: se un dataset non contiene,
        per esempio, HbA1c oppure insulina, il training può proseguire con le
        restanti feature disponibili.
        """
        selected = []

        for feature in CANDIDATE_FEATURES:
            if feature not in dataframe.columns:
                continue

            if dataframe[feature].notna().any():
                selected.append(feature)

        if not selected:
            raise ValueError(
                "Nessuna feature clinica disponibile per addestrare la SVM."
            )

        return selected

    @staticmethod
    def _create_binary_target(dataframe: pd.DataFrame) -> pd.Series:
        """
        Converte il target a tre classi in target binario per la SVM.

        - type2_diabetes -> 1
        - normal / prediabetes -> 0

        La Rete Bayesiana mantiene invece le tre classi. Questa scelta rende
        chiara la differenza tra i due modelli nel progetto ibrido.
        """
        return (
            dataframe[TARGET_COLUMN]
            .astype(str)
            .eq(POSITIVE_CLASS)
            .astype(int)
        )

    @staticmethod
    def _balance_binary_classes(
        dataframe: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Bilancia le due classi tramite sottocampionamento della classe maggiore.

        Senza questo passaggio, un dataset con molti più casi non diabetici
        potrebbe produrre un modello apparentemente accurato ma poco capace
        di identificare i casi positivi.
        """
        working = dataframe.copy()
        working["__target__"] = target.values

        counts = working["__target__"].value_counts()
        if len(counts) < 2:
            raise ValueError(
                "Sono necessarie sia la classe positiva sia la classe negativa "
                "per addestrare la SVM."
            )

        samples_per_class = int(counts.min())
        balanced_parts = []

        for class_value, group in working.groupby("__target__"):
            print(
                f"Classe {class_value}: disponibili={len(group)}, "
                f"usate={samples_per_class}"
            )

            balanced_parts.append(
                group.sample(
                    n=samples_per_class,
                    random_state=RANDOM_STATE,
                )
            )

        balanced = pd.concat(
            balanced_parts,
            ignore_index=True,
        )

        # Mescola le righe per evitare blocchi consecutivi della stessa classe.
        balanced = balanced.sample(
            frac=1,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

        balanced_target = balanced.pop("__target__").astype(int)
        return balanced, balanced_target

    # -----------------------------------------------------------------------
    # Pipeline di preprocessing e modello SVM
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_pipeline(
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> Pipeline:
        """
        Crea la pipeline completa: preprocessing + SVM calibrata.

        Parte numerica:
        1. imputazione mediana dei valori mancanti;
        2. StandardScaler, essenziale per SVM basate su distanza.

        Parte categorica:
        1. imputazione con valore più frequente;
        2. One-Hot Encoding del sesso; gestisce valori mai visti in fase
           di predizione senza generare errori.

        Modello:
        - SVC con kernel RBF e class_weight='balanced';
        - CalibratedClassifierCV trasforma il punteggio SVM in probabilità.
        """
        transformers = []

        if numeric_features:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("numeric", numeric_pipeline, numeric_features)
            )

        if categorical_features:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        if not transformers:
            raise ValueError("Impossibile creare la pipeline: nessuna feature.")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        # probability=False: la probabilità viene gestita dalla calibrazione
        # esterna, più esplicita e adatta a questo progetto didattico.
        base_svm = SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=RANDOM_STATE,
        )

        calibrated_svm = CalibratedClassifierCV(
            estimator=base_svm,
            cv=3,
            method="sigmoid",
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("svm", calibrated_svm),
            ]
        )

    # -----------------------------------------------------------------------
    # Addestramento
    # -----------------------------------------------------------------------
    def train(self) -> None:
        """Esegue lettura, bilanciamento, training, valutazione e salvataggio."""
        self._check_dataset_exists()

        print("=" * 68)
        print("ADDESTRAMENTO SUPPORT VECTOR MACHINE (SVM)")
        print("=" * 68)
        print(f"Dataset: {DATASET_PATH}")

        dataframe = pd.read_csv(DATASET_PATH)

        if TARGET_COLUMN not in dataframe.columns:
            raise ValueError(
                f"La colonna target '{TARGET_COLUMN}' non è presente nel CSV."
            )

        # Le righe senza classe non sono utilizzabili per addestramento.
        dataframe = dataframe[dataframe[TARGET_COLUMN].notna()].copy()

        if dataframe.empty:
            raise ValueError(
                "Non sono presenti righe etichettate nel dataset unificato."
            )

        self.features = self._select_available_features(dataframe)
        target = self._create_binary_target(dataframe)

        print(f"Righe etichettate disponibili: {len(dataframe)}")
        print("Distribuzione target prima del bilanciamento:")
        print(target.value_counts().sort_index())
        print("Feature selezionate:")
        for feature in self.features:
            print(f"  - {feature}")

        # Manteniamo soltanto le feature, così la SVM non usa accidentalmente
        # source_dataset, source_row oppure il target originale.
        features_dataframe = dataframe[self.features].copy()
        features_dataframe, target = self._balance_binary_classes(
            features_dataframe,
            target,
        )

        numeric_features = [
            feature
            for feature in self.features
            if feature != "sex"
        ]
        categorical_features = [
            feature
            for feature in self.features
            if feature == "sex"
        ]

        x_train, x_test, y_train, y_test = train_test_split(
            features_dataframe,
            target,
            test_size=TEST_SIZE,
            stratify=target,
            random_state=RANDOM_STATE,
        )

        self.model = self._build_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        print("\nAvvio addestramento SVM...")
        self.model.fit(x_train, y_train)

        print("\n=== REPORT DI CLASSIFICAZIONE ===")
        predictions = self.model.predict(x_test)
        print(
            classification_report(
                y_test,
                predictions,
                target_names=["non_diabete", "diabete_tipo2"],
                zero_division=0,
            )
        )

        self.metadata = {
            "algorithm": "SVC_RBF_calibrated",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_path": DATASET_PATH,
            "rows_before_balancing": int(len(dataframe)),
            "rows_after_balancing": int(len(features_dataframe)),
            "features": self.features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "positive_class": POSITIVE_CLASS,
            "random_state": RANDOM_STATE,
        }

        self._save()

        print("\nAddestramento completato correttamente.")
        print(f"Modello SVM salvato: {MODEL_PATH}")

    def _save(self) -> None:
        """Salva modello, feature e metadati nello stesso file joblib."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "features": self.features,
                "metadata": self.metadata,
            },
            MODEL_PATH,
        )

    # -----------------------------------------------------------------------
    # Caricamento e predizione su un singolo paziente
    # -----------------------------------------------------------------------
    def load(self) -> None:
        """Carica da disco una SVM già addestrata."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Modello SVM non trovato.\n\n"
                "Esegui prima:\n"
                "  python ml_engine/svm_model.py"
            )

        saved_data = joblib.load(MODEL_PATH)
        self.model = saved_data["model"]
        self.features = saved_data["features"]
        self.metadata = saved_data.get("metadata", {})

    def predict(self, patient_values: dict[str, Any]) -> dict[str, float]:
        """
        Restituisce le probabilità binarie per un singolo paziente.

        I valori non forniti vengono impostati a NaN: la pipeline userà
        l'imputazione configurata durante l'addestramento.
        """
        if self.model is None:
            raise RuntimeError(
                "SVM non caricata. Chiama load() oppure train() prima di predict()."
            )

        patient_row = {
            feature: patient_values.get(feature, None)
            for feature in self.features
        }

        dataframe = pd.DataFrame([patient_row])
        probability_positive = float(
            self.model.predict_proba(dataframe)[0][1]
        )

        return {
            "normal": 1.0 - probability_positive,
            "type2_diabetes": probability_positive,
        }


if __name__ == "__main__":
    svm_model = SVMModel()
    svm_model.train()