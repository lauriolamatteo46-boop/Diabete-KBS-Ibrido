# Diabete KBS Ibrido 

> **Sistema ibrido di supporto alla valutazione del rischio di diabete di tipo 2**, basato su
> **regole cliniche simboliche (stile Datalog)**, **ontologia leggera**, **Support Vector Machine (SVM)**
> e **Rete Bayesiana (BN)**, con interfaccia grafica desktop offline ed esportazione del referto in PDF.

> ⚠️ **Avvertenza.** Questo è un progetto **didattico** (Ingegneria della Conoscenza / Knowledge-Based Systems).
> **Non è un dispositivo medico**, non produce diagnosi e non sostituisce in alcun modo la valutazione di un
> professionista sanitario.

---

## Indice

1. [Panoramica](#1-panoramica)
2. [Cosa fa il programma, in sintesi](#2-cosa-fa-il-programma-in-sintesi)
3. [Architettura](#3-architettura)
4. [Struttura del progetto](#4-struttura-del-progetto)
5. [Requisiti e installazione](#5-requisiti-e-installazione)
6. [Dataset utilizzati](#6-dataset-utilizzati)
7. [Guida all'esecuzione (pipeline e test)](#7-guida-allesecuzione-pipeline-e-test)
8. [Documentazione dei moduli: funzioni, classi e come funzionano](#8-documentazione-dei-moduli-funzioni-classi-e-come-funzionano)
   - [8.1 `dataset_builder/`](#81-dataset_builder)
   - [8.2 `ml_engine/feature_engineering.py`](#82-ml_enginefeature_engineeringpy)
   - [8.3 `kbs_engine/`](#83-kbs_engine)
   - [8.4 `ml_engine/svm_model.py`](#84-ml_enginesvm_modelpy)
   - [8.5 `ml_engine/bn_model.py`](#85-ml_enginebn_modelpy)
   - [8.6 `diagnosis_engine.py`](#86-diagnosis_enginepy)
   - [8.7 `report_generator.py`](#87-report_generatorpy)
   - [8.8 `main_gui.py`](#88-main_guipy)
   - [8.9 `test_pipeline.py`](#89-test_pipelinepy)
9. [La fusione dei risultati](#9-la-fusione-dei-risultati)
10. [Esempi d'uso](#10-esempi-duso)
11. [Uso programmatico (API Python)](#11-uso-programmatico-api-python)
12. [Riepilogo delle librerie e delle funzioni usate](#12-riepilogo-delle-librerie-e-delle-funzioni-usate)
13. [Possibili sviluppi futuri](#14-possibili-sviluppi-futuri)

---

## 1. Panoramica

**Diabete KBS Ibrido** è un'applicazione Python che, a partire da una serie di **valori clinici numerici**
(età, sesso, peso, altezza, pressione arteriosa, glicemia a digiuno, HbA1c, insulina, profilo lipidico, ecc.),
esegue una diagnosi differenziale rigorosa e restituisce una **valutazione differenziata a cinque classi**:

| Classe finale       | Nome a display                                              | Significato clinico                                                            |
|---------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------|
| `healthy`           | Stato in salute ottimale                                    | parametri glicemici, pressori, lipidici e ponderali nei limiti fisiologici     |
| `normal`            | Non diabete (Fattori di rischio da monitorare)              | glicemia e HbA1c normali, ma presenti fattori cardiovascolari/metabolici (es. PA, lipidi o BMI) |
| `prediabetes`       | Prediabete (Rischio glicemico aumentato)                    | alterata glicemia a digiuno (100–125 mg/dL) o HbA1c borderline (5.7–6.4%)      |
| `type1_diabetes`    | Diabete Mellito di Tipo 1                                   | iperglicemia con marcato deficit insulinico (< 5 µIU/mL), esordio giovanile o normopeso |
| `type2_diabetes`    | Diabete Mellito di Tipo 2                                   | iperglicemia correlata a insulino-resistenza, obesità/sovrappeso, età matura   |

L'idea centrale è **l'ibridazione di paradigmi diversi dell'Intelligenza Artificiale**:

| Livello | Paradigma | Ruolo nel sistema | Peso nella fusione |
|---|---|---|---|
| **Regole** (`kbs_engine/rules.py` + `datalog_engine.py`) | IA simbolica / basi di conoscenza | Applica le soglie cliniche standard (glicemia, HbA1c, pressione, BMI, lipidi) e produce **fattori espliciti e spiegabili** | **0,40** |
| **SVM** (`ml_engine/svm_model.py`) | Machine Learning (apprendimento supervisionato) | Stima la probabilità binaria *non diabete* / *diabete tipo 2* a partire dai dati | **0,25** |
| **Rete Bayesiana** (`ml_engine/bn_model.py`) | Modelli grafici probabilistici | Stima la probabilità sulle **tre** classi a partire da fattori discretizzati | **0,35** |
| **Ontologia** (`kbs_engine/ontology_manager.py`) | Rappresentazione della conoscenza | **Non vota**: arricchisce l'esito finale con descrizione, gravità e raccomandazioni | — |

Una caratteristica importante: **l'utente inserisce solo misure numeriche**. Non deve dichiarare
"ho la pressione alta sì/no" o "sono obeso sì/no": il programma **calcola il BMI** da peso e altezza e
**deriva i fattori** (ipertensione, obesità, trigliceridi alti, HDL basso, ...) applicando le soglie
cliniche definite nelle regole.

---

## 2. Cosa fa il programma, in sintesi

1. **Costruisce un dataset unificato** a partire da tre dataset pubblici eterogenei sul diabete
   (Mendeley, DiaBD, Pima), portandoli in uno **schema canonico comune** (24 colonne) e conservando
   la tracciabilità della riga di origine.
2. **Addestra una SVM** (kernel RBF, classi bilanciate, probabilità calibrate) per distinguere
   *diabete tipo 2* da *non diabete*.
3. **Addestra una Rete Bayesiana discreta** con struttura dichiarata a mano
   (6 fattori clinici discretizzati → classe diagnostica) e CPD stimate con prior BDeu.
4. **Valuta un nuovo paziente** combinando regole, SVM e BN con una media pesata, e arricchisce il
   risultato con l'ontologia.
5. **Mostra tutto in una GUI Tkinter** (tre pulsanti: *Valuta*, *Ripristina valori*, *Esporta PDF*)
   e permette di **esportare un referto PDF** in `reports/`.
6. **Degrada con grazia**: se SVM e/o BN non sono ancora stati addestrati, il sistema funziona comunque
   usando solo le regole e l'ontologia.

---

## 3. Architettura

### 3.1 Flusso complessivo

```text
                       ┌──────────────────────── FASE OFFLINE (training) ────────────────────────┐
                       │                                                                         │
 data/raw/*.csv        │  dataset_builder/build_unified_dataset.py                               │
 (Mendeley, DiaBD,  ───┼─►  adapters.adapt()  ──►  schema canonico (canonical_schema.COLUMNS)    │
  Pima)                │          │                                                              │
                       │          ▼                                                              │
                       │  data/processed/diabetes_clinical_unified.csv                           │
                       │          │                                                              │
                       │          ├──► ml_engine/svm_model.py ──► data/processed/svm_diabetes_model.pkl
                       │          │                                                              │
                       │          └──► ml_engine/bn_model.py  ──► data/processed/bayesian_diabetes_model.bif
                       └─────────────────────────────────────────────────────────────────────────┘

                       ┌──────────────────────── FASE ONLINE (uso) ──────────────────────────────┐
                       │                                                                         │
  main_gui.py  ──────► │  diagnosis_engine.diagnose(valori)                                      │
  (valori numerici,    │        │                                                                │
   calcolo BMI)        │        ├─► DatalogEngine.diagnosi()  → label, score, fattori  (peso 0,40)│
                       │        ├─► SVMModel.predict()        → P(normal), P(T2D)      (peso 0,25)│
                       │        ├─► BayesianModel.predict()   → P(normal/pre/T2D)      (peso 0,35)│
                       │        └─► OntologyManager.get_info()→ gravità, descrizione, raccomandazioni
                       │                                                                         │
                       │  ◄──────────────── dizionario risultato ──────────────────              │
                       └─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                       report_generator.export_pdf()  ──►  reports/report_YYYYMMDD_HHMMSS.pdf
```

### 3.2 Principi progettuali

- **Separazione delle responsabilità**: ogni livello (dati, regole, ML, fusione, interfaccia, report) sta in un
  modulo dedicato.
- **Un'unica fonte di verità per le soglie di discretizzazione**: `ml_engine/feature_engineering.discretize()` è
  usata sia in fase di addestramento della BN sia in fase di inferenza su un singolo paziente, così le soglie
  non possono divergere tra training e uso reale.
- **Struttura della BN dichiarata, non appresa**: il grafo è scritto a mano (tutti i fattori sono genitori del
  nodo target), per massimizzare la spiegabilità.
- **Nessun dato inventato**: il builder lascia `NaN` dove una fonte non possiede una misura. `source_dataset` e
  `source_row` consentono di risalire all'origine di ogni record.
- **Funzionamento offline**: nessuna chiamata di rete in fase di esecuzione; la GUI è Tkinter e il PDF è generato
  localmente con ReportLab.

---

## 4. Struttura del progetto

```text
icon_diabete_V6/
├── main_gui.py                      # Interfaccia grafica Tkinter (punto di ingresso dell'applicazione)
├── diagnosis_engine.py              # Motore di fusione: HybridDiagnosis + funzione diagnose()
├── report_generator.py              # Esportazione del referto PDF (ReportLab)
├── test_pipeline.py                 # Suite di test automatici (Feature Engineering, BN e Motore Ibrido)
├── test_full_project.py             # Suite globale di collaudo (42 test: dati, modelli, regole, Tipo 1/2/Prediabete/Salute)
├── requirements.txt                 # Dipendenze Python
├── README.md                        # Questo file
│
├── dataset_builder/                 # Costruzione del dataset unificato
│   ├── __init__.py
│   ├── canonical_schema.py          # COLUMNS, NUMERIC, CLASSES
│   ├── adapters.py                  # Adattatori Mendeley / DiaBD / Pima + dispatcher adapt()
│   └── build_unified_dataset.py     # Script: legge data/raw/*.csv e scrive il CSV unificato
│
├── kbs_engine/                      # Conoscenza simbolica
│   ├── __init__.py
│   ├── rules.py                     # derive() e diagnose_rules(): soglie cliniche e punteggio
│   ├── datalog_engine.py            # DatalogEngine: facade che espone le regole
│   └── ontology_manager.py          # OntologyManager: descrizione, gravità, raccomandazioni
│
├── ml_engine/                       # Apprendimento automatico
│   ├── __init__.py
│   ├── feature_engineering.py       # discretize(): fasce cliniche condivise (usate dalla BN)
│   ├── svm_model.py                 # SVMModel: training, salvataggio, caricamento, predict
│   └── bn_model.py                  # BayesianModel: training, salvataggio BIF, caricamento, predict
│
├── data/
│   ├── raw/                         # CSV originali scaricati dalle fonti
│   │   ├── Dataset of Diabetes .csv                                   # Mendeley (1000 righe)
│   │   ├── diabetes (1).csv                                           # Pima Indians (768 righe)
│   │   └── DiaBD_A Diabetes Dataset for Enhanced Risk Analysis
│   │       and Research in Bangladesh.csv                             # DiaBD (5288 righe)
│   └── processed/                   # Prodotti della pipeline
│       ├── diabetes_clinical_unified.csv                              # 7056 righe × 24 colonne
│       ├── svm_diabetes_model.pkl                                     # SVM addestrata (joblib)
│       └── bayesian_diabetes_model.bif                                # Rete Bayesiana (formato BIF)
│
└── reports/                         # PDF generati dalla GUI (contiene solo .gitkeep all'inizio)
```

> Le cartelle `__pycache__/` presenti nell'archivio contengono bytecode compilato con **Python 3.14**
> (`*.cpython-314.pyc`) e non fanno parte del codice sorgente.

---

## 5. Requisiti e installazione

### 5.1 Requisiti

- **Python 3.10 o superiore** consigliato (il codice usa annotazioni moderne come `dict[str, Any]` e `X | None`;
  il bytecode incluso è stato generato con Python 3.14).
- **Tkinter** (incluso in Python su Windows e macOS; su Linux va installato a parte, ad esempio
  `sudo apt install python3-tk`).

### 5.2 Dipendenze (`requirements.txt`)

| Pacchetto | Versione minima | A cosa serve nel progetto |
|---|---|---|
| `pandas` | ≥ 2.0 | Lettura/scrittura CSV, manipolazione dei DataFrame, discretizzazione (`pd.cut`), concatenazione dei dataset |
| `numpy` | ≥ 1.24 | Dipendenza numerica di pandas, scikit-learn e pgmpy (non importato direttamente nel codice) |
| `scikit-learn` | ≥ 1.3 | Pipeline di preprocessing, SVM (`SVC`), calibrazione, split train/test, metriche |
| `joblib` | ≥ 1.3 | Salvataggio/caricamento del modello SVM in `.pkl` |
| `pgmpy` | ≥ 0.1.25 | Rete Bayesiana: struttura, stima CPD, inferenza esatta, lettura/scrittura BIF |
| `reportlab` | ≥ 4.0 | Generazione del referto PDF |

> Il codice importa `DiscreteBayesianNetwork` da `pgmpy.models`: è il nome usato nelle versioni recenti di pgmpy.
> Se con una versione molto vecchia si ottiene un `ImportError`, aggiornare pgmpy (`pip install -U pgmpy`).

### 5.3 Installazione

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Dataset utilizzati

Il progetto fonde **tre dataset pubblici** con strutture, unità di misura e definizioni di "diabete" diverse.
I file originali vanno in `data/raw/`.

| Dataset | Fonte | File in `data/raw/` | Righe | Contenuto principale |
|---|---|---|---|---|
| **Mendeley – Diabetes Dataset** | <https://data.mendeley.com/datasets/wj9rwkp9c2/1> | `Dataset of Diabetes .csv` | 1000 | Età, sesso, urea, creatinina, HbA1c, colesterolo, trigliceridi, HDL, LDL, VLDL, BMI, classe (N / P / Y) |
| **DiaBD** (Bangladesh) | <https://data.mendeley.com/datasets/m8cgwxs9s6/2> | `DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv` | 5288 | Età, sesso, polso, pressione sistolica/diastolica, glicemia, altezza, peso, BMI, familiarità, ipertensione, malattie cardiovascolari, ictus, diabetico (Yes/No) |
| **Pima Indians Diabetes** | <https://archive.ics.uci.edu/dataset/34/diabetes> | `diabetes (1).csv` | 768 | Gravidanze, glucosio, pressione diastolica, spessore cutaneo, insulina, BMI, funzione di pedigree, età, esito (0/1) |

> I nomi dei file **non devono** essere obbligatoriamente quelli suggeriti nella vecchia documentazione
> (`mendeley_diabetes.csv`, `diabd.csv`, `pima_diabetes.csv`): il builder riconosce la sorgente anche
> **dalle colonne** presenti (vedi [`adapt()`](#adaptpath-df)).

### 6.1 Schema canonico (24 colonne)

Definito in `dataset_builder/canonical_schema.py` come lista `COLUMNS`:

| Gruppo | Colonne |
|---|---|
| Tracciabilità | `source_dataset`, `source_row` |
| Anagrafica | `age_years`, `sex` |
| Antropometria | `weight_kg`, `height_cm`, `bmi` |
| Cardiovascolare | `systolic_bp_mmhg`, `diastolic_bp_mmhg`, `heart_rate_bpm` |
| Glicemico | `fasting_glucose_mg_dl`, `hba1c_percent`, `insulin_uiu_ml` |
| Lipidico | `total_cholesterol_mg_dl`, `hdl_mg_dl`, `ldl_mg_dl`, `vldl_mg_dl`, `triglycerides_mg_dl` |
| Renale | `creatinine_mg_dl`, `urea_mg_dl` |
| Anamnesi | `family_history_diabetes`, `hypertension_history`, `heart_disease` |
| Target | `diabetes_class` (`normal` / `prediabetes` / `type2_diabetes`) |

Il suffisso dei nomi (`_mg_dl`, `_mmhg`, `_percent`, ...) indica l'**unità di misura prevista**
dallo schema (vedi anche [§13](#13-limiti-noti-e-osservazioni-sul-codice) per le eccezioni reali nei dati).

### 6.2 Mappatura colonne originali → schema canonico

| Colonna canonica | Mendeley | DiaBD | Pima |
|---|---|---|---|
| `age_years` | `AGE` | `age` | `Age` |
| `sex` | `Gender` | `gender` | — |
| `weight_kg` | — | `weight` | — |
| `height_cm` | — | `height` | — |
| `bmi` | `BMI` | `bmi` | `BMI` |
| `systolic_bp_mmhg` | — | `systolic_bp` | — |
| `diastolic_bp_mmhg` | — | `diastolic_bp` | `BloodPressure` |
| `heart_rate_bpm` | — | `pulse_rate` | — |
| `fasting_glucose_mg_dl` | — | `glucose` | `Glucose` |
| `hba1c_percent` | `HbA1c` | — | — |
| `insulin_uiu_ml` | — | — | `Insulin` |
| `total_cholesterol_mg_dl` | `Chol` | — | — |
| `hdl_mg_dl` / `ldl_mg_dl` / `vldl_mg_dl` | `HDL` / `LDL` / `VLDL` | — | — |
| `triglycerides_mg_dl` | `TG` | — | — |
| `urea_mg_dl` | `Urea` | — | — |
| `heart_disease` | — | `cardiovascular_disease` | — |
| `diabetes_class` | `CLASS` | `diabetic` | `Outcome` |

### 6.3 Dataset unificato: numeri reali

`data/processed/diabetes_clinical_unified.csv` contiene **7056 righe × 24 colonne**.

**Distribuzione del target per sorgente**

| Sorgente | `normal` | `prediabetes` | `type2_diabetes` | Totale |
|---|---:|---:|---:|---:|
| `diabd` | 4946 | 0 | 342 | 5288 |
| `mendeley` | 156 | 0 | 844 | 1000 |
| `pima` | 500 | 0 | 268 | 768 |
| **Totale** | **5602** | **0** | **1454** | **7056** |

**Quali misure fornisce ciascuna sorgente** (✔ = colonna popolata, – = vuota)

| Colonna | Mendeley | DiaBD | Pima |
|---|:--:|:--:|:--:|
| età | ✔ | ✔ | ✔ |
| sesso | ✔ | ✔ | – |
| peso, altezza | – | ✔ | – |
| BMI | ✔ | ✔ | ✔ |
| pressione sistolica | – | ✔ | – |
| pressione diastolica | – | ✔ | ✔ |
| frequenza cardiaca | – | ✔ | – |
| glicemia | – | ✔ | ✔ |
| HbA1c | ✔ | – | – |
| insulina | – | – | ✔ |
| colesterolo, HDL, LDL, VLDL, trigliceridi, urea | ✔ | – | – |
| `heart_disease` | – | ✔ | – |
| creatinina, familiarità, ipertensione (anamnesi) | – | – | – |

Il risultato è un dataset **molto sparso per costruzione**: ad esempio HbA1c e profilo lipidico sono presenti
solo nel 14,2 % delle righe, l'insulina nel 10,9 %. Per questo SVM e BN gestiscono esplicitamente i valori
mancanti (imputazione per la SVM, stato `unknown` per la BN).

---

## 7. Guida all'esecuzione (pipeline e test)

Tutti i comandi si lanciano **dalla cartella principale** del progetto (`icon_diabete_V6/`).

### Passo 1 — Crea il dataset unificato

```bash
python dataset_builder/build_unified_dataset.py
```

Cosa succede:

1. cerca tutti i `*.csv` in `data/raw/` (errore esplicito se non ce ne sono);
2. per ciascun file sceglie l'adattatore giusto e converte nello schema canonico;
3. concatena i risultati e forza le colonne numeriche a tipo numerico;
4. **ricalcola il BMI solo dove manca e peso/altezza sono disponibili**;
5. salva `data/processed/diabetes_clinical_unified.csv` e stampa righe totali e distribuzione delle classi.

Output a console (esempio reale):

```text
Importato Dataset of Diabetes .csv | righe: 1000 | sorgente: mendeley
Importato DiaBD_A Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv | righe: 5288 | sorgente: diabd
Importato diabetes (1).csv | righe: 768 | sorgente: pima

CSV unificato creato: .../data/processed/diabetes_clinical_unified.csv
Righe: 7056
diabetes_class
normal            5549
type2_diabetes    1454
prediabetes         53
```

### Passo 2 — Addestra la SVM

```bash
python ml_engine/svm_model.py
```

Stampa le feature selezionate, il bilanciamento delle classi, il **report di classificazione** sul test set e
salva `data/processed/svm_diabetes_model.pkl`.

### Passo 3 — Addestra la Rete Bayesiana

```bash
python ml_engine/bn_model.py
```

Stampa distribuzione delle classi e stati discreti di ogni nodo, stima le CPD e salva
`data/processed/bayesian_diabetes_model.bif`.

### Passo 4 — Avvia la GUI

```bash
python main_gui.py
```

Compila i campi e premi **✓ Valuta**. Se hai saltato i passi 2 e/o 3, la GUI funziona comunque (solo regole +
ontologia, o regole + il modello disponibile).

### Passo 5 — Verifica automatica della pipeline (Test)

```bash
# Test rapido di pipeline:
python test_pipeline.py

# Test globale e approfondito di collaudo (42 test su tutti i componenti):
python test_full_project.py
```

`test_full_project.py` esegue una suite completa di 42 verifiche suddivise in 6 sezioni:
1. **Dati e Modelli**: integrità del CSV unificato, presenza dei modelli addestrati `.bif` e `.pkl`;
2. **Feature Engineering**: rispetto delle soglie cliniche e gestione di `None`/valori assenti;
3. **Conoscenza Simbolica (Datalog)**: corretta attivazione dei fatti e diagnosi differenziale delle regole;
4. **Modelli Statistici (SVM + BN)**: inferenza probabilistica, somma delle masse a 1 e stabilità su valori estremi;
5. **Diagnosi Differenziale Completa**: test su 12 scenari clinici (Diabete Tipo 1, Diabete Tipo 2, Prediabete, Non Diabete con ipertensione/lipidi, Stato in salute, stress test);
6. **Ontologia e Report PDF**: coerenza semantica ed effettiva generazione del referto PDF.

> **Ordine obbligatorio:** il passo 1 deve precedere i passi 2 e 3. Se il CSV unificato non esiste, SVM e BN
> sollevano `FileNotFoundError` con il messaggio *"Prima esegui: python dataset_builder/build_unified_dataset.py"*.

---

## 8. Documentazione dei moduli: funzioni, classi e come funzionano

### 8.1 `dataset_builder/`

Il package trasforma CSV eterogenei in un unico CSV con lo **stesso schema**.

#### `canonical_schema.py`

Tre costanti di modulo, nessuna funzione:

| Costante | Contenuto |
|---|---|
| `COLUMNS` | Lista ordinata delle 24 colonne dello schema canonico (vedi [§6.1](#61-schema-canonico-24-colonne)). Definisce anche l'**ordine** delle colonne nel CSV finale. |
| `NUMERIC` | `COLUMNS` privata di `source_dataset`, `source_row`, `sex`, `diabetes_class` → le **20 colonne** forzate a tipo numerico dal builder. |
| `CLASSES` | `["normal", "prediabetes", "type2_diabetes"]`: vocabolario ufficiale delle classi (riferimento; non è importata dagli altri moduli). |

#### `adapters.py`

Contiene le funzioni che **leggono i nomi di colonna in modo tollerante** e convertono ogni dataset nello schema canonico.

##### Funzioni di utilità

| Funzione | Cosa fa e come |
|---|---|
| `normalize(name)` | Porta un nome di colonna in forma normalizzata: minuscolo e **solo lettere/cifre** (regex `[^a-z0-9]` → rimossi). `"Systolic_BP"` → `"systolicbp"`, `"HbA1c"` → `"hba1c"`. Rende il matching insensibile a maiuscole, spazi, underscore e punteggiatura. |
| `find_column(df, aliases)` | Costruisce un dizionario `{nome_normalizzato: nome_originale}` delle colonne del DataFrame e restituisce la **prima colonna** che corrisponde a uno degli alias (nell'ordine dato), oppure `None`. |
| `numeric(df, aliases)` | Trova la colonna con `find_column` e la converte con `pd.to_numeric(errors="coerce")` (valori non numerici → `NaN`). Se la colonna non esiste restituisce una `Series` di `pd.NA` con lo stesso indice. |
| `text(df, aliases)` | Come `numeric`, ma per testo: `astype(str)`, `strip()`, `lower()`. Usata per il sesso. |
| `standard_target(series)` | Converte l'etichetta originale nelle classi canoniche (vedi sotto). |
| `empty_frame(df, source)` | Crea un DataFrame con lo stesso indice, imposta `source_dataset = source`, `source_row = df.index` (riga di origine) e inizializza **tutte** le colonne canoniche a `pd.NA`. |

##### `standard_target(series)`

Parte da `"normal"` per **tutte** le righe e poi sovrascrive:

1. se il testo (minuscolo, senza spazi ai bordi) **contiene `prediabet`** → `prediabetes`;
2. se è uno tra `1`, `1.0`, `yes`, `y`, `true`, `positive`, **oppure** contiene `diabet` ma **non** `pre` → `type2_diabetes`.

Quindi: Pima `0/1` → `normal/type2_diabetes`; DiaBD `No/Yes` → `normal/type2_diabetes`; Mendeley `N`/`Y` → `normal`/`type2_diabetes`.
Vedi [§13, punto 1](#13-limiti-noti-e-osservazioni-sul-codice) per la gestione della classe Mendeley `P`.

##### Adattatori per dataset

Ognuno chiama `empty_frame`, poi riempie le colonne canoniche usando un dizionario
`{colonna_canonica: [alias possibili]}` e le funzioni `numeric` / `text`, infine imposta `diabetes_class`
tramite `standard_target` se trova la colonna di classe.

| Funzione | Sorgente | Note |
|---|---|---|
| `adapt_mendeley(df)` | Mendeley | alias multipli per tollerare varianti (es. glicemia: `sugarlevelblood`, `bloodsugar`, `glucose`; colesterolo: `cholesterol`, `chol`; trigliceridi: `triglycerides`, `tg`). Colonna di classe cercata tra `class`, `diagnosis`, `diabetic`, `outcome`. |
| `adapt_diabd(df)` | DiaBD | importa pressioni, frequenza cardiaca, peso/altezza, glicemia, `cardiovascular_disease` → `heart_disease`. Classe: `diabetic` / `diabetes` / `outcome`. |
| `adapt_pima(df)` | Pima | importa glucosio, pressione diastolica (`BloodPressure`), insulina, BMI, età. Classe: `outcome` / `diabetes`. |

##### `adapt(path, df)`

Funzione "dispatcher": decide **quale adattatore usare**. Normalizza il nome del file in minuscolo e concatena i nomi
normalizzati delle colonne, poi applica nell'ordine:

1. **DiaBD** se il nome del file contiene `diabd` **oppure** le colonne contengono sia `systolic` sia `pulse`;
2. **Pima** se il nome contiene `pima` **oppure** le colonne contengono sia `pregnancies` sia `insulin`;
3. altrimenti **Mendeley** (default).

Per questo `diabetes (1).csv` viene riconosciuto come Pima anche senza "pima" nel nome.

#### `build_unified_dataset.py`

Script eseguibile con la sola funzione `main()`:

1. imposta `ROOT` (cartella principale) e lo aggiunge a `sys.path`, così funziona lanciato direttamente da terminale;
2. `glob.glob("data/raw/*.csv")` → se vuoto solleva `FileNotFoundError("Nessun CSV in data/raw/. Leggi README.md.")`;
3. per ogni file: `pd.read_csv` → `adapt(path, df)` → stampa nome, numero di righe e sorgente;
4. `pd.concat(..., ignore_index=True)[COLUMNS]` per unire e ordinare le colonne;
5. forza le colonne `NUMERIC` a numerico (`pd.to_numeric(errors="coerce")`);
6. **calcola il BMI** come `peso / (altezza/100)²` **solo** dove `bmi` è mancante, `weight_kg` è presente e `height_cm > 0`
   (i valori esistenti non vengono mai sovrascritti);
7. crea `data/processed/` se serve, salva il CSV e stampa righe totali e `value_counts` del target.

---

### 8.2 `ml_engine/feature_engineering.py`

#### `discretize(df)`

Converte i valori continui in **fasce cliniche categoriali**. È il modulo che **garantisce soglie identiche** tra training della
Rete Bayesiana e valutazione di un singolo paziente. Restituisce un DataFrame con 6 colonne di stringhe.

Dettagli tecnici: una piccola lambda `num(c)` legge la colonna `c` e la converte in numerico; `pd.cut(valori, bins, labels)`
assegna la fascia; i `NaN` diventano la stringa **`"unknown"`** (`astype(str).replace("nan", "unknown")`).

| Colonna prodotta | Colonna di input | Fasce (stati) |
|---|---|---|
| `age_group` | `age_years` | `young` ≤ 44 · `middle` 45–64 · `older` ≥ 65 |
| `bmi_group` | `bmi` | `normal` ≤ 24,9 · `overweight` 25–29,9 · `obese` > 29,9 |
| `glucose_group` | `fasting_glucose_mg_dl` | `normal` ≤ 99 · `borderline` 100–125 · `high` ≥ 126 |
| `hba1c_group` | `hba1c_percent` | `normal` ≤ 5,6 · `borderline` 5,7–6,4 · `high` ≥ 6,5 |
| `bp_group` | `systolic_bp_mmhg`, `diastolic_bp_mmhg` | `high` se sistolica ≥ 140 **o** diastolica ≥ 90 · `unknown` se entrambe mancanti · altrimenti `normal` |
| `lipid_risk` | `triglycerides_mg_dl` | `high` se ≥ 150 · `unknown` se mancante · altrimenti `normal` |

> Gli stati `unknown` sono intenzionali: rappresentano **esami non disponibili**, evitando di scartare righe.

---

### 8.3 `kbs_engine/`

#### `rules.py` — regole cliniche esplicite

Implementa il ragionamento simbolico in due fasi: **(1) derivazione dei fatti**, **(2) conclusione diagnostica**.

##### `derive(v)` — dai valori numerici ai fatti simbolici

Prende il dizionario del paziente `v`. Internamente definisce `g = lambda k: float(v.get(k) or 0)` (un valore **assente, vuoto o `None` vale 0**)
e legge il sesso (`v.get("sex", "female")`, in minuscolo). Restituisce **8 fatti booleani**:

| Fatto | Condizione | Soglia clinica di riferimento |
|---|---|---|
| `glicemia_diagnostica` | glicemia ≥ 126 | ≥ 126 mg/dL a digiuno |
| `glicemia_borderline` | 100 ≤ glicemia < 126 | 100–125 mg/dL (alterata glicemia a digiuno) |
| `hba1c_diagnostica` | HbA1c ≥ 6,5 | ≥ 6,5 % |
| `hba1c_borderline` | 5,7 ≤ HbA1c < 6,5 | 5,7–6,4 % |
| `ipertensione` | sistolica ≥ 140 **oppure** diastolica ≥ 90 | ≥ 140/90 mmHg |
| `obesita` | BMI ≥ 30 | BMI ≥ 30 |
| `trigliceridi_alti` | trigliceridi ≥ 150 | ≥ 150 mg/dL |
| `hdl_basso` | HDL > 0 **e** HDL < 40 (maschi) / < 50 (femmine) | < 40 mg/dL uomini, < 50 mg/dL donne |

Note: il sesso è considerato maschile se il testo **inizia per `m`**; per HDL la condizione `> 0` evita che un valore mancante
(0) venga scambiato per "HDL basso".

##### `diagnose_rules(f)` — dai fatti alla classe

Calcola uno **score** da 0 a 1:

```text
score_base = 0,80   se hba1c_diagnostica  OPPURE  glicemia_diagnostica
           = 0,45   altrimenti, se hba1c_borderline OPPURE glicemia_borderline
           = 0      altrimenti

score = min(1, score_base + 0,07 × (numero di fattori veri tra:
                 ipertensione, obesita, trigliceridi_alti, hdl_basso))

label = type2_diabetes  se score ≥ 0,70
      = prediabetes     se score ≥ 0,30
      = normal          altrimenti
```

Restituisce `{"label": ..., "score": ..., "factors": [nomi dei fatti veri]}`.

Conseguenze dirette della formula:

- un solo marcatore glicemico **diagnostico** (glicemia ≥ 126 *o* HbA1c ≥ 6,5) porta già a `type2_diabetes` (0,80 ≥ 0,70);
- un marcatore **borderline** dà `prediabetes` (0,45), e resta tale con 0–3 fattori di rischio (max 0,66); con **tutti e 4** i fattori
  supera 0,70 (0,73) e diventa `type2_diabetes`;
- i soli fattori di rischio (senza alterazioni glicemiche) valgono al massimo 0,28 → restano `normal`.

In notazione logica illustrativa (stile Datalog):

```prolog
glicemia_diagnostica(P) :- glicemia(P, G), G >= 126.
glicemia_borderline(P)  :- glicemia(P, G), G >= 100, G < 126.
obesita(P)              :- bmi(P, B), B >= 30.
ipertensione(P)         :- sistolica(P, S), S >= 140.
ipertensione(P)         :- diastolica(P, D), D >= 90.
...
type2_diabetes(P)       :- score(P, S), S >= 0.70.
```

#### `datalog_engine.py`

```python
class DatalogEngine:
    def diagnosi(self, values):
        return diagnose_rules(derive(values))
```

È una **facade**: espone un'unica interfaccia `diagnosi(values)` al resto del sistema e tiene separato "il motore" dalle "regole".
Il motore di fusione dipende solo da questa interfaccia, quindi le regole potrebbero essere sostituite da un vero motore Datalog
(es. pyDatalog, Souffle) senza modificare `diagnosis_engine.py`. **Nota:** nella versione attuale le regole sono funzioni Python pure
(derivazione dei fatti → conclusione, in stile *forward chaining*), non è usata una libreria Datalog esterna.

#### `ontology_manager.py` — ontologia leggera

Classe `OntologyManager` con un dizionario di classe `INFO`:

| Classe | `severity` | `description` | `recommendations` |
|---|---|---|---|
| `normal` | bassa | Nessun pattern clinico significativo rilevato. | Monitoraggio periodico · Alimentazione equilibrata · Attività fisica regolare |
| `prediabetes` | moderata | Pattern compatibile con rischio glicemico aumentato. | Controllo glicemico · Monitoraggio HbA1c · Valutazione professionale |
| `type2_diabetes` | alta | Pattern compatibile con diabete tipo 2 o rischio elevato. | Consulto sanitario · Monitoraggio glicemia e HbA1c · Valutazione cardiovascolare |

Metodo `get_info(label)`: restituisce la voce corrispondente; per un'etichetta sconosciuta ricade su `normal`.
**L'ontologia non partecipa al voto**: aggiunge conoscenza descrittiva all'esito già deciso.

---

### 8.4 `ml_engine/svm_model.py`

Gestisce il ciclo di vita della **Support Vector Machine**.

#### Costanti di configurazione

| Costante | Valore | Significato |
|---|---|---|
| `DATASET_PATH` | `data/processed/diabetes_clinical_unified.csv` | input del training |
| `MODEL_PATH` | `data/processed/svm_diabetes_model.pkl` | output (usato anche da `diagnosis_engine`) |
| `CANDIDATE_FEATURES` | 14 colonne: `age_years`, `sex`, `bmi`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`, `fasting_glucose_mg_dl`, `hba1c_percent`, `insulin_uiu_ml`, `total_cholesterol_mg_dl`, `hdl_mg_dl`, `ldl_mg_dl`, `triglycerides_mg_dl`, `creatinine_mg_dl`, `urea_mg_dl` | feature *candidate* |
| `TARGET_COLUMN` / `POSITIVE_CLASS` | `diabetes_class` / `type2_diabetes` | classe positiva |
| `RANDOM_STATE` | 42 | riproducibilità |
| `TEST_SIZE` | 0,25 | quota del test set |

Un `ROOT_DIR` + `sys.path.insert` iniziale permette di lanciare il file direttamente (`python ml_engine/svm_model.py`).

#### Classe `SVMModel`

Attributi: `model` (la `Pipeline`), `features` (lista ordinata delle feature usate), `metadata` (dizionario di tracciabilità).

| Metodo | Cosa fa |
|---|---|
| `_check_dataset_exists()` *(static)* | Verifica l'esistenza del CSV unificato; altrimenti `FileNotFoundError` con istruzioni. |
| `_select_available_features(df)` *(static)* | Tiene solo le `CANDIDATE_FEATURES` **presenti e con almeno un valore non nullo**. Rende il training robusto a dataset privi di alcune misure (es. `creatinine_mg_dl` è tutta vuota ed è quindi scartata automaticamente). `ValueError` se non ne resta nessuna. |
| `_create_binary_target(df)` *(static)* | Trasforma il target a 3 classi in **binario**: `type2_diabetes → 1`; `normal` e `prediabetes → 0`. |
| `_balance_binary_classes(df, target)` *(static)* | **Sottocampiona la classe maggioritaria** fino alla numerosità della minoritaria (`group.sample(n=min, random_state=42)`), unisce, rimescola. Stampa quante righe per classe sono disponibili/usate. `ValueError` se manca una delle due classi. |
| `_build_pipeline(numeric, categorical)` *(static)* | Costruisce la pipeline completa (vedi sotto). |
| `train()` | Orchestra l'intero addestramento (vedi sotto). |
| `_save()` | Salva con `joblib.dump` un dizionario `{"model", "features", "metadata"}` in `MODEL_PATH`. |
| `load()` | Rilegge il file `.pkl` e ripristina `model`, `features`, `metadata`. `FileNotFoundError` se assente. |
| `predict(patient_values)` | Restituisce `{"normal": 1-p, "type2_diabetes": p}`. |

#### Pipeline di preprocessing + modello

```text
Pipeline
├── preprocessor: ColumnTransformer (remainder="drop")
│   ├── "numeric"      → [SimpleImputer(strategy="median")  →  StandardScaler()]   su tutte le feature numeriche
│   └── "categorical"  → [SimpleImputer(strategy="most_frequent")  →  OneHotEncoder(handle_unknown="ignore")]   su "sex"
└── svm: CalibratedClassifierCV(
            estimator = SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=False, random_state=42),
            cv = 3, method = "sigmoid")
```

Perché ogni scelta:

- **Imputazione con mediana** → gestisce i valori mancanti (molto frequenti, vedi §6.3), robusta agli outlier.
- **`StandardScaler`** → le SVM con kernel RBF sono basate su distanze: senza normalizzazione le variabili con scala maggiore dominerebbero.
- **`OneHotEncoder(handle_unknown="ignore")`** → il sesso è categoriale; un valore mai visto in predizione non genera errori.
- **Kernel RBF** → permette un confine decisionale non lineare.
- **`CalibratedClassifierCV(method="sigmoid")`** (scaling di Platt, 3 fold) → una SVM standard fornisce solo un margine, non una probabilità; la calibrazione
  produce probabilità utilizzabili nella fusione con regole e BN.
- **Bilanciamento** → con 5602 negativi vs 1454 positivi un modello "pigro" sembrerebbe accurato ma troverebbe pochi casi positivi.

#### `train()` passo per passo

1. controlla il CSV; lo legge; scarta le righe **senza `diabetes_class`**;
2. seleziona le feature disponibili; costruisce il target binario; stampa distribuzione e feature;
3. isola **solo** le colonne feature (così non entrano per errore `source_dataset`, `source_row` o il target);
4. **bilancia** le classi → con il dataset attuale: 1454 + 1454 = **2908 righe**;
5. separa numeriche e categoriche (`sex` è l'unica categorica);
6. `train_test_split(test_size=0.25, stratify=target, random_state=42)` → 2181 train / 727 test;
7. `fit` sul train, poi `classification_report` sul test (`zero_division=0`, etichette `non_diabete` / `diabete_tipo2`);
8. compila i metadati (`algorithm="SVC_RBF_calibrated"`, `created_at`, `dataset_path`, `rows_before_balancing`, `rows_after_balancing`,
   `features`, `numeric_features`, `categorical_features`, `positive_class`, `random_state`) e salva.

**Feature effettivamente usate con il dataset attuale (13):** `age_years`, `sex`, `bmi`, `systolic_bp_mmhg`, `diastolic_bp_mmhg`,
`fasting_glucose_mg_dl`, `hba1c_percent`, `insulin_uiu_ml`, `total_cholesterol_mg_dl`, `hdl_mg_dl`, `ldl_mg_dl`, `triglycerides_mg_dl`, `urea_mg_dl`
(12 numeriche + `sex`).

**Esempio di report ottenuto rieseguendo il training** (scikit-learn 1.9.1, Python 3.13; i valori possono variare con versioni diverse):

```text
               precision    recall  f1-score   support

  non_diabete       0.78      0.95      0.86       364
diabete_tipo2       0.93      0.74      0.82       363

     accuracy                           0.84       727
```

> Le metriche vanno lette con cautela: vedi [§13](#13-limiti-noti-e-osservazioni-sul-codice), punti 2 e 6.

#### `predict(patient_values)`

Costruisce una riga con **tutte** le feature del modello (`patient_values.get(feature, None)`): quelle non fornite diventano `NaN` e vengono
**imputate** dalla pipeline con i valori appresi in training. Chiama `predict_proba` e prende la colonna della classe positiva
(`[0][1]`). Output: `{"normal": 1 − p, "type2_diabetes": p}`; qui "normal" significa *non diabete tipo 2* (comprende anche prediabete).

---

### 8.5 `ml_engine/bn_model.py`

Gestisce la **Rete Bayesiana discreta** tramite `pgmpy`.

#### Struttura della rete

La struttura è **fissata a mano** (non appresa): ognuno dei 6 fattori clinici discretizzati è genitore del nodo target.

```text
    age_group ────────┐
    bmi_group ────────┤
    glucose_group ────┤
    hba1c_group ──────┼──►  diabetes_class   (normal / prediabetes / type2_diabetes)
    bp_group ─────────┤
    lipid_risk ───────┘
```

È una rete a struttura convergente (tutti i genitori → un unico figlio): il nodo target ha una tabella di probabilità condizionata (CPD) con una colonna per ciascuna
combinazione di stati dei sei genitori (fino a 4·4·4·4·3·3 = 2304 combinazioni se tutti gli stati, `unknown` inclusi, sono presenti nei dati).
I nodi genitori sono nodi radice con CPD marginali.

#### Costanti

| Costante | Significato |
|---|---|
| `DATASET_PATH` / `MODEL_PATH` | CSV unificato / `data/processed/bayesian_diabetes_model.bif` |
| `TARGET_COLUMN` | `diabetes_class` |
| `EVIDENCE_NODES` | `["age_group", "bmi_group", "glucose_group", "hba1c_group", "bp_group", "lipid_risk"]` (gli stessi nomi prodotti da `discretize()`) |
| `EXPECTED_CLASSES` | `["normal", "prediabetes", "type2_diabetes"]` |

#### Classe `BayesianModel`

Attributi: `model` (`DiscreteBayesianNetwork`), `inference` (`VariableElimination`), `metadata`.

| Metodo | Cosa fa |
|---|---|
| `_check_dataset_exists()` *(static)* | Verifica il CSV unificato. |
| `_load_training_data()` *(static)* | Legge il CSV e tiene **solo le righe con `diabetes_class`** (le righe non etichettate non possono contribuire all'apprendimento supervisionato delle CPD). |
| `_prepare_discrete_data(df)` *(static)* | Applica `discretize()`, aggiunge il target come stringa, converte **tutto a stringa** (`fillna("unknown").astype(str)`) per evitare incoerenze di tipo con pgmpy, e restituisce le 7 colonne (6 nodi + target). |
| `_build_network_structure()` *(static)* | Crea `DiscreteBayesianNetwork` con gli archi `(nodo, "diabetes_class")` per ogni nodo di evidenza. |
| `_estimate_cpds(discrete_data)` | `BayesianEstimator(model, data).get_parameters(prior_type="BDeu", equivalent_sample_size=5)` → `model.add_cpds(*cpds)` → `model.check_model()` (errore se la rete non è valida). |
| `train()` | Orchestra il training e salva (vedi sotto). |
| `_save()` | Scrive il modello con `BIFWriter(model).write_bif(MODEL_PATH)` (formato **BIF**, testuale e portabile). |
| `load()` | `BIFReader(MODEL_PATH).get_model()` e inizializza `VariableElimination`. |
| `_build_evidence(patient_values)` | Trasforma un singolo paziente in evidenze: crea un DataFrame di una riga, chiama **la stessa `discretize()`** del training, e converte ogni stato a stringa (`NaN` → `"unknown"`). |
| `predict(patient_values)` | Interroga la rete e restituisce le probabilità delle classi. |

**Stima delle CPD con prior BDeu.** `BayesianEstimator` esegue una stima bayesiana (conteggi + pseudo-conteggi) invece della massima verosimiglianza.
Con `prior_type="BDeu"` e `equivalent_sample_size=5` si distribuisce un piccolo numero equivalente di osservazioni "virtuali" in modo uniforme: così una
combinazione rara o mai osservata **non ottiene probabilità esattamente zero**. Le CPD sono calcolate esplicitamente e aggiunte al modello
(più compatibile fra versioni di pgmpy rispetto a `model.fit(..., prior_type=...)`).

**Inferenza.** `VariableElimination` è l'algoritmo di **inferenza esatta**: `inference.query(variables=["diabetes_class"], evidence=evidenze, show_progress=False)`
restituisce `P(diabetes_class | evidenze)`. Il codice trasforma il risultato in `{stato: probabilità}` e, tramite `setdefault`, **garantisce sempre le tre chiavi**
`normal`, `prediabetes`, `type2_diabetes` (0.0 se una classe non è stata osservata in training), così il motore di fusione riceve una struttura coerente.

**Metadati** (non scritti nel file BIF, ma tenuti in `self.metadata` durante `train()`): `algorithm`, `created_at`, `dataset_path`, `rows_used`, `target`,
`evidence_nodes`, `structure_edges`, `prior_type`, `equivalent_sample_size`.

---

### 8.6 `diagnosis_engine.py`

Il **motore centrale** che orchestra i quattro componenti.

- In testa: aggiunge la cartella del file a `sys.path` (per poter essere importato anche avviando direttamente `main_gui.py`) e importa
  `DatalogEngine`, `OntologyManager`, `SVMModel` e `BayesianModel`, oltre ai rispettivi `MODEL_PATH` (rinominati `SVM_MODEL_PATH` e `BN_MODEL_PATH`).

#### Classe `HybridDiagnosis`

| Elemento | Descrizione |
|---|---|
| `RULE_WEIGHT = 0.40`, `SVM_WEIGHT = 0.25`, `BN_WEIGHT = 0.35` | Pesi della fusione (somma = 1,00). |
| `__init__()` | Crea `DatalogEngine` e `OntologyManager`, imposta `svm = bn = None` e chiama `_load_optional_models()`. |
| `_load_optional_models()` | Carica SVM e BN **solo se i file addestrati esistono** e, se il caricamento fallisce, stampa un avviso (`"Avviso: SVM non caricata: ..."`) e prosegue con `None`. Permette alla GUI di funzionare anche **prima** del training. |
| `evaluate(patient_values)` | Esegue la valutazione completa (vedi [§9](#9-la-fusione-dei-risultati)). |

#### Funzione pubblica `diagnose(patient_values)`

```python
def diagnose(patient_values):
    return HybridDiagnosis().evaluate(patient_values)
```

È il punto d'ingresso usato dalla GUI. Poiché istanzia un nuovo `HybridDiagnosis` a ogni chiamata, **modelli e BIF vengono riletti da disco a ogni valutazione**
(soluzione semplice che riflette sempre l'ultimo training).

#### Dizionario restituito da `evaluate()`

| Chiave | Contenuto |
|---|---|
| `label` | classe finale (`normal` / `prediabetes` / `type2_diabetes`) |
| `final_score` | punteggio combinato della classe vincente |
| `all_scores` | dizionario con i punteggi combinati delle tre classi |
| `rule_result` | `{"label", "score", "factors"}` dal motore a regole |
| `svm` | `{"normal", "type2_diabetes"}` oppure `None` se la SVM non è caricata |
| `bn` | `{"normal", "prediabetes", "type2_diabetes"}` oppure `None` |
| `ontology` | `{"severity", "description", "recommendations"}` della classe finale |
| `models_loaded` | `{"svm": bool, "bayesian_network": bool}` |

---

### 8.7 `report_generator.py`

#### `export_pdf(values, result)`

Genera un PDF A4 con **ReportLab** e ne restituisce il percorso.

1. crea la cartella `reports/` se manca;
2. nome file: `reports/report_YYYYMMDD_HHMMSS.pdf` (timestamp con `datetime.now().strftime`);
3. costruisce lo "story" `platypus`:
   - titolo *"Diabete KBS Ibrido — Report didattico"* (stile `Title`);
   - **tabella** `Parametro | Valore` con tutti i valori inseriti (BMI incluso), intestazione con sfondo `#087e8b` e testo bianco, griglia grigia (`Table` + `TableStyle`);
   - *"Esito: …"* (stile `Heading2`);
   - *"Punteggio finale: 0.xx"*;
   - *"Fattori: …"* (elenco dei fatti veri dal motore a regole, oppure "nessuno");
   - *"Raccomandazioni: …"* (dall'ontologia);
   - avvertenza che il progetto è didattico e non sostituisce un professionista sanitario;
4. `SimpleDocTemplate(path, pagesize=A4).build(story)`.

---

### 8.8 `main_gui.py`

Interfaccia **Tkinter** (finestra 1120×720) con tre aree: intestazione blu (`#12304a`), pannello sinistro con i campi di input e pannello destro con il risultato.

#### Campi di input (15)

| Chiave (`dict`) | Etichetta | Valore predefinito previsto (`DEFAULTS`) |
|---|---|---|
| `age_years` | Età (anni) | 45 |
| `sex` | Sesso (female/male) | female |
| `weight_kg` | Peso (kg) | 75 |
| `height_cm` | Altezza (cm) | 170 |
| `systolic_bp_mmhg` | Pressione sistolica (mmHg) | 120 |
| `diastolic_bp_mmhg` | Pressione diastolica (mmHg) | 80 |
| `fasting_glucose_mg_dl` | Glicemia a digiuno (mg/dL) | 95 |
| `hba1c_percent` | HbA1c (%) | *(vuoto)* |
| `insulin_uiu_ml` | Insulina (µIU/mL) | *(vuoto)* |
| `total_cholesterol_mg_dl` | Colesterolo totale (mg/dL) | *(vuoto)* |
| `hdl_mg_dl` | HDL (mg/dL) | *(vuoto)* |
| `ldl_mg_dl` | LDL (mg/dL) | *(vuoto)* |
| `triglycerides_mg_dl` | Trigliceridi (mg/dL) | *(vuoto)* |
| `creatinine_mg_dl` | Creatinina (mg/dL) | *(vuoto)* |
| `urea_mg_dl` | Urea (mg/dL) | *(vuoto)* |

> Vedi [§13, punto 9](#13-limiti-noti-e-osservazioni-sul-codice): nella versione fornita i valori predefiniti non vengono effettivamente inseriti nei campi.

#### Classe `App`

| Metodo | Cosa fa |
|---|---|
| `__init__(root)` | Costruisce la finestra: titolo, `LabelFrame` "Valori clinici numerici" (una riga `Label` + `Entry` per ciascun campo di `LABELS`), `LabelFrame` "Risultato: Regole + SVM + BN + Ontologia" con un widget `Text` (font Consolas), e i pulsanti. Al termine chiama `reset()`. |
| `reset()` | Pulsante **↻ Ripristina valori**: svuota i campi (e dovrebbe reinserire i valori di `DEFAULTS`, vedi [§13.9](#13-limiti-noti-e-osservazioni-sul-codice)), reimposta il messaggio *"Inserisci i valori clinici e premi Valuta."*, disabilita *Esporta PDF* e azzera `values`/`result`. |
| `evaluate()` | Pulsante **✓ Valuta**: legge i campi → costruisce il dizionario `values` → calcola il BMI → chiama `diagnose(values)` → mostra il risultato → abilita *Esporta PDF*. Qualsiasi eccezione viene mostrata in un `messagebox.showerror("Errore", ...)`. |
| `export()` | Pulsante **⇩ Esporta PDF**: chiama `export_pdf(self.values, self.result)` e mostra in un `messagebox.showinfo` il percorso del file creato. |

**Parsing dell'input in `evaluate()`:**

- ogni campo viene `strip()`-ato; i campi **vuoti vengono ignorati** (non entrano nel dizionario);
- la **virgola decimale** è accettata (`"5,8"` → `"5.8"`);
- `sex` resta testo; tutti gli altri campi sono convertiti con `float(...)` (un valore non numerico produce l'errore mostrato nel messagebox);
- se sono presenti `weight_kg` e `height_cm`: `bmi = peso / (altezza/100)²` e viene aggiunto a `values`.

**Testo mostrato nel pannello risultato:**

```text
ESITO: <label>
Score fusione: <final_score con 2 decimali>
BMI: <bmi con 1 decimale>

Datalog: <rule_result>

SVM: <probabilità SVM o None>

Rete Bayesiana: <probabilità BN o None>

Ontologia: <severity, description, recommendations>
```

Il pulsante *Esporta PDF* è disabilitato finché non è stata eseguita almeno una valutazione riuscita.

### 8.9 `test_pipeline.py`

Script dedicato ai test automatici dell'intera pipeline ibrida, avviabile sia come script standalone che in test CI:

- **Scopo:** Valutare la robustezza della discretizzazione (`ml_engine/feature_engineering.py`), della Rete Bayesiana (`ml_engine/bn_model.py`) e del motore di fusione (`diagnosis_engine.py`) su profili clinici sintetici.
- **Casi di test implementati (differenziazione clinica a 5 classi):**
  1. `Stato in Salute Ottimale`: 28 anni, BMI 21.5, glicemia 85, HbA1c 4.9, PA 115/75, TG 90. Verifica l'attribuzione della classe `healthy`.
  2. `Non Diabete (Ipertensione e Sovrappeso)`: 54 anni, BMI 28.5, glicemia 92, HbA1c 5.1, PA 145/92, TG 180. Verifica che la presenza di fattori di rischio cardiovascolare non induca una falsa diagnosi di diabete, attribuendo correttamente `normal`.
  3. `Prediabete`: 48 anni, BMI 27.2, glicemia 112, HbA1c 5.9. Valuta la classe intermedia `prediabetes`.
  4. `Diabete di Tipo 1`: 19 anni, normopeso (BMI 19.2), glicemia 230, HbA1c 9.5, insulina 2.5 µIU/mL. Valuta il riconoscimento del fenotipo autoimmune/insulino-privo (`type1_diabetes`).
  5. `Diabete di Tipo 2`: 62 anni, obeso (BMI 34), glicemia 175, HbA1c 8.4, insulina 24.0 µIU/mL, PA 150/95, TG 260. Valuta il riconoscimento del fenotipo metabolico/insulino-resistente (`type2_diabetes`).
  6. `Valori Mancanti (Robustezza)`: paziente con parametri `None`. Verifica che i dati assenti vengano convertiti nello stato `'unknown'` e che l'inferenza si completi senza crash.
- **Esecuzione:**
  ```bash
  python test_pipeline.py
  ```

---

## 9. La fusione dei risultati e diagnosi differenziale

Implementata in `HybridDiagnosis.evaluate()`. Il sistema integra la conoscenza simbolica del dominio con i modelli statistico-probabilistici supervisionati operando su cinque classi distinte:
`healthy`, `normal`, `prediabetes`, `type1_diabetes`, `type2_diabetes`.

1. **Regole Datalog (peso 0,40).** Il motore a regole valuta le soglie cliniche oggettive e conduce la **diagnosi differenziale**:
   - differenzia Diabete Tipo 1 vs Tipo 2 sulla base di marcatori specifici (insulina < 5 µIU/mL, età < 35, normopeso vs insulino-resistenza, età matura, obesità e sindrome metabolica);
   - distingue lo stato in salute ottimale dal non diabete con fattori di rischio cardiovascolare/metabolico (ipertensione, dislipidemia, sovrappeso).
   Lo score di confidenza (da 0,80 a 0,95) viene assegnato alla classe specifica.
2. **SVM (peso 0,25).** Fornisce la stima supervisionata binaria (non-diabete / diabete). La probabilità di non-diabete supporta le classi `healthy` e `normal`, mentre la probabilità di diabete converge verso la variante differenziata (`type1_diabetes` o `type2_diabetes`).
3. **Rete Bayesiana (peso 0,35).** Fornisce la distribuzione probabilistica sulle tre classi di base (`normal`, `prediabetes`, `type2_diabetes`), allocando i pesi coerentemente con il fenotipo del paziente.
4. **Salvaguardia di Consistenza Clinica:**
   Un paziente con glicemia e HbA1c perfettamente normali (< 100 mg/dL e < 5.7%) non può in alcun caso ricevere una diagnosi di diabete conclamato; analogamente, in presenza di iperglicemia diagnostica (≥ 126 mg/dL o HbA1c ≥ 6.5%), l'esito è rigorosamente vincolato a `type1_diabetes` o `type2_diabetes`.
5. **Ontologia**: arricchisce l'esito finale con nome clinico esteso, categoria, descrizione, livello di gravità e raccomandazioni pratiche.

### Esempio numerico (valori ipotetici, a scopo illustrativo)

| Componente | normal | prediabetes | type2_diabetes |
|---|---:|---:|---:|
| Regole: label `type2_diabetes`, score 0,80 → 0,40 × 0,80 | – | – | 0,320 |
| SVM: P = (0,30 ; – ; 0,70) → 0,25 × P | 0,075 | – | 0,175 |
| BN: P = (0,20 ; 0,30 ; 0,50) → 0,35 × P | 0,070 | 0,105 | 0,175 |
| **Totale** | **0,145** | **0,105** | **0,670** |

→ `label = "type2_diabetes"`, `final_score = 0,67`.

### Proprietà utili da conoscere

- **Modalità degradata**: senza SVM e BN decide solo il motore a regole e `final_score ≤ 0,40` (0,40 × score massimo 1).
  Con tutti e tre i componenti caricati il totale delle masse è 0,40·score + 0,25 + 0,35.
- `final_score` **non è una probabilità**: è una somma pesata e non è normalizzata.
- In caso di **pareggio**, `max` restituisce la prima chiave nell'ordine del dizionario: `normal` → `prediabetes` → `type2_diabetes`.
- `models_loaded` indica sempre quali componenti hanno effettivamente contribuito.

---

## 10. Esempi d'uso

### 10.1 Risultati del motore a regole (output reali)

Pazienti di esempio valutati con `DatalogEngine().diagnosi(...)`; il BMI è calcolato come fa la GUI.

| Caso | Valori inseriti | BMI | Esito regole | Score | Fattori attivati |
|---|---|---:|---|---:|---|
| **A** – nessuna alterazione | 45 anni, F, 75 kg, 170 cm, PA 120/80, glicemia 95 | 26,0 | `normal` | 0,00 | nessuno |
| **B** – borderline | 52 anni, M, 88 kg, 172 cm, PA 135/85, glicemia 110, HbA1c 6,0 | 29,7 | `prediabetes` | 0,45 | `glicemia_borderline`, `hba1c_borderline` |
| **C** – marcatori diagnostici | 60 anni, M, 98 kg, 170 cm, PA 150/92, glicemia 148, HbA1c 7,4, trigliceridi 190, HDL 35 | 33,9 | `type2_diabetes` | 1,00 | `glicemia_diagnostica`, `hba1c_diagnostica`, `ipertensione`, `obesita`, `trigliceridi_alti`, `hdl_basso` |
| **D** – solo fattori di rischio | 50 anni, M, 100 kg, 170 cm, PA 150/95, glicemia 90, trigliceridi 200, HDL 35 | 34,6 | `normal` | 0,28 | `ipertensione`, `obesita`, `trigliceridi_alti`, `hdl_basso` |
| **E** – nessun dato | *(dizionario vuoto)* | – | `normal` | 0,00 | nessuno |

Nel caso C lo score grezzo sarebbe 0,80 + 4 × 0,07 = 1,08, ma viene **limitato a 1** da `min(1, …)`.

### 10.2 Esempio in modalità "solo regole" (SVM e BN non addestrate)

Per il caso C, `diagnose(...)` restituirebbe (valori calcolati dalla formula di fusione):

```python
{
  "label": "type2_diabetes",
  "final_score": 0.40,                     # 0,40 × 1,00
  "all_scores": {"normal": 0.0, "prediabetes": 0.0, "type2_diabetes": 0.40},
  "rule_result": {"label": "type2_diabetes", "score": 1,
                  "factors": ["glicemia_diagnostica", "hba1c_diagnostica", "ipertensione",
                              "obesita", "trigliceridi_alti", "hdl_basso"]},
  "svm": None,
  "bn": None,
  "ontology": {"severity": "alta",
               "description": "Pattern compatibile con diabete tipo 2 o rischio elevato.",
               "recommendations": ["Consulto sanitario",
                                   "Monitoraggio glicemia e HbA1c",
                                   "Valutazione cardiovascolare"]},
  "models_loaded": {"svm": False, "bayesian_network": False}
}
```

### 10.3 Sessione tipica con la GUI

1. Avvia `python main_gui.py`.
2. Inserisci i valori (virgola o punto per i decimali; i campi che non conosci lasciali vuoti).
3. Premi **✓ Valuta**: a destra compaiono esito, score di fusione, BMI, e il dettaglio di Datalog/SVM/BN/Ontologia.
4. Premi **⇩ Esporta PDF**: il referto viene salvato in `reports/report_<data>_<ora>.pdf` e il percorso è mostrato in una finestra di dialogo.
5. **↻ Ripristina valori** riporta il form allo stato iniziale.

---

## 11. Uso programmatico (API Python)

Dalla cartella principale del progetto:

```python
from diagnosis_engine import diagnose
from report_generator import export_pdf

paziente = {
    "age_years": 52,
    "sex": "male",
    "weight_kg": 88,
    "height_cm": 172,
    "systolic_bp_mmhg": 135,
    "diastolic_bp_mmhg": 85,
    "fasting_glucose_mg_dl": 110,
    "hba1c_percent": 6.0,
    "triglycerides_mg_dl": 140,
}
paziente["bmi"] = paziente["weight_kg"] / (paziente["height_cm"] / 100) ** 2   # il BMI va calcolato da chi chiama

risultato = diagnose(paziente)
print(risultato["label"], risultato["final_score"])
print(risultato["rule_result"]["factors"])
print(risultato["ontology"]["recommendations"])

percorso_pdf = export_pdf(paziente, risultato)
```

Singoli componenti:

```python
from kbs_engine.datalog_engine import DatalogEngine
from kbs_engine.ontology_manager import OntologyManager
from ml_engine.svm_model import SVMModel
from ml_engine.bn_model import BayesianModel

DatalogEngine().diagnosi(paziente)            # {"label", "score", "factors"}
OntologyManager().get_info("prediabetes")      # {"severity", "description", "recommendations"}

svm = SVMModel(); svm.load(); svm.predict(paziente)          # {"normal", "type2_diabetes"}
bn = BayesianModel(); bn.load(); bn.predict(paziente)        # {"normal", "prediabetes", "type2_diabetes"}
```

Riaddestrare un modello da codice: `SVMModel().train()` e `BayesianModel().train()`.

> La chiave `bmi` **non** viene calcolata da `diagnose()`: la calcola la GUI (o chi chiama). Senza `bmi` il fatto `obesita` non può attivarsi.

---

## 12. Riepilogo delle librerie e delle funzioni usate

### Librerie esterne

| Libreria | Classi / funzioni usate | Dove | Scopo |
|---|---|---|---|
| **pandas** | `read_csv`, `to_csv`, `DataFrame`, `Series`, `concat`, `to_numeric(errors="coerce")`, `cut`, `groupby`, `.sample`, `.value_counts`, `.notna`, `.fillna`, `.astype` | builder, adapters, feature_engineering, svm, bn | Lettura/scrittura dati, pulizia, discretizzazione |
| **scikit-learn** | `SVC`, `CalibratedClassifierCV`, `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `classification_report` | `svm_model.py` | Preprocessing, classificatore, calibrazione, valutazione |
| **joblib** | `dump`, `load` | `svm_model.py` | Persistenza del modello SVM |
| **pgmpy** | `DiscreteBayesianNetwork`, `BayesianEstimator` (`get_parameters`, prior `BDeu`), `VariableElimination` (`query`), `BIFWriter` (`write_bif`), `BIFReader` (`get_model`), `check_model`, `add_cpds` | `bn_model.py` | Rete Bayesiana: struttura, parametri, inferenza, file BIF |
| **reportlab** | `SimpleDocTemplate`, `Paragraph`, `Spacer`, `Table`, `TableStyle`, `getSampleStyleSheet`, `A4`, `colors` | `report_generator.py` | Referto PDF |
| **tkinter** | `Tk`, `Frame`, `LabelFrame`, `Label`, `Entry`, `Text`, `Button`, `messagebox` | `main_gui.py` | Interfaccia grafica |

### Libreria standard Python

`os` (percorsi, `makedirs`), `sys` (`sys.path`), `glob` (ricerca dei CSV), `re` (normalizzazione dei nomi di colonna),
`datetime` (timestamp di PDF e metadati), `typing.Any`.

### Funzioni e metodi del progetto — indice rapido

| Modulo | Funzione / metodo | Scopo in una riga |
|---|---|---|
| `adapters.py` | `normalize`, `find_column`, `numeric`, `text` | lettura tollerante delle colonne |
| | `standard_target` | etichette originali → classi canoniche |
| | `empty_frame` | scheletro dello schema canonico |
| | `adapt_mendeley`, `adapt_diabd`, `adapt_pima`, `adapt` | conversione per sorgente e scelta dell'adattatore |
| `build_unified_dataset.py` | `main` | pipeline di costruzione del CSV unificato |
| `feature_engineering.py` | `discretize` | fasce cliniche condivise |
| `rules.py` | `derive`, `diagnose_rules` | fatti simbolici e classe/score |
| `datalog_engine.py` | `DatalogEngine.diagnosi` | facade del motore a regole |
| `ontology_manager.py` | `OntologyManager.get_info` | descrizione/gravità/raccomandazioni |
| `svm_model.py` | `SVMModel.train / load / predict` + helper privati | ciclo di vita della SVM |
| `bn_model.py` | `BayesianModel.train / load / predict` + helper privati | ciclo di vita della BN |
| `diagnosis_engine.py` | `HybridDiagnosis.evaluate`, `diagnose` | fusione dei componenti |
| `report_generator.py` | `export_pdf` | referto PDF |
| `main_gui.py` | `App.reset / evaluate / export` | interazione utente |
| `test_pipeline.py` | `esegui_test` | suite di test su profili clinici e robustezza |
| `test_full_project.py` | `main` | collaudo globale a 42 test su tutti i componenti |

---


## 13. Possibili sviluppi futuri

- **Armonizzare le unità di misura** negli adattatori (mmol/L → mg/dL: glicemia ×18, colesterolo ×38,67, trigliceridi ×88,57; metri → cm) e filtrare gli outlier.
- **Mappare la classe `P` di Mendeley a `prediabetes`** e rivalutare BN e fusione con tre classi reali.
- **Normalizzare `sex`**, completare gli alias (`Cr`, `family_diabetes`, `hypertensive`) e importare le colonne ancora vuote.
- **Rendere robusta `discretize()`** ai campi mancanti e gestire gli stati `unknown` assenti nel training.
- **Validazione più rigorosa dei modelli**: cross-validation, split per sorgente, metriche su più classi, salvataggio delle metriche accanto ai modelli.
- **Vero motore Datalog** (pyDatalog/Souffle) dietro la facade `DatalogEngine`, e **ontologia formale** (OWL/owlready2) al posto del dizionario `INFO`.
- **Normalizzare il punteggio di fusione** e rendere i pesi configurabili; includere nel PDF anche i contributi di SVM e BN.
- **Validazione dell'input nella GUI** (intervalli plausibili, selezione del sesso da menu, scelta dell'unità di misura).
- **Test automatici** (pytest) per regole, discretizzazione, adattatori e fusione.

---

Progetto didattico: non è un dispositivo medico.*
