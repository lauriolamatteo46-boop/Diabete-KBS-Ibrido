"""
ontology_manager.py
===================

Ontologia clinica leggera per il Diabete KBS Ibrido.
Definisce la conoscenza tassonomica, le descrizioni, la severità e le raccomandazioni
per ciascuna classe clinica differenziata:
- healthy: Stato in salute ottimale
- normal: Non diabete (con fattori di rischio metabolico/cardiovascolare)
- prediabetes: Prediabete (alterata regolazione glicemica)
- type1_diabetes: Diabete Mellito di Tipo 1 (deficit assoluto di insulina)
- type2_diabetes: Diabete Mellito di Tipo 2 (insulino-resistenza e disfunzione metabolica)
"""

from typing import Any


class OntologyManager:
    """Gestisce la conoscenza semantica e le raccomandazioni cliniche."""

    INFO: dict[str, dict[str, Any]] = {
        "healthy": {
            "name": "healthy",
            "display_name": "Stato in salute ottimale",
            "category": "Salute e Prevenzione",
            "severity": "ottimale",
            "description": (
                "Nessun segno di diabete o prediabete. Parametri glicemici, pressori, "
                "lipidici e indice di massa corporea (BMI) nei limiti fisiologici di normalità."
            ),
            "recommendations": [
                "Mantenere uno stile di vita sano con alimentazione varia ed equilibrata",
                "Attività fisica aerobica regolare (almeno 150 minuti a settimana)",
                "Controlli periodici di prevenzione (check-up routinario ogni 1-2 anni)",
            ],
        },
        "normal": {
            "name": "normal",
            "display_name": "Non diabete (Fattori di rischio da monitorare)",
            "category": "Rischio Cardiovascolare / Metabolico",
            "severity": "bassa",
            "description": (
                "Glicemia a digiuno e HbA1c nella norma (assenza di diabete), ma presenza di uno o più "
                "fattori di rischio cardiovascolare o metabolico (es. pressione arteriosa elevata, "
                "dislipidemia o sovrappeso) che richiedono attenzione preventiva."
            ),
            "recommendations": [
                "Monitoraggio periodico della pressione arteriosa e del profilo lipidico",
                "Dieta a ridotto contenuto di sale e grassi saturi (stile mediterraneo)",
                "Attività fisica costante per favorire il calo ponderale e l'elasticità vascolare",
                "Ripetere gli esami del sangue annualmente per prevenire complicanze metaboliche",
            ],
        },
        "prediabetes": {
            "name": "prediabetes",
            "display_name": "Prediabete (Rischio glicemico aumentato)",
            "category": "Disglicemia Reversibile",
            "severity": "moderata",
            "description": (
                "Alterata glicemia a digiuno (IFG 100-125 mg/dL) o emoglobina glicata borderline (HbA1c 5.7-6.4%). "
                "Rischio marcatamente aumentato di progressione verso il diabete di tipo 2, "
                "ma quadro reversibile con tempestivi interventi sullo stile di vita."
            ),
            "recommendations": [
                "Intervento intensivo sullo stile di vita: obiettivo riduzione ponderale del 5-7%",
                "Attività fisica moderata-intensa per almeno 30 minuti al giorno",
                "Consulenza nutrizionale con riduzione di zuccheri semplici e carboidrati raffinati",
                "Ripetere curva glicemica (OGTT) o dosaggio HbA1c ogni 6 mesi",
            ],
        },
        "type1_diabetes": {
            "name": "type1_diabetes",
            "display_name": "Diabete Mellito di Tipo 1",
            "category": "Malattia Autoimmune / Deficit Insulinico Assoluto",
            "severity": "molto alta",
            "description": (
                "Quadro clinico compatibile con Diabete di Tipo 1: iperglicemia diagnostica caratterizzata "
                "da deficit assoluto o severo di produzione insulinica. Si manifesta tipicamente in età giovanile "
                "o in individui normopeso/sottopeso, senza la tipica sindrome metabolica dell'adulto."
            ),
            "recommendations": [
                "Consulto diabetologico specialistico URGENTE per inquadramento clinico",
                "Dosaggio degli autoanticorpi specifici (anti-GAD65, IA-2, ZnT8, ICA) e C-peptide",
                "Avvio immediato della terapia insulinica sostitutiva (basal-bolus o microinfusore)",
                "Educazione all'automonitoraggio glicemico continuo (CGM) e prevenzione chetoacidosi",
            ],
        },
        "type2_diabetes": {
            "name": "type2_diabetes",
            "display_name": "Diabete Mellito di Tipo 2",
            "category": "Malattia Metabolica / Insulino-Resistenza",
            "severity": "alta",
            "description": (
                "Quadro clinico compatibile con Diabete di Tipo 2: iperglicemia conclamata legata a "
                "insulino-resistenza periferica e progressiva disfunzione beta-cellulare. È comunemente "
                "associato a sovrappeso/obesità, dislipidemia aterogena, ipertensione ed età matura."
            ),
            "recommendations": [
                "Consulto medico e diabetologico per definire il piano terapeutico personalizzato",
                "Valutazione per terapia ipoglicemizzante (es. metformina, agonisti GLP-1, SGLT2-inibitori)",
                "Piano nutrizionale personalizzato e programma strutturato di attività fisica",
                "Screening complicanze croniche (fondo oculare, microalbuminuria, profilo lipidico e renale)",
            ],
        },
    }

    def get_info(self, label: str) -> dict[str, Any]:
        """Restituisce le informazioni ontologiche per l'etichetta specificata."""
        return self.INFO.get(label, self.INFO["normal"])
