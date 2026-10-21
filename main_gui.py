"""
main_gui.py
===========

Interfaccia grafica desktop offline (Tkinter):
- Inserimento dei valori clinici numerici del paziente;
- Valutazione ibrida (Regole Datalog + SVM + Rete Bayesiana + Ontologia);
- Differenziazione chiara: Diabete Tipo 1, Diabete Tipo 2, Prediabete, Non diabete, Stato in salute;
- Esportazione del referto clinico in PDF.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox

# Permette gli import quando lo script viene avviato direttamente.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from diagnosis_engine import diagnose
from report_generator import export_pdf

DEFAULTS = {
    "age_years": "45",
    "sex": "female",
    "weight_kg": "75",
    "height_cm": "170",
    "systolic_bp_mmhg": "120",
    "diastolic_bp_mmhg": "80",
    "fasting_glucose_mg_dl": "95",
    "hba1c_percent": "",
    "insulin_uiu_ml": "",
    "total_cholesterol_mg_dl": "",
    "hdl_mg_dl": "",
    "ldl_mg_dl": "",
    "triglycerides_mg_dl": "",
    "creatinine_mg_dl": "",
    "urea_mg_dl": "",
}

LABELS = {
    "age_years": "Età (anni)",
    "sex": "Sesso (female/male)",
    "weight_kg": "Peso (kg)",
    "height_cm": "Altezza (cm)",
    "systolic_bp_mmhg": "Pressione sistolica (mmHg)",
    "diastolic_bp_mmhg": "Pressione diastolica (mmHg)",
    "fasting_glucose_mg_dl": "Glicemia a digiuno (mg/dL)",
    "hba1c_percent": "HbA1c (%)",
    "insulin_uiu_ml": "Insulina a digiuno (µIU/mL)",
    "total_cholesterol_mg_dl": "Colesterolo totale (mg/dL)",
    "hdl_mg_dl": "HDL (mg/dL)",
    "ldl_mg_dl": "LDL (mg/dL)",
    "triglycerides_mg_dl": "Trigliceridi (mg/dL)",
    "creatinine_mg_dl": "Creatinina (mg/dL)",
    "urea_mg_dl": "Urea (mg/dL)",
}


class App:
    def __init__(self, root: tk.Tk) -> None:
        root.title("Diabete KBS Ibrido — Sistema di Supporto Decisionale")
        root.geometry("1140x740")
        self.root = root

        header = tk.Frame(root, bg="#12304a", pady=12)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Diabete KBS Ibrido",
            font=("Segoe UI", 20, "bold"),
            bg="#12304a",
            fg="white",
        ).pack()
        tk.Label(
            header,
            text="Differenziazione: Diabete Tipo 1 • Diabete Tipo 2 • Prediabete • Non Diabete / Salute",
            font=("Segoe UI", 10),
            bg="#12304a",
            fg="#cfe2f3",
        ).pack()

        body = tk.Frame(root)
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left = tk.LabelFrame(body, text="Parametri clinici del paziente", padx=10, pady=8)
        left.pack(side="left", fill="y")

        right = tk.LabelFrame(body, text="Referto di Valutazione Ibrida (Datalog + SVM + BN + Ontologia)", padx=10, pady=8)
        right.pack(side="right", fill="both", expand=True, padx=10)

        self.entries: dict[str, tk.Entry] = {}
        self.values = None
        self.result = None

        for key, label in LABELS.items():
            row = tk.Frame(left)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, width=35, anchor="w").pack(side="left")
            entry = tk.Entry(row, width=14)
            entry.pack(side="left")
            self.entries[key] = entry

        self.output = tk.Text(right, wrap="word", font=("Consolas", 10), bg="#fdfdfd", padx=8, pady=8)
        self.output.pack(fill="both", expand=True)

        buttons = tk.Frame(root)
        buttons.pack(fill="x", padx=16, pady=(0, 14))

        tk.Button(
            buttons,
            text="↻ Ripristina valori",
            command=self.reset,
            font=("Segoe UI", 10),
            padx=10,
            pady=4,
        ).pack(side="left")

        self.pdf = tk.Button(
            buttons,
            text="⇩ Esporta Referto PDF",
            command=self.export,
            state="disabled",
            font=("Segoe UI", 10, "bold"),
            bg="#087e8b",
            fg="white",
            padx=12,
            pady=4,
        )
        self.pdf.pack(side="right")

        tk.Button(
            buttons,
            text="✓ Valuta Paziente",
            command=self.evaluate,
            font=("Segoe UI", 10, "bold"),
            bg="#2e7d32",
            fg="white",
            padx=14,
            pady=4,
        ).pack(side="right", padx=10)

        self.reset()

    def reset(self) -> None:
        for key, entry in self.entries.items():
            entry.delete(0, "end")
            entry.insert(0, DEFAULTS[key])
        self.output.delete("1.0", "end")
        self.output.insert(
            "end",
            "Inserisci i dati clinici del paziente nel pannello di sinistra e premi '✓ Valuta Paziente'.\n\n"
            "Il sistema valuterà automaticamente e differenzierà:\n"
            "  1. Diabete Mellito di Tipo 1 (carenza insulinica marcata, esordio giovanile/normopeso)\n"
            "  2. Diabete Mellito di Tipo 2 (insulino-resistenza, età matura, obesità, sindrome metabolica)\n"
            "  3. Prediabete (alterata glicemia a digiuno o HbA1c borderline)\n"
            "  4. Non diabete (glicemia normale con fattori pressori/metabolici da monitorare)\n"
            "  5. Stato in salute ottimale (tutti i parametri clinici nella norma)\n"
        )
        self.pdf.config(state="disabled")
        self.values = None
        self.result = None

    def evaluate(self) -> None:
        try:
            values = {}
            for key, entry in self.entries.items():
                raw = entry.get().strip().replace(",", ".")
                if raw:
                    values[key] = raw if key == "sex" else float(raw)

            if "weight_kg" in values and "height_cm" in values:
                h_m = values["height_cm"] / 100.0
                if h_m > 0:
                    values["bmi"] = values["weight_kg"] / (h_m ** 2)

            self.values = values
            self.result = diagnose(values)
            r = self.result
            onto = r.get("ontology", {})
            rules = r.get("rule_result", {})

            # Costruzione del testo del referto
            lines = [
                "=" * 68,
                f" ESITO CLINICO:  {r.get('display_name', r['label']).upper()}",
                "=" * 68,
                f"• Categoria:       {onto.get('category', 'Non specificata')}",
                f"• Livello gravità: {onto.get('severity', 'bassa').upper()}",
                f"• Score fusione:   {r['final_score']:.2f}",
                f"• BMI calcolato:   {values.get('bmi', 0):.1f} kg/m²",
                "",
                "-" * 68,
                " INQUADRAMENTO E DIFFERENZIAZIONE CLINICA",
                "-" * 68,
                f"{rules.get('reason', onto.get('description', ''))}",
                "",
                "Fattori clinici attivi rilevati:",
            ]

            factors = rules.get("factors", [])
            if factors:
                for f in factors:
                    lines.append(f"  [+] {f}")
            else:
                lines.append("  [i] Nessun fattore di rischio o anomalia rilevata.")

            lines.extend([
                "",
                "-" * 68,
                " DETTAGLIO MOTORI DI INTEGRAZIONE IBRIDA",
                "-" * 68,
                f"• Regole Simboliche (Datalog):  {rules.get('label')} (confidenza: {rules.get('score', 0):.2f})",
                f"• Support Vector Machine (SVM): {r.get('svm')}",
                f"• Rete Bayesiana (BN):          {r.get('bn')}",
                "",
                "-" * 68,
                " RACCOMANDAZIONI CLINICHE (ONTOLOGIA)",
                "-" * 68,
            ])

            for idx, rec in enumerate(onto.get("recommendations", []), 1):
                lines.append(f"  {idx}. {rec}")

            lines.append("=" * 68)

            self.output.delete("1.0", "end")
            self.output.insert("end", "\n".join(lines))
            self.pdf.config(state="normal")

        except Exception as error:
            messagebox.showerror("Errore", str(error))

    def export(self) -> None:
        try:
            path = export_pdf(self.values, self.result)
            messagebox.showinfo("PDF Esportato", f"Referto PDF generato con successo:\n\n{path}")
        except Exception as error:
            messagebox.showerror("Errore PDF", str(error))


if __name__ == "__main__":
    tk_root = tk.Tk()
    App(tk_root)
    tk_root.mainloop()
