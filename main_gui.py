"""Interfaccia desktop offline: valori numerici, valutazione, reset e PDF."""
import tkinter as tk
from tkinter import messagebox
from diagnosis_engine import diagnose
from report_generator import export_pdf
DEFAULTS={"age_years":"45","sex":"female","weight_kg":"75","height_cm":"170","systolic_bp_mmhg":"120","diastolic_bp_mmhg":"80","fasting_glucose_mg_dl":"95","hba1c_percent":"","insulin_uiu_ml":"","total_cholesterol_mg_dl":"","hdl_mg_dl":"","ldl_mg_dl":"","triglycerides_mg_dl":"","creatinine_mg_dl":"","urea_mg_dl":""}
LABELS={"age_years":"Età (anni)","sex":"Sesso (female/male)","weight_kg":"Peso (kg)","height_cm":"Altezza (cm)","systolic_bp_mmhg":"Pressione sistolica (mmHg)","diastolic_bp_mmhg":"Pressione diastolica (mmHg)","fasting_glucose_mg_dl":"Glicemia a digiuno (mg/dL)","hba1c_percent":"HbA1c (%)","insulin_uiu_ml":"Insulina (µIU/mL)","total_cholesterol_mg_dl":"Colesterolo totale (mg/dL)","hdl_mg_dl":"HDL (mg/dL)","ldl_mg_dl":"LDL (mg/dL)","triglycerides_mg_dl":"Trigliceridi (mg/dL)","creatinine_mg_dl":"Creatinina (mg/dL)","urea_mg_dl":"Urea (mg/dL)"}
class App:
 def __init__(self,root):
  root.title("Diabete KBS Ibrido");
  root.geometry("1120x720");
  self.root=root;tk.Label(root,text="Diabete KBS Ibrido",font=("Segoe UI",22,"bold"),bg="#12304a",fg="white",pady=15).pack(fill="x");
  body=tk.Frame(root);
  body.pack(fill="both",expand=True,padx=18,pady=14);
  left=tk.LabelFrame(body,text="Valori clinici numerici",padx=10,pady=8);
  left.pack(side="left",fill="y");
  right=tk.LabelFrame(body,text="Risultato: Regole + SVM + BN + Ontologia",padx=10,pady=8);
  right.pack(side="right",fill="both",expand=True,padx=12);
  self.entries={};
  self.values=self.result=None
  for key,label in LABELS.items():
   row=tk.Frame(left);
   row.pack(fill="x",pady=2);
   tk.Label(row,text=label,width=37,anchor="w").pack(side="left");
   entry=tk.Entry(row,width=15);entry.pack(side="left");
   self.entries[key]=entry
  self.output=tk.Text(right,wrap="word",font=("Consolas",10));
  self.output.pack(fill="both",expand=True);
  buttons=tk.Frame(root);
  buttons.pack(fill="x",padx=18,pady=(0,16));
  tk.Button(buttons,text="↻ Ripristina valori",command=self.reset).pack(side="left");
  self.pdf=tk.Button(buttons,text="⇩ Esporta PDF",command=self.export,state="disabled");
  self.pdf.pack(side="right");
  tk.Button(buttons,text="✓ Valuta",command=self.evaluate).pack(side="right",padx=8);
  self.reset()
 def reset(self):
  for key,entry in self.entries.items():entry.delete(0,"end");
  entry.insert(0,DEFAULTS[key])
  self.output.delete("1.0","end");
  self.output.insert("end","Inserisci i valori clinici e premi Valuta.");
  self.pdf.config(state="disabled");
  self.values=self.result=None
 def evaluate(self):
  try:
   values={}
   for key,entry in self.entries.items():
    raw=entry.get().strip().replace(",",".")
    if raw:values[key]=raw if key=="sex" else float(raw)
   if "weight_kg" in values and "height_cm" in values:values["bmi"]=values["weight_kg"]/(values["height_cm"]/100)**2
   self.values=values;self.result=diagnose(values);r=self.result;self.output.delete("1.0","end");self.output.insert("end",f"ESITO: {r['label']}\nScore fusione: {r['final_score']:.2f}\nBMI: {values.get('bmi',0):.1f}\n\nDatalog: {r['rule_result']}\n\nSVM: {r['svm']}\n\nRete Bayesiana: {r['bn']}\n\nOntologia: {r['ontology']}");self.pdf.config(state="normal")
  except Exception as error:messagebox.showerror("Errore",str(error))
 def export(self):
  try:messagebox.showinfo("PDF",export_pdf(self.values,self.result))
  except Exception as error:messagebox.showerror("Errore PDF",str(error))
if __name__=="__main__":
 root=tk.Tk();
 App(root);
 root.mainloop()
