"""Costruisce data/processed/diabetes_clinical_unified.csv dai CSV in data/raw/."""
import os,sys,glob,pandas as pd
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."));sys.path.insert(0,ROOT)
from dataset_builder.adapters import adapt
from dataset_builder.canonical_schema import COLUMNS,NUMERIC
RAW=os.path.join(ROOT,"data","raw");OUTPUT=os.path.join(ROOT,"data","processed","diabetes_clinical_unified.csv")
def main():
 paths=glob.glob(os.path.join(RAW,"*.csv"))
 if not paths:raise FileNotFoundError("Nessun CSV in data/raw/. Leggi README.md.")
 frames=[]
 for path in paths:
  converted=adapt(path,pd.read_csv(path));frames.append(converted);print("Importato",os.path.basename(path),"| righe:",len(converted),"| sorgente:",converted.source_dataset.iloc[0])
 final=pd.concat(frames,ignore_index=True)[COLUMNS]
 for c in NUMERIC:final[c]=pd.to_numeric(final[c],errors="coerce")
 # Calcola BMI soltanto quando peso e altezza sono realmente disponibili.
 missing=final.bmi.isna()&final.weight_kg.notna()&final.height_cm.gt(0);final.loc[missing,"bmi"]=final.loc[missing,"weight_kg"]/(final.loc[missing,"height_cm"]/100)**2
 os.makedirs(os.path.dirname(OUTPUT),exist_ok=True);final.to_csv(OUTPUT,index=False);print("\nCSV unificato creato:",OUTPUT);print("Righe:",len(final));print(final.diabetes_class.value_counts(dropna=False))
if __name__=="__main__":main()
