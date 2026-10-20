"""Esporta un PDF locale con dati, esito, fattori e raccomandazioni."""
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib import colors
ROOT=os.path.dirname(__file__)
def export_pdf(values,result):
 folder=os.path.join(ROOT,"reports");os.makedirs(folder,exist_ok=True);path=os.path.join(folder,"report_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".pdf");styles=getSampleStyleSheet();story=[Paragraph("Diabete KBS Ibrido — Report didattico",styles["Title"]),Spacer(1,10)]
 rows=[["Parametro","Valore"]]+[[k,str(v)] for k,v in values.items()];table=Table(rows);table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#087e8b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.3,colors.grey)]));story += [table,Spacer(1,10),Paragraph("Esito: "+result["label"],styles["Heading2"]),Paragraph("Punteggio finale: %.2f"%result["final_score"],styles["BodyText"]),Paragraph("Fattori: "+(", ".join(result["rule_result"]["factors"]) or "nessuno"),styles["BodyText"]),Paragraph("Raccomandazioni: "+", ".join(result["ontology"]["recommendations"]),styles["BodyText"]),Paragraph("Avvertenza: progetto didattico; non sostituisce un professionista sanitario.",styles["BodyText"])];SimpleDocTemplate(path,pagesize=A4).build(story);return path
