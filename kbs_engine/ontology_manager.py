"""Ontologia leggera: conoscenza descrittiva e raccomandazioni per ogni esito."""
class OntologyManager:
 INFO={"normal":{"severity":"bassa","description":"Nessun pattern clinico significativo rilevato.","recommendations":["Monitoraggio periodico","Alimentazione equilibrata","Attività fisica regolare"]},"prediabetes":{"severity":"moderata","description":"Pattern compatibile con rischio glicemico aumentato.","recommendations":["Controllo glicemico","Monitoraggio HbA1c","Valutazione professionale"]},"type2_diabetes":{"severity":"alta","description":"Pattern compatibile con diabete tipo 2 o rischio elevato.","recommendations":["Consulto sanitario","Monitoraggio glicemia e HbA1c","Valutazione cardiovascolare"]}}
 def get_info(self,label):return self.INFO.get(label,self.INFO["normal"])
