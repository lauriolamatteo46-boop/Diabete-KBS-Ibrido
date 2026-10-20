"""Facade Datalog: mantiene separata l'interfaccia del motore dalle regole."""
from kbs_engine.rules import derive,diagnose_rules
class DatalogEngine:
 def diagnosi(self,values):return diagnose_rules(derive(values))
