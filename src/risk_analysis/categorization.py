from typing import List, Dict
from src.models import Risk

def categorize_risks(risks: List[Risk]) -> Dict[str, List[Risk]]:
    categories = {}
    for risk in risks:
        if risk.category not in categories:
            categories[risk.category] = []
        categories[risk.category].append(risk)
    return categories

def categorize_risks_multi_level(risks: List[Risk]) -> Dict[str, Dict[str, List[Risk]]]:
    categories = {}
    for risk in risks:
        if risk.category not in categories:
            categories[risk.category] = {}
        if risk.subcategory not in categories[risk.category]:
            categories[risk.category][risk.subcategory] = []
        categories[risk.category][risk.subcategory].append(risk)
    return categories

def assign_risk_priority(risk: Risk) -> str:
    if risk.impact > 0.7 and risk.likelihood > 0.7:
        return "High"
    elif risk.impact > 0.3 and risk.likelihood > 0.3:
        return "Medium"
    else:
        return "Low"

def prioritize_risks(risks: List[Risk]) -> Dict[str, List[Risk]]:
    priorities = {"High": [], "Medium": [], "Low": []}
    for risk in risks:
        priority = assign_risk_priority(risk)
        priorities[priority].append(risk)
    return priorities