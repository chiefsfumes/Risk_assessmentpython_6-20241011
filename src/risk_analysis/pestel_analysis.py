from typing import List, Dict
from src.models import Risk, ExternalData

def perform_pestel_analysis(risks: List[Risk], external_data: Dict[str, ExternalData]) -> Dict[str, List[Dict[str, str]]]:
    pestel_categories = {
        "Political": [],
        "Economic": [],
        "Social": [],
        "Technological": [],
        "Environmental": [],
        "Legal": []
    }
    
    for risk in risks:
        category = categorize_risk_pestel(risk)
        pestel_categories[category].append({
            "risk_id": risk.id,
            "description": risk.description,
            "impact": risk.impact
        })
    
    # Enrich with external data
    enrich_pestel_with_external_data(pestel_categories, external_data)
    
    return pestel_categories

def categorize_risk_pestel(risk: Risk) -> str:
    # Implement logic to categorize risk into PESTEL categories
    # This is a simplified example and should be expanded with more sophisticated logic
    if "regulation" in risk.description.lower() or "policy" in risk.description.lower():
        return "Political"
    elif "economic" in risk.description.lower() or "financial" in risk.description.lower():
        return "Economic"
    elif "social" in risk.description.lower() or "demographic" in risk.description.lower():
        return "Social"
    elif "technology" in risk.description.lower() or "innovation" in risk.description.lower():
        return "Technological"
    elif "environmental" in risk.description.lower() or "climate" in risk.description.lower():
        return "Environmental"
    elif "legal" in risk.description.lower() or "liability" in risk.description.lower():
        return "Legal"
    else:
        return "Environmental"  # Default category for climate risks

def enrich_pestel_with_external_data(pestel_categories: Dict[str, List[Dict[str, str]]], external_data: Dict[str, ExternalData]):
    # Add relevant external data to each PESTEL category
    latest_data = list(external_data.values())[-1]  # Assuming the last entry is the most recent
    
    pestel_categories["Economic"].append({
        "factor": "GDP Growth",
        "value": f"{latest_data.gdp_growth}%"
    })
    
    pestel_categories["Social"].append({
        "factor": "Population",
        "value": str(latest_data.population)
    })
    
    pestel_categories["Environmental"].append({
        "factor": "Energy Demand",
        "value": f"{latest_data.energy_demand} TWh"
    })