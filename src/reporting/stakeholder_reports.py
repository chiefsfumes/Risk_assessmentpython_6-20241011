from typing import Dict, List
import json
import os
from src.config import OUTPUT_DIR

def generate_stakeholder_reports(main_report: Dict, company_industry: str) -> Dict[str, str]:
    stakeholder_reports = {
        "board_executive": generate_board_executive_report(main_report, company_industry),
        "investors": generate_investor_report(main_report, company_industry),
        "regulators": generate_regulatory_report(main_report, company_industry),
        "public": generate_public_report(main_report, company_industry)
    }
    
    # Save reports to files
    for stakeholder, report in stakeholder_reports.items():
        file_path = os.path.join(OUTPUT_DIR, f"{stakeholder}_report.json")
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return stakeholder_reports

def generate_board_executive_report(main_report: Dict, company_industry: str) -> Dict:
    return {
        "summary": main_report["executive_summary"],
        "top_risks": get_top_risks(main_report, 5),
        "strategic_implications": extract_strategic_implications(main_report),
        "key_scenarios": summarize_key_scenarios(main_report),
        "mitigation_priorities": identify_mitigation_priorities(main_report)
    }

def generate_investor_report(main_report: Dict, company_industry: str) -> Dict:
    return {
        "risk_overview": main_report["risk_overview"],
        "scenario_analysis": main_report["scenario_analysis"],
        "financial_implications": extract_financial_implications(main_report),
        "comparative_analysis": perform_comparative_analysis(main_report, company_industry),
        "long_term_outlook": extract_long_term_outlook(main_report)
    }

def generate_regulatory_report(main_report: Dict, company_industry: str) -> Dict:
    return {
        "risk_assessment_methodology": extract_methodology(main_report),
        "compliance_status": assess_compliance_status(main_report, company_industry),
        "scenario_analysis_details": main_report["scenario_analysis"],
        "risk_mitigation_plans": main_report["mitigation_strategies"]
    }

def generate_public_report(main_report: Dict, company_industry: str) -> Dict:
    return {
        "company_climate_strategy": extract_climate_strategy(main_report),
        "key_climate_risks": get_top_risks(main_report, 3),
        "sustainability_initiatives": extract_sustainability_initiatives(main_report),
        "future_commitments": extract_future_commitments(main_report)
    }

def get_top_risks(main_report: Dict, n: int) -> List[Dict]:
    risks = main_report["risk_overview"]["high_impact_risks"]
    return sorted(risks, key=lambda x: x["impact"], reverse=True)[:n]

def extract_strategic_implications(main_report: Dict) -> List[str]:
    return main_report["executive_insights"]["strategic_implications"]

def summarize_key_scenarios(main_report: Dict) -> Dict[str, str]:
    return {scenario: analysis["summary"] for scenario, analysis in main_report["scenario_analysis"].items()}

def identify_mitigation_priorities(main_report: Dict) -> List[str]:
    return main_report["mitigation_strategies"]["priorities"]

def extract_financial_implications(main_report: Dict) -> Dict[str, float]:
    return main_report["financial_analysis"]["implications"]

def perform_comparative_analysis(main_report: Dict, company_industry: str) -> Dict[str, str]:
    industry_data = main_report["industry_comparison"].get(company_industry, {})
    return {
        "risk_profile": industry_data.get("risk_profile", "No data available"),
        "performance": industry_data.get("performance", "No data available"),
        "mitigation_efforts": industry_data.get("mitigation_efforts", "No data available")
    }

def extract_long_term_outlook(main_report: Dict) -> str:
    return main_report["long_term_projections"]["summary"]

def extract_methodology(main_report: Dict) -> Dict[str, str]:
    return main_report["methodology"]

def assess_compliance_status(main_report: Dict, company_industry: str) -> Dict[str, str]:
    compliance_data = main_report["compliance_assessment"]
    return {
        "overall_status": compliance_data["status"],
        "key_regulations": compliance_data["key_regulations"],
        "areas_of_concern": compliance_data["areas_of_concern"]
    }

def extract_climate_strategy(main_report: Dict) -> str:
    return main_report["climate_strategy"]["summary"]

def extract_sustainability_initiatives(main_report: Dict) -> List[str]:
    return main_report["sustainability_initiatives"]

def extract_future_commitments(main_report: Dict) -> List[str]:
    return main_report["future_commitments"]