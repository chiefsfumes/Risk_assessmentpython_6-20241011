import pytest
import json
import os
from src.reporting import generate_report
from src.models import Risk, RiskInteraction, SimulationResult, Scenario

@pytest.fixture
def sample_data():
    risks = [
        Risk(id=1, description="Risk 1", category="Physical", likelihood=0.8, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Risk 2", category="Transition", likelihood=0.6, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
    ]
    categorized_risks = {"Physical": [risks[0]], "Transition": [risks[1]]}
    risk_interactions = [RiskInteraction(risk1_id=1, risk2_id=2, interaction_score=0.5, interaction_type="Moderate")]
    scenario_impacts = {"Scenario1": [(risks[0], 0.95), (risks[1], 0.75)]}
    simulation_results = {
        "Scenario1": {
            1: SimulationResult(1, "Scenario1", [0.9, 0.91, 0.89], [0.8, 0.81, 0.79]),
            2: SimulationResult(2, "Scenario1", [0.7, 0.71, 0.69], [0.6, 0.61, 0.59])
        }
    }
    scenarios = {"Scenario1": Scenario(name="Scenario1", temp_increase=2.0, carbon_price=100, renewable_energy=0.5, policy_stringency=0.7, biodiversity_loss=0.2, ecosystem_degradation=0.3, financial_stability=0.8, supply_chain_disruption=0.4)}
    
    return {
        "risks": risks,
        "categorized_risks": categorized_risks,
        "risk_interactions": risk_interactions,
        "scenario_impacts": scenario_impacts,
        "simulation_results": simulation_results,
        "scenarios": scenarios
    }

def test_generate_report(sample_data, tmp_path):
    os.environ['OUTPUT_DIR'] = str(tmp_path)
    report_json = generate_report(
        sample_data["risks"],
        sample_data["categorized_risks"],
        sample_data["risk_interactions"],
        sample_data["scenario_impacts"],
        sample_data["simulation_results"],
        {},  # clustered_risks
        {},  # risk_entities
        {},  # sensitivity_results
        {},  # time_series_results
        sample_data["scenarios"],
        {"comprehensive_analysis": {}, "risk_narratives": {}, "executive_insights": ""}  # advanced_analysis
    )
    
    assert isinstance(report_json, str)
    report = json.loads(report_json)
    
    assert "executive_summary" in report
    assert "risk_overview" in report
    assert "scenario_analysis" in report
    assert "monte_carlo_results" in report
    
    assert os.path.exists(tmp_path / "climate_risk_report.json")
    assert os.path.exists(tmp_path / "climate_risk_report.html")