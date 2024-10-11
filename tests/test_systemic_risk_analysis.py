import pytest
import networkx as nx
from src.risk_analysis.systemic_risk_analysis import (
    analyze_systemic_risks, identify_trigger_points, assess_resilience,
    is_systemic_risk, identify_systemic_factor, identify_relevant_external_factors,
    calculate_scenario_resilience
)
from src.models import Risk, ExternalData, SimulationResult

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Global supply chain disruption", category="Physical", likelihood=0.7, impact=0.9, subcategory="Acute", tertiary_category="", time_horizon="Short-term", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Carbon pricing policy changes", category="Transition", likelihood=0.8, impact=0.7, subcategory="Policy", tertiary_category="", time_horizon="Medium-term", industry_specific=True, sasb_category="Energy"),
        Risk(id=3, description="Shift in consumer preferences", category="Market", likelihood=0.6, impact=0.6, subcategory="Demand", tertiary_category="", time_horizon="Long-term", industry_specific=False, sasb_category=""),
    ]

@pytest.fixture
def sample_external_data():
    return {
        "2020": ExternalData(year=2020, gdp_growth=2.3, population=7794798739, energy_demand=173340, carbon_price=35, renewable_energy_share=0.29, biodiversity_index=0.7, deforestation_rate=0.5),
        "2021": ExternalData(year=2021, gdp_growth=5.7, population=7874965732, energy_demand=176431, carbon_price=40, renewable_energy_share=0.31, biodiversity_index=0.68, deforestation_rate=0.48),
    }

@pytest.fixture
def sample_risk_network(sample_risks):
    G = nx.Graph()
    for risk in sample_risks:
        G.add_node(risk.id, **risk.__dict__)
    G.add_edge(1, 2, weight=0.7)
    G.add_edge(1, 3, weight=0.5)
    G.add_edge(2, 3, weight=0.6)
    return G

def test_analyze_systemic_risks(sample_risks):
    company_industry = "Energy"
    key_dependencies = ["Oil suppliers", "Renewable energy technology"]
    systemic_risks = analyze_systemic_risks(sample_risks, company_industry, key_dependencies)
    
    assert len(systemic_risks) > 0
    for risk_id, risk_info in systemic_risks.items():
        assert "description" in risk_info
        assert "impact" in risk_info
        assert "systemic_factor" in risk_info

def test_identify_trigger_points(sample_risks, sample_risk_network, sample_external_data):
    trigger_points = identify_trigger_points(sample_risks, sample_risk_network, sample_external_data)
    
    assert len(trigger_points) > 0
    for risk_id, trigger_info in trigger_points.items():
        assert "description" in trigger_info
        assert "connected_risks" in trigger_info
        assert "external_factors" in trigger_info

def test_assess_resilience(sample_risks):
    scenario_impacts = {
        "Scenario1": [(risk, 0.8) for risk in sample_risks],
        "Scenario2": [(risk, 0.6) for risk in sample_risks],
    }
    simulation_results = {
        "Scenario1": {risk.id: SimulationResult(risk.id, "Scenario1", [0.7, 0.8, 0.9], [0.6, 0.7, 0.8]) for risk in sample_risks},
        "Scenario2": {risk.id: SimulationResult(risk.id, "Scenario2", [0.5, 0.6, 0.7], [0.4, 0.5, 0.6]) for risk in sample_risks},
    }
    resilience_scores = assess_resilience(sample_risks, scenario_impacts, simulation_results)
    
    assert len(resilience_scores) == 2
    assert all(0 <= score <= 1 for score in resilience_scores.values())

def test_is_systemic_risk():
    risk1 = Risk(id=1, description="Global market disruption", category="Market", likelihood=0.7, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    risk2 = Risk(id=2, description="Local operational issue", category="Operational", likelihood=0.5, impact=0.4, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    
    assert is_systemic_risk(risk1, "Energy", ["Global markets"])
    assert not is_systemic_risk(risk2, "Energy", ["Global markets"])

def test_identify_systemic_factor():
    risk1 = Risk(id=1, description="Financial market collapse", category="Market", likelihood=0.7, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    risk2 = Risk(id=2, description="Supply chain disruption", category="Operational", likelihood=0.6, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    
    assert identify_systemic_factor(risk1) == "Financial System"
    assert identify_systemic_factor(risk2) == "Supply Chain"

def test_identify_relevant_external_factors(sample_external_data):
    risk = Risk(id=1, description="Economic downturn impact on energy demand", category="Market", likelihood=0.7, impact=0.8, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    factors = identify_relevant_external_factors(risk, sample_external_data)
    
    assert len(factors) > 0
    assert any("GDP Growth" in factor for factor in factors)

def test_calculate_scenario_resilience():
    impacts = [
        (Risk(id=1, description="Risk 1", category="Physical", likelihood=0.7, impact=0.8, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""), 0.9),
        (Risk(id=2, description="Risk 2", category="Transition", likelihood=0.6, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""), 0.7),
    ]
    simulation_results = {
        1: SimulationResult(1, "Scenario1", [0.8, 0.9, 1.0], [0.7, 0.8, 0.9]),
        2: SimulationResult(2, "Scenario1", [0.6, 0.7, 0.8], [0.5, 0.6, 0.7]),
    }
    resilience = calculate_scenario_resilience(impacts, simulation_results)
    
    assert 0 <= resilience <= 1