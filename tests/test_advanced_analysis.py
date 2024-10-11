import pytest
from src.risk_analysis.advanced_analysis import (
    conduct_advanced_risk_analysis, generate_risk_narratives,
    generate_executive_insights, perform_cross_scenario_analysis,
    identify_key_uncertainties, generate_mitigation_strategies
)
from src.models import Risk, Scenario

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Physical Risk 1", category="Physical", likelihood=0.7, impact=0.8, subcategory="Acute", tertiary_category="", time_horizon="Short-term", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Transition Risk 1", category="Transition", likelihood=0.6, impact=0.7, subcategory="Policy", tertiary_category="", time_horizon="Medium-term", industry_specific=True, sasb_category="Energy"),
        Risk(id=3, description="Market Risk 1", category="Market", likelihood=0.5, impact=0.6, subcategory="Demand", tertiary_category="", time_horizon="Long-term", industry_specific=False, sasb_category=""),
    ]

@pytest.fixture
def sample_scenarios():
    return {
        "Net Zero 2050": Scenario(name="Net Zero 2050", temp_increase=1.5, carbon_price=250, renewable_energy=0.75, policy_stringency=0.9, biodiversity_loss=0.1, ecosystem_degradation=0.2, financial_stability=0.8, supply_chain_disruption=0.3),
        "Delayed Transition": Scenario(name="Delayed Transition", temp_increase=2.5, carbon_price=125, renewable_energy=0.55, policy_stringency=0.6, biodiversity_loss=0.3, ecosystem_degradation=0.4, financial_stability=0.6, supply_chain_disruption=0.5),
    }

@pytest.fixture
def sample_external_data():
    return {
        "2020": ExternalData(year=2020, gdp_growth=2.3, population=7794798739, energy_demand=173340, carbon_price=35, renewable_energy_share=0.29, biodiversity_index=0.7, deforestation_rate=0.5),
        "2021": ExternalData(year=2021, gdp_growth=5.7, population=7874965732, energy_demand=176431, carbon_price=40, renewable_energy_share=0.31, biodiversity_index=0.68, deforestation_rate=0.48),
    }

def test_conduct_advanced_risk_analysis(sample_risks, sample_scenarios, sample_external_data):
    company_industry = "Energy"
    key_dependencies = ["Oil suppliers", "Renewable energy technology"]
    analysis_results = conduct_advanced_risk_analysis(sample_risks, sample_scenarios, company_industry, key_dependencies, sample_external_data)
    
    assert "comprehensive_analysis" in analysis_results
    assert "risk_narratives" in analysis_results
    assert "executive_insights" in analysis_results
    assert "systemic_risks" in analysis_results
    assert "cross_scenario_results" in analysis_results
    assert "key_uncertainties" in analysis_results
    assert "mitigation_strategies" in analysis_results
    assert "pestel_analysis" in analysis_results
    assert "sasb_material_risks" in analysis_results
    assert "trigger_points" in analysis_results
    assert "resilience_assessment" in analysis_results

def test_generate_risk_narratives(sample_risks):
    comprehensive_analysis = {
        "Scenario1": {risk.id: f"Analysis for Risk {risk.id} in Scenario1" for risk in sample_risks},
        "Scenario2": {risk.id: f"Analysis for Risk {risk.id} in Scenario2" for risk in sample_risks},
    }
    narratives = generate_risk_narratives(sample_risks, comprehensive_analysis)
    
    assert len(narratives) == len(sample_risks)
    for risk_id, narrative in narratives.items():
        assert isinstance(narrative, str)
        assert len(narrative) > 100  # Assuming a minimum length for a meaningful narrative

def test_generate_executive_insights(sample_risks):
    comprehensive_analysis = {
        "Scenario1": {risk.id: f"Analysis for Risk {risk.id} in Scenario1" for risk in sample_risks},
        "Scenario2": {risk.id: f"Analysis for Risk {risk.id} in Scenario2" for risk in sample_risks},
    }
    insights = generate_executive_insights(comprehensive_analysis, sample_risks)
    
    assert isinstance(insights, str)
    assert len(insights) > 200  # Assuming a minimum length for meaningful insights
    assert "trends" in insights.lower()
    assert "critical risks" in insights.lower()
    assert "opportunities" in insights.lower()
    assert "recommendations" in insights.lower()

def test_perform_cross_scenario_analysis():
    comprehensive_analysis = {
        "Scenario1": {
            1: "Impact: 0.8, Likelihood: 0.7, Adaptability: 0.6",
            2: "Impact: 0.7, Likelihood: 0.6, Adaptability: 0.5",
        },
        "Scenario2": {
            1: "Impact: 0.9, Likelihood: 0.8, Adaptability: 0.5",
            2: "Impact: 0.6, Likelihood: 0.5, Adaptability: 0.7",
        },
    }
    cross_scenario_results = perform_cross_scenario_analysis(comprehensive_analysis)
    
    assert len(cross_scenario_results) == 2
    for risk_id, scenarios in cross_scenario_results.items():
        assert "Scenario1" in scenarios
        assert "Scenario2" in scenarios
        for scenario_data in scenarios.values():
            assert "impact" in scenario_data
            assert "likelihood" in scenario_data
            assert "adaptability" in scenario_data

def test_identify_key_uncertainties():
    cross_scenario_results = {
        1: {
            "Scenario1": {"impact": 0.8, "likelihood": 0.7},
            "Scenario2": {"impact": 0.9, "likelihood": 0.8},
        },
        2: {
            "Scenario1": {"impact": 0.7, "likelihood": 0.6},
            "Scenario2": {"impact": 0.6, "likelihood": 0.5},
        },
    }
    key_uncertainties = identify_key_uncertainties(cross_scenario_results)
    
    assert isinstance(key_uncertainties, list)
    assert 1 in key_uncertainties  # Risk 1 should be identified as uncertain
    assert 2 not in key_uncertainties  # Risk 2 should not be identified as uncertain

def test_generate_mitigation_strategies(sample_risks):
    comprehensive_analysis = {
        "Scenario1": {risk.id: f"Analysis for Risk {risk.id} in Scenario1" for risk in sample_risks},
        "Scenario2": {risk.id: f"Analysis for Risk {risk.id} in Scenario2" for risk in sample_risks},
    }
    mitigation_strategies = generate_mitigation_strategies(sample_risks, comprehensive_analysis)
    
    assert len(mitigation_strategies) == len(sample_risks)
    for risk_id, strategies in mitigation_strategies.items():
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert all(isinstance(strategy, str) for strategy in strategies)