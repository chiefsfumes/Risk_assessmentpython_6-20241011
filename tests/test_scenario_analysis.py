import pytest
import numpy as np
from src.risk_analysis.scenario_analysis import (
    simulate_scenario_impact, monte_carlo_simulation, analyze_scenario_sensitivity,
    calculate_var_cvar, perform_stress_testing, generate_scenario_narratives
)
from src.models import Risk, ExternalData, Scenario, SimulationResult

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Physical Risk 1", category="Physical", likelihood=0.7, impact=0.8, subcategory="Acute", tertiary_category="", time_horizon="Short-term", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Transition Risk 1", category="Transition", likelihood=0.6, impact=0.7, subcategory="Policy", tertiary_category="", time_horizon="Medium-term", industry_specific=True, sasb_category="Energy"),
        Risk(id=3, description="Market Risk 1", category="Market", likelihood=0.5, impact=0.6, subcategory="Demand", tertiary_category="", time_horizon="Long-term", industry_specific=False, sasb_category=""),
    ]

@pytest.fixture
def sample_external_data():
    return {
        "2020": ExternalData(year=2020, gdp_growth=2.3, population=7794798739, energy_demand=173340, carbon_price=35, renewable_energy_share=0.29, biodiversity_index=0.7, deforestation_rate=0.5),
        "2021": ExternalData(year=2021, gdp_growth=5.7, population=7874965732, energy_demand=176431, carbon_price=40, renewable_energy_share=0.31, biodiversity_index=0.68, deforestation_rate=0.48),
    }

@pytest.fixture
def sample_scenarios():
    return {
        "Net Zero 2050": Scenario(name="Net Zero 2050", temp_increase=1.5, carbon_price=250, renewable_energy=0.75, policy_stringency=0.9, biodiversity_loss=0.1, ecosystem_degradation=0.2, financial_stability=0.8, supply_chain_disruption=0.3),
        "Delayed Transition": Scenario(name="Delayed Transition", temp_increase=2.5, carbon_price=125, renewable_energy=0.55, policy_stringency=0.6, biodiversity_loss=0.3, ecosystem_degradation=0.4, financial_stability=0.6, supply_chain_disruption=0.5),
    }

def test_simulate_scenario_impact(sample_risks, sample_external_data, sample_scenarios):
    scenario = sample_scenarios["Net Zero 2050"]
    impacts = simulate_scenario_impact(sample_risks, sample_external_data, scenario)
    
    assert len(impacts) == len(sample_risks)
    for risk, impact in impacts:
        assert isinstance(risk, Risk)
        assert 0 <= impact <= 1
    
    # Test that higher impact risks have higher simulated impacts
    impacts_dict = dict(impacts)
    assert impacts_dict[1] > impacts_dict[3]  # Physical Risk 1 should have higher impact than Market Risk 1

def test_monte_carlo_simulation(sample_risks, sample_external_data, sample_scenarios):
    results = monte_carlo_simulation(sample_risks, sample_external_data, sample_scenarios)
    
    assert len(results) == len(sample_scenarios)
    for scenario_name, scenario_results in results.items():
        assert len(scenario_results) == len(sample_risks)
        for risk_id, sim_result in scenario_results.items():
            assert isinstance(sim_result, SimulationResult)
            assert len(sim_result.impact_distribution) == 1000  # Default NUM_SIMULATIONS
            assert len(sim_result.likelihood_distribution) == 1000
            assert all(0 <= impact <= 1 for impact in sim_result.impact_distribution)
            assert all(0 <= likelihood <= 1 for likelihood in sim_result.likelihood_distribution)
    
    # Test that Net Zero 2050 scenario has lower average impacts than Delayed Transition
    net_zero_impacts = np.mean([np.mean(sim.impact_distribution) for sim in results["Net Zero 2050"].values()])
    delayed_transition_impacts = np.mean([np.mean(sim.impact_distribution) for sim in results["Delayed Transition"].values()])
    assert net_zero_impacts < delayed_transition_impacts

def test_analyze_scenario_sensitivity(sample_risks, sample_scenarios):
    scenario = sample_scenarios["Net Zero 2050"]
    sensitivity_result = analyze_scenario_sensitivity(sample_risks, scenario, "carbon_price", 0.2)
    
    assert "variable" in sensitivity_result
    assert "base_impact" in sensitivity_result
    assert "high_impact" in sensitivity_result
    assert "low_impact" in sensitivity_result
    assert "sensitivity" in sensitivity_result
    assert sensitivity_result["variable"] == "carbon_price"
    assert isinstance(sensitivity_result["sensitivity"], float)
    
    # Test that increasing carbon price increases impact
    assert sensitivity_result["high_impact"] > sensitivity_result["base_impact"] > sensitivity_result["low_impact"]

def test_calculate_var_cvar():
    simulation_results = {
        "Scenario1": {
            1: SimulationResult(1, "Scenario1", np.random.normal(0.5, 0.1, 1000).tolist(), np.random.normal(0.6, 0.1, 1000).tolist()),
            2: SimulationResult(2, "Scenario1", np.random.normal(0.4, 0.1, 1000).tolist(), np.random.normal(0.5, 0.1, 1000).tolist()),
        }
    }
    
    var_cvar_results = calculate_var_cvar(simulation_results)
    
    assert "Scenario1" in var_cvar_results
    assert 1 in var_cvar_results["Scenario1"]
    assert 2 in var_cvar_results["Scenario1"]
    assert "VaR" in var_cvar_results["Scenario1"][1]
    assert "CVaR" in var_cvar_results["Scenario1"][1]
    assert 0 <= var_cvar_results["Scenario1"][1]["VaR"] <= 1
    assert 0 <= var_cvar_results["Scenario1"][1]["CVaR"] <= 1
    assert var_cvar_results["Scenario1"][1]["CVaR"] >= var_cvar_results["Scenario1"][1]["VaR"]
    
    # Test that higher risk has higher VaR and CVaR
    assert var_cvar_results["Scenario1"][1]["VaR"] > var_cvar_results["Scenario1"][2]["VaR"]
    assert var_cvar_results["Scenario1"][1]["CVaR"] > var_cvar_results["Scenario1"][2]["CVaR"]

def test_perform_stress_testing(sample_risks, sample_external_data, sample_scenarios):
    stress_test_results = perform_stress_testing(sample_risks, sample_scenarios, sample_external_data)
    
    assert len(stress_test_results) == len(sample_scenarios)
    for scenario_name, stressed_impacts in stress_test_results.items():
        assert len(stressed_impacts) == len(sample_risks)
        for risk, impact in stressed_impacts:
            assert isinstance(risk, Risk)
            assert 0 <= impact <= 1
        
        # Test that stressed impacts are higher than original impacts
        original_impacts = dict(simulate_scenario_impact(sample_risks, sample_external_data, sample_scenarios[scenario_name]))
        stressed_impacts_dict = dict(stressed_impacts)
        for risk_id in original_impacts:
            assert stressed_impacts_dict[risk_id] > original_impacts[risk_id]

def test_generate_scenario_narratives(sample_scenarios):
    narratives = generate_scenario_narratives(sample_scenarios)
    
    assert len(narratives) == len(sample_scenarios)
    for scenario_name, narrative in narratives.items():
        assert isinstance(narrative, str)
        assert len(narrative) > 100  # Assuming a minimum length for a meaningful narrative
        
        # Test that scenario-specific keywords are present in the narrative
        assert scenario_name in narrative
        assert "temperature increase" in narrative.lower()
        assert "carbon price" in narrative.lower()
        assert "renewable energy" in narrative.lower()

# Add edge case tests
def test_simulate_scenario_impact_edge_cases(sample_external_data):
    # Test with extreme risks
    extreme_risks = [
        Risk(id=1, description="Extreme Risk", category="Physical", likelihood=1.0, impact=1.0, subcategory="Acute", tertiary_category="", time_horizon="Short-term", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Negligible Risk", category="Transition", likelihood=0.0, impact=0.0, subcategory="Policy", tertiary_category="", time_horizon="Long-term", industry_specific=False, sasb_category=""),
    ]
    
    extreme_scenario = Scenario(name="Extreme Scenario", temp_increase=5.0, carbon_price=1000, renewable_energy=1.0, policy_stringency=1.0, biodiversity_loss=1.0, ecosystem_degradation=1.0, financial_stability=0.0, supply_chain_disruption=1.0)
    
    impacts = simulate_scenario_impact(extreme_risks, sample_external_data, extreme_scenario)
    impacts_dict = dict(impacts)
    
    assert impacts_dict[1] == 1.0  # Extreme risk should have maximum impact
    assert impacts_dict[2] == 0.0  # Negligible risk should have no impact

def test_monte_carlo_simulation_consistency(sample_risks, sample_external_data, sample_scenarios):
    # Test that multiple runs produce consistent results
    results1 = monte_carlo_simulation(sample_risks, sample_external_data, sample_scenarios)
    results2 = monte_carlo_simulation(sample_risks, sample_external_data, sample_scenarios)
    
    for scenario in sample_scenarios:
        for risk in sample_risks:
            np.testing.assert_allclose(
                np.mean(results1[scenario][risk.id].impact_distribution),
                np.mean(results2[scenario][risk.id].impact_distribution),
                rtol=0.1  # Allow for some variation due to randomness
            )