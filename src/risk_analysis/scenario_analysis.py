from typing import List, Dict, Tuple
from src.models import Risk, ExternalData, Scenario, SimulationResult
from src.config import NUM_SIMULATIONS, LLM_MODEL, LLM_API_KEY
from src.prompts import RISK_ASSESSMENT_PROMPT
import openai
import numpy as np
from scipy.stats import norm

openai.api_key = LLM_API_KEY

def simulate_scenario_impact(risks: List[Risk], external_data: Dict[str, ExternalData], scenario: Scenario) -> List[Tuple[Risk, float]]:
    impacts = []
    for risk in risks:
        impact = calculate_risk_impact(risk, external_data, scenario)
        impacts.append((risk, impact))
    return impacts

def monte_carlo_simulation(risks: List[Risk], external_data: Dict[str, ExternalData], scenarios: Dict[str, Scenario]) -> Dict[str, Dict[int, SimulationResult]]:
    results = {}
    for scenario_name, scenario in scenarios.items():
        scenario_results = {}
        for risk in risks:
            impact_distribution = []
            likelihood_distribution = []
            for _ in range(NUM_SIMULATIONS):
                perturbed_scenario = perturb_scenario(scenario)
                perturbed_external_data = perturb_external_data(external_data)
                impact = calculate_risk_impact(risk, perturbed_external_data, perturbed_scenario)
                likelihood = calculate_risk_likelihood(risk, perturbed_external_data, perturbed_scenario)
                impact_distribution.append(impact)
                likelihood_distribution.append(likelihood)
            scenario_results[risk.id] = SimulationResult(risk.id, scenario_name, impact_distribution, likelihood_distribution)
        results[scenario_name] = scenario_results
    return results

def calculate_risk_impact(risk: Risk, external_data: Dict[str, ExternalData], scenario: Scenario) -> float:
    base_impact = risk.impact
    temp_factor = 1 + (scenario.temp_increase - 1.5) * 0.1  # 10% increase per degree above 1.5°C
    carbon_price_factor = 1 + (scenario.carbon_price / 100) * 0.05  # 5% increase per $100 carbon price
    renewable_factor = 1 - scenario.renewable_energy * 0.2  # 20% decrease at 100% renewable energy
    biodiversity_factor = 1 + scenario.biodiversity_loss * 0.15  # 15% increase for complete biodiversity loss
    ecosystem_factor = 1 + scenario.ecosystem_degradation * 0.25  # 25% increase for complete ecosystem degradation
    
    # Consider external data
    latest_year = max(external_data.keys())
    gdp_growth = external_data[latest_year].gdp_growth
    gdp_factor = 1 - gdp_growth * 0.1  # Higher GDP growth slightly reduces impact
    
    impact = (base_impact * temp_factor * carbon_price_factor * renewable_factor * 
              biodiversity_factor * ecosystem_factor * gdp_factor)
    return min(1.0, max(0.0, impact))  # Ensure impact is between 0 and 1

def calculate_risk_likelihood(risk: Risk, external_data: Dict[str, ExternalData], scenario: Scenario) -> float:
    base_likelihood = risk.likelihood
    policy_factor = 1 - scenario.policy_stringency * 0.3  # Stricter policies reduce likelihood
    ecosystem_factor = 1 + scenario.ecosystem_degradation * 0.4  # Ecosystem degradation increases likelihood
    financial_stability_factor = 1 - scenario.financial_stability * 0.2  # Higher financial stability reduces likelihood
    supply_chain_factor = 1 + scenario.supply_chain_disruption * 0.3  # Supply chain disruption increases likelihood
    
    # Consider external data
    latest_year = max(external_data.keys())
    population = external_data[latest_year].population
    population_factor = 1 + (population / 1e10) * 0.1  # Population growth slightly increases likelihood
    
    likelihood = (base_likelihood * policy_factor * ecosystem_factor * 
                  financial_stability_factor * supply_chain_factor * population_factor)
    return min(1.0, max(0.0, likelihood))  # Ensure likelihood is between 0 and 1

def perturb_scenario(scenario: Scenario, perturbation_scale: float = 0.1) -> Scenario:
    perturbed_values = {}
    for attr, value in vars(scenario).items():
        if attr != 'name' and isinstance(value, (int, float)):
            perturbed_values[attr] = max(0, value * (1 + np.random.normal(0, perturbation_scale)))
    return Scenario(name=scenario.name, **perturbed_values)

def perturb_external_data(external_data: Dict[str, ExternalData], perturbation_scale: float = 0.05) -> Dict[str, ExternalData]:
    perturbed_data = {}
    for year, data in external_data.items():
        perturbed_values = {}
        for attr, value in vars(data).items():
            if isinstance(value, (int, float)):
                perturbed_values[attr] = max(0, value * (1 + np.random.normal(0, perturbation_scale)))
        perturbed_data[year] = ExternalData(**perturbed_values)
    return perturbed_data

def analyze_scenario_sensitivity(risks: List[Risk], base_scenario: Scenario, variable: str, range_pct: float) -> Dict[str, float]:
    base_impact = sum(calculate_risk_impact(risk, {}, base_scenario) for risk in risks)
    
    high_scenario = perturb_scenario(base_scenario, {variable: range_pct})
    low_scenario = perturb_scenario(base_scenario, {variable: -range_pct})
    
    high_impact = sum(calculate_risk_impact(risk, {}, high_scenario) for risk in risks)
    low_impact = sum(calculate_risk_impact(risk, {}, low_scenario) for risk in risks)
    
    sensitivity = (high_impact - low_impact) / (2 * range_pct * getattr(base_scenario, variable))
    
    return {
        "variable": variable,
        "base_impact": base_impact,
        "high_impact": high_impact,
        "low_impact": low_impact,
        "sensitivity": sensitivity
    }

def calculate_var_cvar(simulation_results: Dict[str, Dict[int, SimulationResult]], confidence_level: float = 0.95) -> Dict[str, Dict[int, Dict[str, float]]]:
    var_cvar_results = {}
    for scenario, risks in simulation_results.items():
        var_cvar_results[scenario] = {}
        for risk_id, result in risks.items():
            impacts = np.array(result.impact_distribution)
            var = np.percentile(impacts, (1 - confidence_level) * 100)
            cvar = np.mean(impacts[impacts > var])
            var_cvar_results[scenario][risk_id] = {
                "VaR": var,
                "CVaR": cvar
            }
    return var_cvar_results

def perform_stress_testing(risks: List[Risk], scenarios: Dict[str, Scenario], external_data: Dict[str, ExternalData]) -> Dict[str, List[Tuple[Risk, float]]]:
    stress_test_results = {}
    for scenario_name, scenario in scenarios.items():
        # Create a stressed scenario
        stressed_scenario = Scenario(
            name=f"Stressed_{scenario_name}",
            temp_increase=scenario.temp_increase * 1.5,
            carbon_price=scenario.carbon_price * 2,
            renewable_energy=scenario.renewable_energy * 0.5,
            policy_stringency=scenario.policy_stringency * 0.5,
            biodiversity_loss=min(1.0, scenario.biodiversity_loss * 1.5),
            ecosystem_degradation=min(1.0, scenario.ecosystem_degradation * 1.5),
            financial_stability=scenario.financial_stability * 0.5,
            supply_chain_disruption=min(1.0, scenario.supply_chain_disruption * 2)
        )
        
        # Simulate impacts under stressed conditions
        stress_test_results[scenario_name] = simulate_scenario_impact(risks, external_data, stressed_scenario)
    
    return stress_test_results

def generate_scenario_narratives(scenarios: Dict[str, Scenario]) -> Dict[str, str]:
    narratives = {}
    for scenario_name, scenario in scenarios.items():
        prompt = f"""
        Generate a detailed narrative for the following climate scenario:

        Scenario: {scenario_name}
        - Temperature increase: {scenario.temp_increase}°C
        - Carbon price: ${scenario.carbon_price}/ton
        - Renewable energy share: {scenario.renewable_energy * 100}%
        - Policy stringency: {scenario.policy_stringency * 100}%
        - Biodiversity loss: {scenario.biodiversity_loss * 100}%
        - Ecosystem degradation: {scenario.ecosystem_degradation * 100}%
        - Financial stability: {scenario.financial_stability * 100}%
        - Supply chain disruption: {scenario.supply_chain_disruption * 100}%

        Provide a compelling narrative that describes the overall state of the world in this scenario, including key challenges and opportunities for businesses, major societal and environmental changes, potential technological advancements or setbacks, and the general economic landscape.
        """

        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in climate scenario analysis and futurism."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        narratives[scenario_name] = response.choices[0].message['content']
    
    return narratives