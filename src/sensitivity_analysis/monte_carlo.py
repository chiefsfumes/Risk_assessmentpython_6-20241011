import numpy as np
from typing import List, Dict
from src.models import Risk, Scenario, SimulationResult

def perform_monte_carlo_simulations(risks: List[Risk], scenarios: Dict[str, Scenario], num_simulations: int = 10000) -> Dict[str, Dict[int, SimulationResult]]:
    results = {}
    for scenario_name, scenario in scenarios.items():
        scenario_results = {}
        for risk in risks:
            impact_distribution = []
            likelihood_distribution = []
            for _ in range(num_simulations):
                perturbed_scenario = perturb_scenario(scenario)
                impact = calculate_risk_impact(risk, perturbed_scenario)
                likelihood = calculate_risk_likelihood(risk, perturbed_scenario)
                impact_distribution.append(impact)
                likelihood_distribution.append(likelihood)
            scenario_results[risk.id] = SimulationResult(risk.id, scenario_name, impact_distribution, likelihood_distribution)
        results[scenario_name] = scenario_results
    return results

def perturb_scenario(scenario: Scenario) -> Scenario:
    perturbed_values = vars(scenario).copy()
    for var in perturbed_values:
        if var != 'name':
            perturbed_values[var] *= np.random.normal(1, 0.1)  # 10% standard deviation
    return Scenario(**perturbed_values)

def calculate_risk_impact(risk: Risk, scenario: Scenario) -> float:
    # Implement a more sophisticated impact calculation
    base_impact = risk.impact
    temp_factor = 1 + (scenario.temp_increase - 1.5) * 0.1
    carbon_price_factor = 1 + (scenario.carbon_price / 100) * 0.05
    renewable_factor = 1 - scenario.renewable_energy * 0.2
    
    impact = base_impact * temp_factor * carbon_price_factor * renewable_factor
    return min(1.0, max(0.0, impact))

def calculate_risk_likelihood(risk: Risk, scenario: Scenario) -> float:
    # Implement a more sophisticated likelihood calculation
    base_likelihood = risk.likelihood
    policy_factor = 1 - scenario.policy_stringency * 0.3
    ecosystem_factor = 1 + scenario.ecosystem_degradation * 0.4
    
    likelihood = base_likelihood * policy_factor * ecosystem_factor
    return min(1.0, max(0.0, likelihood))