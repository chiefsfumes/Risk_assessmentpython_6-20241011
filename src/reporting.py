import json
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import os
from src.models import Risk, RiskInteraction, SimulationResult, Scenario
from src.config import OUTPUT_DIR

def generate_report(risks: List[Risk], categorized_risks: Dict[str, List[Risk]], 
                    risk_interactions: List[RiskInteraction], scenario_impacts: Dict[str, List[Tuple[Risk, float]]],
                    simulation_results: Dict[str, Dict[int, SimulationResult]], clustered_risks: Dict[int, List[int]],
                    risk_entities: Dict[str, List[str]], sensitivity_results: Dict[str, Dict[str, float]],
                    time_series_results: Dict[int, List[float]], scenarios: Dict[str, Scenario],
                    advanced_analysis: Dict) -> str:
    report = {
        "executive_summary": generate_executive_summary(risks, scenario_impacts, simulation_results, advanced_analysis),
        "risk_overview": {
            "total_risks": len(risks),
            "risk_categories": {category: len(risks) for category, risks in categorized_risks.items()},
            "high_impact_risks": [risk.to_dict() for risk in risks if risk.impact > 0.7],
        },
        "risk_interactions": {
            "summary": summarize_risk_interactions(risk_interactions),
            "detailed_interactions": [interaction.__dict__ for interaction in risk_interactions]
        },
        "scenario_analysis": {
            scenario: {
                "summary": summarize_scenario_impact(impacts),
                "detailed_impacts": [
                    {"risk_id": risk.id, "impact": impact} 
                    for risk, impact in sorted(impacts, key=lambda x: x[1], reverse=True)
                ],
                "llm_analysis": advanced_analysis["comprehensive_analysis"][scenario]
            } for scenario, impacts in scenario_impacts.items()
        },
        "monte_carlo_results": {
            scenario: {
                risk_id: {
                    "mean_impact": np.mean(results.impact_distribution),
                    "std_impact": np.std(results.impact_distribution),
                    "5th_percentile_impact": np.percentile(results.impact_distribution, 5),
                    "95th_percentile_impact": np.percentile(results.impact_distribution, 95),
                    "mean_likelihood": np.mean(results.likelihood_distribution),
                    "std_likelihood": np.std(results.likelihood_distribution)
                } for risk_id, results in scenario_results.items()
            } for scenario, scenario_results in simulation_results.items()
        },
        "risk_clusters": clustered_risks,
        "risk_entities": risk_entities,
        "sensitivity_analysis": sensitivity_results,
        "time_series_projection": {risk_id: projections for risk_id, projections in time_series_results.items()},
        "risk_narratives": advanced_analysis["risk_narratives"],
        "executive_insights": advanced_analysis["executive_insights"],
        "mitigation_strategies": generate_mitigation_strategies(risks, scenario_impacts, simulation_results)
    }
    
    report_json = json.dumps(report, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, 'climate_risk_report.json'), 'w') as f:
        f.write(report_json)
    
    generate_html_report(report)
    
    return report_json

def generate_executive_summary(risks: List[Risk], scenario_impacts: Dict[str, List[Tuple[Risk, float]]], 
                               simulation_results: Dict[str, Dict[int, SimulationResult]],
                               advanced_analysis: Dict) -> str:
    num_risks = len(risks)
    high_impact_risks = sum(1 for risk in risks if risk.impact > 0.7)
    worst_scenario = max(scenario_impacts.items(), key=lambda x: sum(impact for _, impact in x[1]))
    
    summary = f"""
    Executive Summary:
    
    This climate risk assessment identified {num_risks} distinct risks, with {high_impact_risks} classified as high-impact.
    The '{worst_scenario[0]}' scenario presents the most significant challenges, with potential for severe impacts across multiple risk categories.
    
    Key findings:
    1. {summarize_top_risks(scenario_impacts[worst_scenario[0]])}
    2. {summarize_monte_carlo_results(simulation_results)}
    
    Advanced Analysis Insights:
    {advanced_analysis['executive_insights']}
    
    Immediate attention is required to develop and implement comprehensive mitigation strategies, particularly focusing on the high-impact risks identified in this assessment.
    """
    
    return summary

def summarize_risk_interactions(risk_interactions: List[RiskInteraction]) -> str:
    strong_interactions = sum(1 for interaction in risk_interactions if interaction.interaction_type == "Strong")
    moderate_interactions = sum(1 for interaction in risk_interactions if interaction.interaction_type == "Moderate")
    weak_interactions = sum(1 for interaction in risk_interactions if interaction.interaction_type == "Weak")
    
    summary = f"""
    Risk Interaction Summary:
    - Strong interactions: {strong_interactions}
    - Moderate interactions: {moderate_interactions}
    - Weak interactions: {weak_interactions}
    
    The analysis reveals a complex web of risk interactions, with {strong_interactions} strong interactions 
    indicating potential compounding effects that require careful consideration in risk mitigation strategies.
    """
    
    return summary

def summarize_scenario_impact(impacts: List[Tuple[Risk, float]]) -> str:
    sorted_impacts = sorted(impacts, key=lambda x: x[1], reverse=True)
    top_3_risks = sorted_impacts[:3]
    
    summary = f"""
    Scenario Impact Summary:
    Top 3 impacted risks:
    1. Risk {top_3_risks[0][0].id}: Impact score {top_3_risks[0][1]:.2f}
    2. Risk {top_3_risks[1][0].id}: Impact score {top_3_risks[1][1]:.2f}
    3. Risk {top_3_risks[2][0].id}: Impact score {top_3_risks[2][1]:.2f}
    
    This scenario shows significant impacts on the above risks, requiring targeted mitigation strategies.
    """
    
    return summary

def summarize_top_risks(impacts: List[Tuple[Risk, float]]) -> str:
    sorted_impacts = sorted(impacts, key=lambda x: x[1], reverse=True)
    top_3_risks = sorted_impacts[:3]
    
    summary = f"""
    The top 3 risks under this scenario are:
    1. {top_3_risks[0][0].description} (Impact: {top_3_risks[0][1]:.2f})
    2. {top_3_risks[1][0].description} (Impact: {top_3_risks[1][1]:.2f})
    3. {top_3_risks[2][0].description} (Impact: {top_3_risks[2][1]:.2f})
    """
    
    return summary

def summarize_monte_carlo_results(simulation_results: Dict[str, Dict[int, SimulationResult]]) -> str:
    scenario_summaries = []
    
    for scenario, results in simulation_results.items():
        max_impact_risk = max(results.items(), key=lambda x: np.mean(x[1].impact_distribution))
        max_likelihood_risk = max(results.items(), key=lambda x: np.mean(x[1].likelihood_distribution))
        
        scenario_summary = f"""
        {scenario} Scenario:
        - Highest impact risk: Risk {max_impact_risk[0]} (Mean impact: {np.mean(max_impact_risk[1].impact_distribution):.2f})
        - Highest likelihood risk: Risk {max_likelihood_risk[0]} (Mean likelihood: {np.mean(max_likelihood_risk[1].likelihood_distribution):.2f})
        """
        scenario_summaries.append(scenario_summary)
    
    return "\n".join(scenario_summaries)

def generate_mitigation_strategies(risks: List[Risk], scenario_impacts: Dict[str, List[Tuple[Risk, float]]],
                                   simulation_results: Dict[str, Dict[int, SimulationResult]]) -> Dict[int, List[str]]:
    mitigation_strategies = {}
    
    for risk in risks:
        strategies = []
        
        # Analyze scenario impacts
        max_impact_scenario = max(scenario_impacts.items(), key=lambda x: next((impact for r, impact in x[1] if r.id == risk.id), 0))
        
        if max_impact_scenario[0] == "Net Zero 2050":
            strategies.append("Accelerate transition to low-carbon technologies")
        elif max_impact_scenario[0] == "Delayed Transition":
            strategies.append("Prepare for abrupt policy changes and market shifts")
        elif max_impact_scenario[0] == "Current Policies":
            strategies.append("Enhance resilience to physical climate risks")
        
        # Analyze Monte Carlo simulation results
        risk_simulation = {scenario: results[risk.id] for scenario, results in simulation_results.items() if risk.id in results}
        high_variability_scenarios = [scenario for scenario, results in risk_simulation.items() if np.std(results.impact_distribution) > 0.5]
        
        if high_variability_scenarios:
            strategies.append(f"Develop flexible strategies to address high uncertainty in {', '.join(high_variability_scenarios)} scenarios")
        
        # Category-specific strategies
        if risk.category == "Physical Risk":
            strategies.append("Invest in climate-resilient infrastructure and operations")
        elif risk.category == "Transition Risk":
            strategies.append("Diversify product/service portfolio to align with low-carbon economy")
        elif risk.category == "Market Risk":
            strategies.append("Monitor and adapt to changing consumer preferences and market dynamics")
        elif risk.category == "Policy Risk":
            strategies.append("Engage in policy discussions and prepare for various regulatory scenarios")
        elif risk.category == "Reputation Risk":
            strategies.append("Enhance sustainability reporting and stakeholder communication")
        
        mitigation_strategies[risk.id] = strategies
    
    return mitigation_strategies

def generate_html_report(report: Dict) -> None:
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Climate Risk Assessment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .risk-category {{ margin-bottom: 20px; }}
            .scenario {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Climate Risk Assessment Report</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>{report['executive_summary']}</p>
        </div>
        
        <h2>Risk Overview</h2>
        <p>Total Risks: {report['risk_overview']['total_risks']}</p>
        <h3>Risk Categories</h3>
        <ul>
            {' '.join(f'<li>{category}: {count}</li>' for category, count in report['risk_overview']['risk_categories'].items())}
        </ul>
        
        <h3>High Impact Risks</h3>
        <ul>
            {' '.join(f'<li>Risk {risk["id"]}: {risk["description"]} (Impact: {risk["impact"]})</li>' for risk in report['risk_overview']['high_impact_risks'])}
        </ul>
        
        <h2>Scenario Analysis</h2>
        {' '.join(f'''
        <div class="scenario">
            <h3>{scenario}</h3>
            <p>{data['summary']}</p>
            <h4>Top 3 Risks:</h4>
            <ul>
                {' '.join(f'<li>Risk {impact["risk_id"]}: Impact {impact["impact"]:.2f}</li>' for impact in data['detailed_impacts'][:3])}
            </ul>
            <h4>LLM Analysis:</h4>
            {' '.join(f'<p>{analysis}</p>' for analysis in data['llm_analysis'].values())}
        </div>
        ''' for scenario, data in report['scenario_analysis'].items())}
        
        <h2>Monte Carlo Simulation Results</h2>
        {' '.join(f'''
        <h3>{scenario}</h3>
        <table>
            <tr>
                <th>Risk ID</th>
                <th>Mean Impact</th>
                <th>Std Dev Impact</th>
                <th>5th Percentile</th>
                <th>95th Percentile</th>
            </tr>
            {' '.join(f'''
            <tr>
                <td>{risk_id}</td>
                <td>{results['mean_impact']:.2f}</td>
                <td>{results['std_impact']:.2f}</td>
                <td>{results['5th_percentile_impact']:.2f}</td>
                <td>{results['95th_percentile_impact']:.2f}</td>
            </tr>
            ''' for risk_id, results in scenario_results.items())}
        </table>
        ''' for scenario, scenario_results in report['monte_carlo_results'].items())}
        
        <h2>Sensitivity Analysis</h2>
        <table>
            <tr>
                <th>Scenario</th>
                {' '.join(f'<th>{variable}</th>' for variable in next(iter(report['sensitivity_analysis'].values())).keys())}
            </tr>
            {' '.join(f'''
            <tr>
                <td>{scenario}</td>
                {' '.join(f'<td>{sensitivity:.2f}</td>' for sensitivity in sensitivities.values())}
            </tr>
            ''' for scenario, sensitivities in report['sensitivity_analysis'].items())}
        </table>
        
        <h2>Risk Narratives</h2>
        {' '.join(f'''
        <div class="risk-category">
            <h3>Risk {risk_id}</h3>
            <p>{narrative}</p>
        </div>
        ''' for risk_id, narrative in report['risk_narratives'].items())}
        
        <h2>Executive Insights</h2>
        <p>{report['executive_insights']}</p>
        
        <h2>Mitigation Strategies</h2>
        {' '.join(f'''
        <div class="risk-category">
            <h3>Risk {risk_id}</h3>
            <ul>
                {' '.join(f'<li>{strategy}</li>' for strategy in strategies)}
            </ul>
        </div>
        ''' for risk_id, strategies in report['mitigation_strategies'].items())}
        
    </body>
    </html>
    """
    
    with open(os.path.join(OUTPUT_DIR, 'climate_risk_report.html'), 'w') as f:
        f.write(html_content)