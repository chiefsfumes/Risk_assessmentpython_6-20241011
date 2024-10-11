import os
import logging
import argparse
from typing import Dict, List

from src.data_loader import load_risk_data, load_external_data
from src.risk_analysis.categorization import categorize_risks, categorize_risks_multi_level, prioritize_risks
from src.risk_analysis.interaction_analysis import analyze_risk_interactions, build_risk_network, identify_central_risks, detect_risk_clusters, analyze_risk_cascades
from src.risk_analysis.scenario_analysis import simulate_scenario_impact, monte_carlo_simulation, llm_risk_assessment, analyze_scenario_sensitivity
from src.risk_analysis.time_series_analysis import time_series_analysis, analyze_impact_trends, identify_critical_periods, forecast_cumulative_impact
from src.risk_analysis.advanced_analysis import conduct_advanced_risk_analysis
from src.visualization import generate_visualizations
from src.reporting import generate_report
from src.config import SCENARIOS, OUTPUT_DIR, setup_logging
from src.data_collection.nlp_extraction import extract_risk_statements_from_10k
from src.risk_analysis.pestel_analysis import perform_pestel_analysis
from src.risk_analysis.sasb_integration import integrate_sasb_materiality
from src.risk_analysis.systemic_risk_analysis import analyze_systemic_risks, identify_trigger_points, assess_resilience
from src.sensitivity_analysis.monte_carlo import perform_monte_carlo_simulations
from src.reporting.stakeholder_reports import generate_stakeholder_reports
from src.models import Risk, ExternalData, Scenario

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Climate Risk Assessment Tool")
    parser.add_argument("--risk_data", type=str, default="data/risk_data.csv", help="Path to risk data CSV file")
    parser.add_argument("--external_data", type=str, default="data/external_data.csv", help="Path to external data CSV file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Advanced Climate Risk Assessment Tool")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Data Collection and Preprocessing
        risk_statements = extract_risk_statements_from_10k('data/10k_filings')
        risks: List[Risk] = load_risk_data(args.risk_data)
        external_data: Dict[str, ExternalData] = load_external_data(args.external_data)
        
        # Enhanced Risk Categorization
        categorized_risks = categorize_risks(risks)
        multi_level_categorized_risks = categorize_risks_multi_level(risks)
        prioritized_risks = prioritize_risks(risks)
        industry_specific_risks = integrate_sasb_materiality(risks, 'Energy')  # Replace 'Energy' with actual industry
        pestel_analysis = perform_pestel_analysis(risks, external_data)
        
        # Risk Interaction Analysis
        risk_interactions = analyze_risk_interactions(risks)
        risk_network = build_risk_network(risks, risk_interactions)
        central_risks = identify_central_risks(risk_network)
        risk_clusters = detect_risk_clusters(risk_network)
        risk_cascades = analyze_risk_cascades(risk_network, [r.id for r in risks if r.impact > 0.8])
        
        # Scenario Analysis
        scenario_impacts = {
            scenario_name: simulate_scenario_impact(risks, external_data, scenario_params)
            for scenario_name, scenario_params in SCENARIOS.items()
        }
        
        simulation_results = monte_carlo_simulation(risks, external_data, SCENARIOS)
        
        # Sensitivity Analysis
        sensitivity_results = {
            scenario_name: analyze_scenario_sensitivity(risks, scenario, 'carbon_price', 0.2)
            for scenario_name, scenario in SCENARIOS.items()
        }
        
        # Time Series Analysis
        time_series_results = time_series_analysis(risks, external_data)
        impact_trends = analyze_impact_trends(time_series_results)
        critical_periods = identify_critical_periods(time_series_results, threshold=0.7)
        cumulative_impact = forecast_cumulative_impact(time_series_results)
        
        # Advanced LLM-based Analysis
        company_industry = "Energy"  # This should be dynamically determined or provided as input
        key_dependencies = ["Oil suppliers", "Renewable energy technology", "Grid infrastructure"]
        advanced_analysis = conduct_advanced_risk_analysis(risks, SCENARIOS, company_industry, key_dependencies)
        
        # Systemic Risk Analysis
        systemic_risks = analyze_systemic_risks(risks, company_industry, key_dependencies)
        trigger_points = identify_trigger_points(risks, risk_network, external_data)
        resilience_assessment = assess_resilience(risks, scenario_impacts, simulation_results)
        
        # Monte Carlo Simulations
        monte_carlo_results = perform_monte_carlo_simulations(risks, SCENARIOS, num_simulations=10000)
        
        # Generate Visualizations
        generate_visualizations(risks, risk_interactions, simulation_results, 
                                sensitivity_results, time_series_results,
                                risk_network, risk_clusters, cumulative_impact)
        
        # Generate Reports
        main_report = generate_report(risks, categorized_risks, multi_level_categorized_risks, prioritized_risks,
                                      risk_interactions, central_risks, risk_clusters, risk_cascades,
                                      scenario_impacts, simulation_results, sensitivity_results, 
                                      time_series_results, impact_trends, critical_periods, cumulative_impact,
                                      SCENARIOS, advanced_analysis, systemic_risks, trigger_points, 
                                      resilience_assessment, monte_carlo_results)
        
        stakeholder_reports = generate_stakeholder_reports(main_report, company_industry)

        logger.info("Risk Assessment Report and stakeholder reports generated successfully.")
        logger.info(f"Main report saved to: {os.path.join(args.output_dir, 'climate_risk_report.json')}")
        logger.info(f"Stakeholder reports saved in: {args.output_dir}")
        logger.info(f"Visualizations saved in: {args.output_dir}")

    except Exception as e:
        logger.error(f"An error occurred during the risk assessment process: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    main(args)