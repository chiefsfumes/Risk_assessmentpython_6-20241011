# API Reference

## Overview
This document provides a detailed reference for the key classes and functions in the Climate Risk Assessment Tool.

## Classes

### Risk
Represents a climate-related risk.

Attributes:
- `id` (int): Unique identifier for the risk
- `description` (str): Description of the risk
- `category` (str): Category of the risk (e.g., "Physical", "Transition")
- `subcategory` (str): Subcategory of the risk
- `likelihood` (float): Current likelihood of the risk (0-1)
- `impact` (float): Current impact of the risk (0-1)
- ...

### Scenario
Represents a climate scenario.

Attributes:
- `name` (str): Name of the scenario
- `temp_increase` (float): Temperature increase in °C
- `carbon_price` (float): Carbon price in $/ton
- `renewable_energy` (float): Renewable energy share (0-1)
- ...

## Functions

### Data Loading

`load_risk_data(file_path: str) -> List[Risk]`
Loads risk data from a CSV file.

`load_external_data(file_path: str) -> Dict[str, ExternalData]`
Loads external data from a CSV file.

### Risk Analysis

`categorize_risks(risks: List[Risk]) -> Dict[str, List[Risk]]`
Categorizes risks based on their primary category.

`simulate_scenario_impact(risks: List[Risk], external_data: Dict[str, ExternalData], scenario: Scenario) -> List[Tuple[Risk, float]]`
Simulates the impact of a given scenario on the provided risks.

`monte_carlo_simulation(risks: List[Risk], external_data: Dict[str, ExternalData], scenarios: Dict[str, Scenario]) -> Dict[str, Dict[int, SimulationResult]]`
Performs Monte Carlo simulations for each risk under different scenarios.

### Reporting

`generate_report(risks: List[Risk], ...) -> str`
Generates a comprehensive report in JSON format and creates an HTML version.

### Visualization

`generate_visualizations(risks: List[Risk], ...)`
Generates a set of visualizations based on the risk assessment results.

## Usage Examples

```python
# Load data
risks = load_risk_data('data/risk_data.csv')
external_data = load_external_data('data/external_data.csv')

# Perform analysis
categorized_risks = categorize_risks(risks)
scenario_impacts = simulate_scenario_impact(risks, external_data, scenarios['Net Zero 2050'])

# Generate report
report = generate_report(risks, categorized_risks, ...)

# Create visualizations
generate_visualizations(risks, ...)
```

For more detailed information on each function and class, refer to the inline documentation in the source code.