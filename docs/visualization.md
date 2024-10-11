# Visualization

## Overview

The visualization module provides functions to create various visual representations of the risk assessment results. These visualizations help in understanding the complex relationships between risks, their impacts under different scenarios, and other key insights from the analysis.

## Key Functions

### `generate_visualizations(risks: List[Risk], risk_interactions: List[RiskInteraction], simulation_results: Dict[str, Dict[int, SimulationResult]], sensitivity_results: Dict[str, Dict[str, float]], time_series_results: Dict[int, List[float]])`

This function generates a comprehensive set of visualizations based on the risk assessment results.

#### Input
- `risks`: A list of `Risk` objects.
- `risk_interactions`: A list of `RiskInteraction` objects.
- `simulation_results`: Results from Monte Carlo simulations.
- `sensitivity_results`: Results from sensitivity analysis.
- `time_series_results`: Results from time series analysis.

#### Output
- This function saves various visualization files (PNG format) in the specified output directory.

#### Example

```python
generate_visualizations(risks, risk_interactions, simulation_results, sensitivity_results, time_series_results)
```

### `risk_matrix(risks: List[Risk])`

This function creates a risk matrix visualization, plotting risks based on their likelihood and impact.

#### Input
- `risks`: A list of `Risk` objects.

#### Output
- Saves a 'risk_matrix.png' file in the output directory.

#### Illustration

![Risk Matrix](https://example.com/risk_matrix.png)

*The risk matrix shows risks plotted on a 2D grid, with likelihood on the x-axis and impact on the y-axis. Each risk is represented by a colored dot, with colors indicating different risk categories.*

### `interaction_heatmap(risks: List[Risk], risk_interactions: List[RiskInteraction])`

This function creates a heatmap visualization of risk interactions.

#### Input
- `risks`: A list of `Risk` objects.
- `risk_interactions`: A list of `RiskInteraction` objects.

#### Output
- Saves an 'interaction_heatmap.png' file in the output directory.

#### Illustration

![Interaction Heatmap](https://example.com/interaction_heatmap.png)

*The interaction heatmap shows the strength of interactions between risks. Darker colors indicate stronger interactions.*

### `interaction_network(risks: List[Risk], risk_interactions: List[RiskInteraction])`

This function creates a network graph visualization of risk interactions.

#### Input
- `risks`: A list of `Risk` objects.
- `risk_interactions`: A list of `RiskInteraction` objects.

#### Output
- Saves an 'interaction_network.png' file in the output directory.

#### Illustration

![Interaction Network](https://example.com/interaction_network.png)

*The interaction network shows risks as nodes and their interactions as edges. The thickness of edges represents the strength of interactions.*

### `monte_carlo_results(simulation_results: Dict[str, Dict[int, SimulationResult]])`

This function visualizes the results of Monte Carlo simulations.

#### Input
- `simulation_results`: A dictionary containing simulation results for each scenario and risk.

#### Output
- Saves a 'monte_carlo_results.png' file in the output directory.

#### Illustration

![Monte Carlo Results](https://example.com/monte_carlo_results.png)

*This plot shows the distribution of impact values for each risk under different scenarios, typically as a series of overlapping density plots.*

### `sensitivity_analysis_heatmap(sensitivity_results: Dict[str, Dict[str, float]])`

This function creates a heatmap visualization of sensitivity analysis results.

#### Input
- `sensitivity_results`: A dictionary containing sensitivity analysis results.

#### Output
- Saves a 'sensitivity_analysis_heatmap.png' file in the output directory.

#### Illustration

![Sensitivity Analysis Heatmap](https://example.com/sensitivity_heatmap.png)

*The sensitivity analysis heatmap shows how different variables affect the overall risk impact across scenarios. Darker colors indicate higher sensitivity.*

### `time_series_projection(risks: List[Risk], time_series_results: Dict[int, List[float]])`

This function visualizes the projected time series of risk impacts.

#### Input
- `risks`: A list of `Risk` objects.
- `time_series_results`: A dictionary containing time series projections for each risk.

#### Output
- Saves a 'time_series_projection.png' file in the output directory.

#### Illustration

![Time Series Projection](https://example.com/time_series_projection.png)

*This plot shows the projected impact of each risk over time, with different lines representing different risks.*

## Usage Notes

- All visualizations are saved as PNG files in the directory specified by the `OUTPUT_DIR` configuration variable.
- The visualizations use a consistent color scheme and styling for coherence across different plots.
- Some visualizations (like the interaction network) may become cluttered with a large number of risks. In such cases, consider filtering to show only the most significant interactions or risks.
- The `matplotlib` and `seaborn` libraries are used for creating these visualizations. Ensure these libraries are properly installed and configured in your environment.

These visualizations provide powerful tools for communicating complex risk assessment results to stakeholders, allowing for quick identification of key risks, their interactions, and their behavior under different scenarios.