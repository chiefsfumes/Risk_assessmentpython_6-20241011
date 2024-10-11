# Configuration

## Overview

The configuration module (`src/config.py`) contains essential settings and parameters used throughout the Climate Risk Assessment Tool. It provides a centralized location for managing various aspects of the tool's behavior, including scenario definitions, simulation parameters, and file paths.

## Key Components

### Scenario Definitions

The `SCENARIOS` dictionary defines different climate scenarios used in the risk assessment:

```python
SCENARIOS: Dict[str, Scenario] = {
    "Net Zero 2050": Scenario(
        name="Net Zero 2050",
        temp_increase=1.5,
        carbon_price=250,
        renewable_energy=0.75,
        policy_stringency=0.9,
        biodiversity_loss=0.1,
        ecosystem_degradation=0.2,
        financial_stability=0.8,
        supply_chain_disruption=0.3
    ),
    "Delayed Transition": Scenario(...),
    "Current Policies": Scenario(...),
    "Nature Positive": Scenario(...),
    "Global Instability": Scenario(...)
}
```

Each scenario is defined with specific parameters that influence risk calculations.

### Simulation Parameters

- `NUM_SIMULATIONS = 10000`: The number of iterations for Monte Carlo simulations.
- `NUM_CLUSTERS = 3`: The default number of clusters for risk clustering analysis.

### File Paths

- `BASE_DIR`: The base directory of the project.
- `DATA_DIR`: Directory for input data files.
- `OUTPUT_DIR`: Directory for output files (reports, visualizations).

### Model Configuration

- `NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"`: The pre-trained model used for Named Entity Recognition.
- `LLM_MODEL = "gpt-3.5-turbo"`: The language model used for advanced text analysis.
- `LLM_API_KEY`: The API key for accessing the language model (loaded from environment variables).

### Visualization Settings

- `VIZ_DPI = 300`: The resolution (dots per inch) for saved visualizations.
- `HEATMAP_CMAP = 'YlOrRd'`: The color map used for heatmap visualizations.

### Analysis Parameters

- `TIME_SERIES_HORIZON = 10`: The number of years to project in time series analysis.
- `SENSITIVITY_VARIABLES`: List of variables to consider in sensitivity analysis.
- `SENSITIVITY_RANGE = 0.2`: The range (+/- 20%) for sensitivity analysis perturbations.

## Usage

To use these configurations in other parts of the project, import the necessary variables or functions:

```python
from src.config import SCENARIOS, NUM_SIMULATIONS, OUTPUT_DIR, LLM_MODEL
```

## Customization

To customize the tool's behavior:

1. Modify scenario definitions in the `SCENARIOS` dictionary to reflect different climate pathways.
2. Adjust `NUM_SIMULATIONS` to balance between accuracy and computation time.
3. Update file paths if using a different project structure.
4. Change the `LLM_MODEL` if using a different language model for analysis.

## Environment Variables

Ensure that the `OPENAI_API_KEY` environment variable is set with your API key for language model access:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Logging Configuration

The `setup_logging` function configures the logging behavior:

```python
def setup_logging(log_level: str = "INFO") -> None:
    # ... logging setup code ...
```

Call this function at the start of the main script to initialize logging with the desired level.

By centralizing these configurations, the Climate Risk Assessment Tool maintains consistency across its modules and allows for easy adjustments to its behavior and parameters.