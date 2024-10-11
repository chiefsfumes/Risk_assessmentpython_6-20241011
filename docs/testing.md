# Testing

## Overview

The testing module ensures the reliability and correctness of the Climate Risk Assessment Tool. It uses the `pytest` framework to run a comprehensive suite of unit tests covering various components of the system.

## Test Structure

The tests are organized in the `tests/` directory, with separate files for each major component of the system:

- `test_data_loader.py`
- `test_risk_analysis.py`
- `test_scenario_analysis.py`
- `test_interaction_analysis.py`
- `test_time_series_analysis.py`
- `test_advanced_analysis.py`
- `test_visualization.py`
- `test_reporting.py`
- `test_pestel_analysis.py`
- `test_sasb_integration.py`
- `test_systemic_risk_analysis.py`

## Running Tests

To run the entire test suite, use the following command from the project root directory:

```bash
pytest
```

To run tests for a specific module:

```bash
pytest tests/test_risk_analysis.py
```

## Key Test Cases

### Data Loading

- Test successful loading of risk data and external data.
- Test error handling for missing files or incorrect data formats.

### Risk Analysis

- Test risk categorization and prioritization.
- Verify the correctness of scenario impact calculations.
- Ensure Monte Carlo simulations produce expected distributions.

### Interaction Analysis

- Test the identification of risk interactions.
- Verify the construction and analysis of the risk network.

### Time Series Analysis

- Test the ARIMA model projections.
- Verify the identification of critical periods and trends.

### Advanced Analysis

- Test the integration with language models for risk assessment.
- Verify the generation of risk narratives and executive insights.

### Visualization

- Test the generation of various plots and charts.
- Ensure visualizations are saved correctly.

### Reporting

- Test the generation of comprehensive JSON and HTML reports.
- Verify the creation of stakeholder-specific reports.

## Example Test Function

Here's an example of a test function from `test_risk_analysis.py`:

```python
def test_categorize_risks(sample_risks):
    categorized = categorize_risks(sample_risks)
    assert len(categorized) == 2
    assert len(categorized["Physical"]) == 2
    assert len(categorized["Transition"]) == 1
```

This test ensures that the `categorize_risks` function correctly categorizes a sample set of risks.

## Fixtures

Pytest fixtures are used to provide sample data for tests. For example:

```python
@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Physical Risk 1", category="Physical", likelihood=0.7, impact=0.8, ...),
        Risk(id=2, description="Transition Risk 1", category="Transition", likelihood=0.6, impact=0.7, ...),
        Risk(id=3, description="Physical Risk 2", category="Physical", likelihood=0.5, impact=0.6, ...)
    ]
```

This fixture provides a consistent set of sample risks for various tests.

## Mocking

For tests that involve external APIs or complex computations, mocking is used to isolate the tested functionality:

```python
@patch('src.risk_analysis.advanced_analysis.openai.ChatCompletion.create')
def test_llm_risk_assessment(mock_create):
    mock_create.return_value.choices[0].message = {'content': 'Mocked LLM response'}
    result = llm_risk_assessment(sample_risk, sample_scenario, "Energy")
    assert "Mocked LLM response" in result
```

This test mocks the OpenAI API call to test the `llm_risk_assessment` function in isolation.

## Continuous Integration

The test suite is integrated into the project's CI/CD pipeline, ensuring that all tests pass before changes are merged into the main branch.

## Coverage

To measure test coverage, run:

```bash
pytest --cov=src
```

This command will show the percentage of code covered by tests for each module.

By maintaining a comprehensive test suite, we ensure the reliability and correctness of the Climate Risk Assessment Tool, facilitating confident development and deployment of new features and improvements.