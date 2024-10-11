import pytest
import numpy as np
from src.risk_analysis.time_series_analysis import (
    time_series_analysis, project_risk_impact_arima, analyze_impact_trends,
    identify_critical_periods, forecast_cumulative_impact
)
from src.models import Risk, ExternalData

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Physical Risk 1", category="Physical", likelihood=0.7, impact=0.8, subcategory="Acute", tertiary_category="", time_horizon="Short-term", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Transition Risk 1", category="Transition", likelihood=0.6, impact=0.7, subcategory="Policy", tertiary_category="", time_horizon="Medium-term", industry_specific=True, sasb_category="Energy"),
    ]

@pytest.fixture
def sample_external_data():
    return {
        str(year): ExternalData(
            year=year,
            gdp_growth=np.random.normal(2, 0.5),
            population=7_500_000_000 + year * 80_000_000,
            energy_demand=150_000 + year * 1000,
            carbon_price=20 + year * 2,
            renewable_energy_share=0.2 + year * 0.01,
            biodiversity_index=1 - year * 0.01,
            deforestation_rate=0.5 - year * 0.01
        )
        for year in range(2010, 2021)
    }

def test_time_series_analysis(sample_risks, sample_external_data):
    results = time_series_analysis(sample_risks, sample_external_data)
    
    assert len(results) == len(sample_risks)
    for risk_id, projections in results.items():
        assert len(projections) == 10  # Assuming TIME_SERIES_HORIZON = 10
        assert all(0 <= impact <= 1 for impact in projections)
    
    # Test that projections are different for different risks
    assert not np.allclose(results[1], results[2])

def test_project_risk_impact_arima(sample_risks, sample_external_data):
    risk = sample_risks[0]
    projections = project_risk_impact_arima(risk, sample_external_data)
    
    assert len(projections) == 10  # Assuming TIME_SERIES_HORIZON = 10
    assert all(0 <= impact <= 1 for impact in projections)
    
    # Test that projections are not constant
    assert len(set(projections)) > 1

def test_analyze_impact_trends():
    time_series_results = {
        1: [0.5, 0.55, 0.6, 0.65, 0.7],
        2: [0.7, 0.65, 0.6, 0.55, 0.5],
    }
    trend_analysis = analyze_impact_trends(time_series_results)
    
    assert len(trend_analysis) == len(time_series_results)
    for risk_id, analysis in trend_analysis.items():
        assert "slope" in analysis
        assert "average_impact" in analysis
        assert "max_impact" in analysis
        assert "min_impact" in analysis
        assert "volatility" in analysis
    
    # Test that slopes have correct signs
    assert trend_analysis[1]["slope"] > 0
    assert trend_analysis[2]["slope"] < 0

def test_identify_critical_periods():
    time_series_results = {
        1: [0.5, 0.6, 0.7, 0.8, 0.9],
        2: [0.4, 0.5, 0.6, 0.7, 0.8],
    }
    critical_periods = identify_critical_periods(time_series_results, threshold=0.7)
    
    assert 1 in critical_periods
    assert 2 in critical_periods
    assert critical_periods[1] == [3, 4]
    assert critical_periods[2] == [4]
    
    # Test with no critical periods
    no_critical = identify_critical_periods(time_series_results, threshold=0.95)
    assert len(no_critical) == 0

def test_forecast_cumulative_impact():
    time_series_results = {
        1: [0.5, 0.6, 0.7, 0.8, 0.9],
        2: [0.4, 0.5, 0.6, 0.7, 0.8],
    }
    cumulative_impact = forecast_cumulative_impact(time_series_results)
    
    assert len(cumulative_impact) == 5
    assert cumulative_impact == [0.9, 1.1, 1.3, 1.5, 1.7]
    
    # Test that cumulative impact is always increasing
    assert all(cumulative_impact[i] <= cumulative_impact[i+1] for i in range(len(cumulative_impact)-1))

# Add edge case tests
def test_time_series_analysis_edge_cases(sample_risks):
    # Test with no external data
    empty_external_data = {}
    with pytest.raises(ValueError):
        time_series_analysis(sample_risks, empty_external_data)
    
    # Test with constant external data
    constant_external_data = {
        str(year): ExternalData(
            year=year,
            gdp_growth=2.0,
            population=7_500_000_000,
            energy_demand=150_000,
            carbon_price=20,
            renewable_energy_share=0.2,
            biodiversity_index=1.0,
            deforestation_rate=0.5
        )
        for year in range(2010, 2021)
    }
    results = time_series_analysis(sample_risks, constant_external_data)
    for risk_id, projections in results.items():
        assert len(set(projections)) == 1  # All projections should be the same

def test_analyze_impact_trends_edge_cases():
    # Test with constant time series
    constant_time_series = {
        1: [0.5, 0.5, 0.5, 0.5, 0.5],
    }
    trend_analysis = analyze_impact_trends(constant_time_series)
    assert trend_analysis[1]["slope"] == 0
    assert trend_analysis[1]["volatility"] == 0
    
    # Test with extreme volatility
    volatile_time_series = {
        1: [0.1, 0.9, 0.1, 0.9, 0.1],
    }
    trend_analysis = analyze_impact_trends(volatile_time_series)
    assert trend_analysis[1]["volatility"] > 0.3  # High volatility

def test_forecast_cumulative_impact_edge_cases():
    # Test with negative impacts
    negative_impacts = {
        1: [-0.5, -0.4, -0.3, -0.2, -0.1],
        2: [-0.4, -0.3, -0.2, -0.1, 0.0],
    }
    cumulative_impact = forecast_cumulative_impact(negative_impacts)
    assert all(impact < 0 for impact in cumulative_impact)
    
    # Test with very large impacts
    large_impacts = {
        1: [1e6, 2e6, 3e6, 4e6, 5e6],
        2: [1e6, 2e6, 3e6, 4e6, 5e6],
    }
    cumulative_impact = forecast_cumulative_impact(large_impacts)
    assert cumulative_impact[-1] == 10e6