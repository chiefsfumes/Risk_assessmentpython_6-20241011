from typing import List, Dict
from src.models import Risk, ExternalData
from src.config import TIME_SERIES_HORIZON
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def time_series_analysis(risks: List[Risk], external_data: Dict[str, ExternalData]) -> Dict[int, List[float]]:
    time_series_results = {}
    for risk in risks:
        projections = project_risk_impact_arima(risk, external_data)
        time_series_results[risk.id] = projections
    return time_series_results

def project_risk_impact_arima(risk: Risk, external_data: Dict[str, ExternalData]) -> List[float]:
    # Prepare historical data
    historical_impacts = [calculate_historical_impact(risk, data) for data in external_data.values()]
    
    # Fit ARIMA model
    model = ARIMA(historical_impacts, order=(1, 1, 1))  # Example order, adjust based on your data
    model_fit = model.fit()
    
    # Make future projections
    forecast = model_fit.forecast(steps=TIME_SERIES_HORIZON)
    
    return list(forecast)

def calculate_historical_impact(risk: Risk, data: ExternalData) -> float:
    # Implement logic to calculate historical impact based on risk characteristics and external data
    base_impact = risk.impact
    gdp_factor = 1 + (data.gdp_growth - 2) * 0.05  # Assume 2% as baseline GDP growth
    population_factor = 1 + (data.population / 1e10) * 0.1
    energy_factor = 1 + (data.energy_demand / 1e5) * 0.05
    
    historical_impact = base_impact * gdp_factor * population_factor * energy_factor
    return min(1.0, max(0.0, historical_impact))

def analyze_impact_trends(time_series_results: Dict[int, List[float]]) -> Dict[int, Dict[str, float]]:
    trend_analysis = {}
    for risk_id, projections in time_series_results.items():
        trend = np.polyfit(range(len(projections)), projections, 1)
        slope = trend[0]
        
        trend_analysis[risk_id] = {
            "slope": slope,
            "average_impact": np.mean(projections),
            "max_impact": max(projections),
            "min_impact": min(projections),
            "volatility": np.std(projections)
        }
    
    return trend_analysis

def identify_critical_periods(time_series_results: Dict[int, List[float]], threshold: float) -> Dict[int, List[int]]:
    critical_periods = {}
    for risk_id, projections in time_series_results.items():
        critical = [i for i, impact in enumerate(projections) if impact > threshold]
        if critical:
            critical_periods[risk_id] = critical
    return critical_periods

def forecast_cumulative_impact(time_series_results: Dict[int, List[float]]) -> List[float]:
    num_periods = len(next(iter(time_series_results.values())))
    cumulative_impact = [sum(risk_projections[i] for risk_projections in time_series_results.values())
                         for i in range(num_periods)]
    return cumulative_impact