import pytest
import os
from src.visualization import generate_visualizations
from src.models import Risk, RiskInteraction, SimulationResult

@pytest.fixture
def sample_data():
    risks = [
        Risk(id=1, description="Risk 1", category="Physical", likelihood=0.8, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Risk 2", category="Transition", likelihood=0.6, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
    ]
    risk_interactions = [RiskInteraction(risk1_id=1, risk2_id=2, interaction_score=0.5, interaction_type="Moderate")]
    simulation_results = {
        "Scenario1": {
            1: SimulationResult(1, "Scenario1", [0.9, 0.91, 0.89], [0.8, 0.81, 0.79]),
            2: SimulationResult(2, "Scenario1", [0.7, 0.71, 0.69], [0.6, 0.61, 0.59])
        }
    }
    sensitivity_results = {
        "Scenario1": {
            "temp_increase": 0.1,
            "carbon_price": 0.2
        }
    }
    time_series_results = {
        1: [0.9, 0.91, 0.92, 0.93, 0.94],
        2: [0.7, 0.71, 0.72, 0.73, 0.74]
    }
    
    return {
        "risks": risks,
        "risk_interactions": risk_interactions,
        "simulation_results": simulation_results,
        "sensitivity_results": sensitivity_results,
        "time_series_results": time_series_results
    }

def test_generate_visualizations(sample_data, tmp_path):
    os.environ['OUTPUT_DIR'] = str(tmp_path)
    generate_visualizations(
        sample_data["risks"],
        sample_data["risk_interactions"],
        sample_data["simulation_results"],
        sample_data["sensitivity_results"],
        sample_data["time_series_results"]
    )
    
    assert os.path.exists(tmp_path / "risk_matrix.png")
    assert os.path.exists(tmp_path / "interaction_heatmap.png")
    assert os.path.exists(tmp_path / "interaction_network.png")
    assert os.path.exists(tmp_path / "monte_carlo_results.png")
    assert os.path.exists(tmp_path / "sensitivity_analysis_heatmap.png")
    assert os.path.exists(tmp_path / "time_series_projection.png")