import pytest
from src.risk_analysis.categorization import categorize_risks, prioritize_risks
from src.models import Risk

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Risk 1", category="Physical", likelihood=0.8, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Risk 2", category="Transition", likelihood=0.6, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
        Risk(id=3, description="Risk 3", category="Physical", likelihood=0.4, impact=0.5, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
    ]

def test_categorize_risks(sample_risks):
    categorized = categorize_risks(sample_risks)
    assert len(categorized) == 2
    assert len(categorized["Physical"]) == 2
    assert len(categorized["Transition"]) == 1

def test_prioritize_risks(sample_risks):
    prioritized = prioritize_risks(sample_risks)
    assert len(prioritized["High"]) == 1
    assert len(prioritized["Medium"]) == 1
    assert len(prioritized["Low"]) == 1
    assert prioritized["High"][0].id == 1
    assert prioritized["Medium"][0].id == 2
    assert prioritized["Low"][0].id == 3