import pytest
from src.risk_analysis.pestel_analysis import perform_pestel_analysis, categorize_risk_pestel, enrich_pestel_with_external_data
from src.models import Risk, ExternalData

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="New environmental regulations", category="Policy", likelihood=0.8, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Economic downturn", category="Market", likelihood=0.6, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
        Risk(id=3, description="Technological disruption", category="Technology", likelihood=0.7, impact=0.8, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
    ]

@pytest.fixture
def sample_external_data():
    return {
        "2020": ExternalData(year=2020, gdp_growth=2.3, population=7794798739, energy_demand=173340, carbon_price=35, renewable_energy_share=0.29, biodiversity_index=0.7, deforestation_rate=0.5),
        "2021": ExternalData(year=2021, gdp_growth=5.7, population=7874965732, energy_demand=176431, carbon_price=40, renewable_energy_share=0.31, biodiversity_index=0.68, deforestation_rate=0.48),
    }

def test_perform_pestel_analysis(sample_risks, sample_external_data):
    pestel_categories = perform_pestel_analysis(sample_risks, sample_external_data)
    
    assert set(pestel_categories.keys()) == {"Political", "Economic", "Social", "Technological", "Environmental", "Legal"}
    assert len(pestel_categories["Political"]) > 0
    assert len(pestel_categories["Economic"]) > 0
    assert len(pestel_categories["Technological"]) > 0
    
    for category, risks in pestel_categories.items():
        for risk in risks:
            assert "risk_id" in risk
            assert "description" in risk
            assert "impact" in risk

def test_categorize_risk_pestel():
    risk1 = Risk(id=1, description="New environmental regulations", category="Policy", likelihood=0.8, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    risk2 = Risk(id=2, description="Economic downturn", category="Market", likelihood=0.6, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    risk3 = Risk(id=3, description="Technological disruption", category="Technology", likelihood=0.7, impact=0.8, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")
    
    assert categorize_risk_pestel(risk1) == "Political"
    assert categorize_risk_pestel(risk2) == "Economic"
    assert categorize_risk_pestel(risk3) == "Technological"

def test_enrich_pestel_with_external_data(sample_external_data):
    pestel_categories = {
        "Political": [],
        "Economic": [],
        "Social": [],
        "Technological": [],
        "Environmental": [],
        "Legal": []
    }
    
    enrich_pestel_with_external_data(pestel_categories, sample_external_data)
    
    assert any("GDP Growth" in factor["factor"] for factor in pestel_categories["Economic"])
    assert any("Population" in factor["factor"] for factor in pestel_categories["Social"])
    assert any("Energy Demand" in factor["factor"] for factor in pestel_categories["Environmental"])