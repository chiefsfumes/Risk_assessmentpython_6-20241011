import pytest
from src.risk_analysis.sasb_integration import integrate_sasb_materiality
from src.models import Risk

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="GHG emissions from operations", category="Environmental", likelihood=0.8, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=True, sasb_category=""),
        Risk(id=2, description="Water management in water-stressed areas", category="Environmental", likelihood=0.6, impact=0.8, subcategory="", tertiary_category="", time_horizon="", industry_specific=True, sasb_category=""),
        Risk(id=3, description="Employee health and safety", category="Social", likelihood=0.7, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=True, sasb_category=""),
        Risk(id=4, description="Business ethics and transparency", category="Governance", likelihood=0.5, impact=0.8, subcategory="", tertiary_category="", time_horizon="", industry_specific=True, sasb_category=""),
    ]

def test_integrate_sasb_materiality(sample_risks):
    industry = "Energy"
    material_risks = integrate_sasb_materiality(sample_risks, industry)
    
    assert "GHG Emissions" in material_risks
    assert "Water Management" in material_risks
    assert "Employee Health & Safety" in material_risks
    assert "Business Ethics & Transparency" in material_risks
    
    assert len(material_risks["GHG Emissions"]) == 1
    assert len(material_risks["Water Management"]) == 1
    assert len(material_risks["Employee Health & Safety"]) == 1
    assert len(material_risks["Business Ethics & Transparency"]) == 1
    
    assert material_risks["GHG Emissions"][0].id == 1
    assert material_risks["Water Management"][0].id == 2
    assert material_risks["Employee Health & Safety"][0].id == 3
    assert material_risks["Business Ethics & Transparency"][0].id == 4

def test_integrate_sasb_materiality_non_material():
    risks = [
        Risk(id=1, description="Non-material risk", category="Other", likelihood=0.5, impact=0.5, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category=""),
    ]
    industry = "Energy"
    material_risks = integrate_sasb_materiality(risks, industry)
    
    assert all(len(category_risks) == 0 for category_risks in material_risks.values())

def test_integrate_sasb_materiality_multiple_categories():
    risks = [
        Risk(id=1, description="GHG emissions and water management", category="Environmental", likelihood=0.8, impact=0.9, subcategory="", tertiary_category="", time_horizon="", industry_specific=True, sasb_category=""),
    ]
    industry = "Energy"
    material_risks = integrate_sasb_materiality(risks, industry)
    
    assert len(material_risks["GHG Emissions"]) == 1
    assert len(material_risks["Water Management"]) == 1
    assert material_risks["GHG Emissions"][0].id == 1
    assert material_risks["Water Management"][0].id == 1

def test_integrate_sasb_materiality_different_industry():
    risks = [
        Risk(id=1, description="GHG emissions from operations", category="Environmental", likelihood=0.8, impact=0.7, subcategory="", tertiary_category="", time_horizon="", industry_specific=True, sasb_category=""),
    ]
    industry = "Technology"
    material_risks = integrate_sasb_materiality(risks, industry)
    
    assert all(len(category_risks) == 0 for category_risks in material_risks.values())