import pytest
from src.data_loader import load_risk_data, load_external_data
from src.models import Risk, ExternalData

def test_load_risk_data():
    risks = load_risk_data('data/risk_data.csv')
    assert len(risks) > 0
    assert isinstance(risks[0], Risk)
    assert risks[0].id == 1
    assert risks[0].description == "Increased frequency of extreme weather events"
    assert risks[0].category == "Physical Risk"
    assert risks[0].likelihood == 0.8
    assert risks[0].impact == 0.9

def test_load_external_data():
    external_data = load_external_data('data/external_data.csv')
    assert len(external_data) > 0
    assert isinstance(external_data['2020'], ExternalData)
    assert external_data['2020'].year == 2020
    assert external_data['2020'].gdp_growth == 2.3
    assert external_data['2020'].population == 7794798739
    assert external_data['2020'].energy_demand == 173340

def test_load_risk_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_risk_data('nonexistent_file.csv')

def test_load_external_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_external_data('nonexistent_file.csv')