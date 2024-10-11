from dataclasses import dataclass
from typing import List, Dict
from pydantic import BaseModel, Field, validator

class Risk(BaseModel):
    id: int
    description: str
    category: str
    subcategory: str
    tertiary_category: str
    likelihood: float
    impact: float
    time_horizon: str
    industry_specific: bool
    sasb_category: str

    @validator('likelihood', 'impact')
    def check_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Probability must be between 0 and 1')
        return v

    def to_dict(self) -> Dict:
        return self.dict()

class ExternalData(BaseModel):
    year: int
    gdp_growth: float
    population: int
    energy_demand: float
    carbon_price: float
    renewable_energy_share: float
    biodiversity_index: float
    deforestation_rate: float

@dataclass
class RiskInteraction:
    risk1_id: int
    risk2_id: int
    interaction_score: float
    interaction_type: str

@dataclass
class SimulationResult:
    risk_id: int
    scenario: str
    impact_distribution: List[float]
    likelihood_distribution: List[float]

class PESTELAnalysis(BaseModel):
    political: List[Dict[str, str]]
    economic: List[Dict[str, str]]
    social: List[Dict[str, str]]
    technological: List[Dict[str, str]]
    environmental: List[Dict[str, str]]
    legal: List[Dict[str, str]]

class SASBMaterialRisk(BaseModel):
    risk_id: int
    sasb_category: str
    description: str
    impact: float

    @validator('impact')
    def check_impact(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Impact must be between 0 and 1')
        return v

class SystemicRisk(BaseModel):
    risk_id: int
    description: str
    impact: float
    systemic_factor: str
    connected_risks: List[int]
    trigger_points: List[str]

    @validator('impact')
    def check_impact(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Impact must be between 0 and 1')
        return v