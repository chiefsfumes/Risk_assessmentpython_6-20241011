import pandas as pd
from typing import List, Dict
from src.models import Risk, ExternalData

def load_risk_data(file_path: str) -> List[Risk]:
    try:
        df = pd.read_csv(file_path)
        risks = []
        for _, row in df.iterrows():
            try:
                risk = Risk(
                    id=row['id'],
                    description=row['description'],
                    category=row['category'],
                    subcategory=row.get('subcategory', ''),
                    tertiary_category=row.get('tertiary_category', ''),
                    likelihood=row['likelihood'],
                    impact=row['impact'],
                    time_horizon=row.get('time_horizon', ''),
                    industry_specific=row.get('industry_specific', False),
                    sasb_category=row.get('sasb_category', '')
                )
                risks.append(risk)
            except ValueError as e:
                print(f"Error processing row {row['id']}: {e}")
        return risks
    except FileNotFoundError:
        raise FileNotFoundError(f"Risk data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Risk data file is empty: {file_path}")

def load_external_data(file_path: str) -> Dict[str, ExternalData]:
    try:
        df = pd.read_csv(file_path)
        external_data = {}
        for _, row in df.iterrows():
            try:
                data = ExternalData(
                    year=row['year'],
                    gdp_growth=row['gdp_growth'],
                    population=row['population'],
                    energy_demand=row['energy_demand'],
                    carbon_price=row.get('carbon_price', 0),
                    renewable_energy_share=row.get('renewable_energy_share', 0),
                    biodiversity_index=row.get('biodiversity_index', 0),
                    deforestation_rate=row.get('deforestation_rate', 0)
                )
                external_data[str(row['year'])] = data
            except ValueError as e:
                print(f"Error processing row for year {row['year']}: {e}")
        return external_data
    except FileNotFoundError:
        raise FileNotFoundError(f"External data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"External data file is empty: {file_path}")