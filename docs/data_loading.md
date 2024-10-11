# Data Loading and Preprocessing

## Overview

The data loading and preprocessing module is responsible for importing risk data and external data into the system. It ensures that the data is properly formatted and ready for analysis.

## Key Functions

### `load_risk_data(file_path: str) -> List[Risk]`

This function loads risk data from a CSV file and converts it into a list of `Risk` objects.

#### Input
- `file_path`: A string representing the path to the CSV file containing risk data.

#### Output
- A list of `Risk` objects.

#### Example

```python
risks = load_risk_data('data/risk_data.csv')
```

#### Illustration

Input CSV:
```
id,description,category,likelihood,impact
1,"Increased frequency of extreme weather events","Physical Risk",0.8,0.9
2,"Transition to low-carbon technologies","Transition Risk",0.7,0.8
```

Output:
```python
[
    Risk(id=1, description="Increased frequency of extreme weather events", category="Physical Risk", likelihood=0.8, impact=0.9, ...),
    Risk(id=2, description="Transition to low-carbon technologies", category="Transition Risk", likelihood=0.7, impact=0.8, ...)
]
```

### `load_external_data(file_path: str) -> Dict[str, ExternalData]`

This function loads external data from a CSV file and converts it into a dictionary of `ExternalData` objects, keyed by year.

#### Input
- `file_path`: A string representing the path to the CSV file containing external data.

#### Output
- A dictionary with years as keys and `ExternalData` objects as values.

#### Example

```python
external_data = load_external_data('data/external_data.csv')
```

#### Illustration

Input CSV:
```
year,gdp_growth,population,energy_demand
2020,2.3,7794798739,173340
2021,5.7,7874965732,176431
```

Output:
```python
{
    "2020": ExternalData(year=2020, gdp_growth=2.3, population=7794798739, energy_demand=173340, ...),
    "2021": ExternalData(year=2021, gdp_growth=5.7, population=7874965732, energy_demand=176431, ...)
}
```

## Error Handling

Both functions include error handling for common issues:

- `FileNotFoundError`: Raised if the specified file does not exist.
- `pd.errors.EmptyDataError`: Raised if the CSV file is empty.
- `ValueError`: Raised if there are issues with data formatting or values.

## Data Models

The module uses two main data models:

1. `Risk`: Represents a climate risk with attributes such as id, description, category, likelihood, and impact.
2. `ExternalData`: Represents external factors for a given year, including GDP growth, population, energy demand, etc.

These models are defined in the `src/models.py` file and are used throughout the entire project for consistent data representation.