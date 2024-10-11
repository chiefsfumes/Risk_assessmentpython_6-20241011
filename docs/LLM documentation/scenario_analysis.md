# LLM Calls in Scenario Analysis

This document details the LLM calls made in the `src/risk_analysis/scenario_analysis.py` script.

## 1. Generate Scenario Narratives

### Function: `generate_scenario_narratives(scenarios: Dict[str, Scenario]) -> Dict[str, str]`

#### Context
This function uses the LLM to generate detailed narratives for each climate scenario, providing a rich description of the potential future state under different climate pathways.

#### Prompt Template
```
Generate a detailed narrative for the following climate scenario:

Scenario: {scenario_name}
- Temperature increase: {scenario.temp_increase}°C
- Carbon price: ${scenario.carbon_price}/ton
- Renewable energy share: {scenario.renewable_energy * 100}%
- Policy stringency: {scenario.policy_stringency * 100}%
- Biodiversity loss: {scenario.biodiversity_loss * 100}%
- Ecosystem degradation: {scenario.ecosystem_degradation * 100}%
- Financial stability: {scenario.financial_stability * 100}%
- Supply chain disruption: {scenario.supply_chain_disruption * 100}%

Provide a compelling narrative that describes the overall state of the world in this scenario, including key challenges and opportunities for businesses, major societal and environmental changes, potential technological advancements or setbacks, and the general economic landscape.
```

#### Simulated Call
```python
generate_scenario_narratives({
    "Net Zero 2050": Scenario(
        name="Net Zero 2050",
        temp_increase=1.5,
        carbon_price=250,
        renewable_energy=0.75,
        policy_stringency=0.9,
        biodiversity_loss=0.1,
        ecosystem_degradation=0.2,
        financial_stability=0.8,
        supply_chain_disruption=0.3
    )
})
```

#### Simulated Response
```
Net Zero 2050 Scenario Narrative

In the Net Zero 2050 scenario, the world has successfully limited global temperature increase to 1.5°C above pre-industrial levels, marking a significant achievement in the fight against climate change. This future is characterized by rapid and far-reaching transitions across all sectors of the global economy.

Economic Landscape:
The global economy has undergone a profound transformation. With a high carbon price of $250 per ton, carbon-intensive industries have either adapted or faced decline. This has led to a boom in clean technologies and green jobs, offsetting losses in traditional sectors. The high level of financial stability (80%) indicates that the transition, while challenging, has been managed without triggering a major economic crisis. However, moderate supply chain disruptions (30%) persist as the global economy continues to adapt to new patterns of production and consumption.

Energy Sector:
Renewable energy dominates the energy landscape, providing 75% of global energy needs. This shift has democratized energy production, with many communities and businesses generating their own power. Traditional energy companies have either transformed into clean energy providers or become obsolete. The remaining fossil fuel use is largely coupled with carbon capture and storage technologies.

Policy and Governance:
Stringent policies (90%) have been crucial in driving the transition. Governments worldwide have implemented comprehensive climate action plans, including carbon pricing, strict emissions standards, and substantial investments in green infrastructure. International cooperation on climate issues has strengthened, with global carbon markets and technology transfer programs playing key roles.

Environment and Biodiversity:
The limited temperature increase has helped contain biodiversity loss (10%) and ecosystem degradation (20%). While some damage has occurred, many ecosystems show signs of recovery. Reforestation and rewilding projects have gained momentum, supported by carbon offset programs. Ocean health has improved, with marine protected areas expanding significantly.

Societal Changes:
Society has embraced a more sustainable lifestyle. Cities have been redesigned for climate resilience and low-carbon living, featuring extensive green spaces, efficient public transportation, and energy-positive buildings. There's a strong emphasis on circular economy principles, with recycling and upcycling becoming the norm.

Technological Advancements:
Innovation has accelerated across multiple fronts. Energy storage technologies have made great strides, solving the intermittency issues of renewable energy. Green hydrogen has become a viable fuel for heavy industry and long-distance transportation. Carbon removal technologies, both natural and engineered, have scaled up significantly. Artificial intelligence plays a crucial role in optimizing energy use and predicting climate patterns.

Challenges:
Despite overall success, challenges remain. Some regions struggle with the pace of transition, particularly those historically dependent on fossil fuels. Climate justice issues persist, with ongoing debates about the fair distribution of transition costs and benefits. Adaptation remains crucial, as some climate change impacts are already locked in.

Opportunities for Businesses:
Companies that have embraced sustainability and clean technologies are thriving. There are vast opportunities in renewable energy, energy efficiency, sustainable agriculture, and climate adaptation technologies. The circular economy has opened new markets for innovative products and services. Businesses with strong environmental, social, and governance (ESG) profiles attract more investment and customer loyalty.

In conclusion, the Net Zero 2050 scenario presents a world that has risen to the challenge of climate change through innovation, policy action, and societal change. While significant hurdles have been overcome, the global community remains vigilant, continuously working to maintain and improve upon the gains made in creating a sustainable, low-carbon world.
```

#### Processing and Usage
The generated narrative is stored in a dictionary with the scenario name as the key. These narratives are included in the final report, providing stakeholders with a vivid description of each potential future scenario. They help in understanding the context of the risk assessments and inform strategic planning discussions.