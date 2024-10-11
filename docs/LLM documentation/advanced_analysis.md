# LLM Calls in Advanced Analysis

This document details the LLM calls made in the `src/risk_analysis/advanced_analysis.py` script.

## 1. LLM Risk Assessment

### Function: `llm_risk_assessment(risk: Risk, scenario: Scenario, company_industry: str) -> str`

#### Context
This function uses the LLM to provide a detailed analysis of how a specific risk's likelihood and impact may change under a given scenario.

#### Prompt Template
```
Analyze the following risk for the {company_industry} industry under the given scenario:

Risk: {risk.description}
Category: {risk.category}
Subcategory: {risk.subcategory}
Current likelihood: {risk.likelihood}
Current impact: {risk.impact}

Scenario: {scenario.name}
- Temperature increase: {scenario.temp_increase}°C
- Carbon price: ${scenario.carbon_price}/ton
- Renewable energy share: {scenario.renewable_energy * 100}%
- Policy stringency: {scenario.policy_stringency * 100}%
- Biodiversity loss: {scenario.biodiversity_loss * 100}%
- Ecosystem degradation: {scenario.ecosystem_degradation * 100}%
- Financial stability: {scenario.financial_stability * 100}%
- Supply chain disruption: {scenario.supply_chain_disruption * 100}%

Provide a detailed analysis of how this risk's likelihood and impact may change under the given scenario.
```

#### Simulated Call
```python
llm_risk_assessment(
    Risk(id=1, description="Increased frequency of extreme weather events", category="Physical", subcategory="Acute", likelihood=0.7, impact=0.8),
    Scenario(name="Net Zero 2050", temp_increase=1.5, carbon_price=250, renewable_energy=0.75, policy_stringency=0.9, biodiversity_loss=0.1, ecosystem_degradation=0.2, financial_stability=0.8, supply_chain_disruption=0.3),
    "Energy"
)
```

#### Simulated Response
```
In the Net Zero 2050 scenario for the Energy industry, the risk of increased frequency of extreme weather events is likely to see changes in both likelihood and impact:

Likelihood: The likelihood may slightly decrease from 0.7 to around 0.65. While the scenario aims to limit temperature increase to 1.5°C, which could help mitigate the frequency of extreme weather events, the inertia in the climate system means that some increase in these events is still likely in the near term.

Impact: The impact may increase from 0.8 to about 0.85. Although the scenario presents a more resilient economy with high financial stability (80%) and lower supply chain disruption (30%), the energy sector remains vulnerable to extreme weather events. The high renewable energy share (75%) might introduce new vulnerabilities to weather-dependent energy sources.

Key factors influencing this assessment:
1. The limited temperature increase helps contain the escalation of extreme weather events.
2. High policy stringency (90%) suggests better preparedness and adaptation measures.
3. Low biodiversity loss (10%) and ecosystem degradation (20%) may provide some natural buffers against extreme weather impacts.
4. The energy transition itself (high renewable share) may temporarily increase vulnerability to weather events.

Adaptability: The energy sector in this scenario shows high adaptability potential due to the strong policy framework and financial stability. Companies should focus on diversifying energy sources, improving grid resilience, and investing in weather prediction technologies to mitigate this risk.
```

#### Processing and Usage
The response is stored and used as part of the comprehensive analysis for each risk under each scenario. It contributes to the generation of risk narratives and executive insights.

## 2. Generate Risk Narratives

### Function: `generate_risk_narratives(risks: List[Risk], comprehensive_analysis: Dict[str, Dict[int, str]]) -> Dict[int, str]`

#### Context
This function generates narrative summaries for each risk across all scenarios, synthesizing the insights from the comprehensive analysis.

#### Prompt Template
```
As an expert in climate risk assessment and scenario analysis, create a concise narrative summary for the following risk across different scenarios. Focus on key trends, variations in impact and likelihood, and overarching mitigation strategies.

Risk: {risk_description}
Category: {risk_category}
Subcategory: {risk_subcategory}
Current likelihood: {risk_likelihood}
Current impact: {risk_impact}

Scenario Analyses:
{scenario_analyses}

Provide a concise narrative summary (about 200 words) that synthesizes the insights from all scenarios, highlighting key trends and strategic implications.
```

#### Simulated Call
```python
generate_risk_narratives(
    [Risk(id=1, description="Increased frequency of extreme weather events", category="Physical", subcategory="Acute", likelihood=0.7, impact=0.8)],
    {
        "Net Zero 2050": {1: "Analysis for Net Zero 2050..."},
        "Delayed Transition": {1: "Analysis for Delayed Transition..."},
        "Current Policies": {1: "Analysis for Current Policies..."}
    }
)
```

#### Simulated Response
```
The risk of increased frequency of extreme weather events shows varying trajectories across different climate scenarios, but remains a significant concern in all futures. In the Net Zero 2050 scenario, the risk's likelihood slightly decreases due to aggressive mitigation efforts, but its impact remains high as the energy sector transitions to more weather-dependent renewable sources. The Delayed Transition scenario sees both likelihood and impact increase, as delayed action exacerbates climate effects before mitigation strategies take hold. Under Current Policies, this risk becomes severe, with both likelihood and impact rising significantly due to unabated climate change.

Across all scenarios, the energy sector faces challenges in adapting infrastructure and operations to more frequent and intense weather events. Key strategic implications include the need for robust resilience planning, diversification of energy sources, and investment in climate-adaptive technologies. The variation in risk profiles across scenarios underscores the importance of flexible, adaptive strategies that can be adjusted as the global climate response evolves. Proactive measures taken now, such as strengthening infrastructure and improving early warning systems, will be beneficial in all future scenarios.
```

#### Processing and Usage
The generated narratives are stored in a dictionary with risk IDs as keys. These narratives provide a synthesized view of each risk across different scenarios and are included in the final report.

## 3. Generate Executive Insights

### Function: `generate_executive_insights(comprehensive_analysis: Dict[str, Dict[int, str]], risks: List[Risk]) -> str`

#### Context
This function generates high-level executive insights based on the comprehensive analysis of all risks across all scenarios.

#### Prompt Template
```
As a senior climate risk analyst, review the following comprehensive risk analyses across multiple scenarios for a company. Provide high-level executive insights focusing on:

1. Key overarching trends across scenarios
2. Most critical risks requiring immediate attention
3. Potential strategic opportunities arising from climate change
4. Recommendations for enhancing overall climate resilience

Analyses:
{all_analyses}

Provide a concise executive summary (about 400 words) with key insights and strategic recommendations.
```

#### Simulated Call
```python
generate_executive_insights(
    {
        "Net Zero 2050": {
            1: "Analysis for Risk 1 in Net Zero 2050...",
            2: "Analysis for Risk 2 in Net Zero 2050..."
        },
        "Delayed Transition": {
            1: "Analysis for Risk 1 in Delayed Transition...",
            2: "Analysis for Risk 2 in Delayed Transition..."
        }
    },
    [
        Risk(id=1, description="Increased frequency of extreme weather events", category="Physical", subcategory="Acute", likelihood=0.7, impact=0.8),
        Risk(id=2, description="Carbon pricing policies", category="Transition", subcategory="Policy", likelihood=0.6, impact=0.7)
    ]
)
```

#### Simulated Response
```
Executive Summary: Climate Risk Assessment Insights

Key Overarching Trends:
1. Physical risks, particularly extreme weather events, persist across all scenarios, requiring robust adaptation strategies.
2. Transition risks vary significantly between scenarios, with earlier action scenarios presenting near-term challenges but long-term opportunities.
3. The interconnectedness of risks is evident, with physical risks often triggering or exacerbating transition and liability risks.

Critical Risks Requiring Immediate Attention:
1. Increased frequency of extreme weather events: This physical risk shows high impact across all scenarios, threatening operations and infrastructure.
2. Carbon pricing policies: This transition risk varies greatly between scenarios but has potentially significant financial implications in all cases.

Strategic Opportunities:
1. Early adoption of low-carbon technologies could provide a competitive advantage, especially in delayed action scenarios.
2. Development of climate-resilient products and services presents new market opportunities.
3. Proactive engagement in policy discussions could help shape favorable regulatory environments.

Recommendations for Enhancing Climate Resilience:
1. Implement a robust climate scenario analysis process for strategic planning.
2. Diversify operations and supply chains to reduce vulnerability to localized climate impacts.
3. Invest in R&D for climate-adaptive technologies and low-carbon solutions.
4. Enhance disclosure and stakeholder communication on climate risks and opportunities.
5. Develop flexible strategies that can be adjusted as the global climate response evolves.

By addressing these critical risks, capitalizing on emerging opportunities, and enhancing overall resilience, the company can navigate the complex landscape of climate change while positioning itself for long-term success in a low-carbon future.
```

#### Processing and Usage
The executive insights are included in the final report, providing a high-level summary of key findings and recommendations for decision-makers.

## 4. Generate Mitigation Strategies

### Function: `generate_mitigation_strategies(risks: List[Risk], comprehensive_analysis: Dict[str, Dict[int, str]]) -> Dict[int, List[str]]`

#### Context
This function generates specific mitigation strategies for each risk based on the comprehensive analysis across all scenarios.

#### Prompt Template
```
As an expert in climate risk mitigation and adaptation strategies, analyze the following risk and its assessment across different scenarios:

Risk: {risk_description}
Category: {risk_category}
Subcategory: {risk_subcategory}

Scenario Analyses:
{scenario_analyses}

Based on this information, please provide:

1. 3-5 concrete mitigation strategies that address this risk across multiple scenarios.
2. For each strategy, briefly explain its potential effectiveness and any challenges in implementation.
3. Prioritize these strategies based on their potential impact and feasibility.

Please number each strategy and provide a concise explanation for each.
```

#### Simulated Call
```python
generate_mitigation_strategies(
    [Risk(id=1, description="Increased frequency of extreme weather events", category="Physical", subcategory="Acute", likelihood=0.7, impact=0.8)],
    {
        "Net Zero 2050": {1: "Analysis for Net Zero 2050..."},
        "Delayed Transition": {1: "Analysis for Delayed Transition..."},
        "Current Policies": {1: "Analysis for Current Policies..."}
    }
)
```

#### Simulated Response
```
Mitigation Strategies for Increased Frequency of Extreme Weather Events:

1. Enhance Infrastructure Resilience:
   - Implement climate-proofing measures for existing infrastructure.
   - Design new facilities with extreme weather resilience in mind.
   Effectiveness: High, as it directly addresses the physical impact of extreme weather.
   Challenges: High upfront costs and potential disruptions during implementation.

2. Diversify Supply Chains:
   - Establish multiple suppliers in different geographic locations.
   - Develop alternative transportation routes and methods.
   Effectiveness: Medium to high, reduces vulnerability to localized extreme weather events.
   Challenges: Increased complexity in supply chain management and potential cost increases.

3. Implement Advanced Early Warning Systems:
   - Invest in or partner with weather forecasting services for tailored predictions.
   - Develop protocols for rapid response to weather warnings.
   Effectiveness: Medium, improves preparedness and reduces potential impacts.
   Challenges: Requires ongoing investment and may not prevent all impacts.

4. Develop Business Continuity Plans:
   - Create comprehensive plans for various extreme weather scenarios.
   - Regularly conduct drills and update plans based on new information.
   Effectiveness: Medium to high, ensures faster recovery and reduces downtime.
   Challenges: Requires significant time investment and regular updates.

5. Invest in Climate-Adaptive Technologies:
   - Research and implement technologies that can operate in extreme conditions.
   - Explore energy storage solutions to maintain operations during outages.
   Effectiveness: High, particularly in long-term adaptation to changing climate conditions.
   Challenges: High upfront costs and potential technological uncertainties.

Prioritized order: 1, 3, 4, 2, 5 - based on a balance of immediate impact, feasibility, and long-term effectiveness.
```

#### Processing and Usage
The generated mitigation strategies are stored in a dictionary with risk IDs as keys. These strategies are included in the final report, providing actionable recommendations for addressing each identified risk.