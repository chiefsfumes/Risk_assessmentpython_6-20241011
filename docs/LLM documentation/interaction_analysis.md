# LLM Calls in Interaction Analysis

This document details the LLM calls made in the `src/risk_analysis/interaction_analysis.py` script.

## 1. Analyze Risk Interactions

### Function: `analyze_risk_interactions(risks: List[Risk]) -> List[RiskInteraction]`

#### Context
This function uses the LLM to analyze potential interactions between pairs of risks, determining the strength and nature of their relationship.

#### Prompt Template
```
As an expert in climate risk assessment and risk interactions, analyze the potential interaction between the following two risks:

Risk 1: {risk1_description}
Category: {risk1_category}
Subcategory: {risk1_subcategory}

Risk 2: {risk2_description}
Category: {risk2_category}
Subcategory: {risk2_subcategory}

Please provide:
1. A brief explanation of how these risks might interact or influence each other.
2. An assessment of the potential compounding effects if both risks materialize simultaneously.
3. Any potential mitigating factors that could reduce the combined impact of these risks.
4. A suggested interaction score on a scale of 0 (no interaction) to 1 (strong interaction), with a brief justification.

Structure your response with clear headings for each point.
```

#### Simulated Call
```python
analyze_risk_interactions([
    Risk(id=1, description="Increased frequency of extreme weather events", category="Physical", subcategory="Acute", likelihood=0.7, impact=0.8),
    Risk(id=2, description="Carbon pricing policies", category="Transition", subcategory="Policy", likelihood=0.6, impact=0.7)
])
```

#### Simulated Response
```
1. Interaction Explanation:
The increased frequency of extreme weather events (Risk 1) and carbon pricing policies (Risk 2) have a complex interaction. Extreme weather events can disrupt energy infrastructure and operations, potentially leading to increased emissions in the short term (e.g., use of backup generators). This could result in higher carbon costs under strict pricing policies. Conversely, carbon pricing policies may drive investment in resilient, low-carbon infrastructure, which could help mitigate the impacts of extreme weather events in the long term.

2. Compounding Effects:
If both risks materialize simultaneously, the compounding effects could be significant. Extreme weather events could damage energy infrastructure, leading to increased emissions and higher carbon costs. This financial strain could reduce resources available for climate adaptation, creating a negative feedback loop. Additionally, public perception of the energy industry could worsen if it's seen as both vulnerable to climate impacts and a major contributor to emissions.

3. Mitigating Factors:
- Investment in climate-resilient, low-carbon technologies could address both risks simultaneously.
- Diversification of energy sources, including distributed renewable energy, could reduce vulnerability to both physical impacts and carbon pricing.
- Advanced weather forecasting and risk management systems could help minimize the impact of extreme weather on operations and emissions.
- Policy mechanisms that allow for flexibility in carbon pricing during extreme events could help manage short-term compounding effects.

4. Interaction Score:
Suggested interaction score: 0.8 (Strong interaction)

Justification: The interaction between these risks is strong due to their potential to exacerbate each other's impacts and the significant implications for the energy sector's operations, finances, and strategic planning. The score reflects the direct influence each risk has on the other's likelihood and impact, as well as the potential for compounding effects. However, it's not a perfect 1.0 as there are mitigating factors and potential long-term synergies in addressing both risks simultaneously.
```

#### Processing and Usage
The response is processed to extract the interaction score (0.8 in this case). The interaction type is determined based on the score (e.g., "Strong" for scores ≥ 0.7). A `RiskInteraction` object is created with this information and added to the list of interactions. This data is used to build the risk network and inform further analysis of risk relationships and potential cascading effects.

## 2. Generate Risk Interaction Summary

### Function: `generate_risk_interaction_summary(interactions: List[RiskInteraction], central_risks: Dict[int, float], clusters: Dict[int, int]) -> str`

#### Context
This function generates a summary of the risk interaction analysis, including insights on critical interactions, central risks, and risk clustering.

#### Prompt Template
```
Summarize the key findings from the risk interaction analysis:

1. Top 5 strongest interactions:
{top_5_interactions}

2. Top 3 central risks:
{top_3_central_risks}

3. Risk clusters:
{risk_clusters}

Provide a concise summary of:
1. The most critical risk interactions and their potential implications
2. The role of central risks in the overall risk landscape
3. Insights from the risk clustering and what it reveals about the company's risk profile
4. Recommendations for risk management based on these findings
```

#### Simulated Call
```python
generate_risk_interaction_summary(
    [RiskInteraction(risk1_id=1, risk2_id=2, interaction_score=0.8, interaction_type="Strong"),
     RiskInteraction(risk1_id=1, risk2_id=3, interaction_score=0.6, interaction_type="Moderate"),
     RiskInteraction(risk1_id=2, risk2_id=3, interaction_score=0.7, interaction_type="Strong")],
    {1: 0.9, 2: 0.8, 3: 0.7},
    {1: 0, 2: 0, 3: 1}
)
```

#### Simulated Response
```
Risk Interaction Analysis Summary

1. Critical Risk Interactions:
The analysis reveals strong interactions between several key risks, particularly between extreme weather events, carbon pricing policies, and supply chain disruptions. These interactions suggest potential compounding effects that could significantly amplify the overall risk exposure. For instance, extreme weather events could disrupt operations and supply chains, leading to increased emissions and higher carbon costs under strict pricing policies. This interconnectedness underscores the need for integrated risk management strategies that address multiple risks simultaneously.

2. Role of Central Risks:
The most central risks, including extreme weather events and carbon pricing policies, act as key nodes in the risk network. Their high centrality indicates that they have wide-ranging impacts across various aspects of the business. These risks should be prioritized in risk management efforts as addressing them effectively could have cascading positive effects on other connected risks. However, their central position also means that if these risks materialize, they could trigger a series of related risk events.

3. Risk Clustering Insights:
The risk clustering reveals two main groups of interconnected risks. The first cluster, containing extreme weather events and carbon pricing policies, represents closely linked physical and transition risks. The second cluster, including supply chain disruptions, suggests a group of operational and market-related risks. This clustering indicates that the company's risk profile is characterized by a mix of direct climate-related risks and indirect, climate-influenced business risks. It highlights the need for a multi-faceted approach to climate risk management that addresses both immediate physical threats and longer-term transition challenges.

4. Risk Management Recommendations:
a) Develop integrated mitigation strategies that address multiple interconnected risks simultaneously, focusing on the most central risks as priority areas.
b) Enhance scenario planning and stress testing capabilities to account for the potential compounding effects of interacting risks.
c) Implement a cross-functional risk management approach that breaks down silos between different risk categories (e.g., physical, transition, operational).
d) Invest in improving the resilience of critical systems and processes that are exposed to multiple high-impact risks.
e) Regularly reassess the risk landscape and interactions, as the dynamics may change with evolving climate conditions and policy environments.
f) Develop early warning systems and response plans that consider the cascading effects of central risks.

By addressing these critical risk interactions and focusing on central risks, the company can enhance its overall resilience to climate-related challenges and position itself more effectively in a changing business environment.
```

#### Processing and Usage
This summary is included in the final report, providing decision-makers with a high-level overview of the risk interaction analysis. It helps in understanding the complex relationships between different risks and informs the development of comprehensive risk management strategies.