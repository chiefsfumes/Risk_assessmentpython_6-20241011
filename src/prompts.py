# Climate Risk Assessment Prompts

RISK_ASSESSMENT_PROMPT = """
As an expert in climate risk assessment for the {industry} sector, analyze the following risk statement:

Risk: {risk_description}
Category: {risk_category}
Subcategory: {risk_subcategory}
Current likelihood: {risk_likelihood}
Current impact: {risk_impact}
Time horizon: {risk_time_horizon}

Scenario: {scenario_name}
- Temperature increase: {temp_increase}°C
- Carbon price: ${carbon_price}/ton
- Renewable energy share: {renewable_energy}%
- Policy stringency: {policy_stringency}%
- Biodiversity loss: {biodiversity_loss}%
- Ecosystem degradation: {ecosystem_degradation}%
- Financial stability: {financial_stability}%
- Supply chain disruption: {supply_chain_disruption}%

Provide a detailed analysis addressing the following points:
1. How does this risk's likelihood and impact change under the given scenario?
2. What are the potential financial implications for the company over the next 5 years?
3. Are there any emerging opportunities related to this risk in this scenario?
4. What additional challenges might arise from this risk in this specific context?
5. Suggest 2-3 possible mitigation strategies tailored to this scenario.

Please structure your response with clear headings for each point.
"""

RISK_NARRATIVE_PROMPT = """
As an expert in climate risk assessment, create a concise narrative summary for the following risk across different scenarios. Focus on key trends, variations in impact and likelihood, and overarching mitigation strategies.

Risk: {risk_description}
Category: {risk_category}
Subcategory: {risk_subcategory}
Current likelihood: {risk_likelihood}
Current impact: {risk_impact}
Time horizon: {risk_time_horizon}

Scenario Analyses:
{scenario_analyses}

Provide a concise narrative summary (about 200 words) that synthesizes the insights from all scenarios, highlighting key trends and strategic implications.
"""

EXECUTIVE_INSIGHTS_PROMPT = """
As a senior climate risk analyst, review the following comprehensive risk analyses across multiple scenarios for a company. Provide high-level executive insights focusing on:

1. Key overarching trends across scenarios
2. Most critical risks requiring immediate attention
3. Potential strategic opportunities arising from climate change
4. Recommendations for enhancing overall climate resilience

Analyses:
{all_analyses}

Provide a concise executive summary (about 400 words) with key insights and strategic recommendations.
"""

INTERACTION_ANALYSIS_PROMPT = """
As an expert in climate risk assessment, analyze the potential interaction between the following two risks:

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
"""

SYSTEMIC_RISK_PROMPT = """
As an expert in systemic risk analysis, evaluate the following risk in the context of broader systems:

Risk: {risk_description}
Category: {risk_category}
Subcategory: {risk_subcategory}

Company Industry: {company_industry}
Key Dependencies: {key_dependencies}

Please provide an analysis addressing the following points:
1. How might this risk contribute to or be affected by systemic vulnerabilities in the {company_industry} sector?
2. Identify potential trigger points or critical thresholds related to this risk that could lead to cascading effects across systems.
3. Assess the company's potential role in mitigating this systemic risk, considering its position in the industry.
4. Suggest collaborative initiatives or policy engagements that could help address this risk at a systemic level.

Structure your response with clear headings for each point and provide concise, actionable insights.
"""

MITIGATION_STRATEGY_PROMPT = """
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
"""