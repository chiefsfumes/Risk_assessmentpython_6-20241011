import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import numpy as np
import networkx as nx
import os
from src.models import Risk, RiskInteraction, SimulationResult
from src.config import OUTPUT_DIR, VIZ_DPI, HEATMAP_CMAP, TIME_SERIES_HORIZON

def generate_visualizations(risks: List[Risk], risk_interactions: List[RiskInteraction], 
                            simulation_results: Dict[str, Dict[int, SimulationResult]],
                            sensitivity_results: Dict[str, Dict[str, float]],
                            time_series_results: Dict[int, List[float]]):
    risk_matrix(risks)
    interaction_heatmap(risks, risk_interactions)
    interaction_network(risks, risk_interactions)
    monte_carlo_results(simulation_results)
    sensitivity_analysis_heatmap(sensitivity_results)
    time_series_projection(risks, time_series_results)

def risk_matrix(risks: List[Risk]):
    plt.figure(figsize=(12, 10))
    categories = set(risk.category for risk in risks)
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    for risk in risks:
        plt.scatter(risk.likelihood, risk.impact, s=100, c=[color_map[risk.category]], alpha=0.7)
        plt.annotate(risk.id, (risk.likelihood, risk.impact), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Likelihood')
    plt.ylabel('Impact')
    plt.title('Risk Matrix')
    plt.colorbar(plt.cm.ScalarMappable(cmap='tab10'), label='Risk Category', ticks=[])
    plt.savefig(os.path.join(OUTPUT_DIR, 'risk_matrix.png'), dpi=VIZ_DPI)
    plt.close()

def interaction_heatmap(risks: List[Risk], risk_interactions: List[RiskInteraction]):
    n = len(risks)
    interaction_matrix = np.zeros((n, n))
    for interaction in risk_interactions:
        i = next(index for index, risk in enumerate(risks) if risk.id == interaction.risk1_id)
        j = next(index for index, risk in enumerate(risks) if risk.id == interaction.risk2_id)
        interaction_matrix[i, j] = interaction_matrix[j, i] = interaction.interaction_score
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(interaction_matrix, annot=True, cmap=HEATMAP_CMAP, xticklabels=[r.id for r in risks], yticklabels=[r.id for r in risks])
    plt.title('Risk Interaction Heatmap')
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_heatmap.png'), dpi=VIZ_DPI)
    plt.close()

def interaction_network(risks: List[Risk], risk_interactions: List[RiskInteraction]):
    G = nx.Graph()
    for risk in risks:
        G.add_node(risk.id, category=risk.category)
    for interaction in risk_interactions:
        if interaction.interaction_score > 0.5:  # Only show strong interactions
            G.add_edge(interaction.risk1_id, interaction.risk2_id, weight=interaction.interaction_score)
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[color_map[G.nodes[node]['category']] for node in G.nodes()],
            node_size=1000, font_size=8, font_weight='bold', edge_color='gray', width=[G[u][v]['weight'] * 2 for u, v in G.edges()])
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()})
    plt.title('Risk Interaction Network')
    plt.savefig(os.path.join(OUTPUT_DIR, 'interaction_network.png'), dpi=VIZ_DPI)
    plt.close()

def monte_carlo_results(simulation_results: Dict[str, Dict[int, SimulationResult]]):
    plt.figure(figsize=(16, 12))
    for scenario, results in simulation_results.items():
        for risk_id, sim_result in results.items():
            sns.kdeplot(sim_result.impact_distribution, label=f'Risk {risk_id} - {scenario}')
    plt.xlabel('Risk Impact')
    plt.ylabel('Density')
    plt.title('Monte Carlo Simulation Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'monte_carlo_results.png'), dpi=VIZ_DPI)
    plt.close()

def sensitivity_analysis_heatmap(sensitivity_results: Dict[str, Dict[str, float]]):
    plt.figure(figsize=(14, 10))
    sensitivity_df = pd.DataFrame(sensitivity_results).T
    sns.heatmap(sensitivity_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Sensitivity Analysis Heatmap')
    plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_analysis_heatmap.png'), dpi=VIZ_DPI)
    plt.close()

def time_series_projection(risks: List[Risk], time_series_results: Dict[int, List[float]]):
    plt.figure(figsize=(16, 12))
    for risk_id, projections in time_series_results.items():
        risk = next(r for r in risks if r.id == risk_id)
        plt.plot(range(1, TIME_SERIES_HORIZON + 1), projections, label=f'Risk {risk_id} ({risk.category})')
    plt.xlabel('Years into the future')
    plt.ylabel('Projected Impact')
    plt.title('Time Series Projection of Risk Impacts')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_series_projection.png'), dpi=VIZ_DPI)
    plt.close()