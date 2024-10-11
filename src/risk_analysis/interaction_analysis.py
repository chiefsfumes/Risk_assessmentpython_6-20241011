from typing import List, Dict, Tuple
from src.models import Risk, RiskInteraction
from src.config import LLM_MODEL, LLM_API_KEY
from src.prompts import INTERACTION_ANALYSIS_PROMPT
import openai
import networkx as nx
import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

openai.api_key = LLM_API_KEY

def analyze_risk_interactions(risks: List[Risk]) -> List[RiskInteraction]:
    interactions = []
    for i, risk1 in enumerate(risks):
        for j, risk2 in enumerate(risks[i+1:], start=i+1):
            prompt = INTERACTION_ANALYSIS_PROMPT.format(
                risk1_description=risk1.description,
                risk1_category=risk1.category,
                risk1_subcategory=risk1.subcategory,
                risk2_description=risk2.description,
                risk2_category=risk2.category,
                risk2_subcategory=risk2.subcategory
            )

            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in climate risk assessment and risk interactions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )

            analysis = response.choices[0].message['content']
            interaction_score = extract_interaction_score(analysis)
            interaction_type = determine_interaction_type(interaction_score)
            interactions.append(RiskInteraction(risk1.id, risk2.id, interaction_score, interaction_type, analysis))

    return interactions

def extract_interaction_score(analysis: str) -> float:
    import re
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", analysis)
    return float(numbers[-1]) if numbers else 0.5

def determine_interaction_type(score: float) -> str:
    if score < 0.3:
        return "Weak"
    elif score < 0.7:
        return "Moderate"
    else:
        return "Strong"

def build_risk_network(risks: List[Risk], interactions: List[RiskInteraction]) -> nx.Graph:
    G = nx.Graph()
    for risk in risks:
        G.add_node(risk.id, **risk.__dict__)
    for interaction in interactions:
        G.add_edge(interaction.risk1_id, interaction.risk2_id, 
                   weight=interaction.interaction_score, 
                   type=interaction.interaction_type)
    return G

def identify_central_risks(G: nx.Graph) -> Dict[int, float]:
    centrality_measures = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight='weight'),
        "eigenvector": nx.eigenvector_centrality(G, weight='weight'),
        "pagerank": nx.pagerank(G, weight='weight')
    }
    
    combined_centrality = {}
    for node in G.nodes():
        combined_centrality[node] = np.mean([measure[node] for measure in centrality_measures.values()])
    
    return combined_centrality

def detect_risk_clusters(G: nx.Graph, num_clusters: int = 3) -> Dict[int, int]:
    # Convert graph to a matrix representation
    adj_matrix = nx.to_numpy_array(G)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(adj_matrix)
    
    return {node: label for node, label in zip(G.nodes(), cluster_labels)}

def analyze_risk_cascades(G: nx.Graph, initial_risks: List[int], threshold: float = 0.5, max_steps: int = 10) -> Dict[int, List[float]]:
    cascade_progression = {risk: [1.0] for risk in initial_risks}
    for _ in range(max_steps):
        new_activations = {}
        for node in G.nodes():
            if node not in cascade_progression:
                neighbor_influence = sum(cascade_progression.get(neighbor, [0])[-1] * G[node][neighbor]['weight']
                                         for neighbor in G.neighbors(node))
                if neighbor_influence > threshold:
                    new_activations[node] = neighbor_influence
        
        if not new_activations:
            break
        
        for node, activation in new_activations.items():
            cascade_progression[node] = [0.0] * (len(next(iter(cascade_progression.values()))) - 1) + [activation]
        
        for progression in cascade_progression.values():
            progression.append(progression[-1])
    
    return cascade_progression

def calculate_risk_correlations(risks: List[Risk], simulation_results: Dict[str, Dict[int, List[float]]]) -> Dict[Tuple[int, int], float]:
    correlations = {}
    risk_ids = [risk.id for risk in risks]
    for i, risk1_id in enumerate(risk_ids):
        for risk2_id in risk_ids[i+1:]:
            correlation, _ = pearsonr(simulation_results[risk1_id], simulation_results[risk2_id])
            correlations[(risk1_id, risk2_id)] = correlation
    return correlations

def identify_risk_feedback_loops(G: nx.Graph) -> List[List[int]]:
    feedback_loops = list(nx.simple_cycles(G))
    return [loop for loop in feedback_loops if len(loop) > 2]

def analyze_network_resilience(G: nx.Graph) -> Dict[str, float]:
    resilience_metrics = {
        "average_clustering": nx.average_clustering(G),
        "average_shortest_path_length": nx.average_shortest_path_length(G, weight='weight'),
        "graph_density": nx.density(G),
        "assortativity": nx.degree_assortativity_coefficient(G)
    }
    return resilience_metrics

def generate_risk_interaction_summary(interactions: List[RiskInteraction], central_risks: Dict[int, float], clusters: Dict[int, int]) -> str:
    prompt = f"""
    Summarize the key findings from the risk interaction analysis:

    1. Top 5 strongest interactions:
    {', '.join([f"Risk {i.risk1_id} - Risk {i.risk2_id} (Score: {i.interaction_score:.2f})" for i in sorted(interactions, key=lambda x: x.interaction_score, reverse=True)[:5]])}

    2. Top 3 central risks:
    {', '.join([f"Risk {risk_id} (Centrality: {centrality:.2f})" for risk_id, centrality in sorted(central_risks.items(), key=lambda x: x[1], reverse=True)[:3]])}

    3. Risk clusters:
    {', '.join([f"Cluster {cluster}: {', '.join([str(risk_id) for risk_id, c in clusters.items() if c == cluster])}" for cluster in set(clusters.values())])}

    Provide a concise summary of:
    1. The most critical risk interactions and their potential implications
    2. The role of central risks in the overall risk landscape
    3. Insights from the risk clustering and what it reveals about the company's risk profile
    4. Recommendations for risk management based on these findings
    """

    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in climate risk assessment and network analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message['content']