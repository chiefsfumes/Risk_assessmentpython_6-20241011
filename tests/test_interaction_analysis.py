import pytest
import networkx as nx
import numpy as np
from src.risk_analysis.interaction_analysis import (
    analyze_risk_interactions, build_risk_network, identify_central_risks,
    detect_risk_clusters, analyze_risk_cascades, calculate_risk_correlations,
    identify_risk_feedback_loops, analyze_network_resilience, generate_risk_interaction_summary
)
from src.models import Risk, RiskInteraction

@pytest.fixture
def sample_risks():
    return [
        Risk(id=1, description="Physical Risk 1", category="Physical", likelihood=0.7, impact=0.8, subcategory="Acute", tertiary_category="", time_horizon="Short-term", industry_specific=False, sasb_category=""),
        Risk(id=2, description="Transition Risk 1", category="Transition", likelihood=0.6, impact=0.7, subcategory="Policy", tertiary_category="", time_horizon="Medium-term", industry_specific=True, sasb_category="Energy"),
        Risk(id=3, description="Market Risk 1", category="Market", likelihood=0.5, impact=0.6, subcategory="Demand", tertiary_category="", time_horizon="Long-term", industry_specific=False, sasb_category=""),
        Risk(id=4, description="Physical Risk 2", category="Physical", likelihood=0.8, impact=0.9, subcategory="Chronic", tertiary_category="", time_horizon="Long-term", industry_specific=False, sasb_category=""),
    ]

@pytest.fixture
def sample_interactions(sample_risks):
    return [
        RiskInteraction(risk1_id=1, risk2_id=2, interaction_score=0.7, interaction_type="Strong"),
        RiskInteraction(risk1_id=1, risk2_id=3, interaction_score=0.4, interaction_type="Moderate"),
        RiskInteraction(risk1_id=2, risk2_id=3, interaction_score=0.6, interaction_type="Moderate"),
        RiskInteraction(risk1_id=2, risk2_id=4, interaction_score=0.8, interaction_type="Strong"),
        RiskInteraction(risk1_id=3, risk2_id=4, interaction_score=0.5, interaction_type="Moderate"),
    ]

def test_analyze_risk_interactions(sample_risks):
    interactions = analyze_risk_interactions(sample_risks)
    
    assert len(interactions) == (len(sample_risks) * (len(sample_risks) - 1)) // 2
    for interaction in interactions:
        assert isinstance(interaction, RiskInteraction)
        assert 0 <= interaction.interaction_score <= 1
        assert interaction.interaction_type in ["Weak", "Moderate", "Strong"]
    
    # Test that interactions are symmetric
    risk_pairs = set((min(i.risk1_id, i.risk2_id), max(i.risk1_id, i.risk2_id)) for i in interactions)
    assert len(risk_pairs) == len(interactions)

def test_build_risk_network(sample_risks, sample_interactions):
    G = build_risk_network(sample_risks, sample_interactions)
    
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == len(sample_risks)
    assert len(G.edges) == len(sample_interactions)
    for _, _, data in G.edges(data=True):
        assert "weight" in data
        assert "type" in data
    
    # Test that all risks are connected
    assert nx.is_connected(G)

def test_identify_central_risks(sample_risks, sample_interactions):
    G = build_risk_network(sample_risks, sample_interactions)
    central_risks = identify_central_risks(G)
    
    assert len(central_risks) == len(sample_risks)
    for risk_id, centrality in central_risks.items():
        assert 0 <= centrality <= 1
    
    # Test that the risk with most connections has highest centrality
    most_connected_risk = max(G.degree, key=lambda x: x[1])[0]
    assert most_connected_risk == max(central_risks, key=central_risks.get)

def test_detect_risk_clusters(sample_risks, sample_interactions):
    G = build_risk_network(sample_risks, sample_interactions)
    clusters = detect_risk_clusters(G, num_clusters=2)
    
    assert len(clusters) == len(sample_risks)
    assert set(clusters.values()) == {0, 1}  # Assuming 2 clusters
    
    # Test that strongly connected risks are in the same cluster
    strong_interactions = [i for i in sample_interactions if i.interaction_type == "Strong"]
    for interaction in strong_interactions:
        assert clusters[interaction.risk1_id] == clusters[interaction.risk2_id]

def test_analyze_risk_cascades(sample_risks, sample_interactions):
    G = build_risk_network(sample_risks, sample_interactions)
    initial_risks = [1, 2]
    cascade_progression = analyze_risk_cascades(G, initial_risks)
    
    assert len(cascade_progression) >= len(initial_risks)
    for risk_id, progression in cascade_progression.items():
        assert len(progression) > 0
        assert all(0 <= value <= 1 for value in progression)
    
    # Test that initial risks have highest initial values
    for risk_id in initial_risks:
        assert cascade_progression[risk_id][0] == 1.0

def test_calculate_risk_correlations(sample_risks):
    simulation_results = {
        1: [0.5, 0.6, 0.7, 0.8, 0.9],
        2: [0.4, 0.5, 0.6, 0.7, 0.8],
        3: [0.3, 0.4, 0.5, 0.6, 0.7],
    }
    correlations = calculate_risk_correlations(sample_risks, simulation_results)
    
    assert len(correlations) == 3  # Number of unique pairs for 3 risks
    for (risk1_id, risk2_id), correlation in correlations.items():
        assert -1 <= correlation <= 1
    
    # Test that perfectly correlated risks have correlation 1
    simulation_results[4] = simulation_results[1]
    correlations = calculate_risk_correlations(sample_risks, simulation_results)
    assert correlations[(1, 4)] == 1.0

def test_identify_risk_feedback_loops(sample_risks, sample_interactions):
    G = build_risk_network(sample_risks, sample_interactions)
    feedback_loops = identify_risk_feedback_loops(G)
    
    assert isinstance(feedback_loops, list)
    for loop in feedback_loops:
        assert len(loop) > 2
        assert loop[0] == loop[-1]  # First and last node should be the same in a cycle
    
    # Test with a known feedback loop
    G.add_edge(3, 1, weight=0.5, type="Moderate")
    feedback_loops = identify_risk_feedback_loops(G)
    assert any(set(loop) == {1, 2, 3} for loop in feedback_loops)

def test_analyze_network_resilience(sample_risks, sample_interactions):
    G = build_risk_network(sample_risks, sample_interactions)
    resilience_metrics = analyze_network_resilience(G)
    
    assert "average_clustering" in resilience_metrics
    assert "average_shortest_path_length" in resilience_metrics
    assert "graph_density" in resilience_metrics
    assert "assortativity" in resilience_metrics
    assert all(0 <= value <= 1 for value in resilience_metrics.values())
    
    # Test that adding more edges increases density
    G.add_edge(1, 4, weight=0.5, type="Moderate")
    new_resilience_metrics = analyze_network_resilience(G)
    assert new_resilience_metrics["graph_density"] > resilience_metrics["graph_density"]

def test_generate_risk_interaction_summary(sample_interactions, sample_risks):
    G = build_risk_network(sample_risks, sample_interactions)
    central_risks = identify_central_risks(G)
    clusters = detect_risk_clusters(G)
    summary = generate_risk_interaction_summary(sample_interactions, central_risks, clusters)
    
    assert isinstance(summary, str)
    assert len(summary) > 100  # Assuming a minimum length for a meaningful summary
    assert "critical risk interactions" in summary.lower()
    assert "central risks" in summary.lower()
    assert "risk clustering" in summary.lower()
    assert "recommendations" in summary.lower()
    
    # Test that the most central risk is mentioned in the summary
    most_central_risk = max(central_risks, key=central_risks.get)
    assert f"Risk {most_central_risk}" in summary

# Add edge case tests
def test_analyze_risk_interactions_edge_cases():
    # Test with a single risk
    single_risk = [Risk(id=1, description="Single Risk", category="Physical", likelihood=0.5, impact=0.5, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="")]
    interactions = analyze_risk_interactions(single_risk)
    assert len(interactions) == 0

    # Test with identical risks
    identical_risks = [Risk(id=i, description="Identical Risk", category="Physical", likelihood=0.5, impact=0.5, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="") for i in range(1, 4)]
    interactions = analyze_risk_interactions(identical_risks)
    assert len(interactions) == 3
    assert all(i.interaction_score == interactions[0].interaction_score for i in interactions)

def test_network_analysis_edge_cases():
    # Test with disconnected network
    risks = [Risk(id=i, description=f"Risk {i}", category="Physical", likelihood=0.5, impact=0.5, subcategory="", tertiary_category="", time_horizon="", industry_specific=False, sasb_category="") for i in range(1, 5)]
    interactions = [
        RiskInteraction(risk1_id=1, risk2_id=2, interaction_score=0.5, interaction_type="Moderate"),
        RiskInteraction(risk1_id=3, risk2_id=4, interaction_score=0.5, interaction_type="Moderate"),
    ]
    G = build_risk_network(risks, interactions)
    
    resilience_metrics = analyze_network_resilience(G)
    assert np.isinf(resilience_metrics["average_shortest_path_length"])
    
    clusters = detect_risk_clusters(G, num_clusters=2)
    assert len(set(clusters.values())) == 2
    assert clusters[1] == clusters[2] and clusters[3] == clusters[4] and clusters[1] != clusters[3]