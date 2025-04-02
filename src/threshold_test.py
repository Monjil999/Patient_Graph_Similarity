import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from patient_similarity_graph import PatientSimilarityGraph

def test_threshold(threshold):
    """
    Test the patient similarity graph with a specific threshold value.
    
    Parameters:
    -----------
    threshold : float
        The similarity threshold to test (between 0 and 1).
    """
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs('../plots', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Initialize and run the patient similarity graph model
    psg = PatientSimilarityGraph(data_path='../data/synthea/csv')
    
    try:
        # Load and preprocess the data
        psg.load_data()
        psg.preprocess_data()
        
        # Create features and build the graph with the specified threshold
        psg.create_patient_features(max_patients=1000)
        psg.build_similarity_graph(threshold=threshold)
        
        # Analyze the graph
        try:
            psg.analyze_graph()
        except Exception as e:
            print(f"Error analyzing graph: {e}")
        
        # Visualize the graph
        output_file = f"../plots/patient_similarity_graph_threshold_{threshold:.2f}.png"
        psg.visualize_graph(output_file=output_file)
        
        # Calculate additional metrics
        # Number of connected components
        num_components = nx.number_connected_components(psg.graph)
        print(f"Number of connected components: {num_components}")
        
        # Size of largest connected component
        largest_cc = max(nx.connected_components(psg.graph), key=len)
        print(f"Size of largest connected component: {len(largest_cc)}/{psg.graph.number_of_nodes()} nodes" 
              f" ({len(largest_cc)/psg.graph.number_of_nodes()*100:.2f}%)")
        
        # Number of isolated nodes
        isolated_nodes = list(nx.isolates(psg.graph))
        print(f"Number of isolated nodes: {len(isolated_nodes)}")
        
        # Average degree
        degrees = [d for n, d in psg.graph.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        print(f"Average node degree: {avg_degree:.2f}")
        
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        return {
            'threshold': threshold,
            'num_nodes': psg.graph.number_of_nodes(),
            'num_edges': psg.graph.number_of_edges(),
            'graph_density': nx.density(psg.graph),
            'avg_clustering': nx.average_clustering(psg.graph),
            'num_components': num_components,
            'largest_cc_size': len(largest_cc),
            'isolated_nodes': len(isolated_nodes),
            'avg_degree': avg_degree,
            'execution_time': end_time - start_time
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """
    Test multiple threshold values and compare the results.
    """
    # Define thresholds to test
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Run tests for each threshold
    results = []
    for threshold in thresholds:
        print(f"\n===== Testing threshold {threshold} =====")
        result = test_threshold(threshold)
        if result:
            results.append(result)
    
    # Create a summary table
    if results:
        summary_df = pd.DataFrame(results)
        print("\n===== Summary of Results =====")
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv('../models/threshold_test_results.csv', index=False)
        print("Results saved to ../models/threshold_test_results.csv")
        
        # Plot key metrics vs threshold
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Number of edges
        plt.subplot(2, 2, 1)
        plt.plot(summary_df['threshold'], summary_df['num_edges'], 'o-', color='blue')
        plt.title('Number of Edges vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Number of Edges')
        plt.grid(True)
        
        # Plot 2: Number of connected components
        plt.subplot(2, 2, 2)
        plt.plot(summary_df['threshold'], summary_df['num_components'], 'o-', color='red')
        plt.title('Number of Connected Components vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Number of Components')
        plt.grid(True)
        
        # Plot 3: Size of largest connected component
        plt.subplot(2, 2, 3)
        plt.plot(summary_df['threshold'], 
                 summary_df['largest_cc_size'] / summary_df['num_nodes'] * 100, 
                 'o-', color='green')
        plt.title('Size of Largest Component (%) vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('% of Nodes in Largest Component')
        plt.grid(True)
        
        # Plot 4: Average node degree
        plt.subplot(2, 2, 4)
        plt.plot(summary_df['threshold'], summary_df['avg_degree'], 'o-', color='purple')
        plt.title('Average Node Degree vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Average Degree')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../plots/threshold_test_summary.png')
        print("Summary plots saved to ../plots/threshold_test_summary.png")

if __name__ == "__main__":
    main() 