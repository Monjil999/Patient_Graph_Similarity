import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our patient similarity graph model
from patient_similarity_graph import PatientSimilarityGraph

def main():
    print("=== Testing Graph-Based Patient Similarity Model with Different Parameters ===")
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs('../plots', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Initialize the patient similarity graph
    psg = PatientSimilarityGraph(data_path='../data/synthea/csv')
    
    # Load the data
    psg.load_data()
    
    # Preprocess the data
    psg.preprocess_data()
    
    # Create patient features (limit to 1000 patients for faster processing)
    psg.create_patient_features(max_patients=1000)
    
    # Test different similarity thresholds
    thresholds = [0.5, 0.7, 0.8]
    
    for threshold in thresholds:
        print(f"\n=== Testing with similarity threshold: {threshold} ===")
        
        # Build the similarity graph with the current threshold
        psg.build_similarity_graph(threshold=threshold)
        
        # Analyze the graph
        try:
            psg.analyze_graph()
        except Exception as e:
            print(f"Error during graph analysis: {e}")
        
        # Visualize the graph
        psg.visualize_graph(output_file=f'../plots/patient_similarity_graph_threshold_{threshold}.png')
        
        # Find similar patients for a specific patient
        patient_id = psg.patients_df['Id'].iloc[0]
        print(f"\nFinding similar patients for patient {patient_id}:")
        
        try:
            similar_patients = psg.find_similar_patients(patient_id, top_n=5)
            for idx, (similar_id, similarity) in enumerate(similar_patients, 1):
                print(f"{idx}. Patient {similar_id}: Similarity = {similarity:.4f}")
        except Exception as e:
            print(f"Error finding similar patients: {e}")
    
    # Print execution time
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 