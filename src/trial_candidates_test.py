import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our patient similarity graph model
from patient_similarity_graph import PatientSimilarityGraph

def main():
    """
    Test the clinical trial candidates functionality with different conditions.
    """
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs('../plots', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Initialize and run the patient similarity graph model
    print("=== Clinical Trial Candidates Test ===")
    psg = PatientSimilarityGraph(data_path='../data/synthea/csv')
    
    # Load and preprocess the data
    psg.load_data()
    psg.preprocess_data()
    
    # Create features and build the graph
    psg.create_patient_features(max_patients=1000)
    psg.build_similarity_graph(threshold=0.7)  # Using a higher threshold for more selective graph
    
    # List of conditions to test for clinical trial candidates
    conditions = [
        "Acute viral pharyngitis (disorder)",
        "Viral sinusitis (disorder)",
        "Acute bronchitis (disorder)",
        "Hypertension",
        "Type 2 diabetes mellitus"
    ]
    
    # Test each condition
    results = []
    for condition in conditions:
        print(f"\n=== Finding candidates for condition: {condition} ===")
        
        # Find candidates for this condition
        candidates = psg.find_clinical_trial_candidates(
            condition=condition,
            min_age=18,
            max_age=None,
            gender=None,
            top_n=5
        )
        
        # Display results
        if candidates.empty:
            print(f"No candidates found for condition: {condition}")
        else:
            print(f"Top {len(candidates)} candidates:")
            print(candidates)
            
            # Store summary data
            results.append({
                'condition': condition,
                'num_candidates': len(candidates),
                'avg_age': candidates['AGE'].mean() if not candidates.empty else None,
                'avg_centrality': candidates['centrality'].mean() if not candidates.empty else None
            })
    
    # Create a summary table
    if results:
        summary_df = pd.DataFrame(results)
        print("\n=== Summary of Results ===")
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv('../models/trial_candidates_results.csv', index=False)
        print("Results saved to ../models/trial_candidates_results.csv")
    
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 