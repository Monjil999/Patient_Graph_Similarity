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
    print("=== Graph-Based Patient Similarity Model ===")
    start_time = time.time()
    
    # Create output directories if they don't exist
    os.makedirs('../plots', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Initialize the patient similarity graph with the correct path
    # When running from src/ directory, we need to go up one level
    psg = PatientSimilarityGraph(data_path='../data/synthea/csv')
    
    # Load the data
    psg.load_data()
    
    # Preprocess the data
    psg.preprocess_data()
    
    # Create patient features (limit to 1000 patients for faster processing)
    psg.create_patient_features(max_patients=1000)
    
    # Build the similarity graph
    psg.build_similarity_graph(threshold=0.6)
    
    # Analyze the graph
    try:
        psg.analyze_graph()
    except Exception as e:
        print(f"Error during graph analysis: {e}")
    
    # Visualize the graph
    psg.visualize_graph(output_file='../plots/patient_similarity_graph.png')
    
    # Example: Find similar patients for a specific patient
    # First, get a patient ID from the dataset
    patient_id = psg.patients_df['Id'].iloc[0]
    print(f"\nFinding similar patients for patient {patient_id}:")
    
    try:
        similar_patients = psg.find_similar_patients(patient_id, top_n=5)
        for idx, (similar_id, similarity) in enumerate(similar_patients, 1):
            print(f"{idx}. Patient {similar_id}: Similarity = {similarity:.4f}")
    except Exception as e:
        print(f"Error finding similar patients: {e}")
    
    # Example: Find similar patients for therapy recommendation
    print("\nTherapy Recommendation Example:")
    try:
        # First, get a common condition from the dataset
        common_condition = psg.conditions_df['DESCRIPTION'].value_counts().index[0]
        print(f"Using condition: {common_condition}")
        
        # Find a patient with this condition
        patient_with_condition = psg.conditions_df[psg.conditions_df['DESCRIPTION'] == common_condition]['PATIENT'].iloc[0]
        print(f"Finding therapy recommendations for patient {patient_with_condition}")
        
        # Get similar patients with the same condition
        similar_for_therapy = psg.recommend_similar_patients_for_therapy(
            patient_with_condition, common_condition, top_n=5
        )
        
        for idx, (similar_id, similarity) in enumerate(similar_for_therapy, 1):
            print(f"{idx}. Patient {similar_id}: Similarity = {similarity:.4f}")
    except Exception as e:
        print(f"Error in therapy recommendation: {e}")
    
    # Example: Find clinical trial candidates
    print("\nClinical Trial Candidate Matching Example:")
    try:
        # First, get a common condition from the dataset
        common_condition = psg.conditions_df['DESCRIPTION'].value_counts().index[1]  # Use the second most common condition
        print(f"Finding clinical trial candidates for condition: {common_condition}")
        
        # Find candidates
        candidates = psg.find_clinical_trial_candidates(
            condition=common_condition,
            min_age=18,
            max_age=65,
            gender=None,
            top_n=5
        )
        
        print("Top candidates:")
        print(candidates)
    except Exception as e:
        print(f"Error in clinical trial matching: {e}")
    
    # Print execution time
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 