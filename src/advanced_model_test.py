import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from patient_similarity_graph import PatientSimilarityGraph
from gcn_patient_model import GCNPatientModel
from node2vec_patient_model import Node2VecPatientModel

def load_patient_data(sample_size=1000):
    """Load patient data from CSV files"""
    data_dir = '../data/synthea/csv'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist. Please download the Synthea dataset.")
        return None, None, None
    
    # Load patients
    print("Loading patient data...")
    patients = pd.read_csv(f'{data_dir}/patients.csv')
    patients = patients.sample(sample_size, random_state=42) if sample_size else patients
    
    # Load observations
    observations = pd.read_csv(f'{data_dir}/observations.csv')
    observations = observations[observations['PATIENT'].isin(patients['Id'])]
    
    # Load conditions
    conditions = pd.read_csv(f'{data_dir}/conditions.csv')
    conditions = conditions[conditions['PATIENT'].isin(patients['Id'])]
    
    return patients, observations, conditions

def prepare_condition_map(conditions, patient_ids_map):
    """Prepare a map of patient index to list of conditions"""
    condition_map = {}
    
    # Group conditions by patient
    for _, row in conditions.iterrows():
        patient_id = row['PATIENT']
        if patient_id in patient_ids_map:
            idx = patient_ids_map[patient_id]
            if idx not in condition_map:
                condition_map[idx] = []
            condition_map[idx].append(row['DESCRIPTION'])
    
    return condition_map

def main():
    """Main function to test and compare the models"""
    # Load data
    patients, observations, conditions = load_patient_data(sample_size=1000)
    if patients is None:
        return
    
    # Create feature matrix and patient ID list
    print("Preprocessing data...")
    patient_graph = PatientSimilarityGraph()
    feature_matrix = patient_graph.preprocess_data(patients, observations)
    patient_ids = patients['Id'].tolist()
    
    # Create map from patient ID to index
    patient_ids_map = {patient_id: i for i, patient_id in enumerate(patient_ids)}
    
    # Create condition map
    condition_map = prepare_condition_map(conditions, patient_ids_map)
    
    # Select a sample condition for testing
    test_condition = "Hypertension"
    
    # Create a standardized feature matrix for the neural network
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    print(f"\nFeature matrix shape: {feature_matrix.shape}\n")
    
    # Test original similarity model
    print("="*80)
    print("Testing original similarity model...")
    start_time = time.time()
    
    # Build graph with threshold 0.6
    patient_graph.build_similarity_graph(feature_matrix, patient_ids, threshold=0.6)
    original_graph = patient_graph.get_graph()
    
    print(f"Graph has {original_graph.number_of_nodes()} nodes and {original_graph.number_of_edges()} edges")
    
    # Find similar patients for a sample patient
    sample_patient_idx = 0
    similar_patients = patient_graph.find_similar_patients(patient_ids[sample_patient_idx], top_n=5)
    
    print(f"\nSimilar patients to {patient_ids[sample_patient_idx]}:")
    for patient_id, similarity in similar_patients:
        print(f"Patient ID: {patient_id}, Similarity: {similarity:.4f}")
    
    # Find clinical trial candidates
    candidates = patient_graph.find_clinical_trial_candidates(test_condition, top_n=5)
    
    print(f"\nClinical trial candidates for {test_condition}:")
    if candidates:
        for i, (patient_id, centrality) in enumerate(candidates, 1):
            patient = patients[patients['Id'] == patient_id].iloc[0]
            print(f"{i}. ID: {patient_id}, Centrality: {centrality:.4f}, "
                  f"Age: {2023 - int(patient['BIRTHDATE'][:4])}, Gender: {patient['GENDER']}")
    else:
        print("No candidates found.")
    
    original_time = time.time() - start_time
    print(f"\nOriginal model execution time: {original_time:.2f} seconds")
    
    # Test GCN model
    print("\n" + "="*80)
    print("Testing Graph Convolutional Network model...")
    start_time = time.time()
    
    try:
        gcn_model = GCNPatientModel(hidden_channels=64, num_layers=2)
        
        # Preprocess data for GCN
        gcn_model.preprocess_data(scaled_features, patient_ids, similarity_threshold=0.6)
        
        # Build and train the model
        gcn_model.build_model()
        gcn_model.train(epochs=50, lr=0.01)
        
        # Find similar patients
        gcn_similar = gcn_model.find_similar_patients(sample_patient_idx, top_n=5)
        
        print(f"\nGCN similar patients to {patient_ids[sample_patient_idx]}:")
        for patient_id, similarity in gcn_similar:
            print(f"Patient ID: {patient_id}, Similarity: {similarity:.4f}")
        
        # Find clinical trial candidates
        gcn_candidates = gcn_model.find_clinical_trial_candidates(test_condition, condition_map, top_n=5)
        
        print(f"\nGCN clinical trial candidates for {test_condition}:")
        if gcn_candidates:
            for i, (patient_id, centrality) in enumerate(gcn_candidates, 1):
                patient = patients[patients['Id'] == patient_id].iloc[0]
                print(f"{i}. ID: {patient_id}, Centrality: {centrality:.4f}, "
                      f"Age: {2023 - int(patient['BIRTHDATE'][:4])}, Gender: {patient['GENDER']}")
        else:
            print("No candidates found.")
    except Exception as e:
        print(f"Error in GCN model: {e}")
        print("GCN model requires PyTorch and PyTorch Geometric packages.")
    
    gcn_time = time.time() - start_time
    print(f"\nGCN model execution time: {gcn_time:.2f} seconds")
    
    # Test node2vec model
    print("\n" + "="*80)
    print("Testing node2vec model...")
    start_time = time.time()
    
    try:
        node2vec_model = Node2VecPatientModel(dimensions=64, walk_length=30, num_walks=100, p=1, q=1)
        
        # Build graph
        node2vec_model.build_graph(feature_matrix, patient_ids, similarity_threshold=0.6)
        
        # Train model
        node2vec_model.train()
        
        # Find similar patients
        node2vec_similar = node2vec_model.find_similar_patients(sample_patient_idx, top_n=5)
        
        print(f"\nNode2Vec similar patients to {patient_ids[sample_patient_idx]}:")
        for patient_id, similarity in node2vec_similar:
            print(f"Patient ID: {patient_id}, Similarity: {similarity:.4f}")
        
        # Find clinical trial candidates
        node2vec_candidates = node2vec_model.find_clinical_trial_candidates(test_condition, condition_map, top_n=5)
        
        print(f"\nNode2Vec clinical trial candidates for {test_condition}:")
        if node2vec_candidates:
            for i, (patient_id, centrality) in enumerate(node2vec_candidates, 1):
                patient = patients[patients['Id'] == patient_id].iloc[0]
                print(f"{i}. ID: {patient_id}, Centrality: {centrality:.4f}, "
                      f"Age: {2023 - int(patient['BIRTHDATE'][:4])}, Gender: {patient['GENDER']}")
        else:
            print("No candidates found.")
        
        # Visualize embeddings
        node2vec_model.visualize_embeddings(color_by_condition=condition_map, condition=test_condition)
        
    except Exception as e:
        print(f"Error in node2vec model: {e}")
        print("Node2Vec model requires gensim package.")
    
    node2vec_time = time.time() - start_time
    print(f"\nNode2Vec model execution time: {node2vec_time:.2f} seconds")
    
    # Compare models
    print("\n" + "="*80)
    print("Model Comparison:")
    print(f"Original model execution time: {original_time:.2f} seconds")
    print(f"GCN model execution time: {gcn_time:.2f} seconds")
    print(f"Node2Vec model execution time: {node2vec_time:.2f} seconds")
    
    # Find overlap in similar patients
    if 'gcn_similar' in locals() and 'node2vec_similar' in locals():
        original_similar_ids = set([p[0] for p in similar_patients])
        gcn_similar_ids = set([p[0] for p in gcn_similar])
        node2vec_similar_ids = set([p[0] for p in node2vec_similar])
        
        print("\nOverlap in similar patients:")
        print(f"Original & GCN: {len(original_similar_ids.intersection(gcn_similar_ids))}/5")
        print(f"Original & Node2Vec: {len(original_similar_ids.intersection(node2vec_similar_ids))}/5")
        print(f"GCN & Node2Vec: {len(gcn_similar_ids.intersection(node2vec_similar_ids))}/5")
    
    # Find overlap in clinical trial candidates
    if 'gcn_candidates' in locals() and 'node2vec_candidates' in locals() and candidates:
        original_candidate_ids = set([p[0] for p in candidates])
        gcn_candidate_ids = set([p[0] for p in gcn_candidates]) if gcn_candidates else set()
        node2vec_candidate_ids = set([p[0] for p in node2vec_candidates]) if node2vec_candidates else set()
        
        print("\nOverlap in clinical trial candidates:")
        print(f"Original & GCN: {len(original_candidate_ids.intersection(gcn_candidate_ids))}/5")
        print(f"Original & Node2Vec: {len(original_candidate_ids.intersection(node2vec_candidate_ids))}/5")
        print(f"GCN & Node2Vec: {len(gcn_candidate_ids.intersection(node2vec_candidate_ids))}/5")
    
    print("\nAdvanced models comparison completed!")

if __name__ == "__main__":
    main() 