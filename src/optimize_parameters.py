import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

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

def prepare_data():
    """Prepare data for model evaluation"""
    # Load data
    patients, observations, conditions = load_patient_data(sample_size=1000)
    if patients is None:
        return None, None, None, None, None
    
    # Create feature matrix and patient ID list
    print("Preprocessing data...")
    patient_graph = PatientSimilarityGraph()
    feature_matrix = patient_graph.preprocess_data(patients, observations)
    patient_ids = patients['Id'].tolist()
    
    # Create map from patient ID to index
    patient_ids_map = {patient_id: i for i, patient_id in enumerate(patient_ids)}
    
    # Create condition map
    condition_map = prepare_condition_map(conditions, patient_ids_map)
    
    # Create a standardized feature matrix for the neural network
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    return patients, feature_matrix, scaled_features, patient_ids, condition_map

def create_evaluation_pairs(patient_ids, feature_matrix, n_samples=5000):
    """Create patient pairs for similarity evaluation with known ground truth"""
    n_patients = len(patient_ids)
    
    # Normalize features for cosine similarity
    normalized_features = feature_matrix / np.linalg.norm(feature_matrix, axis=1)[:, np.newaxis]
    
    # Calculate true cosine similarities for all pairs
    pair_indices = []
    true_similarities = []
    
    # Sample random pairs
    pairs = set()
    while len(pairs) < n_samples:
        i = np.random.randint(0, n_patients)
        j = np.random.randint(0, n_patients)
        if i != j and (i, j) not in pairs and (j, i) not in pairs:
            pairs.add((i, j))
            similarity = np.dot(normalized_features[i], normalized_features[j])
            pair_indices.append((i, j))
            true_similarities.append(similarity)
    
    return pair_indices, true_similarities

def evaluate_gcn_parameters(feature_matrix, scaled_features, patient_ids, condition_map, pair_indices, true_similarities):
    """Evaluate different GCN parameters"""
    # Define parameter grid
    param_grid = {
        'hidden_channels': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'learning_rate': [0.01, 0.001],
        'epochs': [50, 100],
        'similarity_threshold': [0.5, 0.6, 0.7]
    }
    
    # Generate all parameter combinations
    param_combinations = []
    
    # Simple grid search (for demonstration - in practice might use itertools.product)
    for hc in param_grid['hidden_channels']:
        for nl in param_grid['num_layers']:
            for lr in param_grid['learning_rate']:
                for ep in param_grid['epochs']:
                    for st in param_grid['similarity_threshold']:
                        param_combinations.append({
                            'hidden_channels': hc,
                            'num_layers': nl,
                            'learning_rate': lr,
                            'epochs': ep,
                            'similarity_threshold': st
                        })
    
    print(f"Evaluating {len(param_combinations)} GCN parameter combinations...")
    
    results = []
    
    # Run evaluation for each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"Testing GCN parameters {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Build and train model
            gcn_model = GCNPatientModel(
                hidden_channels=params['hidden_channels'],
                num_layers=params['num_layers']
            )
            
            # Preprocess data for GCN
            gcn_model.preprocess_data(scaled_features, patient_ids, similarity_threshold=params['similarity_threshold'])
            
            # Build and train the model
            start_time = time.time()
            gcn_model.build_model()
            gcn_model.train(epochs=params['epochs'], lr=params['learning_rate'])
            training_time = time.time() - start_time
            
            # Evaluate model on pairs
            predictions = []
            
            for i, j in pair_indices:
                embeddings = gcn_model.get_embeddings()
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                predictions.append(similarity)
            
            # Calculate metrics
            correlation = np.corrcoef(predictions, true_similarities)[0, 1]
            mse = np.mean((np.array(predictions) - np.array(true_similarities)) ** 2)
            
            # Binary prediction evaluation (is similar or not)
            binary_true = [1 if s > 0.6 else 0 for s in true_similarities]
            binary_pred = [1 if s > 0.6 else 0 for s in predictions]
            
            # Calculate accuracy
            accuracy = np.mean([1 if p == t else 0 for p, t in zip(binary_pred, binary_true)])
            
            # Store results
            results.append({
                'params': params,
                'correlation': correlation,
                'mse': mse,
                'accuracy': accuracy,
                'training_time': training_time
            })
            
            print(f"  Results: Correlation={correlation:.4f}, MSE={mse:.4f}, Accuracy={accuracy:.4f}, Time={training_time:.2f}s")
            
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
    
    # Sort results by correlation
    results.sort(key=lambda x: x['correlation'], reverse=True)
    
    # Save results to file
    with open('../models/gcn_parameter_optimization.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print best parameters
    print("\nTop 3 GCN parameter combinations:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"{i+1}. Correlation: {r['correlation']:.4f}, MSE: {r['mse']:.4f}, Accuracy: {r['accuracy']:.4f}")
        print(f"   Parameters: {r['params']}")
    
    return results

def evaluate_node2vec_parameters(feature_matrix, patient_ids, condition_map, pair_indices, true_similarities):
    """Evaluate different Node2Vec parameters"""
    # Define parameter grid
    param_grid = {
        'dimensions': [32, 64, 128],
        'walk_length': [10, 20, 30],
        'num_walks': [50, 100],
        'p': [0.5, 1.0, 2.0],
        'q': [0.5, 1.0, 2.0],
        'similarity_threshold': [0.5, 0.6, 0.7]
    }
    
    # Generate parameter combinations (subset for demonstration)
    param_combinations = []
    
    # Simplified grid search (fewer combinations for demo)
    for dim in param_grid['dimensions']:
        for st in param_grid['similarity_threshold']:
            for p in param_grid['p']:
                for q in param_grid['q']:
                    # Fix walk_length and num_walks for simplicity
                    param_combinations.append({
                        'dimensions': dim,
                        'walk_length': 20,
                        'num_walks': 100,
                        'p': p,
                        'q': q,
                        'similarity_threshold': st
                    })
    
    print(f"Evaluating {len(param_combinations)} Node2Vec parameter combinations...")
    
    results = []
    
    # Run evaluation for each parameter combination
    for i, params in enumerate(param_combinations):
        print(f"Testing Node2Vec parameters {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Build and train model
            node2vec_model = Node2VecPatientModel(
                dimensions=params['dimensions'],
                walk_length=params['walk_length'],
                num_walks=params['num_walks'],
                p=params['p'],
                q=params['q']
            )
            
            # Build graph
            node2vec_model.build_graph(feature_matrix, patient_ids, similarity_threshold=params['similarity_threshold'])
            
            # Train model
            start_time = time.time()
            node2vec_model.train()
            training_time = time.time() - start_time
            
            # Evaluate model on pairs
            predictions = []
            
            for i, j in pair_indices:
                if i not in node2vec_model.embeddings or j not in node2vec_model.embeddings:
                    # Skip if either node doesn't have an embedding
                    continue
                    
                emb_i = node2vec_model.embeddings[i]
                emb_j = node2vec_model.embeddings[j]
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                predictions.append(similarity)
            
            # Calculate metrics (if we have predictions)
            if predictions:
                # Need to trim true_similarities to match predictions
                trimmed_true = true_similarities[:len(predictions)]
                
                correlation = np.corrcoef(predictions, trimmed_true)[0, 1]
                mse = np.mean((np.array(predictions) - np.array(trimmed_true)) ** 2)
                
                # Binary prediction evaluation
                binary_true = [1 if s > 0.6 else 0 for s in trimmed_true]
                binary_pred = [1 if s > 0.6 else 0 for s in predictions]
                
                # Calculate accuracy
                accuracy = np.mean([1 if p == t else 0 for p, t in zip(binary_pred, binary_true)])
                
                # Store results
                results.append({
                    'params': params,
                    'correlation': correlation,
                    'mse': mse,
                    'accuracy': accuracy,
                    'training_time': training_time
                })
                
                print(f"  Results: Correlation={correlation:.4f}, MSE={mse:.4f}, Accuracy={accuracy:.4f}, Time={training_time:.2f}s")
            else:
                print("  No predictions generated - skipping this parameter combination")
            
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
    
    # Sort results by correlation
    results.sort(key=lambda x: x['correlation'], reverse=True)
    
    # Save results to file
    with open('../models/node2vec_parameter_optimization.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print best parameters
    print("\nTop 3 Node2Vec parameter combinations:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"{i+1}. Correlation: {r['correlation']:.4f}, MSE: {r['mse']:.4f}, Accuracy: {r['accuracy']:.4f}")
        print(f"   Parameters: {r['params']}")
    
    return results

def plot_parameter_effect(gcn_results, node2vec_results):
    """Plot the effect of different parameters on model performance"""
    # Ensure output directory exists
    os.makedirs('../plots', exist_ok=True)
    
    # Plot for GCN
    if gcn_results:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract parameter values and performance metrics
        hidden_channels = [r['params']['hidden_channels'] for r in gcn_results]
        num_layers = [r['params']['num_layers'] for r in gcn_results]
        thresholds = [r['params']['similarity_threshold'] for r in gcn_results]
        correlations = [r['correlation'] for r in gcn_results]
        
        # Plot correlation vs hidden_channels
        for nl in sorted(set(num_layers)):
            data = [(hc, corr) for hc, n, corr in zip(hidden_channels, num_layers, correlations) if n == nl]
            if data:
                hc_values, corr_values = zip(*data)
                axs[0, 0].plot(hc_values, corr_values, 'o-', label=f'Layers={nl}')
        
        axs[0, 0].set_xlabel('Hidden Channels')
        axs[0, 0].set_ylabel('Correlation')
        axs[0, 0].set_title('Effect of Hidden Channels on GCN Performance')
        axs[0, 0].legend()
        
        # Plot correlation vs num_layers
        for hc in sorted(set(hidden_channels)):
            data = [(nl, corr) for nl, h, corr in zip(num_layers, hidden_channels, correlations) if h == hc]
            if data:
                nl_values, corr_values = zip(*data)
                axs[0, 1].plot(nl_values, corr_values, 'o-', label=f'Hidden={hc}')
        
        axs[0, 1].set_xlabel('Number of Layers')
        axs[0, 1].set_ylabel('Correlation')
        axs[0, 1].set_title('Effect of Number of Layers on GCN Performance')
        axs[0, 1].legend()
        
        # Plot correlation vs threshold
        for hc in sorted(set(hidden_channels)):
            data = [(t, corr) for t, h, corr in zip(thresholds, hidden_channels, correlations) if h == hc]
            if data:
                t_values, corr_values = zip(*data)
                axs[1, 0].plot(t_values, corr_values, 'o-', label=f'Hidden={hc}')
        
        axs[1, 0].set_xlabel('Similarity Threshold')
        axs[1, 0].set_ylabel('Correlation')
        axs[1, 0].set_title('Effect of Similarity Threshold on GCN Performance')
        axs[1, 0].legend()
        
        # Plot training time vs performance
        axs[1, 1].scatter([r['training_time'] for r in gcn_results], [r['correlation'] for r in gcn_results])
        axs[1, 1].set_xlabel('Training Time (s)')
        axs[1, 1].set_ylabel('Correlation')
        axs[1, 1].set_title('Performance vs Training Time Trade-off for GCN')
        
        plt.tight_layout()
        plt.savefig('../plots/gcn_parameter_optimization.png')
        plt.close()
    
    # Plot for Node2Vec
    if node2vec_results:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract parameter values and performance metrics
        dimensions = [r['params']['dimensions'] for r in node2vec_results]
        p_values = [r['params']['p'] for r in node2vec_results]
        q_values = [r['params']['q'] for r in node2vec_results]
        thresholds = [r['params']['similarity_threshold'] for r in node2vec_results]
        correlations = [r['correlation'] for r in node2vec_results]
        
        # Plot correlation vs dimensions
        for p in sorted(set(p_values)):
            data = [(d, corr) for d, p_val, corr in zip(dimensions, p_values, correlations) if p_val == p]
            if data:
                d_values, corr_values = zip(*data)
                axs[0, 0].plot(d_values, corr_values, 'o-', label=f'p={p}')
        
        axs[0, 0].set_xlabel('Dimensions')
        axs[0, 0].set_ylabel('Correlation')
        axs[0, 0].set_title('Effect of Dimensions on Node2Vec Performance')
        axs[0, 0].legend()
        
        # Plot correlation vs p value
        for q in sorted(set(q_values)):
            data = [(p, corr) for p, q_val, corr in zip(p_values, q_values, correlations) if q_val == q]
            if data:
                p_val, corr_values = zip(*data)
                axs[0, 1].plot(p_val, corr_values, 'o-', label=f'q={q}')
        
        axs[0, 1].set_xlabel('p value (return parameter)')
        axs[0, 1].set_ylabel('Correlation')
        axs[0, 1].set_title('Effect of p value on Node2Vec Performance')
        axs[0, 1].legend()
        
        # Plot correlation vs threshold
        for d in sorted(set(dimensions)):
            data = [(t, corr) for t, dim, corr in zip(thresholds, dimensions, correlations) if dim == d]
            if data:
                t_values, corr_values = zip(*data)
                axs[1, 0].plot(t_values, corr_values, 'o-', label=f'Dim={d}')
        
        axs[1, 0].set_xlabel('Similarity Threshold')
        axs[1, 0].set_ylabel('Correlation')
        axs[1, 0].set_title('Effect of Similarity Threshold on Node2Vec Performance')
        axs[1, 0].legend()
        
        # Plot training time vs performance
        axs[1, 1].scatter([r['training_time'] for r in node2vec_results], [r['correlation'] for r in node2vec_results])
        axs[1, 1].set_xlabel('Training Time (s)')
        axs[1, 1].set_ylabel('Correlation')
        axs[1, 1].set_title('Performance vs Training Time Trade-off for Node2Vec')
        
        plt.tight_layout()
        plt.savefig('../plots/node2vec_parameter_optimization.png')
        plt.close()

def main():
    """Main function for parameter optimization"""
    print("Starting parameter optimization...")
    
    # Prepare data
    patients, feature_matrix, scaled_features, patient_ids, condition_map = prepare_data()
    if feature_matrix is None:
        print("Failed to prepare data. Exiting.")
        return
    
    # Create evaluation pairs
    print("Creating evaluation pairs...")
    pair_indices, true_similarities = create_evaluation_pairs(patient_ids, feature_matrix)
    
    # Ensure output directories exist
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../plots', exist_ok=True)
    
    # Evaluate GCN parameters
    print("\n" + "="*80)
    print("Evaluating GCN parameters...")
    gcn_results = evaluate_gcn_parameters(feature_matrix, scaled_features, patient_ids, condition_map, pair_indices, true_similarities)
    
    # Evaluate Node2Vec parameters
    print("\n" + "="*80)
    print("Evaluating Node2Vec parameters...")
    node2vec_results = evaluate_node2vec_parameters(feature_matrix, patient_ids, condition_map, pair_indices, true_similarities)
    
    # Plot parameter effects
    print("\n" + "="*80)
    print("Generating parameter effect plots...")
    plot_parameter_effect(gcn_results, node2vec_results)
    
    print("\nParameter optimization completed!")
    print(f"Results saved to '../models/gcn_parameter_optimization.json' and '../models/node2vec_parameter_optimization.json'")
    print(f"Plots saved to '../plots/gcn_parameter_optimization.png' and '../plots/node2vec_parameter_optimization.png'")

if __name__ == "__main__":
    main() 