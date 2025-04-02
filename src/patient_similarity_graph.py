import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from collections import defaultdict

# Set the plotting style to a valid style
plt.style.use('seaborn-v0_8-whitegrid')

class PatientSimilarityGraph:
    def __init__(self, data_path='../data/synthea/csv'):
        """
        Initialize the PatientSimilarityGraph with the path to the data.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the Synthea CSV files.
        """
        self.data_path = data_path
        self.patients_df = None
        self.conditions_df = None
        self.medications_df = None
        self.encounters_df = None
        self.observations_df = None
        self.procedures_df = None
        self.patient_features = None
        self.graph = None
        
    def load_data(self):
        """Load all necessary datasets from the data path."""
        print("Loading data...")
        # Load the main datasets
        self.patients_df = pd.read_csv(os.path.join(self.data_path, 'patients.csv'))
        self.conditions_df = pd.read_csv(os.path.join(self.data_path, 'conditions.csv'))
        self.medications_df = pd.read_csv(os.path.join(self.data_path, 'medications.csv'))
        self.encounters_df = pd.read_csv(os.path.join(self.data_path, 'encounters.csv'))
        self.observations_df = pd.read_csv(os.path.join(self.data_path, 'observations.csv'))
        self.procedures_df = pd.read_csv(os.path.join(self.data_path, 'procedures.csv'))
        
        print(f"Loaded data for {len(self.patients_df)} patients")
        return self
    
    def preprocess_data(self):
        """Preprocess the data, including handling missing values and converting dates."""
        print("Preprocessing data...")
        
        # Convert date columns to datetime
        self.patients_df['BIRTHDATE'] = pd.to_datetime(self.patients_df['BIRTHDATE'])
        
        # Calculate age
        current_date = pd.to_datetime('2023-01-01')
        self.patients_df['AGE'] = (current_date - self.patients_df['BIRTHDATE']).dt.days / 365.25
        
        # Create gender dummy variables (one-hot encoding)
        gender_dummies = pd.get_dummies(self.patients_df['GENDER'], prefix='GENDER')
        self.patients_df = pd.concat([self.patients_df, gender_dummies], axis=1)
        
        return self
    
    def create_patient_features(self, max_patients=None):
        """
        Create feature vectors for each patient.
        
        Parameters:
        -----------
        max_patients : int or None
            Maximum number of patients to include in the analysis. None means all patients.
        """
        print("Creating patient features...")
        
        if max_patients is not None:
            patient_ids = self.patients_df['Id'].head(max_patients).tolist()
        else:
            patient_ids = self.patients_df['Id'].tolist()
            
        # Basic demographic features
        demographics = self.patients_df.loc[self.patients_df['Id'].isin(patient_ids), 
                                           ['Id', 'AGE', 'GENDER_M', 'GENDER_F']].set_index('Id')
        
        # Count conditions per patient
        condition_counts = self.conditions_df[self.conditions_df['PATIENT'].isin(patient_ids)]
        condition_counts = condition_counts.groupby(['PATIENT', 'DESCRIPTION']).size().unstack(fill_value=0)
        
        # If there are no conditions, create an empty DataFrame with the same index
        if condition_counts.empty:
            condition_counts = pd.DataFrame(index=patient_ids)
        else:
            # Ensure all patients are included
            condition_counts = condition_counts.reindex(patient_ids, fill_value=0)
        
        # Rename columns to avoid conflicts
        condition_counts.columns = ['COND_' + str(col) for col in condition_counts.columns]
        
        # Count medications per patient
        medication_counts = self.medications_df[self.medications_df['PATIENT'].isin(patient_ids)]
        medication_counts = medication_counts.groupby(['PATIENT', 'DESCRIPTION']).size().unstack(fill_value=0)
        
        # If there are no medications, create an empty DataFrame with the same index
        if medication_counts.empty:
            medication_counts = pd.DataFrame(index=patient_ids)
        else:
            # Ensure all patients are included
            medication_counts = medication_counts.reindex(patient_ids, fill_value=0)
        
        # Rename columns to avoid conflicts
        medication_counts.columns = ['MED_' + str(col) for col in medication_counts.columns]
        
        # Count procedures per patient
        procedure_counts = self.procedures_df[self.procedures_df['PATIENT'].isin(patient_ids)]
        procedure_counts = procedure_counts.groupby(['PATIENT', 'DESCRIPTION']).size().unstack(fill_value=0)
        
        # If there are no procedures, create an empty DataFrame with the same index
        if procedure_counts.empty:
            procedure_counts = pd.DataFrame(index=patient_ids)
        else:
            # Ensure all patients are included
            procedure_counts = procedure_counts.reindex(patient_ids, fill_value=0)
        
        # Rename columns to avoid conflicts
        procedure_counts.columns = ['PROC_' + str(col) for col in procedure_counts.columns]
        
        # Count encounter types per patient
        encounter_counts = self.encounters_df[self.encounters_df['PATIENT'].isin(patient_ids)]
        encounter_counts = encounter_counts.groupby(['PATIENT', 'ENCOUNTERCLASS']).size().unstack(fill_value=0)
        
        # If there are no encounters, create an empty DataFrame with the same index
        if encounter_counts.empty:
            encounter_counts = pd.DataFrame(index=patient_ids)
        else:
            # Ensure all patients are included
            encounter_counts = encounter_counts.reindex(patient_ids, fill_value=0)
        
        # Rename columns to avoid conflicts
        encounter_counts.columns = ['ENC_' + str(col) for col in encounter_counts.columns]
        
        # Select a subset of important observations
        important_observations = [
            'Body Mass Index', 'Body Weight', 'Body Height', 
            'Systolic Blood Pressure', 'Diastolic Blood Pressure',
            'Heart rate', 'Respiratory rate'
        ]
        
        # Get the latest observation value for each important type
        observation_values = defaultdict(dict)
        
        for patient_id in patient_ids:
            patient_obs = self.observations_df[
                (self.observations_df['PATIENT'] == patient_id) & 
                (self.observations_df['DESCRIPTION'].isin(important_observations))
            ]
            
            # Group by observation type and get the latest value
            for desc in important_observations:
                obs_desc = patient_obs[patient_obs['DESCRIPTION'] == desc].copy()  # Create explicit copy
                if not obs_desc.empty:
                    # Get the latest observation - fix the SettingWithCopyWarning by using loc
                    # Use format='mixed' to allow multiple date formats (YYYY-MM-DD, MM/DD/YYYY etc.)
                    obs_desc.loc[:, 'DATE'] = pd.to_datetime(obs_desc['DATE'], format='mixed')
                    latest_obs = obs_desc.loc[obs_desc['DATE'].idxmax()]
                    
                    # Store the value
                    observation_values[patient_id][f'OBS_{desc.replace(" ", "_")}'] = latest_obs['VALUE']
                else:
                    observation_values[patient_id][f'OBS_{desc.replace(" ", "_")}'] = np.nan
        
        # Convert to DataFrame
        observation_df = pd.DataFrame.from_dict(observation_values, orient='index')
        
        # Combine all features
        self.patient_features = pd.concat(
            [demographics, condition_counts, medication_counts, procedure_counts, encounter_counts, observation_df], 
            axis=1, 
            join='outer'
        )
        
        # Fill missing values
        self.patient_features = self.patient_features.fillna(0)
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.patient_features)
        self.patient_features = pd.DataFrame(
            scaled_features, 
            index=self.patient_features.index, 
            columns=self.patient_features.columns
        )
        
        print(f"Created feature matrix with shape: {self.patient_features.shape}")
        return self
    
    def build_similarity_graph(self, threshold=0.7):
        """
        Build a similarity graph where nodes are patients and edges represent similarity.
        
        Parameters:
        -----------
        threshold : float
            Minimum similarity threshold to create an edge between patients.
        """
        print(f"Building similarity graph with threshold {threshold}...")
        
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(self.patient_features)
        similarity_df = pd.DataFrame(
            similarity_matrix, 
            index=self.patient_features.index, 
            columns=self.patient_features.index
        )
        
        # Create network graph
        self.graph = nx.Graph()
        
        # Add nodes
        for patient_id in self.patient_features.index:
            self.graph.add_node(patient_id)
        
        # Add edges based on similarity threshold
        for i, patient1 in enumerate(self.patient_features.index):
            for j, patient2 in enumerate(self.patient_features.index[i+1:], i+1):
                similarity = similarity_df.loc[patient1, patient2]
                if similarity >= threshold:
                    self.graph.add_edge(patient1, patient2, weight=similarity)
        
        print(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self
    
    def analyze_graph(self):
        """Analyze the graph structure to identify important properties."""
        print("Analyzing graph...")
        
        if self.graph is None:
            raise ValueError("Graph has not been built yet. Call build_similarity_graph() first.")
        
        # Calculate graph metrics
        density = nx.density(self.graph)
        avg_clustering = nx.average_clustering(self.graph)
        avg_shortest_path = nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else None
        
        # Identify communities using Louvain method
        try:
            from community import best_partition
            communities = best_partition(self.graph)
            self.communities = communities
            num_communities = len(set(communities.values()))
        except ImportError:
            print("python-louvain package not found. Community detection skipped.")
            communities = None
            num_communities = None
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Store the most central patients
        self.central_patients = sorted(
            degree_centrality.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print(f"Graph Density: {density:.4f}")
        print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
        if avg_shortest_path:
            print(f"Average Shortest Path Length: {avg_shortest_path:.4f}")
        else:
            print("Graph is not connected, can't compute average shortest path")
        
        if num_communities:
            print(f"Number of Communities: {num_communities}")
        
        return self
    
    def find_similar_patients(self, patient_id, top_n=5):
        """
        Find the most similar patients to a given patient.
        
        Parameters:
        -----------
        patient_id : str
            The ID of the patient to find similar patients for.
        top_n : int
            Number of similar patients to return.
            
        Returns:
        --------
        list of tuples (patient_id, similarity_score)
        """
        if self.graph is None:
            raise ValueError("Graph has not been built yet. Call build_similarity_graph() first.")
            
        if patient_id not in self.graph.nodes:
            raise ValueError(f"Patient {patient_id} not found in the graph.")
        
        # Get all neighbors with their similarity scores
        neighbors = [(neighbor, self.graph[patient_id][neighbor]['weight']) 
                     for neighbor in self.graph.neighbors(patient_id)]
        
        # Sort by similarity score (descending)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        return neighbors[:top_n]
    
    def visualize_graph(self, output_file=None, color_by_community=True):
        """
        Visualize the patient similarity graph.
        
        Parameters:
        -----------
        output_file : str
            Path to save the visualization. If None, the plot is displayed.
        color_by_community : bool
            Whether to color nodes by detected communities.
        """
        if self.graph is None:
            raise ValueError("Graph has not been built yet. Call build_similarity_graph() first.")
            
        # Set up the plot
        plt.figure(figsize=(12, 12))
        
        # Compute layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Node colors
        if color_by_community and hasattr(self, 'communities'):
            communities = self.communities
            colors = [communities[node] for node in self.graph.nodes()]
        else:
            colors = 'skyblue'
        
        # Draw the network
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, alpha=0.8, node_color=colors, cmap=plt.cm.rainbow)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2)
        
        plt.title("Patient Similarity Network")
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
        
        return self
    
    def recommend_similar_patients_for_therapy(self, patient_id, therapy_condition, top_n=5):
        """
        Find similar patients with a specific condition for therapy recommendation.
        
        Parameters:
        -----------
        patient_id : str
            The ID of the patient to find similar patients for.
        therapy_condition : str
            The medical condition to filter similar patients.
        top_n : int
            Number of similar patients to return.
            
        Returns:
        --------
        list of tuples (patient_id, similarity_score)
        """
        # Find similar patients
        similar_patients = self.find_similar_patients(patient_id, top_n=100)
        similar_patient_ids = [p[0] for p in similar_patients]
        
        # Filter patients with the specific condition
        condition_filter = (
            (self.conditions_df['PATIENT'].isin(similar_patient_ids)) & 
            (self.conditions_df['DESCRIPTION'] == therapy_condition)
        )
        patients_with_condition = self.conditions_df[condition_filter]['PATIENT'].unique()
        
        # Sort by similarity
        recommended_patients = [
            (p_id, score) for p_id, score in similar_patients 
            if p_id in patients_with_condition
        ][:top_n]
        
        return recommended_patients
    
    def find_clinical_trial_candidates(self, condition, min_age=18, max_age=None, gender=None, top_n=10):
        """
        Find patient candidates for a clinical trial based on condition and demographic criteria.
        
        Parameters:
        -----------
        condition : str
            The medical condition for the clinical trial.
        min_age : float
            Minimum age for trial eligibility.
        max_age : float or None
            Maximum age for trial eligibility. None means no upper limit.
        gender : str or None
            If specified, 'M' for male or 'F' for female patients only.
        top_n : int
            Number of patients to recommend.
            
        Returns:
        --------
        DataFrame with candidate patient information
        """
        # Find patients with the condition
        patients_with_condition = self.conditions_df[self.conditions_df['DESCRIPTION'] == condition]['PATIENT'].unique()
        
        # Get demographic data for these patients
        candidate_df = self.patients_df[self.patients_df['Id'].isin(patients_with_condition)].copy()
        
        # Apply demographic filters
        candidate_df = candidate_df[candidate_df['AGE'] >= min_age]
        
        if max_age is not None:
            candidate_df = candidate_df[candidate_df['AGE'] <= max_age]
        
        if gender is not None:
            candidate_df = candidate_df[candidate_df['GENDER'] == gender]
        
        # Rank candidates by their centrality in the graph if they exist in the graph
        centrality_scores = {}
        for patient_id in candidate_df['Id']:
            if patient_id in self.graph:
                # Use degree centrality as a proxy for "typical" patients
                neighbors = list(self.graph.neighbors(patient_id))
                centrality_scores[patient_id] = len(neighbors)
            else:
                centrality_scores[patient_id] = 0
        
        # Add centrality to the DataFrame
        candidate_df['centrality'] = candidate_df['Id'].map(centrality_scores)
        
        # Sort by centrality (descending)
        candidate_df = candidate_df.sort_values('centrality', ascending=False)
        
        # Select top_n candidates
        return candidate_df.head(top_n)[['Id', 'BIRTHDATE', 'AGE', 'GENDER', 'centrality']] 