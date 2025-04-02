import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import collections

class Node2VecPatientModel:
    """
    Node2Vec model for patient similarity.
    This model generates random walks on the patient graph and learns embeddings using Word2Vec.
    """
    
    def __init__(self, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, window=10, min_count=1, workers=4):
        """
        Initialize the Node2Vec model with parameters.
        
        Args:
            dimensions: Dimensionality of the node embeddings
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter (controls likelihood of returning to previous node)
            q: In-out parameter (controls DFS vs. BFS behavior)
            window: Context window size for Word2Vec
            min_count: Minimum count of node appearances for Word2Vec
            workers: Number of parallel workers
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.min_count = min_count
        self.workers = workers
        
        self.graph = None
        self.model = None
        self.patient_ids = None
        self.embeddings = None
    
    def build_graph(self, feature_matrix, patient_ids, similarity_threshold=0.6):
        """
        Build a patient similarity graph using cosine similarity.
        
        Args:
            feature_matrix: Matrix of patient features
            patient_ids: List of patient IDs
            similarity_threshold: Threshold for creating edges
        """
        self.patient_ids = patient_ids
        n_patients = len(patient_ids)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, patient_id in enumerate(patient_ids):
            G.add_node(i, patient_id=patient_id)
        
        # Normalize features for cosine similarity
        normalized_features = feature_matrix / np.linalg.norm(feature_matrix, axis=1)[:, np.newaxis]
        
        # Add edges based on similarity
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                similarity = np.dot(normalized_features[i], normalized_features[j])
                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=float(similarity))
        
        if G.number_of_edges() == 0:
            print("Warning: No edges formed with current threshold. Try lowering similarity_threshold.")
            return
        
        self.graph = G
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    def _get_alias_edge(self, src, dst):
        """
        Get alias edge setup for transition probabilities in node2vec walks.
        Implementation based on the original node2vec paper.
        """
        G = self.graph
        p = self.p
        q = self.q
        
        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:  # Return to the source node
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):  # Common neighbor of src and dst
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:  # Go outward
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(prob)/norm_const for prob in unnormalized_probs]
        
        return self._alias_setup(normalized_probs)
    
    def _alias_setup(self, probs):
        """
        Compute utility lists for non-uniform sampling using Alias method.
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int32)
        
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        # Process until the stacks are empty
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q
    
    def _node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from the given node.
        """
        G = self.graph
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1:  # First step is random
                    # Weight by edge weight
                    weights = [G[cur][nbr]['weight'] for nbr in cur_nbrs]
                    total = sum(weights)
                    probs = [w/total for w in weights]
                    walk.append(np.random.choice(cur_nbrs, p=probs))
                else:
                    prev = walk[-2]
                    # Use node2vec transition probabilities
                    neighbors = list(G.neighbors(cur))
                    weights = [G[cur][nbr]['weight'] for nbr in neighbors]
                    
                    # Adjust weights based on p and q parameters
                    for i, nbr in enumerate(neighbors):
                        if nbr == prev:  # Return to previous node
                            weights[i] /= self.p
                        elif not G.has_edge(nbr, prev):  # Not a common neighbor
                            weights[i] /= self.q
                    
                    total = sum(weights)
                    probs = [w/total for w in weights]
                    walk.append(np.random.choice(neighbors, p=probs))
            else:
                break
        
        return walk
    
    def _generate_walks(self):
        """
        Generate random walks from each node in the graph.
        """
        walks = []
        nodes = list(self.graph.nodes())
        
        print(f"Generating {self.num_walks} walks per node...")
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._node2vec_walk(node)
                # Convert walk to strings for Word2Vec
                walks.append([str(node) for node in walk])
        
        return walks
    
    def train(self):
        """
        Train the node2vec model on the patient graph.
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph first.")
            
        # Generate random walks
        walks = self._generate_walks()
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        self.model = Word2Vec(
            walks, 
            vector_size=self.dimensions,
            window=self.window, 
            min_count=self.min_count, 
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=5
        )
        
        # Extract embeddings
        self.embeddings = {int(node): self.model.wv[str(node)] for node in self.graph.nodes()}
        
        print("Model training complete.")
    
    def find_similar_patients(self, patient_idx, top_n=5):
        """
        Find similar patients to the given patient index based on learned embeddings.
        
        Args:
            patient_idx: Index of the target patient
            top_n: Number of similar patients to return
            
        Returns:
            List of (patient_id, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Model not trained. Call train first.")
            
        # Get target embedding
        target_embedding = self.embeddings[patient_idx]
        
        # Compute similarities
        similarities = []
        for idx, embedding in self.embeddings.items():
            if idx == patient_idx:
                continue
            
            similarity = np.dot(target_embedding, embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(embedding))
            similarities.append((self.patient_ids[idx], float(similarity)))
        
        # Sort by similarity (highest first)
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def find_clinical_trial_candidates(self, condition, patient_condition_map, top_n=5):
        """
        Find candidates for clinical trials based on embedding centrality.
        
        Args:
            condition: The medical condition to find candidates for
            patient_condition_map: Dictionary mapping patient indexes to their conditions
            top_n: Number of candidates to return
            
        Returns:
            List of (patient_id, centrality_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Model not trained. Call train first.")
        
        # Compute a centrality measure using the embeddings
        # We'll use the average similarity to all other patients as a centrality metric
        embedding_matrix = np.array(list(self.embeddings.values()))
        norms = np.linalg.norm(embedding_matrix, axis=1)
        embedding_matrix_norm = embedding_matrix / norms[:, np.newaxis]
        
        sim_matrix = np.dot(embedding_matrix_norm, embedding_matrix_norm.T)
        centrality = np.mean(sim_matrix, axis=1)
        
        # Map centrality scores to node indices
        centrality_dict = {idx: centrality[i] for i, idx in enumerate(self.embeddings.keys())}
        
        # Find patients with the given condition
        candidates = []
        for idx, conditions in patient_condition_map.items():
            if condition in conditions and idx in centrality_dict:
                candidates.append((self.patient_ids[idx], centrality_dict[idx]))
        
        # Sort by centrality (highest first)
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
    
    def visualize_embeddings(self, color_by_condition=None, condition=None, figsize=(10, 8)):
        """
        Visualize the learned embeddings using t-SNE.
        
        Args:
            color_by_condition: Dictionary mapping node indices to conditions
            condition: Specific condition to highlight
            figsize: Figure size
        """
        if self.embeddings is None:
            raise ValueError("Model not trained. Call train first.")
            
        # Get embeddings and node IDs
        nodes = list(self.embeddings.keys())
        embeddings = np.array([self.embeddings[node] for node in nodes])
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        if color_by_condition and condition:
            # Color nodes by condition
            colors = ['r' if condition in color_by_condition.get(node, []) else 'b' for node in nodes]
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.7)
            plt.title(f"Patient Embeddings (Red: {condition}, Blue: Others)")
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
            plt.title("Patient Embeddings")
        
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        
        # Save plot
        plt.savefig("../plots/node2vec_embeddings.png")
        print("Embeddings visualization saved to '../plots/node2vec_embeddings.png'") 