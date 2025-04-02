import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class GCNPatientModel:
    """
    Graph Convolutional Network model for patient similarity.
    This model learns node embeddings through message passing and can be used for
    link prediction between patients and identifying similar patients.
    """
    
    def __init__(self, hidden_channels=64, num_layers=2):
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.model = None
        self.data = None
        self.feature_matrix = None
        self.patient_ids = None
        
    def preprocess_data(self, feature_matrix, patient_ids, similarity_threshold=0.6):
        """
        Preprocess the feature matrix and build initial graph for GCN.
        
        Args:
            feature_matrix: Matrix of patient features
            patient_ids: List of patient IDs
            similarity_threshold: Threshold for creating edges
        """
        self.feature_matrix = feature_matrix
        self.patient_ids = patient_ids
        
        # Create initial graph based on cosine similarity
        n_patients = len(patient_ids)
        edge_index = []
        edge_weight = []
        
        # Normalize features for cosine similarity
        normalized_features = feature_matrix / np.linalg.norm(feature_matrix, axis=1)[:, np.newaxis]
        
        # Build edge list based on similarity threshold
        for i in range(n_patients):
            for j in range(i+1, n_patients):
                similarity = np.dot(normalized_features[i], normalized_features[j])
                if similarity > similarity_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # Add both directions for undirected graph
                    edge_weight.append(float(similarity))
                    edge_weight.append(float(similarity))
        
        if not edge_index:
            print("Warning: No edges formed with current threshold. Try lowering similarity_threshold.")
            return
        
        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        x = torch.tensor(feature_matrix, dtype=torch.float)
        
        # Create PyTorch Geometric data object
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        
        print(f"Graph created with {n_patients} nodes and {len(edge_weight)//2} edges")
        
    def build_model(self):
        """Build the GCN model"""
        self.model = GCNPatientNetwork(
            in_channels=self.feature_matrix.shape[1],
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers
        )
        
    def train(self, epochs=100, lr=0.01):
        """Train the GCN model"""
        if self.data is None or self.model is None:
            raise ValueError("Data not preprocessed or model not built")
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # For link prediction, we use existing edges as positive samples
            z = self.model(self.data.x, self.data.edge_index)
            
            # Simple reconstruction loss to learn good embeddings
            loss = self.model.link_pred_loss(z, self.data.edge_index, self.data.edge_attr)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    def get_embeddings(self):
        """Get learned patient embeddings"""
        self.model.eval()
        with torch.no_grad():
            return self.model(self.data.x, self.data.edge_index).detach().numpy()
    
    def find_similar_patients(self, patient_idx, top_n=5):
        """
        Find similar patients to the given patient index based on learned embeddings.
        
        Args:
            patient_idx: Index of the target patient
            top_n: Number of similar patients to return
            
        Returns:
            List of (patient_id, similarity_score) tuples
        """
        embeddings = self.get_embeddings()
        
        # Compute cosine similarity between the target patient and all others
        target_embedding = embeddings[patient_idx]
        target_norm = np.linalg.norm(target_embedding)
        
        similarities = []
        for i, embedding in enumerate(embeddings):
            if i == patient_idx:
                continue
            
            similarity = np.dot(target_embedding, embedding) / (target_norm * np.linalg.norm(embedding))
            similarities.append((self.patient_ids[i], float(similarity)))
        
        # Sort by similarity (highest first)
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    
    def find_clinical_trial_candidates(self, condition, patient_condition_map, top_n=5):
        """
        Find candidates for clinical trials based on a specific condition using centrality.
        
        Args:
            condition: The medical condition to find candidates for
            patient_condition_map: Dictionary mapping patient indexes to their conditions
            top_n: Number of candidates to return
            
        Returns:
            List of (patient_id, centrality_score) tuples
        """
        embeddings = self.get_embeddings()
        
        # Create a new graph from the learned embeddings
        G = nx.Graph()
        for i in range(len(self.patient_ids)):
            G.add_node(i)
        
        # Create edges based on embedding similarity
        for i in range(len(self.patient_ids)):
            for j in range(i+1, len(self.patient_ids)):
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                if similarity > 0.7:  # Threshold can be adjusted
                    G.add_edge(i, j, weight=similarity)
        
        # Calculate centrality measures
        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
        
        # Find patients with the given condition
        candidates = []
        for idx, conditions in patient_condition_map.items():
            if condition in conditions and idx in centrality:
                candidates.append((self.patient_ids[idx], centrality[idx]))
        
        # Sort by centrality (highest first)
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]


class GCNPatientNetwork(nn.Module):
    """
    Graph Convolutional Network for patient similarity.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.dropout = 0.2
        
    def forward(self, x, edge_index):
        """Forward pass through the GCN layers"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def link_pred_loss(self, z, edge_index, edge_weight):
        """
        Calculate link prediction loss to train the model.
        Uses positive examples from the existing edges.
        """
        src, dst = edge_index
        pos_score = (z[src] * z[dst]).sum(dim=1)
        
        # Generate equal number of negative samples
        neg_src = torch.randint(0, z.size(0), (src.size(0),), device=z.device)
        neg_dst = torch.randint(0, z.size(0), (dst.size(0),), device=z.device)
        neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)
        
        # Use weighted BCE loss
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15) * edge_weight
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15)
        
        return torch.mean(pos_loss) + torch.mean(neg_loss) 