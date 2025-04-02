# Graph-Based Patient Similarity Model

A project that demonstrates the application of graph-based approaches to model patient similarity using electronic health records (EHR) data.

## Project Overview

This project builds a patient similarity model using graph theory to connect patients based on their clinical and demographic similarities. The model can be used to:

1. Find similar patients for a given patient
2. Recommend precision therapies based on similar patients' outcomes
3. Identify suitable candidates for clinical trials
4. Discover natural patient clusters for cohort analysis

## Prediction Capabilities

The graph-based models in this project enable several types of predictions:

1. **Patient Similarity Prediction**: Predicts how similar a patient is to others in the network, enabling personalized medicine approaches.

2. **Clinical Trial Candidate Prediction**: Identifies patients who would be most suitable for specific clinical trials based on their centrality in the similarity network and their medical conditions.

3. **Treatment Outcome Prediction**: By analyzing treatments and outcomes of similar patients, the model can help predict which treatments might be effective for a new patient.

4. **Community/Cluster Prediction**: Identifies natural groupings of patients with similar characteristics, which can inform cohort analysis and targeted interventions.

5. **Network Influence Prediction**: Using centrality measures, the model identifies influential patients in the network who might represent important medical patterns.

## Models Implemented

This project implements three graph-based approaches for patient similarity:

### 1. Basic Similarity Graph
- Uses cosine similarity between patient feature vectors to build connections
- Simple and interpretable approach
- Uses network centrality measures for clinical trial candidate identification
- Primarily focuses on direct feature similarity

### 2. Graph Convolutional Network (GCN)
- Deep learning approach that learns node representations through message passing between patients
- Captures complex, non-linear relationships between patients
- Uses learned node embeddings for finding similar patients and clinical trial candidates
- Better at discovering latent patterns in the network structure
- Makes predictions based on both direct features and higher-order network connections

### 3. Node2Vec
- Uses biased random walks to learn vector representations of patients in the network
- Parameters p and q control the exploration strategy (breadth-first vs. depth-first)
- Captures both structural equivalence and homophily in the network
- Includes visualization of learned embeddings with t-SNE
- Particularly good at identifying communities in the network
- Predicts similarity based on both local and global network structure

## Project Structure

```
├── data/               # Directory for datasets
├── models/             # Directory for saved models
├── notebooks/          # Jupyter notebooks for exploration and visualization
├── plots/              # Visualization outputs
├── src/                # Source code
│   ├── main.py                     # Main script to run basic model
│   ├── patient_similarity_graph.py # Core similarity graph implementation
│   ├── gcn_patient_model.py        # Graph Convolutional Network implementation
│   ├── node2vec_patient_model.py   # Node2Vec implementation
│   ├── advanced_model_test.py      # Script to compare all models
│   ├── threshold_test.py           # Script to test similarity thresholds
│   └── trial_candidates_test.py    # Script to test clinical trial matching
└── README.md           # This file
```

## Getting Started

1. Clone this repository
   ```
   git clone https://github.com/Monjil999/Patient_Graph_Similarity.git
   cd Patient_Graph_Similarity
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```
   cd src
   python main.py
   ```

## Testing Different Models

To compare all three models (Basic, GCN, and Node2Vec):

```
cd src
python advanced_model_test.py
```

This script will:
- Run all models on the same dataset
- Compare their performance in finding similar patients
- Compare their performance in identifying clinical trial candidates
- Measure execution time and overlap in results
- Generate a visualization of the Node2Vec embeddings

## Testing Different Parameters

### Similarity Threshold

The similarity threshold determines how similar two patients must be to be connected in the graph. You can test different thresholds using:

```
cd src
python threshold_test.py
```

This will generate visualizations for different thresholds and analyze how the graph structure changes.

### Clinical Trial Candidate Matching

Test the model's ability to identify candidates for clinical trials with different conditions:

```
cd src
python trial_candidates_test.py
```

## Visualizations

The model generates several visualizations in the `plots` directory:

- Patient similarity graph
- Age distribution
- Gender distribution
- Encounter types
- Top conditions
- Threshold test summary
- Node2Vec embeddings visualization (colored by condition)

## Advanced Model Details

### GCN Model Requirements
The GCN model requires PyTorch and PyTorch Geometric. These are included in the requirements.txt file. Key parameters:
- `hidden_channels`: Dimensionality of the hidden layers (default: 64)
- `num_layers`: Number of graph convolutional layers (default: 2)
- `similarity_threshold`: Threshold for creating initial edges in the graph (default: 0.6)

### Node2Vec Model Parameters
Key parameters that can be tuned:
- `dimensions`: Dimensionality of the embeddings (default: 64)
- `walk_length`: Length of each random walk (default: 30)
- `num_walks`: Number of walks per node (default: 200)
- `p`: Return parameter that controls the likelihood of returning to the previous node (default: 1)
- `q`: In-out parameter that controls whether the walk explores like BFS (q > 1) or DFS (q < 1)
- `window`: Context window size for the Word2Vec algorithm (default: 10)

## Prediction Evaluation

The performance of predictions is evaluated in several ways:

1. **Similarity Prediction Accuracy**: Measured by the overlap between patients identified as similar by different models.

2. **Clinical Trial Candidate Selection**: Evaluated by comparing centrality measures and candidate overlap between models.

3. **Model Efficiency**: Execution time comparison between the different approaches.

4. **Visualization Quality**: The ability to meaningfully visualize patient clusters and relationships.

## Future Work

- Implement more sophisticated similarity metrics
- Add temporal analysis to track patient health over time
- Include treatment outcomes to enhance therapy recommendations
- Experiment with other graph neural network architectures (GAT, GraphSAGE)
- Implement heterogeneous graph representations with different node types
- Explore knowledge graph integration with medical ontologies
- Develop formal evaluation metrics for similarity predictions
- Deploy model as a web service for clinical decision support
