# Graph-Based Patient Similarity Model

A project that demonstrates the application of graph-based approaches to model patient similarity using electronic health records (EHR) data.

## Project Overview

This project builds a patient similarity model using graph theory to connect patients based on their clinical and demographic similarities. The model can be used to:

1. Find similar patients for a given patient
2. Recommend precision therapies based on similar patients' outcomes
3. Identify suitable candidates for clinical trials

## Project Structure

```
├── data/               # Directory for datasets
├── models/             # Directory for saved models
├── notebooks/          # Jupyter notebooks for exploration and visualization
├── plots/              # Visualization outputs
├── src/                # Source code
│   ├── main.py         # Main script to run the model
│   └── patient_similarity_graph.py # Core similarity graph implementation
└── README.md           # This file
```

## Getting Started

1. Clone this repository
2. Ensure you have the necessary dependencies installed:
   - pandas
   - numpy
   - networkx
   - scikit-learn
   - matplotlib

3. Run the main script:
   ```
   cd src
   python main.py
   ```

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

## Future Work

- Implement more sophisticated similarity metrics
- Add temporal analysis to track patient health over time
- Include treatment outcomes to enhance therapy recommendations
- Implement advanced community detection and graph analysis techniques

