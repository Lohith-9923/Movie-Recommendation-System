# Movie Recommendation System using Graph Neural Networks (GNN)

This project implements a movie recommendation system using Graph Neural Networks (GNNs). The system utilizes user-movie interaction data and models the relationships between users and movies as a graph. By applying GNN techniques, the model learns latent representations of users and movies, which are then used to make personalized movie recommendations.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributions](#contributions)
- [Future Work](#future-work)

## Introduction
In this project, we develop a movie recommendation system that leverages the power of Graph Neural Networks (GNNs). Traditional recommendation systems typically rely on matrix factorization or content-based approaches. However, by modeling the interactions as a graph, GNNs can capture more complex relationships and provide more accurate recommendations.

## Dataset
The system is trained on a movie rating dataset where each interaction (user-movie rating) is represented as an edge in the graph. The nodes in the graph represent users and movies.

- **Ratings Dataset**: Contains user IDs, movie IDs, and ratings.
- **Movies Dataset**: Contains movie IDs and corresponding metadata (e.g., titles, genres).

### Data Preprocessing
- The datasets are preprocessed to remove duplicates and handle missing values.
- The graph structure is created from user-movie interactions.
- The dataset is split into training, validation, and test sets.

## Model Architecture
The model uses a Graph Neural Network architecture to learn user and movie embeddings based on the graph structure.

- **Embedding Layers**: Learn low-dimensional representations of users and movies.
- **Graph Convolutional Layers**: Aggregate information from neighboring nodes (i.e., connected users/movies).
- **Output Layer**: Produces the final rating prediction.

### Components:
1. **User and Movie Embeddings**: Initialized randomly and updated during training.
2. **Graph Convolution Layers**: Propagate and aggregate information across the graph.
3. **Prediction Layer**: Outputs predicted ratings or recommendations.

### Loss Function:
- The model is optimized using Mean Squared Error (MSE) loss between predicted and actual ratings.

## Installation
To run this project, follow the steps below:

### Prerequisites
- Python 3.x
- PyTorch
- pandas, numpy
- torch_geometric
- scikit-learn (for evaluation metrics)

### Install the required Python packages:
```bash
pip install torch pandas numpy scikit-learn
```

### If you want to install additional dependencies (optional):
```bash
pip install torch-geometric
```

## Usage
To use the Movie Recommendation System, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lohith-9923/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Run the Jupyter notebook**:
   - Open the `Movie_Recommendation_System_using_GNN.ipynb` notebook.
   - Follow the steps in the notebook to preprocess the data, train the model, and evaluate the results.

## Training
The training process involves feeding the user-movie graph to the GNN model and optimizing the embeddings to minimize the prediction error.

### Steps:
1. Load the datasets.
2. Preprocess the data.
3. Define the model architecture.
4. Train the model using the training dataset.
5. Validate the model using the validation dataset.

### Hyperparameters:
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Number of Epochs**: 50
- **Embedding Dimension**: 64

## Evaluation
The model is evaluated on the test set using the following metrics:

- Mean Squared Error (**MSE**): Measures the average squared difference between predicted and actual ratings.
- **Precision@K**: Measures the accuracy of top-K recommendations.
- **Recall@K**: Measures the coverage of top-K recommendations.
## Results
The model achieves the following performance metrics on the test set:

| Metric                         	| Value 	|
|-------------------------------	|--------------	|
| MSE       	|     1.25        |
| Precision@K        	|     0.82       |
| Recall@K    	|        0.64     	|

## Contributions
Contributions to the Movie Recommendation System using GNN are welcome! Whether it's improving model performance, optimizing code, or adding new features, your input is valued. Please feel free to fork the repository, make your changes, and submit a pull request. Be sure to include clear documentation and ensure your code is well-tested. Thank you for helping to enhance this project!
## Future Work
Outline potential improvements and extensions of the project, such as:

- Using more advanced GNN architectures.
- Incorporating additional features (e.g., movie genres, user demographics).
- Exploring different loss functions and optimization strategies.
