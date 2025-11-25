# Fake News Prediction using Graph Neural Networks (GCN & GAT)

This project implements and compares **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** for detecting fake news. The model leverages the structural information of news propagation (news articles and user interactions) along with textual content to classify news as **Real** or **Fake**.

## 1. Dataset Selection
We utilize two benchmark datasets from **FakeNewsNet**: **Politifact** and **Gossipcop**.

*   **Why these datasets?**
    *   They are standard benchmarks for graph-based fake news detection.
    *   They contain both **textual content** (news titles) and **social context** (user interactions/tweets), which is crucial for GNNs.
    *   **Comparison**: We compare Politifact (smaller, political focus) vs. Gossipcop (larger, entertainment focus).
    *   **Hypothesis**: **Gossipcop** is generally preferred for deep learning models because it is significantly larger (3-4x more data) and has more diverse language patterns. Small datasets like Politifact are prone to **overfitting**, where the model memorizes the training data but fails to generalize.

## 2. Model Architecture
We implemented two GNN variants: **GCN** and **GAT**.

### Why this architecture?
*   **Number of Layers (2 Layers)**:
    *   We use a 2-layer architecture (`conv1` -> `ReLU` -> `Dropout` -> `conv2`).
    *   **Reason**: In GNNs, 2 layers allow the model to aggregate information from a node's immediate neighbors (1-hop) and their neighbors (2-hops). Going deeper (e.g., 3+ layers) often leads to **over-smoothing**, where node representations become indistinguishable.
*   **Hidden Features (64 Channels)**:
    *   We map the high-dimensional input (512 features) to a lower-dimensional hidden space (64 features).
    *   **Reason**: This compresses the information, forcing the model to learn the most salient features while reducing computational cost and overfitting risk.
*   **Heads (4 Heads for GAT)**:
    *   For GAT, we use 4 attention heads.
    *   **Reason**: This allows the model to focus on different parts of the neighborhood simultaneously (e.g., one head might focus on user credibility, another on content similarity).

## 3. Embeddings & Feature Engineering
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** with **Spacy** preprocessing.

### Why TF-IDF?
*   **Efficiency**: TF-IDF is computationally lightweight compared to heavy transformers like BERT, making it ideal for rapid experimentation.
*   **Relevance**: It effectively highlights "signature" words that distinguish fake news from real news (e.g., sensationalist keywords) by penalizing common words.
*   **Why it is better (in this context)**: While BERT captures semantic meaning better, TF-IDF combined with the **structural learning** of GNNs often yields competitive results with a fraction of the training time. The GNN compensates for the lack of deep semantic understanding by learning from the *propagation graph*.

### Preprocessing (Spacy)
*   We use Spacy to **lemmatize** words (convert to base form) and remove **stop words** and punctuation. This reduces noise and ensures the model focuses on meaningful content.

## 4. Predicting on Unseen Data
To predict the label (Real/Fake) for a new, unseen news article, the process is as follows:

1.  **Preprocessing**: The new article's title is cleaned and lemmatized using the same Spacy pipeline.
2.  **Vectorization**: The title is transformed into a 512-dimensional vector using the **already fitted** TF-IDF vectorizer.
3.  **Graph Construction**:
    *   The new article is added as a node.
    *   If there are associated tweets/users, edges are created between the article and these users.
    *   If no social context exists, it relies solely on the content node.
4.  **Inference**:
    *   The graph (or subgraph) is passed through the trained GNN.
    *   The model outputs logits for the two classes.
    *   We apply `argmax` to get the final prediction (0 for Fake, 1 for Real).

## 5. Evaluation Techniques
We evaluate the models using **Accuracy** and **F1-Score**.

*   **Accuracy**: Measures the overall percentage of correct predictions.
*   **F1-Score (Weighted)**:
    *   **Why?** Fake news datasets are often imbalanced (more real news than fake, or vice versa). Accuracy can be misleading in such cases (e.g., predicting "Real" 100% of the time might give high accuracy but is useless).
    *   The F1-score is the harmonic mean of **Precision** and **Recall**, providing a more robust measure of the model's ability to correctly identify both classes.