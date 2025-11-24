
# 📰 Fake News Detection using Graph Neural Networks (GNN)

This project builds a **Fake News Detector** using **Graph Neural Networks (GCN/GAT)** with the **FakeNewsNet (Politifact)** dataset.

We convert news articles and tweets into a **graph**, apply message passing using GNNs, and classify each news article as **real (1)** or **fake (0)**.

---

# 📌 Features

✔ Converts FakeNewsNet CSV files into a **bipartite graph**
✔ Nodes = **News articles + Tweets**
✔ Edges = **Tweet → News** (tweet references a news article)
✔ Node features =
 • News: **TF-IDF of title**
 • Tweets: **zero feature vectors** (placeholder)
✔ Supports **GCN** and **GAT** models
✔ Train/Val/Test split only on **news nodes**
✔ Full training loop with best model selection
✔ Outputs accuracy & F1 score

---

# 🏗 Project Structure

```
full_fakenews_gnn.py
dataset/
    politifact_fake.csv
    politifact_real.csv
README.md
```

---

# 🔧 Installation

### 1. Create virtual environment (optional)

```bash
python3 -m venv gnn_env
source gnn_env/bin/activate
```

### 2. Install dependencies

```bash
pip install torch torch_geometric numpy pandas scikit-learn spacy
python -m spacy download en_core_web_sm
```

If PyTorch Geometric fails, follow installation instructions:
[https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

---

# 📥 Dataset Format

You need two CSV files:

### **politifact_real.csv**

### **politifact_fake.csv**

Each file must contain at least:

| column      | meaning                         |
| ----------- | ------------------------------- |
| `id`        | unique news ID                  |
| `title`     | news headline                   |
| `tweet_ids` | space-separated tweet IDs       |
| `label`     | (added automatically by script) |

Example:

```
id,title,tweet_ids
101,"Breaking news headline","t1 t2 t3"
102,"Another headline","t4 t5"
```

---

# 🧠 How the Graph is Built

### **Nodes**

* First **N** nodes = news
* Next **T** nodes = tweets

### **Edges**

For each tweet referencing a news article:

```
tweet_node → news_node
news_node → tweet_node   (added to make graph undirected)
```

### **Node Features**

* News nodes → TF-IDF of cleaned title (size = 512)
* Tweet nodes → zero vector (no tweet text)

### **Labels**

* News nodes → 0 or 1
* Tweet nodes → -1 (ignored during training)

---

# 🏃 Running the Pipeline

Just run:

```bash
python full_fakenews_gnn.py
```

You can adjust parameters:

```python
run_pipeline(
    fake_csv="dataset/politifact_fake.csv",
    real_csv="dataset/politifact_real.csv",
    tfidf_dim=512,
    hidden_dim=64,
    epochs=100,
    model_type='gcn',   # or 'gat'
    device_str='cpu'
)
```

---

# 📊 Output

During training you'll see logs like:

```
Epoch 010 | Train Loss: 0.5932 | Val Acc: 0.78 F1: 0.79 | Test Acc: 0.76 F1: 0.74
Epoch 020 | Train Loss: 0.4211 | Val Acc: 0.81 F1: 0.82 | Test Acc: 0.78 F1: 0.77
...
Best epoch 37, Best Val F1 0.8741
Final Test Acc: 0.8612, F1: 0.8556
```

---

# 🧩 Code Overview (Simple Explanation)

### **1. Load dataset**

* Fake news labeled **0**
* Real news labeled **1**

### **2. Preprocess titles**

Using spaCy:

* lowercase
* remove stopwords
* lemmatize

### **3. Extract tweet ids**

Convert `"t1 t2 t3"` → `["t1", "t2", "t3"]`

### **4. Build graph**

* Every news gets a node
* Every tweet gets a node
* Edges represent tweet–news relationship

### **5. Build node features**

* News → TF-IDF vectors
* Tweets → zero vectors

### **6. Build masks**

Only **news nodes** are used in train/val/test.

### **7. Train model**

Message passing happens inside:

* `GCNConv`
* `GATConv`

### **8. Evaluate**

Compute accuracy and F1 on val/test nodes.

---

# 🧠 Models Included

### **GCN (Graph Convolutional Network)**

Simple graph smoothing + message passing.

### **GAT (Graph Attention Network)**

Learns importance weights between neighbors.

---

# 📌 Hyperparameters You Can Tune

| Name           | Description                  |
| -------------- | ---------------------------- |
| `tfidf_dim`    | Size of title feature vector |
| `hidden_dim`   | Hidden layer size            |
| `epochs`       | Training loops               |
| `lr`           | Learning rate                |
| `weight_decay` | L2 regularization            |
| `model_type`   | `"gcn"` or `"gat"`           |

---

# 📚 References

* FakeNewsNet Dataset
* Kipf & Welling, **GCN**
* Veličković et al., **GAT**

---


