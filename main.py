import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import spacy
from typing import List, Tuple

def load_minimal_fakenewsnet(fake_path: str, real_path: str) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path, header=0)
    real_df = pd.read_csv(real_path, header=0)
    fake_df['label'] = 0
    real_df['label'] = 1
    df = pd.concat([fake_df, real_df], ignore_index=True)
    return df

def extract_tweet_ids(cell) -> List[str]:
    if isinstance(cell, str) and cell.strip() != "":
        return [t.strip() for t in cell.strip().split()]
    return []

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_title_spacy(title: str) -> str:
    if not isinstance(title, str):
        return ""
    doc = nlp(title.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token.lemma_) > 2
    ]
    return ' '.join(tokens)

def build_graph_from_df(df: pd.DataFrame, tfidf_dim: int = 512) -> Tuple[Data, dict]:
    df['preprocessed_title'] = df['title'].apply(preprocess_title_spacy)
    df['tweet_ids_list'] = df['tweet_ids'].apply(extract_tweet_ids)

    news_nodes = list(df['id'].astype(str))
    tweet_nodes = sorted({t for lst in df['tweet_ids_list'] for t in lst})

    n_news = len(news_nodes)
    n_tweets = len(tweet_nodes)
    print(f"News nodes: {n_news}, Tweet nodes: {n_tweets}, Total nodes: {n_news + n_tweets}")

    news2idx = {nid: i for i, nid in enumerate(news_nodes)}
    tweet2idx = {tid: i + n_news for i, tid in enumerate(tweet_nodes)}  # offset

    # Build edge lists
    edges_src = []
    edges_dst = []
    for _, row in df.iterrows():
        n_id = news2idx[str(row['id'])]
        for t_id in row['tweet_ids_list']:
            if t_id in tweet2idx:
                edges_src.append(tweet2idx[t_id])  # tweet index
                edges_dst.append(n_id)             # news index

    # Make the graph undirected: add reverse edges
    edges_src_full = edges_src + edges_dst
    edges_dst_full = edges_dst + edges_src

    edge_index = torch.tensor([edges_src_full, edges_dst_full], dtype=torch.long)

    # Build node features:
    #  - News: TF-IDF from preprocessed_title -> size tfidf_dim
    #  - Tweets: zero vector
    vectorizer = TfidfVectorizer(max_features=tfidf_dim)

    X_news = vectorizer.fit_transform(df['preprocessed_title'].fillna("")).toarray()  # shape (n_news, tfidf_dim)

    X_tweets = np.zeros((n_tweets, tfidf_dim), dtype=np.float32)
    X_all = np.vstack([X_news.astype(np.float32), X_tweets])

    # Labels: create label vector sized num_nodes. We have labels only for news nodes; tweets get -1 (ignored)
    labels_full = -1 * np.ones((n_news + n_tweets,), dtype=np.int64)
    # map labels from df - ensure order matches news_nodes
    id_to_label = {str(r['id']): int(r['label']) for _, r in df.iterrows()}
    for nid, idx in news2idx.items():
        labels_full[idx] = id_to_label[nid]

    data = Data(
        x=torch.tensor(X_all),
        edge_index=edge_index,
        y=torch.tensor(labels_full),
    )
    data.num_nodes = X_all.shape[0]

    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)
    data.edge_weight = edge_weight

    index_maps = {'news2idx': news2idx, 'tweet2idx': tweet2idx, 'news_nodes': news_nodes, 'tweet_nodes': tweet_nodes, 'vectorizer': vectorizer}
    return data, index_maps

def build_masks_for_news_nodes(data: Data, n_news: int, train_ratio: float = 0.7, val_ratio: float = 0.15, random_seed: int = 42):
    rng = np.random.RandomState(random_seed)
    news_indices = np.arange(n_news)
    train_idx, test_idx = train_test_split(news_indices, train_size=train_ratio, random_state=random_seed, stratify=None)
    remaining = len(test_idx)
    val_size = int(len(news_indices) * val_ratio)
    if val_size > 0:
        val_idx = test_idx[:val_size]
        test_idx = test_idx[val_size:]
    else:
        val_idx = np.array([], dtype=int)

    mask_train = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask_val = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask_test = torch.zeros(data.num_nodes, dtype=torch.bool)

    mask_train[train_idx] = True
    mask_val[val_idx] = True
    mask_test[test_idx] = True

    return mask_train, mask_val, mask_test

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits

class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

def train(model, data, mask_train, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    mask_train = mask_train & (data.y != -1).to(device)
    # Only compute loss on train nodes (where mask_train True and label != -1)
    loss = criterion(out[mask_train], data.y.to(device)[mask_train].long())
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    device = next(model.parameters()).device
    out = model(data.x.to(device), data.edge_index.to(device))
    preds = out.argmax(dim=1).cpu().numpy()
    labels = data.y.cpu().numpy()
    # Only evaluate on nodes where mask True
    mask_np = mask.cpu().numpy()
    true = labels[mask_np]
    pred = preds[mask_np]
    if len(true) == 0:
        return {'acc': 0.0, 'f1': 0.0}
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted', zero_division=0)
    return {'acc': acc, 'f1': f1}

def run_pipeline(fake_csv, real_csv, tfidf_dim=512, hidden_dim=64, epochs=100, model_type='gcn', lr=0.01, weight_decay=5e-4, device_str='cpu'):
    device = torch.device(device_str)
    df = load_minimal_fakenewsnet(fake_csv, real_csv)
    data, maps = build_graph_from_df(df, tfidf_dim=tfidf_dim)
    n_news = len(maps['news2idx'])
    mask_train, mask_val, mask_test = build_masks_for_news_nodes(data, n_news)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)

    n_classes = int(df['label'].nunique())
    in_channels = data.x.shape[1]

    if model_type.lower() == 'gcn':
        model = GCN(in_channels, hidden_dim, n_classes, dropout=0.5).to(device)
    elif model_type.lower() == 'gat':
        model = GAT(in_channels, hidden_dim, n_classes, heads=4, dropout=0.5).to(device)
    else:
        raise ValueError("model_type must be 'gcn' or 'gat'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train(model, data, mask_train, optimizer, criterion, device)
        val_metrics = evaluate(model, data, mask_val)
        test_metrics = evaluate(model, data, mask_test)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val Acc: {val_metrics['acc']:.4f} F1: {val_metrics['f1']:.4f} | Test Acc: {test_metrics['acc']:.4f} F1: {test_metrics['f1']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = evaluate(model, data, mask_test)
    print(f"\nBest epoch {best_epoch}, Best Val F1 {best_val_f1:.4f}")
    print(f"Final Test Acc: {final_test['acc']:.4f}, F1: {final_test['f1']:.4f}")
    return model, data, maps

if __name__ == "__main__":
    fake_csv = "dataset/politifact_fake.csv"
    real_csv = "dataset/politifact_real.csv"
    model, data, maps = run_pipeline(fake_csv, real_csv, tfidf_dim=512, hidden_dim=64, epochs=100, model_type='gat', device_str='cpu')
