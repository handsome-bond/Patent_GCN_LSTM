import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from src.networks import GCNLSTM

def calculate_edge_overlap(data1, data2):
    set1 = set([tuple(edge) for edge in data1.edge_index.t().tolist()])
    set2 = set([tuple(edge) for edge in data2.edge_index.t().tolist()])
    overlap = len(set1.intersection(set2))
    return overlap / max(len(set1), len(set2), 1)

def calculate_weights(data, device):
    overlap_upper_target = calculate_edge_overlap(data[0], data[2])
    overlap_middle_target = calculate_edge_overlap(data[1], data[2])
    overlap_lower_target = 1  
    OR_t_i = [overlap_upper_target, overlap_middle_target, overlap_lower_target]
    sum_OR_t_i = sum(OR_t_i)
    weights = [OR / sum_OR_t_i for OR in OR_t_i]
    return torch.tensor(weights, dtype=torch.float, device=device)

def generate_negative_samples(graph, num_samples):
    negative_samples = []
    all_nodes = np.arange(graph.num_nodes)
    existing_edges = {tuple(sorted((u, v))) for u, v in graph.edge_index.t().tolist()}
    attempts = 0
    while len(negative_samples) < num_samples and attempts < num_samples * 10:
        u, v = np.random.choice(all_nodes, 2, replace=False)
        if (u, v) not in existing_edges and (v, u) not in existing_edges:
            negative_samples.append((u, v))
            existing_edges.add((u, v)) 
            existing_edges.add((v, u))
        attempts += 1
    return negative_samples

def augment_with_negatives(graph_data, negative_ratio=1):
    augmented_data = []
    for graph_sequence in graph_data:
        for graph in graph_sequence:
            pos_samples = graph.edge_index.t().tolist()
            num_negatives = int(len(pos_samples) * negative_ratio)
            neg_samples = generate_negative_samples(graph, num_negatives)
            if neg_samples:
                neg_samples_tensor = torch.tensor(neg_samples, dtype=torch.long).t().to(graph.edge_index.device)
                graph.edge_index = torch.cat([graph.edge_index, neg_samples_tensor], dim=1)
        augmented_data.append(graph_sequence)
    return augmented_data

def evaluate(model, data, weights):
    model.eval()
    with torch.no_grad():
        out, _ = model(data, weights)
        predictions = torch.sigmoid(out).cpu().numpy()
        labels = data[2].y.cpu().numpy()
        
    auc_score = roc_auc_score(labels, predictions)
    sorted_indices = np.argsort(predictions)[::-1]
    top_k_indices = sorted_indices[:100]
    precision = np.mean(labels[top_k_indices])
    return {'auc': auc_score, 'precision': precision}

def train(model, optimizer, train_data, test_data, epochs, eval_interval, device):
    best_metrics = {'auc': 0, 'precision': 0}
    best_embeddings = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data_tuple in train_data:
            optimizer.zero_grad()
            weights = calculate_weights(data_tuple, device)
            predictions, embeddings = model(data_tuple, weights)
            target_data = data_tuple[2].y.to(device)  
            
            loss = F.binary_cross_entropy_with_logits(predictions, target_data.squeeze(-1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % eval_interval == 0:
            eval_metrics = evaluate(model, test_data[0], weights)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_data):.4f} | AUC: {eval_metrics['auc']:.4f} | Prec: {eval_metrics['precision']:.4f}")
            if eval_metrics['auc'] > best_metrics['auc']:
                best_metrics = eval_metrics
                best_embeddings = embeddings
                
    return best_metrics, best_embeddings

def grid_search(train_data, test_data, param_grid, epochs, eval_interval, device):
    best_model = None
    best_metrics = {'auc': 0, 'precision': 0}
    best_params = {}
    best_embeddings = None  
    
    for params in ParameterGrid(param_grid):
        print(f"Testing params: {params}")
        model = GCNLSTM(**params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        metrics, embeddings = train(model, optimizer, train_data, test_data, epochs, eval_interval, device)
        
        if metrics['auc'] > best_metrics['auc']:
            best_metrics = metrics
            best_model = model
            best_params = params
            best_embeddings = embeddings 
            
    return best_model, best_metrics, best_params, best_embeddings

def predict_links(node_embeddings, identifiers, top_k=10, batch_size=1000):
    node_embeddings = node_embeddings.detach().cpu().numpy() 
    num_nodes = node_embeddings.shape[0]
    top_links_heap = []
    
    for i in range(0, num_nodes, batch_size):
        end_i = min(i + batch_size, num_nodes)
        sim_batch = cosine_similarity(node_embeddings[i:end_i], node_embeddings)
        for j in range(sim_batch.shape[0]):
            for k in range(i + j + 1, sim_batch.shape[1]): # 避免重复和自身比较
                if len(top_links_heap) < top_k:
                    heapq.heappush(top_links_heap, (sim_batch[j, k], i + j, k))
                else:
                    heapq.heappushpop(top_links_heap, (sim_batch[j, k], i + j, k))

    top_links = sorted(top_links_heap, reverse=True, key=lambda x: x[0])
    return [(identifiers[link[1]], identifiers[link[2]]) for link in top_links]