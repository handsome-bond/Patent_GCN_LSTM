import torch
import torch.nn.functional as F
import numpy as np
import heapq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

def calculate_edge_overlap(data1, data2):
    # 计算边重叠比例，用于动态权重计算
    set1 = set([tuple(edge) for edge in data1.edge_index.t().tolist()])
    set2 = set([tuple(edge) for edge in data2.edge_index.t().tolist()])
    overlap = len(set1.intersection(set2))
    return overlap / max(len(set1), len(set2), 1)

def calculate_weights(data, device):
    overlap_upper_target = calculate_edge_overlap(data[0], data[2])
    overlap_middle_target = calculate_edge_overlap(data[1], data[2])
    overlap_lower_target = 1.0  
    OR_t_i = [overlap_upper_target, overlap_middle_target, overlap_lower_target]
    sum_OR_t_i = sum(OR_t_i)
    weights = [OR / sum_OR_t_i for OR in OR_t_i]
    return torch.tensor(weights, dtype=torch.float, device=device)

def generate_negative_samples(graph, num_samples):
    # 负采样：选取不存在的边作为负样本
    negative_samples = []
    all_nodes = range(graph.num_nodes)
    existing_edges = set(tuple(sorted((u, v))) for u, v in graph.edge_index.t().tolist())
    
    attempts = 0
    while len(negative_samples) < num_samples and attempts < num_samples * 10:
        u = np.random.choice(all_nodes)
        v = np.random.choice(all_nodes)
        if u != v and (u, v) not in existing_edges:
            negative_samples.append((u, v))
            existing_edges.add((u, v))
        attempts += 1
    return negative_samples

def augment_with_negatives(graph_data, negative_ratio=0.5):
    # 对训练序列中的图进行负采样增强
    for graph_sequence in graph_data:
        for graph in graph_sequence:
            pos_samples = graph.edge_index.t().tolist()
            num_negatives = int(len(pos_samples) * negative_ratio)
            neg_samples = generate_negative_samples(graph, num_negatives)
            if neg_samples:
                neg_tensor = torch.tensor(neg_samples, dtype=torch.long, device=graph.edge_index.device).t()
                graph.edge_index = torch.cat([graph.edge_index, neg_tensor], dim=1)
    return graph_data

def evaluate(model, test_data, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data_tuple in test_data:
            weights = calculate_weights(data_tuple, device)
            logits, _ = model(data_tuple, weights)
            probs = torch.sigmoid(logits)
            
            y_true.append(data_tuple[2].y.view(-1).cpu()) # 解决 Target size 维度报错
            y_pred.append(probs.view(-1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 计算四项核心指标
    return {
        'acc': accuracy_score(y_true, y_pred_binary),
        'prec': precision_score(y_true, y_pred_binary, zero_division=0),
        'rec': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred)
    }

def train_model(model, optimizer, train_data, test_data, epochs, device):
    # 补全训练循环逻辑
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data_tuple in train_data:
            optimizer.zero_grad()
            weights = calculate_weights(data_tuple, device)
            predictions, _ = model(data_tuple, weights)
            target = data_tuple[2].y.view(-1).to(device) #
            
            loss = F.binary_cross_entropy_with_logits(predictions, target.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # 训练结束后获取最后的节点嵌入用于预测
    model.eval()
    with torch.no_grad():
        final_weights = calculate_weights(train_data[-1], device)
        _, final_embeddings = model(train_data[-1], final_weights)
    return final_embeddings

def predict_links_optimized(embeddings, identifiers, top_k=10):
    # 使用堆排序优化的链路预测
    emb_np = embeddings.cpu().numpy()
    num_nodes = emb_np.shape[0]
    sim_matrix = cosine_similarity(emb_np)
    
    heap = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            score = sim_matrix[i, j]
            if len(heap) < top_k:
                heapq.heappush(heap, (score, i, j))
            else:
                heapq.heappushpop(heap, (score, i, j))
    
    top_links = sorted(heap, key=lambda x: x[0], reverse=True)
    return [((identifiers[l[1]], identifiers[l[2]]), l[0]) for l in top_links]