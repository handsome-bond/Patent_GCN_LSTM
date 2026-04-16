import pandas as pd
import numpy as np
import torch
from scipy.sparse import lil_matrix, csr_matrix
from torch_geometric.data import Data

def parse_names_simple(cell):
    if pd.notna(cell) and cell != "-":
        return cell.split(';')
    return []

def create_adjacency_matrix(df, all_names):
    """根据申请人和受让人创建邻接矩阵"""
    size = len(all_names)
    name_index = {name: idx for idx, name in enumerate(all_names)}
    adj_matrix = lil_matrix((size, size), dtype=int)

    for _, row in df.iterrows():
        applicants = parse_names_simple(row['申请人'])
        assignees = parse_names_simple(row['受让人'])
        for app in applicants:
            if app in name_index:  
                app_idx = name_index[app]
                for asg in assignees:
                    if asg in name_index:
                        asg_idx = name_index[asg]
                        adj_matrix[app_idx, asg_idx] = 1  
    return adj_matrix

def reorder_adjacency_matrix(adj_matrix, all_identifiers, identifier_to_index):
    """根据统一的全局标识符重新排序邻接矩阵"""
    new_adj_matrix = csr_matrix(adj_matrix.shape, dtype=int)
    for old_index, identifier in enumerate(all_identifiers):
        new_index = identifier_to_index[identifier]
        new_adj_matrix[new_index, :] = adj_matrix[old_index, :]
    return new_adj_matrix

def unify_all_nodes(features_list):
    """统一所有时间切片的节点特征（补全缺失节点为0）"""
    unified_index = pd.Index([])
    for features in features_list:
        unified_index = unified_index.union(features.index)
    unified_index = unified_index.sort_values()
    
    unified_features_list = []
    for features in features_list:
        extended_features = features.reindex(unified_index, fill_value=0)
        unified_features_list.append(extended_features)
    return unified_features_list, unified_index.tolist()

def process_data(adj_matrix, feature_df, all_identifiers, device):
    """将邻接矩阵和特征转化为 PyG 的 Data 对象"""
    identifier_to_index = {identifier: i for i, identifier in enumerate(all_identifiers)}
    adj_matrix = reorder_adjacency_matrix(adj_matrix, all_identifiers, identifier_to_index)
    
    num_features = feature_df.shape[1]
    mapped_features = np.zeros((len(all_identifiers), num_features))
    for idx, row in feature_df.iterrows():
        if idx in identifier_to_index:
            mapped_features[identifier_to_index[idx]] = row.values
            
    features = torch.tensor(mapped_features, dtype=torch.float, device=device)
    row, col = adj_matrix.nonzero()
    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long, device=device)
    
    # 构造 Target (下一时刻的连接预测)
    chunk_sum = np.array(adj_matrix.sum(axis=1)).flatten()
    target_binary = (chunk_sum > 0).astype(int)
    target_data = torch.tensor(target_binary, dtype=torch.float, device=device).unsqueeze(1)
    
    return Data(x=features, edge_index=edge_index, y=target_data)