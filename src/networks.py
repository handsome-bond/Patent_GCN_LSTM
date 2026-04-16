import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FusionGCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(FusionGCN, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.gcns = nn.ModuleList([
            GCNConv(num_features, hidden_channels), # 处理上层图
            GCNConv(num_features, hidden_channels), # 处理中层图
            GCNConv(num_features, hidden_channels)  # 处理下层图
        ])

    def forward(self, data_upper, data_middle, data_lower, weights):
        # 分别计算三层的卷积结果并加权
        z_upper = F.relu(self.gcns[0](data_upper.x, data_upper.edge_index)) * weights[0]
        z_middle = F.relu(self.gcns[1](data_middle.x, data_middle.edge_index)) * weights[1]
        z_lower = F.relu(self.gcns[2](data_lower.x, data_lower.edge_index)) * weights[2]
        
        z_combined = torch.cat((z_upper, z_middle, z_lower), dim=1)
        return z_combined

class GCNLSTM(nn.Module):
    def __init__(self, num_features, hidden_channels_gcn, hidden_channels_lstm, num_layers_lstm, dropout_rate=0.5, output_dim=1):
        super(GCNLSTM, self).__init__()
        self.fusion_gcn = FusionGCN(num_features, hidden_channels_gcn)
        
        # 【维度调整】：输入维度必须是 hidden_channels_gcn * 3 (因为上面做了 cat)
        self.lstm = nn.LSTM(input_size=hidden_channels_gcn * 3,
                            hidden_size=hidden_channels_lstm,
                            num_layers=num_layers_lstm,
                            dropout=dropout_rate if num_layers_lstm > 1 else 0,
                            batch_first=True)
        self.output_layer = nn.Linear(hidden_channels_lstm, output_dim)

    def forward(self, data_tuple, weights):
        # data_tuple 包含 (data_upper, data_middle, data_lower)
        z_combined = self.fusion_gcn(data_tuple[0], data_tuple[1], data_tuple[2], weights)
        
        # 调整维度以适配 LSTM (Batch, Seq, Features)
        lstm_inputs = z_combined.unsqueeze(1) 
        
        lstm_out, (hn, cn) = self.lstm(lstm_inputs)
        node_embeddings = hn[-1] # 取最后一层的隐藏状态作为节点嵌入
        
        # 预测链路存在概率
        predictions = self.output_layer(lstm_out[:, -1, :]).squeeze(-1)
        return predictions, node_embeddings