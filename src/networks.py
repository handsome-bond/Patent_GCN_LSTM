import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FusionGCN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(FusionGCN, self).__init__()
        self.gcns = nn.ModuleList([
            GCNConv(num_features, hidden_channels),
            GCNConv(num_features, hidden_channels),
            GCNConv(num_features, hidden_channels)
        ])

    def forward(self, data_upper, data_middle, data_lower, weights):
        z_upper = F.relu(self.gcns[0](data_upper.x, data_upper.edge_index)) * weights[0]
        z_middle = F.relu(self.gcns[1](data_middle.x, data_middle.edge_index)) * weights[1]
        z_lower = F.relu(self.gcns[2](data_lower.x, data_lower.edge_index)) * weights[2]
        z_combined = z_upper + z_middle + z_lower
        return z_combined

class GCNLSTM(nn.Module):
    def __init__(self, num_features, hidden_channels_gcn, hidden_channels_lstm, num_layers_lstm, dropout_rate=0.5, output_dim=1):
        super(GCNLSTM, self).__init__()
        self.fusion_gcn = FusionGCN(num_features, hidden_channels_gcn)
        self.lstm = nn.LSTM(input_size=hidden_channels_gcn,
                            hidden_size=hidden_channels_lstm,
                            num_layers=num_layers_lstm,
                            dropout=dropout_rate if num_layers_lstm > 1 else 0,
                            batch_first=True)
        self.output_layer = nn.Linear(hidden_channels_lstm, output_dim)

    def forward(self, data_tuple, weights):
        z_combined = self.fusion_gcn(data_tuple[0], data_tuple[1], data_tuple[2], weights)
        lstm_inputs = z_combined.unsqueeze(0).permute(1, 0, 2)  
        
        lstm_out, (hn, cn) = self.lstm(lstm_inputs)
        node_embeddings = hn[-1]
        predictions = self.output_layer(lstm_out[:, -1, :]).squeeze(-1)
        return predictions, node_embeddings