import torch
import random
import numpy as np
import os

# 设置随机种子，保证结果可复现
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(43)

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

RAW_DATA_PATH = os.path.join(DATA_DIR, 'allnew.xlsx')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'gcn_lstm2_model.pth')

# 确保输出目录存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= 硬件配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 超参数网格 =================
PARAM_GRID = {
    'num_features': [21],  
    'hidden_channels_gcn': [32, 64, 128],  
    'hidden_channels_lstm': [32, 64, 128],  
    'num_layers_lstm': [1, 2, 3],  
    'dropout_rate': [0.3, 0.5, 0.7],  
    'output_dim': [1],  
    'learning_rate': [0.01, 0.005, 0.001]  
}