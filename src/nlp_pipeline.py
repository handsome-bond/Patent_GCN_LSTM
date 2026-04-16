import os
import torch
from src.config import *
from src.networks import GCNLSTM
from src.train_eval import augment_with_negatives, train_model, evaluate, predict_links_optimized

def run_training():
    print("="*50)
    print("▶ 阶段 4: 模型训练、评估与权重保存")
    print("="*50)
    
    # 加载已构建好的图张量
    data_path = os.path.join(PROCESSED_DATA_DIR, 'graph_tensors.pt')
    if not os.path.exists(data_path):
        print("❌ 未找到图数据，请先运行 --stage graph")
        return
        
    loaded = torch.load(data_path, map_location=DEVICE)
    time_series_data = loaded['time_series_data']
    all_identifiers = loaded['all_identifiers']
    
    # 数据拆分与增强
    train_data = time_series_data[:-1]
    test_data = time_series_data[-1:]
    train_data = augment_with_negatives(train_data, negative_ratio=0.5)

    # 初始化模型
    model = GCNLSTM(num_features=21, hidden_channels_gcn=8, 
                    hidden_channels_lstm=6, num_layers_lstm=2, 
                    dropout_rate=0.5, output_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("🚀 开始全量训练...")
    final_embeddings = train_model(model, optimizer, train_data, test_data, epochs=100, device=DEVICE)
    
    # 执行评估
    metrics = evaluate(model, test_data, DEVICE)
    print(f"\n📊 评估结果: Acc={metrics['acc']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

    # 预测并打印结果
    print("\n🔗 预测未来链路 (Top 10):")
    results = predict_links_optimized(final_embeddings, all_identifiers, top_k=10)
    for i, ((u, v), score) in enumerate(results, 1):
        print(f"  {i}. {u} <---> {v} (相似度: {score:.4f})")

    # 【关键补全】：保存模型权重到 models 文件夹
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ 模型权重已保存至: {MODEL_SAVE_PATH}")