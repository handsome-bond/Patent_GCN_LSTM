import os
import glob
import torch
import pandas as pd
from torch.optim import Adam
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from src.config import *
from src.data_preprocessing import apply_entropy_weight, cluster_and_filter, expand_applicant_assignee
from src.nlp_pipeline import analyze_texts_with_lda, group_and_sum_topics
from src.graph_builder import create_adjacency_matrix, unify_all_nodes, process_data
from src.networks import GCNLSTM
from src.train_eval import augment_with_negatives, grid_search, train, predict_links

def expand_rows_for_adj(df):
    """原代码中特有的图拆分逻辑：申请人互相组合以及申请人到受让人"""
    expanded_rows = []
    def parse_names(cell):
        return cell.split(';') if pd.notna(cell) and cell != '-' else []
        
    for _, row in df.iterrows():
        applicants = parse_names(row['申请人'])
        assignees = parse_names(row['受让人'])
        if applicants:
            for applicant in applicants:
                new_row = row.copy()
                new_row['申请人'] = applicant
                if assignees:
                    for assignee in assignees:
                        new_row['受让人'] = assignee
                        expanded_rows.append(new_row.copy())
                else:
                    new_row['受让人'] = None
                    expanded_rows.append(new_row)
            if len(applicants) > 1:
                for i in range(len(applicants)):
                    for j in range(len(applicants)):
                        if i != j:
                            new_row = row.copy()
                            new_row['申请人'] = applicants[i]
                            new_row['受让人'] = applicants[j]
                            expanded_rows.append(new_row)
        elif assignees: 
            for assignee in assignees:
                new_row = row.copy()
                new_row['申请人'] = None
                new_row['受让人'] = assignee
                expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

def run_preprocessing():
    print("="*50)
    print("▶ 阶段 1: 数据预处理 (熵权法 & KMeans & 文件拆分)")
    print("="*50)
    
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"找不到原始数据文件: {RAW_DATA_PATH}")

    df = pd.read_excel(RAW_DATA_PATH)
    columns_for_entropy = ['被引用数量new', '专利权人数量','战略性新兴产业', '引用数量', '权利要求数']
    
    df_scored, weights = apply_entropy_weight(df, columns_for_entropy)
    print("✓ 熵权法权重计算完成")
    
    df_filtered, thresholds = cluster_and_filter(df_scored)
    print(f"✓ KMeans聚类完成，保留阈值以上的文档数量: {len(df_filtered)}")
    
    sources = df_filtered['Source'].unique()
    for source in sources:
        df_source = df_filtered[df_filtered['Source'] == source]
        save_path = os.path.join(PROCESSED_DATA_DIR, f'1shangquan_{source}.xlsx')
        df_source.to_excel(save_path, index=False)
        print(f"  - 生成: 1shangquan_{source}.xlsx")
        
    print("✅ 阶段 1 完成！")

def run_nlp():
    print("="*50)
    print("▶ 阶段 2: NLP 主题提取与特征分组")
    print("="*50)
    
    try:
        dictionary = Dictionary.load(os.path.join(MODEL_DIR, 'your_dict.dict'))
        optimal_model = LdaModel.load(os.path.join(MODEL_DIR, 'your_lda.model'))
        print("✓ 成功加载已训练的 LDA 模型和词典")
    except Exception as e:
        print("❌ 错误：无法加载 LDA 模型！请在 models 文件夹中放入字典和模型文件。")
        return

    files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, '1shangquan_*.xlsx')))
    group1_indices = [1,3,4,7,9,13,15,17]
    group2_indices = [0,8,10,11,19,20]
    group3_indices = [2,5,6,12,14,16,18]

    for filepath in files:
        filename = os.path.basename(filepath)
        df_base = pd.read_excel(filepath)
        
        df_2open = expand_rows_for_adj(df_base)
        df_2open.to_excel(os.path.join(PROCESSED_DATA_DIR, f'2open_{filename}'), index=False)
        topics_2open, df_2open = analyze_texts_with_lda(df_2open, dictionary, optimal_model)
        df_2open['Dominant Topic'] = [max(doc, key=lambda x: x[1])[0] for doc in topics_2open]
        
        for name, indices in zip(["Group 1no", "Group 2no", "Group 3no"], [group1_indices, group2_indices, group3_indices]):
            df_group = df_2open[df_2open['Dominant Topic'].isin(indices)]
            df_group.to_excel(os.path.join(PROCESSED_DATA_DIR, f"2open_{filename[:-5]}_{name}.xlsx"), index=False)

        df_3output = expand_applicant_assignee(df_base)
        df_3output.to_excel(os.path.join(PROCESSED_DATA_DIR, f'3output_{filename}'), index=False)
        topics_3output, df_3output = analyze_texts_with_lda(df_3output, dictionary, optimal_model)
        df_3output['Dominant Topic'] = [max(doc, key=lambda x: x[1])[0] for doc in topics_3output]
        
        for name, indices in zip(["Group 1", "Group 2", "Group 3"], [group1_indices, group2_indices, group3_indices]):
            df_group = df_3output[df_3output['Dominant Topic'].isin(indices)]
            save_path_3out = os.path.join(PROCESSED_DATA_DIR, f"3output_{filename[:-5]}_{name}.xlsx")
            df_group.to_excel(save_path_3out, index=False)
            
            topics_group, _ = analyze_texts_with_lda(df_group, dictionary, optimal_model)
            if len(topics_group) > 0:
                df_summed = group_and_sum_topics(topics_group, df_group['人名'].tolist())
                df_summed.to_excel(save_path_3out[:-5] + "_summed.xlsx", index_label='Applicant')
                
        print(f"  - 完成处理: {filename}")

    print("✅ 阶段 2 完成！")

def run_graph_build():
    print("="*50)
    print("▶ 阶段 3: 构造全局 PyG 图数据")
    print("="*50)
    
    open_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, '2open_*_Group *no.xlsx')))
    summed_files = sorted(glob.glob(os.path.join(PROCESSED_DATA_DIR, '3output_*_Group *_summed.xlsx')))
    
    if len(open_files) == 0 or len(summed_files) == 0:
        print("❌ 未找到阶段2的产物文件，请先运行 --stage nlp")
        return

    all_names = set()
    for f in open_files:
        df = pd.read_excel(f, usecols=["申请人", "受让人"])
        def get_names(cell): return str(cell).split(';') if pd.notna(cell) and cell != "-" else []
        for _, row in df.iterrows():
            all_names.update(get_names(row.get('申请人')))
            all_names.update(get_names(row.get('受让人')))
            
    excel_dataframes = []
    for f in summed_files:
        df = pd.read_excel(f)
        df.set_index(df.columns[0], inplace=True)
        excel_dataframes.append(df)
        all_names.update(df.index)

    all_identifiers = sorted(list(all_names))
    
    all_matrices = []
    for f in open_files:
        df = pd.read_excel(f)
        adj_matrix = create_adjacency_matrix(df, all_identifiers)
        all_matrices.append(adj_matrix)

    unified_features_list, _ = unify_all_nodes(excel_dataframes)

    graph_data_list = []
    for adj, feat in zip(all_matrices, unified_features_list):
        data = process_data(adj, feat, all_identifiers, DEVICE)
        graph_data_list.append(data)

    time_series_data = []
    for i in range(0, len(graph_data_list), 3):
        if i + 2 < len(graph_data_list):
            time_series_data.append((graph_data_list[i], graph_data_list[i+1], graph_data_list[i+2]))

    torch.save({'time_series_data': time_series_data, 'all_identifiers': all_identifiers}, 
               os.path.join(PROCESSED_DATA_DIR, 'graph_tensors.pt'))
    print(f"✅ 阶段 3 完成！共 {len(time_series_data)} 个时序图。")

def run_training():
    print("="*50)
    print("▶ 阶段 4: GCN-LSTM 模型训练与全量预测")
    print("="*50)
    
    data_path = os.path.join(PROCESSED_DATA_DIR, 'graph_tensors.pt')
    if not os.path.exists(data_path):
        print("❌ 未找到图数据张量，请先运行 --stage graph")
        return
        
    data_dict = torch.load(data_path, map_location=DEVICE)
    time_series_data = data_dict['time_series_data']
    all_identifiers = data_dict['all_identifiers']
    
    train_data = time_series_data[:-1]
    test_data = time_series_data[-1:]
    train_data = augment_with_negatives(train_data, negative_ratio=0.5)

    print("🔍 启动超参数网格搜索...")
    best_model, best_metrics, best_params, _ = grid_search(
        train_data, test_data, PARAM_GRID, epochs=100, eval_interval=5, device=DEVICE)
    
    print(f"🏆 网格搜索结束! 最佳参数: {best_params}")
    
    print("\n🧠 使用最佳参数进行全量数据训练以预测未来链路...")
    final_model = GCNLSTM(**best_params).to(DEVICE)
    optimizer = Adam(final_model.parameters(), lr=best_params['learning_rate'])
    _, final_embeddings = train(final_model, optimizer, time_series_data, test_data, epochs=100, eval_interval=5, device=DEVICE)
    
    print("\n🔗 预测未来可能产生的链接 (Top 10):")
    predicted_links = predict_links(final_embeddings, all_identifiers, top_k=10)
    for i, link in enumerate(predicted_links, 1):
        print(f"  {i}. {link[0]} <---> {link[1]}")
    
    torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n💾 最终模型已成功保存至: {MODEL_SAVE_PATH}")