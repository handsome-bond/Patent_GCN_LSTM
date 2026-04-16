import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

def apply_entropy_weight(df, columns):
    """应用熵权法计算综合评价得分"""
    df_selected = df[columns]
    scaler = RobustScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_selected), columns=columns)
    
    epsilon = 1e-3 
    n = len(df)
    entropy = (-1/np.log(n)) * np.sum(df_normalized * np.log(df_normalized + epsilon), axis=0)
    weight = (1 - entropy) / (1 - entropy).sum()
    
    df['综合评价得分'] = np.dot(df_normalized, weight)
    return df, weight

def cluster_and_filter(df, n_clusters=7):
    """KMeans聚类并找出阈值过滤数据"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['聚类结果'] = kmeans.fit_predict(df[['综合评价得分']])
    thresholds = df.groupby('聚类结果')['综合评价得分'].min().sort_values().tolist()
    
    # 删除得分最小的组
    min_threshold = thresholds[1]
    df_filtered = df[df['综合评价得分'] > min_threshold]
    return df_filtered, thresholds

def parse_names(cell):
    return [name for name in str(cell).split(';') if pd.notna(name) and name != '-'] if pd.notna(cell) else []

def expand_applicant_assignee(df):
    """拆分申请人与受让人为独立行，标注角色"""
    expanded_rows = []
    for _, row in df.iterrows():
        applicants = parse_names(row['申请人'])
        assignees = parse_names(row['受让人'])

        for applicant in applicants:
            new_row = row.copy()
            new_row['人名'] = applicant
            new_row['角色'] = '申请人'
            expanded_rows.append(new_row)

        for assignee in assignees:
            new_row = row.copy()
            new_row['人名'] = assignee
            new_row['角色'] = '受让人'
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)