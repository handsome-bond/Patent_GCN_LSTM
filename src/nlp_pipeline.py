import re
import jieba
import pandas as pd

def preprocess_text(text):
    """正则清洗专利摘要文本"""
    if not isinstance(text, str):
        return ""
    patterns_to_remove = [
        r'\b\w+="[^"]*"', r'No\.\d+、SEQ', r'ID', r'\d', 
        r'/>\s*|\s*<Image', r'NO：M', r'[a-zA-Z]', 
        r'[。和或\.～]', r'</>|<>', r':、|:', 
        r'‑“|\.、|\(|\)', r'[^\w\s，、]'
    ]
    combined_pattern = '|'.join(patterns_to_remove)
    return re.sub(combined_pattern, '', text)

def tokenize_texts(texts):
    """文本分词"""
    texts_cleaned = [preprocess_text(text) for text in texts]
    return [list(jieba.cut(text)) for text in texts_cleaned]

def analyze_texts_with_lda(df, dictionary, lda_model):
    """使用已训练的 LDA 模型分析主题概率"""
    texts = df['摘要'].dropna().tolist()
    texts_tokenized = tokenize_texts(texts)
    corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
    topic_probabilities = [lda_model.get_document_topics(bow, minimum_probability=0) for bow in corpus]
    return topic_probabilities, df

def group_and_sum_topics(topics, applicants):
    """根据申请人合并主题概率"""
    num_topics = len(topics[0]) 
    data = {i: [] for i in range(num_topics)}
    for doc in topics:
        for i in range(num_topics):
            prob_dict = dict(doc)
            data[i].append(prob_dict.get(i, 0))  
            
    df_probabilities = pd.DataFrame(data, index=applicants)
    df_summed = df_probabilities.groupby(df_probabilities.index).sum()
    return df_summed