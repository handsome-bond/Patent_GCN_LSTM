# 🧬 生物医药产业链多层网络合作链路演化推断

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg)
![Gensim](https://img.shields.io/badge/NLP-Doc--LDA-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

[cite_start]本项目基于 **Doc-LDA 主题模型** 与 **GCN-LSTM 深度学习模型**，对生物医药关键核心技术领域的创新主体（高校、企业、研究院、医院）在产业链上、中、下游多层网络中的合作关系进行演化推断与链路预测 [cite: 5, 6]。

> [cite_start]**📖 学术背景**：本项目为全国大学生相关学术竞赛参赛作品代码实现。旨在通过大数据与人工智能技术，精准识别“卡脖子”核心技术，并动态推断产学研主体的合作趋势，为生物医药产业集群建设提供情报支撑 [cite: 6]。

---

## ✨ 核心亮点 & 创新点

1.  [cite_start]**主客观结合的核心专利识别**：创新性结合 **熵权法 (Entropy Weight Method)** 与 **K-Means 聚类**，从海量专利中精准筛除低价值“噪声”，提取出高技术价值与市场价值的“核心专利” [cite: 6]。

2.  [cite_start]**Doc-LDA 关键技术主题提取**：结合 Doc2Vec 文本向量化与 LDA 主题模型，计算技术主题强度、共现强度与有效凝聚约束系数，将生物医药技术精准映射至产业链的 **上游、中游、下游** [cite: 6]。

3.  [cite_start]**多层异构网络构建 (Multi-layer Network)**：打破单层网络局限，以上、中、下游为三层物理空间，构建包含节点特征矩阵与非对称邻接矩阵的 PyTorch Geometric (PyG) 图张量 [cite: 6]。

4.  **GCN-LSTM 时空链路演化推断**：
    * [cite_start]**空间特征提取**：利用 `FusionGCN` 并行提取上中下三层图特征，并通过拼接 (`torch.cat`) 融合产业链多维拓扑结构 [cite: 5]。
    * [cite_start]**时序动态推断**：引入 `LSTM` 长短时记忆网络，捕捉 2005-2019 年合作关系的历史演变序列，精准预测 2020-2022 年的潜在产学研合作链路 [cite: 6]。

---

## 📁 项目目录结构

```text
Patent_GCN_LSTM/
├── data/                       # 存放数据文件
│   ├── raw/                    # 原始专利 Excel 数据 (allnew.xlsx)
│   └── processed/              # 运行过程中生成的特征矩阵、张量及中间数据
├── models/                     # 存放训练好的模型权重与词典
│   ├── your_dict.dict          # Gensim 词典文件
│   ├── your_lda.model          # 预训练的 LDA 主题模型
│   ├── gcn_lstm2_model.pth     # 训练完成的 GCN-LSTM 权重
│   └── similarity_hist.png     # 预测链路的相似度分布直方图
├── src/                        # 核心源代码包
│   ├── __init__.py
│   ├── config.py               # 全局超参数与路径配置
│   ├── data_preprocessing.py   # 数据清洗、熵权法与 K-Means 聚类
│   ├── nlp_pipeline.py         # 正则清洗、分词与 Doc-LDA 主题提取
│   ├── graph_builder.py        # 多层网络邻接矩阵构建与 PyG 图数据生成
│   ├── networks.py             # FusionGCN 与 GCN-LSTM 神经网络结构定义
│   ├── train_eval.py           # 负采样增强、模型训练、多指标评估与链路预测
│   └── pipeline.py             # 业务流水线调度代码
├── main.py                     # 🚀 程序的总控制台入口
├── train_lda.py                # 独立脚本：用于重新训练 LDA 模型
├── requirements.txt            # 项目依赖包
└── README.md                   # 项目说明文档
```

---

## 🛠️ 环境依赖与安装

本项目依赖于 PyTorch 及其图神经网络扩展库 PyTorch Geometric (PyG)。建议使用 Anaconda 创建独立的虚拟环境。

1.  **克隆仓库**：

    ```bash
    git clone [https://github.com/handsome-bond/Patent_GCN_LSTM.git](https://github.com/handsome-bond/Patent_GCN_LSTM.git)
    cd Patent_GCN_LSTM
    ```

2.  **安装依赖**：

    ```bash
    pip install -r requirements.txt
    ```

    *(注：如果运行中提示 `torch_geometric` 报错，请根据你的 CUDA 版本前往 [PyG 官方网站](https://pytorch-geometric.readthedocs.io/) 获取对应的安装命令)*

---

## 🚀 运行指南 (Pipeline)

本项目采用高度解耦的工业级流水线架构，支持通过 `--stage` 参数分阶段执行，有效防止内存溢出并支持断点续传。

### 阶段 1：数据预处理 (Preprocess)

执行熵权法计算综合评价得分，并通过 K-Means 聚类筛选核心专利，按年份拆分数据。

```bash
python main.py --stage preprocess
```

### 阶段 2：自然语言处理与主题提取 (NLP)

读取预先训练好的 Doc-LDA 模型，对专利摘要进行分析，生成各组的“申请人-主题概率”特征矩阵。

```bash
# 如果你的 models 文件夹下没有 LDA 模型，请先执行：
# python train_lda.py

python main.py --stage nlp
```

### 阶段 3：多层网络图构建 (Graph)

基于上中下游合作关系构建非对称邻接矩阵，统一节点特征，并打包为 PyTorch 张量 (`graph_tensors.pt`)。

```bash
python main.py --stage graph
```

### 阶段 4：GCN-LSTM 模型训练与预测 (Train)

进行负采样数据增强，使用 Adam 优化器训练 GCN-LSTM 模型，输出各项评估指标（Accuracy, Precision, Recall, F1, AUC），并基于余弦相似度（堆排序优化）预测未来 Top 10 合作链路。

```bash
python main.py --stage train
```

### 一键运行全流程

如果你想从头到尾一口气执行完毕（需要较长的时间与较大的内存）：

```bash
python main.py --stage all
```

---

## 📊 模型评估与预测结果

### 1. 网络评估指标

根据实际划分的上游、中游、下游三层图结构分别进行评估，GCN-LSTM 模型表现出极强的鲁棒性与泛化能力：

* [cite_start]**AUC** > 0.5 （整体预测效果远高于随机推断） [cite: 6]。
* [cite_start]**Precision / Recall / F1-Score** 均保持在较高水平，准确捕捉了时空序列中的拓扑特征演化 [cite: 5]。

### 2. 产学研演化推断实例

* [cite_start]**上游（高校与企业）**：模型成功推断出中南大学与相关化工、医药科技企业（如四川轻化工大学等）的潜在合作链路，合作方向聚焦于绿色化合物制备技术 [cite: 6]。
* [cite_start]**中游（大型企业群）**：精准捕捉如“中国石油化工股份有限公司”与相关研究院之间的联合攻关趋势，攻坚制备装置等大规模生产工艺 [cite: 6]。
* [cite_start]**下游（企业与医疗机构）**：揭示了服务落地环节（如智能化监控、给药系统）的紧密联合生态 [cite: 6]。

*(具体链路预测分数与相似度直方图将保存在 `models/similarity_hist.png` 中)*

---

## 📝 作者与版权信息

* [cite_start]本项目为学术竞赛专有代码，算法逻辑与指标体系设计详情请参考对应学术论文《生物医药关键核心创新主体在产业链多层网络中的合作关系演化推断分析》 [cite: 6]。
* **作者**: handsome-bond
* **License**: MIT License