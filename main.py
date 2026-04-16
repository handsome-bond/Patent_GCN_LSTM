import argparse
from src.pipeline import run_preprocessing, run_nlp, run_graph_build, run_training

def main():
    parser = argparse.ArgumentParser(description="专利链路预测 GCN-LSTM Pipeline")
    parser.add_argument(
        '--stage', 
        type=str, 
        default='all',
        choices=['preprocess', 'nlp', 'graph', 'train', 'all'], 
        help="指定要运行的阶段: preprocess, nlp, graph, train 或者 all"
    )
    args = parser.parse_args()

    # 根据传入的参数决定跑哪一段代码流
    if args.stage in ['preprocess', 'all']:
        run_preprocessing()
        
    if args.stage in ['nlp', 'all']:
        run_nlp()
        
    if args.stage in ['graph', 'all']:
        run_graph_build()
        
    if args.stage in ['train', 'all']:
        run_training()

if __name__ == '__main__':
    main()