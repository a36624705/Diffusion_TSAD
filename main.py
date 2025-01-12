import os
import torch
import numpy as np
from data_loader import get_loader_segment
from model_utils import load_model
from models.register import get_train_fn, get_eval_fn
from evaluation.metrics import get_metrics
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly Detection Framework")

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="ConditionalDiffusion", help="Model name, e.g., 'AE', 'LSTM'.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers (for LSTM).")

    # 数据相关参数
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--win_size", type=int, default=128, help="Sliding window size.")
    parser.add_argument("--step", type=int, default=10, help="Sliding window step size.")
    parser.add_argument("--num_features", type=int, default=25, help="Number of features per time step.")

    # 训练相关参数
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--threshold", type=float, default=0.01, help="Anomaly detection threshold.")

    # 环境相关参数
    parser.add_argument("--dataset", type=str, default="PSM", help="Dataset name, e.g., 'PSM'.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'.")
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()

    # 配置路径
    data_path = os.environ.get("TS_DATASETS")
    if data_path is None:
        raise ValueError("环境变量 $TS_DATASETS 未设置，请设置为数据集的根路径。")
    data_path = os.path.join(data_path, args.dataset)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader = get_loader_segment(data_path, args.batch_size, args.win_size, args.step, mode='train', dataset=args.dataset)
    test_loader = get_loader_segment(data_path, args.batch_size, args.win_size, args.step, mode='test', dataset=args.dataset)

    # 加载模型
    model = load_model(args.model_name, args.num_features, args.win_size, args.hidden_size, args.num_layers).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 动态调用训练和评估函数
    train_fn = get_train_fn(args.model_name)
    eval_fn = get_eval_fn(args.model_name)

    # 训练
    train_fn(
        model=model, 
        train_loader=train_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=args.num_epochs, 
        device=device)

    # 评估
    all_scores, all_labels = eval_fn(model, test_loader, device)

    # 评估指标
    pred = (all_scores > args.threshold).astype(int)
    metrics = get_metrics(score=all_scores, labels=all_labels, pred=pred)
    print(metrics)

if __name__ == "__main__":
    main()
