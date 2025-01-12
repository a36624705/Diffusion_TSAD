import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from data_loader import get_loader_segment
from evaluation.metrics import get_metrics

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义简单的LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 输出 hidden state
        lstm_out, _ = self.lstm(x)
        # 只取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out


# 定义训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # 使用 tqdm 包装训练集加载器
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]") as tbar:
            for batch_data, _ in tbar:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                # 前向传播
                outputs = model(batch_data)
                # 使用 MSE 作为重构误差
                loss = criterion(outputs, batch_data[:, -1, :])
                # 反向传播
                loss.backward()
                optimizer.step()
                # 累积损失
                epoch_loss += loss.item()
                # 在进度条中显示当前损失
                tbar.set_postfix(loss=loss.item())
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate_model(model, test_loader, threshold, device):
    """
    使用 get_metrics 函数对模型进行评估，计算各项指标。
    """
    model.eval()
    all_scores = []  # 存储所有窗口的异常分数
    all_labels = []  # 存储所有窗口的真实标签
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.cpu().numpy()  # 转换为 numpy 格式
            outputs = model(batch_data)
            recon_errors = torch.mean((outputs - batch_data[:, -1, :]) ** 2, dim=1)
            all_scores.extend(recon_errors.cpu().numpy())
            
            # 聚合窗口内的标签 (逻辑 OR 聚合)
            window_labels = (batch_labels.sum(axis=1) > 0).astype(int)
            all_labels.extend(window_labels)

    # 转换为 numpy 数组
    all_scores = np.array(all_scores)
    all_labels = np.squeeze(all_labels)  # 注意：此时长度与 all_scores 一致

    # 基于阈值计算预测标签
    pred = (all_scores > threshold).astype(int)

    # 调用 get_metrics 进行评估
    metrics = get_metrics(score=all_scores, labels=all_labels, pred=pred)
    
    # 打印指标结果
    print("Evaluation Results:")
    print(f"AUC-PR: {metrics['AUC-PR']:.4f}")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")
    print(f"Standard-F1: {metrics['Standard-F1']:.4f}")
    print(f"Affiliation-F: {metrics['Affiliation-F']:.4f}")
    return metrics



# 主函数中加入评价流程
def main():
    # 配置路径与超参数
    data_path = os.environ.get("TS_DATASETS")
    if data_path is None:
        raise ValueError("环境变量 $TS_DATASETS 未设置，请设置为数据集的根路径。")
    data_path = os.path.join(data_path, "PSM")

    batch_size = 32
    win_size = 100
    step = 10
    num_epochs = 10
    learning_rate = 0.001
    input_size = 25  # 假设 PSM 数据集有 25 个特征
    hidden_size = 64
    num_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 PSM 数据加载器加载数据
    train_loader = get_loader_segment(data_path, batch_size, win_size, step, mode='train', dataset='PSM')
    test_loader = get_loader_segment(data_path, batch_size, win_size, step, mode='test', dataset='PSM')

    # 定义模型、损失函数和优化器
    model = SimpleLSTM(input_size, hidden_size, num_layers, input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 设定阈值并检测异常
    threshold = 0.01  # 假设一个简单的阈值
    metrics = evaluate_model(model, test_loader, threshold=threshold, device=device)


if __name__ == "__main__":
    main()

