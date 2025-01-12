import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .register import register_model

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, output_size):
        """
        LSTM 异常检测模型。

        Args:
            num_features (int): 每个时间步的特征数。
            hidden_size (int): LSTM 隐藏层大小。
            num_layers (int): LSTM 层数。
            output_size (int): 输出特征数（一般等于 num_features）。
        """
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, num_features)。

        Returns:
            Tensor: 预测值，形状为 (batch_size, num_features)。
        """
        lstm_out, _ = self.lstm(x)  # LSTM 输出 (batch_size, seq_len, hidden_size)
        out = self.fc(lstm_out[:, -1, :])  # 仅使用最后一个时间步的隐藏状态预测
        return out


def train_lstm(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    带进度条的 LSTM 训练函数。

    Args:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练轮数。
        device (torch.device): 运行设备。
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch") as tbar:
            for batch_data, _ in tbar:
                batch_data = batch_data.to(device)  # 输入数据 (batch_size, seq_len, num_features)
                inputs = batch_data[:, :-1, :]  # 前 n-1 项
                targets = batch_data[:, -1, :]  # 第 n 项

                optimizer.zero_grad()
                outputs = model(inputs)  # 预测第 n 项
                loss = criterion(outputs, targets)  # 计算损失
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tbar.set_postfix(loss=loss.item())
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate_lstm(model, test_loader, device):
    """
    带进度条的 LSTM 验证函数。

    Args:
        model (nn.Module): 要评估的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 运行设备。

    Returns:
        np.array: 所有窗口的预测误差分数。
        np.array: 所有窗口的真实标签。
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
            for batch_data, batch_labels in tbar:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.cpu().numpy()
                inputs = batch_data[:, :-1, :]  # 前 n-1 项
                targets = batch_data[:, -1, :]  # 第 n 项

                outputs = model(inputs)  # 预测第 n 项
                prediction_errors = torch.mean((outputs - targets) ** 2, dim=1)  # 计算预测误差
                all_scores.extend(prediction_errors.cpu().numpy())

                window_labels = (batch_labels.sum(axis=1) > 0).astype(int)
                all_labels.extend(window_labels)
                tbar.set_postfix(avg_pred_error=prediction_errors.mean().item())
    return np.array(all_scores), np.squeeze(all_labels)


# 注册模型到 MODEL_REGISTRY
register_model(
    name="LSTM",
    model_class=LSTMAnomalyDetector,
    train_fn=train_lstm,
    eval_fn=evaluate_lstm
)
