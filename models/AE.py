import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .register import register_model

class Autoencoder(nn.Module):
    def __init__(self, num_features, win_size, hidden_size):
        """
        自编码器模型。

        Args:
            num_features (int): 每个时间步的特征数。
            win_size (int): 滑动窗口的长度。
            hidden_size (int): 隐藏层大小。
        """
        super(Autoencoder, self).__init__()
        self.input_size = num_features * win_size  # 自动计算展平后的输入大小
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.input_size)
        )

    def forward(self, x):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, win_size, num_features)。
        Returns:
            reconstructed (Tensor): 重构后的张量，展平为 (batch_size, input_size)。
        """
        batch_size = x.size(0)
        # 展平输入为 (batch_size, input_size)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_ae(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    带进度条的 Autoencoder 训练函数。

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
        # 包装进度条
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch") as tbar:
            for batch_data, _ in tbar:
                batch_data = batch_data.to(device).view(batch_data.size(0), -1)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # 更新进度条中的当前损失
                tbar.set_postfix(loss=loss.item())

        # 使用 tqdm.write 打印日志，避免与进度条冲突
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(train_loader):.4f}")

def evaluate_ae(model, test_loader, device):
    """
    带进度条的 Autoencoder 验证函数。

    Args:
        model (nn.Module): 要评估的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 运行设备。

    Returns:
        np.array: 所有窗口的重构误差分数。
        np.array: 所有窗口的真实标签。
    """
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        # 包装测试数据加载器为进度条
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
            for batch_data, batch_labels in tbar:
                # 将数据移动到设备并展平
                batch_data = batch_data.to(device).view(batch_data.size(0), -1)
                batch_labels = batch_labels.cpu().numpy()

                # 模型前向推断
                outputs = model(batch_data)
                recon_errors = torch.mean((outputs - batch_data) ** 2, dim=1)

                # 收集结果
                all_scores.extend(recon_errors.cpu().numpy())
                window_labels = (batch_labels.sum(axis=1) > 0).astype(int)
                all_labels.extend(window_labels)

                # 更新进度条显示当前批次的平均重构误差
                tbar.set_postfix(avg_recon_error=recon_errors.mean().item())

    # 返回所有分数和标签
    return np.array(all_scores), np.squeeze(all_labels)

# 注册模型到 MODEL_REGISTRY
register_model(
    name="AE",
    model_class=Autoencoder,
    train_fn=train_ae,
    eval_fn=evaluate_ae
)
