import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .register import register_model


class VariationalAutoencoder(nn.Module):
    def __init__(self, num_features, win_size, hidden_size, latent_size):
        """
        变分自编码器 (VAE) 模型。

        Args:
            num_features (int): 每个时间步的特征数。
            win_size (int): 滑动窗口的长度。
            hidden_size (int): 隐藏层大小。
            latent_size (int): 潜在空间大小。
        """
        super(VariationalAutoencoder, self).__init__()
        self.input_size = num_features * win_size  # 展平后的输入大小

        # 编码器：生成均值和对数方差
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_size // 2, latent_size)  # 均值
        self.log_var_layer = nn.Linear(hidden_size // 2, latent_size)  # 对数方差

        # 解码器：重构输入
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.input_size),
        )

    def reparameterize(self, mu, log_var):
        """
        重参数化技巧：通过随机噪声生成潜在向量。
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # 随机噪声
        return mu + eps * std

    def forward(self, x):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (batch_size, win_size, num_features)。
        Returns:
            Tensor: 重构后的输入。
            Tensor: 均值 (mu)。
            Tensor: 对数方差 (log_var)。
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平输入
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        z = self.reparameterize(mu, log_var)  # 采样潜在向量
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var


def vae_loss_function(reconstructed, original, mu, log_var):
    """
    VAE 损失函数 = 重构误差 + KL 散度。

    Args:
        reconstructed (Tensor): 重构后的输入。
        original (Tensor): 原始输入。
        mu (Tensor): 均值。
        log_var (Tensor): 对数方差。

    Returns:
        Tensor: 总损失。
    """
    # 重构误差 (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction="mean")
    # KL 散度
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / original.size(0)
    return recon_loss + kl_divergence


def train_vae(model, train_loader, optimizer, num_epochs, device, *args, **kwargs):
    """
    带进度条的 VAE 训练函数。

    Args:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练轮数。
        device (torch.device): 运行设备。
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch") as tbar:
            for batch_data, _ in tbar:
                batch_data = batch_data.to(device).view(batch_data.size(0), -1)
                optimizer.zero_grad()
                reconstructed, mu, log_var = model(batch_data)
                loss = vae_loss_function(reconstructed, batch_data, mu, log_var)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tbar.set_postfix(loss=loss.item())
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate_vae(model, test_loader, device):
    """
    带进度条的 VAE 验证函数。

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
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
            for batch_data, batch_labels in tbar:
                batch_data = batch_data.to(device).view(batch_data.size(0), -1)
                batch_labels = batch_labels.cpu().numpy()
                reconstructed, _, _ = model(batch_data)
                recon_errors = torch.mean((reconstructed - batch_data) ** 2, dim=1)
                all_scores.extend(recon_errors.cpu().numpy())
                window_labels = (batch_labels.sum(axis=1) > 0).astype(int)
                all_labels.extend(window_labels)
                tbar.set_postfix(avg_recon_error=recon_errors.mean().item())
    return np.array(all_scores), np.squeeze(all_labels)


# 注册模型到 MODEL_REGISTRY
register_model(
    name="VAE",
    model_class=VariationalAutoencoder,
    train_fn=train_vae,
    eval_fn=evaluate_vae
)
