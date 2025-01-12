import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .register import register_model


class DiffusionModel(nn.Module):
    def __init__(self, input_size, time_steps=100, beta_min=0.1, beta_max=20.0):
        """
        扩散模型用于时序数据重构。

        Args:
            input_size (int): 输入特征维度。
            time_steps (int): 扩散时间步数。
            beta_min (float): 扩散过程的最小 beta 值。
            beta_max (float): 扩散过程的最大 beta 值。
        """
        super(DiffusionModel, self).__init__()
        self.input_size = input_size
        self.time_steps = time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # 定义噪声调度
        self.beta_schedule = torch.linspace(beta_min, beta_max, time_steps)
        self.alpha = 1 - self.beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 模型网络（简单 MLP）
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_size)
        )

    def forward(self, x, t):
        """
        前向传播：扩散过程预测噪声。

        Args:
            x (Tensor): 输入数据，形状为 (batch_size, input_size)。
            t (Tensor): 时间步，形状为 (batch_size,)。

        Returns:
            Tensor: 预测噪声，形状与 x 相同。
        """
        # 时间步嵌入
        time_embed = t.float().unsqueeze(1) / self.time_steps  # 形状变为 (batch_size, 1)
        time_embed = time_embed.repeat(1, x.size(1))  # 扩展到与 x 相同的形状 (batch_size, input_size)
        return self.model(x + time_embed)


def diffusion_process(model, x, time_steps, alpha_bar):
    """
    通过扩散过程重构输入数据。

    Args:
        model (nn.Module): 扩散模型。
        x (Tensor): 输入数据。
        time_steps (int): 扩散时间步数。
        alpha_bar (Tensor): 累积 alpha。

    Returns:
        Tensor: 重构后的数据。
    """
    # 随机采样起始点
    batch_size = x.size(0)
    z = torch.randn_like(x)
    for t in reversed(range(time_steps)):
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
        predicted_noise = model(z, t_tensor)
        z = (z - predicted_noise) / torch.sqrt(alpha_bar[t])  # 去噪过程
    return z


def train_diffusion(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    带进度条的扩散模型训练函数。

    Args:
        model (nn.Module): 要训练的扩散模型。
        train_loader (DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练轮数。
        device (torch.device): 运行设备。
    """
    model.train()
    model.alpha_bar = model.alpha_bar.to(device)  # 将 alpha_bar 移动到指定设备

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch") as tbar:
            for batch_data, _ in tbar:
                batch_data = batch_data.to(device).view(batch_data.size(0), -1)
                optimizer.zero_grad()

                # 随机采样时间步
                t = torch.randint(0, model.time_steps, (batch_data.size(0),), device=device)
                noise = torch.randn_like(batch_data)

                # 扩展 alpha_bar 和 (1 - alpha_bar) 的形状，使其与 batch_data 对齐
                alpha_bar_t = model.alpha_bar[t].view(-1, 1)  # 形状变为 (batch_size, 1)
                noisy_data = batch_data * torch.sqrt(alpha_bar_t) + noise * torch.sqrt(1 - alpha_bar_t)

                # 预测噪声
                predicted_noise = model(noisy_data, t)
                loss = criterion(predicted_noise, noise)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tbar.set_postfix(loss=loss.item())
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate_diffusion(model, test_loader, device):
    """
    带进度条的扩散模型验证函数。

    Args:
        model (nn.Module): 扩散模型。
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

                # 重构输入数据
                reconstructed_data = diffusion_process(model, batch_data, model.time_steps, model.alpha_bar)
                recon_errors = torch.mean((reconstructed_data - batch_data) ** 2, dim=1)  # 计算重构误差

                all_scores.extend(recon_errors.cpu().numpy())
                window_labels = (batch_labels.sum(axis=1) > 0).astype(int)
                all_labels.extend(window_labels)
                tbar.set_postfix(avg_recon_error=recon_errors.mean().item())
    return np.array(all_scores), np.squeeze(all_labels)


# 注册模型到 MODEL_REGISTRY
register_model(
    name="Diffusion",
    model_class=DiffusionModel,
    train_fn=train_diffusion,
    eval_fn=evaluate_diffusion
)
