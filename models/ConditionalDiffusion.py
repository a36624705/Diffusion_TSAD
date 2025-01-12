import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .register import register_model
import torch.nn.functional as F


class UNet1D(nn.Module):
    def __init__(self):
        """
        初始化一维 U-Net 模型，固定为 3 层编码和 3 层解码。
        
        参数：
            input_dim (int): 输入数据的维度（必须是 8 的倍数）
        """
        super(UNet1D, self).__init__()

        # 下采样路径（编码器）
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 上采样路径（解码器）
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 最终输出层
        self.final_layer = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, x):
        """
        前向传播
        
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels=1, seq_length)
        
        返回：
            torch.Tensor: 输出张量，形状为 (batch_size, channels=1, seq_length)
        """
        x = x.unsqueeze(1)

        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # 瓶颈层
        bottleneck = self.bottleneck(enc3)

        # 解码器
        dec3 = self.dec3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 跳跃连接

        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 跳跃连接

        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 跳跃连接

        # 输出层
        output = self.final_layer(dec1)
        output = output.squeeze(1)
        return output


class ConditionalDiffusion(nn.Module):
    def __init__(self, num_features, win_size, hidden_size, time_steps):
        super(ConditionalDiffusion, self).__init__()
        self.time_steps = time_steps
        half_win = win_size // 2

        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(num_features * half_win, hidden_size * half_win),  # 修正输入维度为 num_features * win_size
            nn.ReLU(),
            nn.Linear(hidden_size * half_win, hidden_size)
        )

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_steps, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # UNet 模型
        self.model = UNet1D()


    def positional_encoding(self, t, device):
        position = t.unsqueeze(-1).float()
        div_term = torch.exp(torch.arange(0, self.time_steps, 2, device=device) * -(np.log(10000.0) / self.time_steps))
        encoding = torch.zeros((t.size(0), self.time_steps), device=device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, x, t, h_cond):
        t_embed = self.positional_encoding(t, x.device)
        t_embed = self.time_embed(t_embed)


        x_cond = torch.cat([x, h_cond + t_embed], dim=-1)  # 拼接输入
        return self.model(x_cond)



def compute_alpha_schedule(time_steps, device):
    """
    计算扩散过程中的 alpha 和 beta 参数。
    """
    beta = torch.linspace(1e-4, 0.02, time_steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def train_diffusion(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    条件扩散模型训练函数。

    Args:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数，例如 nn.MSELoss()。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练轮数。
        device (torch.device): 运行设备。
    """
    model.train()
    beta, alpha, alpha_bar = compute_alpha_schedule(model.time_steps, device)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch") as tbar:
            for batch_data, _ in tbar:
                batch_data = batch_data.to(device)
                optimizer.zero_grad()

                # 条件编码
                x_cond = batch_data[:, :batch_data.size(1) // 2, :]
                x_true = batch_data[:, batch_data.size(1) // 2:, :]
                h_cond = model.condition_encoder(x_cond.view(x_cond.size(0), -1))

                # # 随机时间步
                # t = torch.randint(0, model.time_steps, (x_true.size(0),), device=device).long()

                # 固定时间步
                t_fixed = model.time_steps // 2  # 选择中间时间步
                t = torch.full((x_true.size(0),), t_fixed, device=device, dtype=torch.long)

                # 加噪声
                noise = torch.randn_like(x_true)
                x_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1) * x_true + \
                      torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1) * noise

                # 噪声预测
                predicted_noise = model(x_t.view(x_t.size(0), -1), t, h_cond)

                # 使用外部损失函数计算损失
                loss = criterion(predicted_noise, noise.view(noise.size(0), -1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # 更新进度条
                tbar.set_postfix(loss=loss.item())

        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(train_loader):.4f}")



def evaluate_diffusion(model, test_loader, device):
    """
    条件扩散模型评估函数。

    Args:
        model (nn.Module): 要评估的模型。
        test_loader (DataLoader): 测试数据加载器。
        device (torch.device): 运行设备。

    Returns:
        np.array: 所有窗口的重构误差分数。
        np.array: 所有窗口的真实标签。
    """
    model.eval()
    beta, alpha, alpha_bar = compute_alpha_schedule(model.time_steps, device)
    all_scores, all_labels = [], []

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
            for batch_data, batch_labels in tbar:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.cpu().numpy()

                # 条件编码
                x_cond = batch_data[:, :batch_data.size(1) // 2, :]
                x_true = batch_data[:, batch_data.size(1) // 2:, :]
                h_cond = model.condition_encoder(x_cond.view(x_cond.size(0), -1))

                # 初始噪声
                z = torch.randn_like(x_true)
                for t in reversed(range(model.time_steps)):
                    t_tensor = torch.full((x_cond.size(0),), t, device=device, dtype=torch.long)
                    predicted_noise = model(z.view(z.size(0), -1), t_tensor, h_cond)
                    z = (z - (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t]) * predicted_noise.view_as(z)) / torch.sqrt(alpha[t])

                # 计算重构误差
                recon_errors = torch.mean((z - x_true) ** 2, dim=(1, 2))
                all_scores.extend(recon_errors.cpu().numpy())

                # 聚合窗口内的标签
                window_labels = (batch_labels.sum(axis=1) > 0).astype(int)
                all_labels.extend(window_labels)

                tbar.set_postfix(avg_recon_error=recon_errors.mean().item())

    return np.array(all_scores), np.squeeze(all_labels)


# 注册模型到 MODEL_REGISTRY
register_model(
    name="ConditionalDiffusion",
    model_class=ConditionalDiffusion,
    train_fn=train_diffusion,
    eval_fn=evaluate_diffusion
)
