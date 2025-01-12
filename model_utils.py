from models.register import get_model_class

def load_model(name, num_features, win_size, hidden_size, num_layers=None):
    """
    动态加载模型类并实例化。

    Args:
        name (str): 模型名称，如 "AE", "LSTM"。
        num_features (int): 每个时间步的特征数。
        win_size (int): 滑动窗口大小。
        hidden_size (int): 隐藏层大小。
        num_layers (int, optional): LSTM 的层数，其他模型可忽略。

    Returns:
        model: 已实例化的模型。
    """
    model_class = get_model_class(name)
    if name == "AE":
        return model_class(num_features, win_size, hidden_size)
    elif name == "LSTM":
        return model_class(num_features, hidden_size, num_layers, num_features)
    elif name == "VAE":
        return model_class(num_features, win_size, hidden_size, latent_size=32)
    elif name == "LSTM":
        return model_class(num_features, hidden_size, num_layers, output_size=num_features)
    elif name == "Diffusion":
        return model_class(input_size=num_features*win_size, time_steps=100, beta_min=0.1, beta_max=20.0)
    elif name == "ConditionalDiffusion":
        return model_class(num_features, win_size, hidden_size, time_steps=50)
    else:
        raise ValueError(f"Model '{name}' is not supported.")

