import numpy as np


def _gaussian_weights(sigma):
    """
    用高斯公式计算三帧 kernel 权重并归一化。

    公式：g(x) = exp(-x^2 / (2 * sigma^2))，在 x = -1, 0, 1 处采样。
    权重之和归一化为 1。

    Args:
        sigma: 高斯标准差，控制平滑程度。越大越平滑，越小越接近原始值。
    Returns:
        (w_prev, w_curr, w_next) 归一化后的三帧权重 tuple。
    """
    xs = np.array([-1.0, 0.0, 1.0])
    weights = np.exp(-xs ** 2 / (2.0 * sigma ** 2))
    weights /= weights.sum()
    return tuple(weights.tolist())


# 预设的平滑模式，方便直接按名字选用
# gaussian 权重由公式自动计算（sigma=1.0）：约 (0.274, 0.452, 0.274)
_g = _gaussian_weights(sigma=1.0)

SMOOTH_PRESETS = {
    "none":             (0.0,  1.0,  0.0),   # 不平滑，只取当前帧
    "moving_average":   (1/3,  1/3,  1/3),   # 三帧均值
    "ema":              (0.3,  0.7,  0.0),   # Exponential Moving Average（偏重当前帧）
    "gaussian":         _g,                  # Gaussian-temporal, sigma=1.0
}


def smooth_tactile(pressures, frame_idx, weights="none", sigma=1.0):
    """
    对触觉压力序列做时序加权平滑。

    Args:
        pressures:  HDF5 dataset 或类数组，形如 (T, H, W) 或 (T, N)，
                    包含整段 demo 的压力帧序列。
        frame_idx:  当前帧的索引（int）。
        weights:    平滑方式，支持三种传入形式：
                    - str 预设名称：
                        "none"           → (0.0, 1.0, 0.0)       不平滑
                        "moving_average" → (1/3, 1/3, 1/3)       三帧均值
                        "ema"            → (0.3, 0.7, 0.0)        Exponential Moving Average
                        "gaussian"       → 由公式计算，受 sigma 控制
                    - tuple(w_prev, w_curr, w_next)：自定义权重，函数内自动归一化
                    - "gaussian" 配合 sigma 参数可调整平滑强度
        sigma:      仅在 weights="gaussian" 时生效，控制高斯核宽度（默认 1.0）。
                    sigma 越大越平滑，sigma 越小越接近原始值。

    Returns:
        smoothed: np.ndarray (float32)，与单帧 pressure 形状相同的平滑结果。

    Examples:
        smooth_tactile(pressures, i)                          # 不平滑（默认）
        smooth_tactile(pressures, i, weights="moving_average")
        smooth_tactile(pressures, i, weights="gaussian")      # sigma=1.0
        smooth_tactile(pressures, i, weights="gaussian", sigma=0.5)  # 较弱平滑
        smooth_tactile(pressures, i, weights=(0.1, 0.8, 0.1))        # 自定义
    """
    # 解析 weights
    if isinstance(weights, str):
        if weights == "gaussian":
            w_prev, w_curr, w_next = _gaussian_weights(sigma)
        elif weights in SMOOTH_PRESETS:
            w_prev, w_curr, w_next = SMOOTH_PRESETS[weights]
        else:
            raise ValueError(
                f"Unknown preset '{weights}'. "
                f"Available: {list(SMOOTH_PRESETS.keys())}"
            )
    else:
        w_prev, w_curr, w_next = weights

    # 自动归一化，避免权重和不为 1 导致幅度偏移
    total = w_prev + w_curr + w_next
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    w_prev, w_curr, w_next = w_prev / total, w_curr / total, w_next / total

    prev_idx = max(frame_idx - 1, 0)
    next_idx = min(frame_idx + 1, len(pressures) - 1)

    t_prev = pressures[prev_idx].astype(np.float32)
    t_curr = pressures[frame_idx].astype(np.float32)
    t_next = pressures[next_idx].astype(np.float32)

    smoothed = w_prev * t_prev + w_curr * t_curr + w_next * t_next

    return smoothed
