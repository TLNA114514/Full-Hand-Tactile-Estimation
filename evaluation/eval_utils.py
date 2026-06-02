import numpy as np
import torch
import torch.optim as optim
from pytorch3d.transforms import matrix_to_axis_angle



def compute_similarity_transform(pred_points, gt_points, return_transform=False):
    """
    正交普氏分析 (Orthogonal Procrustes Alignment)
    在最小二乘意义上，通过旋转、平移和缩放对齐预测点云与 Ground Truth。
    
    参数:
        pred_points: (N, 3) 预测的三维坐标点 (numpy 数组)
        gt_points: (N, 3) 对应的 Ground Truth 三维坐标点 (numpy 数组)
        return_transform: 是否返回变换参数 (R, t, scale)
    
    返回:
        pred_points_aligned: 对齐后的预测三维点坐标 (N, 3)
    """
    assert pred_points.shape == gt_points.shape, f"预测与GT的维度不匹配: {pred_points.shape} vs {gt_points.shape}"

    # 1. 均值中心化
    mu_pred = pred_points.mean(axis=0, keepdims=True)
    mu_gt = gt_points.mean(axis=0, keepdims=True)
    X_pred = pred_points - mu_pred
    X_gt = gt_points - mu_gt

    # 2. 计算预测坐标方差 (方差用作缩放尺度恢复)
    var_pred = np.sum(X_pred**2)
    if var_pred < 1e-8:
        if return_transform:
            return pred_points + (mu_gt - mu_pred), np.eye(3), mu_gt - mu_pred, 1.0
        return pred_points + (mu_gt - mu_pred)

    # 3. 计算互协方差矩阵 K 并对其进行奇异值分解 (SVD)
    K = X_pred.T @ X_gt
    U, s, Vt = np.linalg.svd(K)
    V = Vt.T

    # 4. 构造反射矩阵 Z，修正方向，确保 R 属于正交群 SO(3) 避免镜像翻转
    Z = np.eye(3)
    Z[-1, -1] *= np.sign(np.linalg.det(U @ V.T))
    R = V @ Z @ U.T

    # 5. 恢复最优化缩放因子 scale
    scale = np.trace(R @ K) / var_pred

    # 6. 恢复最优平移向量 t
    t = mu_gt.T - scale * R @ mu_pred.T

    # 7. 对预测点云进行仿射变换并返回
    pred_points_aligned = (scale * R @ pred_points.T).T + t.T
    if return_transform:
        return pred_points_aligned, R, t, scale
    return pred_points_aligned

def compute_pa_mpvpe(pred_joints, gt_joints, pred_verts, gt_verts):
    """
    标准的 PA-MPVPE 计算方法：
    1. 计算将 pred_joints 普氏对齐到 gt_joints 的变换 (R, t, scale)
    2. 将该变换原封不动地应用到 pred_verts
    3. 计算变换后的 verts 与 gt_verts 之间的平均误差
    """
    is_batched = (pred_joints.ndim == 3)
    if not is_batched:
        pred_j = pred_joints[None, ...]
        gt_j = gt_joints[None, ...]
        pred_v = pred_verts[None, ...]
        gt_v = gt_verts[None, ...]
    else:
        pred_j = pred_joints.copy()
        gt_j = gt_joints.copy()
        pred_v = pred_verts.copy()
        gt_v = gt_verts.copy()
        
    B, N, C = pred_j.shape
    errors = []
    
    for i in range(B):
        # 计算关节的有效掩码
        mask = (~np.isnan(pred_j[i]).any(axis=-1)) & (~np.isnan(gt_j[i]).any(axis=-1))
        if not mask.any():
            continue
            
        p_j = pred_j[i][mask]
        g_j = gt_j[i][mask]
        
        # 获取由关节计算出的变换矩阵
        _, R, t, scale = compute_similarity_transform(p_j, g_j, return_transform=True)
        
        # 将相同的变换应用到所有的顶点上
        p_v = pred_v[i]
        g_v = gt_v[i]
        
        p_v_aligned = (scale * R @ p_v.T).T + t.T
        
        dist = np.linalg.norm(p_v_aligned - g_v, ord=2, axis=-1)
        errors.append(dist.mean() * 1000.0) # 米转毫米
        
    return np.mean(errors) if len(errors) > 0 else 0.0

def compute_mpjpe(pred_joints, gt_joints, alignment='procrustes'):
    """
    计算平均关节点位置误差 (MPJPE)。
    若 alignment='procrustes'，则计算对齐后的误差 PA-MPJPE。
    """
    is_batched = (pred_joints.ndim == 3)
    if not is_batched:
        pred = pred_joints[None, ...]
        gt = gt_joints[None, ...]
    else:
        pred = pred_joints.copy()
        gt = gt_joints.copy()
        
    B, N, C = pred.shape
    errors = []
    
    for i in range(B):
        mask = (~np.isnan(pred[i]).any(axis=-1)) & (~np.isnan(gt[i]).any(axis=-1))
        if not mask.any():
            continue
            
        p = pred[i][mask]
        g = gt[i][mask]
        
        if alignment == 'procrustes':
            p = compute_similarity_transform(p, g)
            
        dist = np.linalg.norm(p - g, ord=2, axis=-1)
        errors.append(dist.mean() * 1000.0) # 米转毫米
        
    return np.mean(errors) if len(errors) > 0 else 0.0

def compute_pck(pred_joints, gt_joints, threshold_mm=5.0, alignment='procrustes'):
    """
    计算三维手部姿态估计的百分比正确关键点 (3D PCK)。
    """
    is_batched = (pred_joints.ndim == 3)
    if not is_batched:
        pred = pred_joints[None, ...]
        gt = gt_joints[None, ...]
    else:
        pred = pred_joints.copy()
        gt = gt_joints.copy()
        
    B, N, C = pred.shape
    total_joints = 0
    correct_joints = 0
    
    for i in range(B):
        mask = (~np.isnan(pred[i]).any(axis=-1)) & (~np.isnan(gt[i]).any(axis=-1))
        if not mask.any():
            continue
            
        p = pred[i][mask]
        g = gt[i][mask]
        
        if alignment == 'procrustes':
            p = compute_similarity_transform(p, g)
            
        dists = np.linalg.norm(p - g, ord=2, axis=-1) * 1000.0 # 转为 mm
        correct_joints += np.sum(dists <= threshold_mm)
        total_joints += len(dists)
        
    return (correct_joints / total_joints) * 100.0 if total_joints > 0 else 0.0

def compute_auc(pred_joints, gt_joints, min_thr=0.0, max_thr=50.0, num_steps=31, alignment='procrustes'):
    """
    计算在指定阈值范围内的 3D PCK 曲线下面积 (3D AUC)。
    """
    thresholds = np.linspace(min_thr, max_thr, num_steps)
    pck_values = []
    
    for thr in thresholds:
        pck = compute_pck(pred_joints, gt_joints, threshold_mm=thr, alignment=alignment)
        pck_values.append(pck / 100.0)
        
    return np.mean(pck_values) * 100.0

def mano_to_mediapipe_torch(mano):
    """
    将 [B, 21, 3] 的 MANO 关节的 PyTorch Tensor 重映射到 MediaPipe 21 关节标准
    """
    B = mano.shape[0]
    device = mano.device
    mp_joints = torch.zeros((B, 21, 3), dtype=mano.dtype, device=device)
    
    mp_joints[:, 0] = mano[:, 0]
    
    mp_joints[:, 1] = mano[:, 13]
    mp_joints[:, 2] = mano[:, 14]
    mp_joints[:, 3] = mano[:, 15]
    mp_joints[:, 4] = mano[:, 20]
    
    mp_joints[:, 5] = mano[:, 1]
    mp_joints[:, 6] = mano[:, 2]
    mp_joints[:, 7] = mano[:, 3]
    mp_joints[:, 8] = mano[:, 16]
    
    mp_joints[:, 9] = mano[:, 4]
    mp_joints[:, 10] = mano[:, 5]
    mp_joints[:, 11] = mano[:, 6]
    mp_joints[:, 12] = mano[:, 17]
    
    mp_joints[:, 13] = mano[:, 10]
    mp_joints[:, 14] = mano[:, 11]
    mp_joints[:, 15] = mano[:, 12]
    mp_joints[:, 16] = mano[:, 19]
    
    mp_joints[:, 17] = mano[:, 7]
    mp_joints[:, 18] = mano[:, 8]
    mp_joints[:, 19] = mano[:, 9]
    mp_joints[:, 20] = mano[:, 18]
    
    return mp_joints

def fit_mano_to_joints(mano_model, gt_joints, device, num_steps=60):
    """
    使用 PyTorch 内部的 L-BFGS 快速优化器将 MANO 模型拟合到 21 个 GT 关节点上，获取 GT 778 个网格顶点。
    这允许我们在没有提供 GT 网格的情况下准确评估 PA-MPVPE 顶点误差。
    """
    import torch.optim as optim
    gt_j = torch.tensor(gt_joints, dtype=torch.float32, device=device).unsqueeze(0) # [1, 21, 3]
    
    # 建立优化参数
    global_orient = torch.eye(3, device=device, dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(True)
    hand_pose = torch.eye(3, device=device, dtype=torch.float32).repeat(1, 15, 1, 1).requires_grad_(True)
    transl = torch.zeros(1, 3, device=device, dtype=torch.float32).requires_grad_(True)
    betas = torch.zeros(1, 10, device=device, dtype=torch.float32).requires_grad_(True)
    
    optimizer = optim.LBFGS([global_orient, hand_pose, transl, betas], lr=0.5, max_iter=num_steps)
    
    def closure():
        optimizer.zero_grad()
        mano_out = mano_model(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl, pose2rot=False)
        pred_j_mp = mano_out.joints # [1, 21, 3] 已在 MANO_wrapper 中对齐到 MediaPipe
        loss = torch.mean((pred_j_mp - gt_j)**2)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    with torch.no_grad():
        mano_out = mano_model(global_orient=global_orient, hand_pose=hand_pose, betas=betas, transl=transl, pose2rot=False)
        gt_vertices = mano_out.vertices[0].cpu().numpy()
        
    return gt_vertices

