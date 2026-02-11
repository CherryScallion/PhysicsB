#!/usr/bin/env python3
"""
【最终可视化脚本】模型预测结果的 3D 渲染与对比
- 使用经过验证的物理仿射矩阵 (Fixing coordinates)
- 使用 MNI 重采样平滑像素 (Smoother blobs)
- 使用红蓝配色与阈值切割 (Professional visualization)
"""

import sys
from pathlib import Path
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets, image
import os
import warnings

# --- 路径设置 ---
# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.loaders import FMRIEEGDataset
from models.classifier_net import PhysicsE2fNet
from utils.paths import get_config_path, get_template_dir, get_checkpoint_dir, resolve_path

# 忽略 nilearn 的一些 warning
warnings.filterwarnings("ignore")

# --- 输出配置 ---
OUTPUT_DIR = resolve_path("./results/final_showcase_color")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自动加载 MNI 模板用于重采样背景
MNI_TEMPLATE = datasets.load_mni152_template()

def compute_optimized_affine(src_shape):
    """
    [核心修复]: 计算一个物理仿射矩阵
    将我们的低分数据 (e.g. 64x64x30) 映射到标准大脑物理尺寸 (mm)。
    这解决了“马赛克”和“显示错位”问题。
    """
    # 目标物理覆盖范围 (mm) - 适配标准成人大脑
    TARGET_FOV = (180.0, 200.0, 95.0) 
    
    # src_shape: [64, 64, 30]
    sx = TARGET_FOV[0] / src_shape[0]
    sy = TARGET_FOV[1] / src_shape[1]
    sz = TARGET_FOV[2] / src_shape[2]
    
    # 缩放
    affine = np.diag([1.12 * sx, sy, 1.4*sz, 1.0])# 1.2 was a little bit too big, see ex5.png
    
    # 平移 (将数据中心对齐到 MNI 坐标原点)
    tx = -sx * (src_shape[0] / 2) - 10 # was 180 * ((1.2 - 2)/2), now 0.12/2 * 180, and a little instinct
    ty = -sy * (src_shape[1] / 2) - 14 # no reason but instinct
    tz = -sz * (src_shape[2] / 2) + 0 # Z axis
    
    affine[:3, 3] = [tx, ty, tz]
    return affine

def reconstruct_mni_img(weights, ica_basis, mask_bool):
    """
    一步到位：权重 -> 3D体积 -> 正确坐标系 -> MNI高清重采样
    """
    # 1. 维度处理与转换 (Tensor -> Numpy)
    if isinstance(weights, torch.Tensor): weights = weights.detach().cpu()
    if isinstance(ica_basis, torch.Tensor): ica_basis = ica_basis.detach().cpu()
    
    if weights.dim() == 1: weights = weights.unsqueeze(0)
    
    # 处理基底维度 [K, D, H, W] -> Flatten [K, Voxels]
    n_mask_voxels = mask_bool.sum()
    
    if ica_basis.dim() == 4:
        # [K, D, H, W]
        ica_basis_flat = ica_basis.reshape(ica_basis.shape[0], -1)
        if ica_basis_flat.shape[1] != n_mask_voxels:
            # 需要 Apply Mask
            mask_flat = mask_bool.reshape(-1)
            ica_basis = ica_basis_flat[:, mask_flat]
        else:
            ica_basis = ica_basis_flat
            
    # 2. 矩阵乘法重建 [1, K] @ [K, V] -> [1, V]
    # 使用 float32 避免精度问题
    activation_vec = torch.matmul(weights.float(), ica_basis.float()).numpy().flatten()
    
    # 3. 填入 3D 盒子
    vol_data = np.zeros(mask_bool.shape) # [D, H, W] (30, 64, 64)
    vol_data[mask_bool] = activation_vec
    
    # 4. [关键] 轴转置 (2, 1, 0)
    # 将 [D, H, W] (30, 64, 64) -> [X, Y, Z] (64, 64, 30)
    vol_nii_data = np.transpose(vol_data, (1, 2, 0)) # AI sucks, I fix it myself
    
    # 5. [关键] 赋予物理坐标
    affine = compute_optimized_affine(vol_nii_data.shape)
    raw_img = nib.Nifti1Image(vol_nii_data, affine)
    
    # 6. MNI 空间重采样 (插值平滑)
    # 这一步把方块变成了看起来很真的脑图
    smooth_img = image.resample_to_img(raw_img, MNI_TEMPLATE, interpolation='continuous')
    
    return smooth_img

def main():
    print("🚀 Starting Final Visualization pipeline...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 寻找必要文件
    template_dir = get_template_dir()
    
    ica_path = next((p for p in [template_dir/"ica_mixing_matrix.pt", template_dir/"ica_basis.pt"] if p.exists()), None)
    mask_path = next((p for p in [template_dir/"mask_dhw.pt", template_dir/"gray_mask.pt"] if p.exists()), None)
    
    model_path = get_checkpoint_dir() / "model_ep50.pth" # 确保读取训练好的权重
    
    if not (ica_path and mask_path and model_path.exists()):
        print(f"❌ Critical files missing.")
        print(f"Model: {model_path} ({model_path.exists()})")
        return

    # 2. 加载基底
    basis = torch.load(ica_path, map_location='cpu')
    mask = torch.load(mask_path, map_location='cpu').numpy().astype(bool)
    if mask.ndim == 4: mask = mask.squeeze() # 容错
    
    print(f"✅ Assets Loaded. Basis: {basis.shape}")

    # 3. 初始化数据集和模型
    config_path = get_config_path()
    # 必须用 lazy_load=True 防止加载所有数据太慢，反正我们只取几个
    ds = FMRIEEGDataset(config_path=str(config_path), lazy_load=True)
    sample_eeg, _ = ds[0]
    
    model = PhysicsE2fNet(
        n_ica_components=64,
        eeg_channels=sample_eeg.shape[0], # 20
        eeg_time_len=sample_eeg.shape[2], # 249
        basis_path=str(ica_path),
        task='regression'
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded.")

    # 4. 选择样本并绘图
    # 选几个 Validation 集的样本 (靠后的索引)
    indices = [len(ds)-10, len(ds)-50, len(ds)-100]
    indices = [i for i in indices if i >= 0]
    
    if not indices: indices = [0] # 兜底

    print(f"Visualizing indices: {indices}")

    for idx in indices:
        eeg, gt_weights = ds[idx]
        
        # 模型推理
        with torch.no_grad():
            input_t = eeg.unsqueeze(0).to(device).float()
            pred_weights = model(input_t).cpu()
            
        # --- 重建为高清脑图 ---
        img_pred = reconstruct_mni_img(pred_weights, basis, mask)
        img_gt = reconstruct_mni_img(gt_weights, basis, mask)
        
        # --- 可视化核心参数 ---
        # 自动计算对比度范围
        # 使用真值的 99% 分位数，保证 Prediction 和 GT 共享同一色标，实现公平对比
        vmax = np.percentile(np.abs(img_gt.get_fdata()), 99.8)
        # 设定底噪阈值 (隐藏掉 25% 以下的微弱信号，只留红蓝主成分)
        thresh = vmax * 0.25 
        
        # 绘图
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Prediction
        plotting.plot_stat_map(
            img_pred, bg_img=MNI_TEMPLATE, 
            display_mode='z', cut_coords=[-40, -20, 0, 20, 40], # 固定5个轴向切片
            threshold=thresh, vmax=vmax,    # 统一度量衡
            cmap='cold_hot',                # 红蓝专业配色
            title=f"Sample {idx}: EEG Prediction",
            axes=axes[0], colorbar=True
        )
        
        # Ground Truth
        plotting.plot_stat_map(
            img_gt, bg_img=MNI_TEMPLATE, 
            display_mode='z', cut_coords=[-40, -20, 0, 20, 40], 
            threshold=thresh, vmax=vmax, 
            cmap='cold_hot',
            title=f"Sample {idx}: fMRI Ground Truth",
            axes=axes[1], colorbar=True
        )
        
        save_file = OUTPUT_DIR / f"Compare_Sample_{idx}.png"
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📸 Saved: {save_file.name} (Max Intensity: {vmax:.2f})")

    print(f"\n🎉 Visualization Done. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()