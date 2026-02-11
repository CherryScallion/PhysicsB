import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# --- 路径处理黑魔法 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.loaders import FMRIEEGDataset
from models.classifier_net import PhysicsE2fNet
from utils.paths import get_config_path, get_template_dir, resolve_path

def find_file(directory: Path, candidates: list):
    """辅助函数：在目录中尝试寻找文件，直到找到一个存在的"""
    for fname in candidates:
        target = directory / fname
        if target.exists():
            return target
    return None

def main():
    print("🔥 Starting RAW Matrix Visualization (Robust Path Version)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    template_dir = get_template_dir()
    
    # -----------------------------------------------------------
    # 自动探测文件名 (Fix: No such file directory)
    # -----------------------------------------------------------
    ica_path = find_file(template_dir, ["ica_mixing_matrix.pt", "ica_basis.pt"])
    mask_path = find_file(template_dir, ["mask_dhw.pt", "gray_mask.pt"])
    
    if ica_path is None:
        print(f"❌ Error: 在 {template_dir} 下找不到 ICA 基底文件。")
        return
    if mask_path is None:
        print(f"❌ Error: 在 {template_dir} 下找不到 Mask 文件。")
        return
        
    print(f"✅ Found ICA: {ica_path.name}")
    print(f"✅ Found Mask: {mask_path.name}")
    # -----------------------------------------------------------

    # Load assets
    try:
        basis = torch.load(ica_path, map_location='cpu') # [64, Vox]
        mask = torch.load(mask_path, map_location='cpu')
        # 处理 mask 可能的 tensor 嵌套
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy().astype(bool)
        if mask.ndim == 4: mask = mask.squeeze()
            
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    # Load Model
    model_path = resolve_path("./checkpoints/model_ep50.pth")
    if not model_path.exists():
        # 如果 ep50 没跑完，尝试找最新的
        chk_dir = resolve_path("./checkpoints")
        if chk_dir.exists():
            files = sorted(list(chk_dir.glob("*.pth")))
            if files:
                model_path = files[-1]
                print(f"⚠️ model_ep50 not found, using latest: {model_path.name}")
            else:
                print("❌ No model checkpoints found!")
                return
        else:
            print("❌ No checkpoints directory!")
            return

    config_path = get_config_path()
    # 强制不使用 lazy load 方便读取 shape，如果内存不够改为 True
    ds = FMRIEEGDataset(config_path=str(config_path), lazy_load=True)
    sample_eeg, _ = ds[0]
    
    model = PhysicsE2fNet(
        n_ica_components=64, 
        eeg_channels=sample_eeg.shape[0], 
        eeg_time_len=sample_eeg.shape[2],
        basis_path=str(ica_path), 
        task='regression'
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 随机取一个样本 (Validation Set range)
    # 假设 validation 在最后
    idx = max(0, len(ds) - 100)
    print(f"🔎 Analysing Sample Index: {idx}")
    
    eeg, w_gt = ds[idx]
    
    # Predict
    with torch.no_grad():
        w_pred = model(eeg.unsqueeze(0).to(device)).cpu().squeeze(0)
    
    print(f"Pred Weight Mean: {w_pred.mean():.4f}, GT Weight Mean: {w_gt.mean():.4f}")
    
    # 创建输出目录
    out_dir = resolve_path("./results/debug")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- 1. 直接对比 Weight Vector (1D) ---
    plt.figure(figsize=(12, 4))
    plt.plot(w_gt.numpy(), label='Ground Truth', marker='o', markersize=3, alpha=0.6)
    plt.plot(w_pred.numpy(), label='Prediction', marker='x', markersize=3, linestyle='--', alpha=0.8)
    plt.title(f"Raw Weights Comparison (Sample {idx}) - No Smoothing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_p1 = out_dir / "raw_weights_compare.png"
    plt.savefig(save_p1)
    plt.close()
    print(f"📸 Saved Weights Plot: {save_p1}")
    
    # --- 2. 重建回体素并切片对比 (3D -> 2D Slice) ---
# --- 修复后的 reconstruct 函数 ---
    def reconstruct(w):
        # 统一转为 Tensor 在 CPU 计算，防止 device 不匹配
        w_t = w.cpu() if isinstance(w, torch.Tensor) else torch.from_numpy(w)
        basis_t = basis.cpu() if isinstance(basis, torch.Tensor) else torch.from_numpy(basis)
        
        # 维度自动适配逻辑
        if basis_t.dim() > 2:
            # Case A: 基底是 3D/4D 的 [K, D, H, W]
            # 直接线性叠加得到 3D 结果: [D, H, W]
            # 使用 einsum 确保维度正确缩减
            vol_t = torch.einsum('k, kdhw -> dhw', w_t.float(), basis_t.float())
            return vol_t.numpy()
        else:
            # Case B: 基底是扁平的 [K, Voxels]
            # 计算得到扁平向量 [Voxels]
            vec = (w_t.float() @ basis_t.float()).numpy()
            
            # 创建空 3D 矩阵并填充
            vol = np.zeros(mask.shape)
            try:
                vol[mask] = vec
            except ValueError as e:
                # 最后的保险：如果长度还不对，说明 Mask 和 Basis 长度不一致
                # 尝试 flatten
                print(f"⚠️ Warning: Shape mismatch (Mask:{mask.sum()} vs Vec:{vec.size}).")
                # 强制填充（可能会截断，但保证不报错）
                min_len = min(mask.sum(), vec.size)
                flat_indices = np.where(mask.flatten())[0]
                vol.flatten()[flat_indices[:min_len]] = vec.flatten()[:min_len]
                
            return vol

    vol_pred = reconstruct(w_pred)
    vol_gt = reconstruct(w_gt)
    vol_diff = vol_pred - vol_gt 
    
    # 寻找信息量最大的切片 (能量最强的那一层)
    energy_map = np.sum(np.abs(vol_gt), axis=(1, 2))
    slice_idx = np.argmax(energy_map)
    print(f"Select Z-Slice: {slice_idx} (Max Energy)")
    
    sl_pred = vol_pred[slice_idx, :, :]
    sl_gt = vol_gt[slice_idx, :, :]
    sl_diff = vol_diff[slice_idx, :, :]
    
    # 统一 Scale
    vmax = np.max(np.abs(sl_gt)) * 0.8
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PRED
    im1 = axes[0].imshow(sl_pred, cmap='seismic', vmin=-vmax, vmax=vmax) 
    axes[0].set_title("Prediction (Raw Slice)")
    plt.colorbar(im1, ax=axes[0])
    
    # GT
    im2 = axes[1].imshow(sl_gt, cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1].set_title("Ground Truth (Raw Slice)")
    plt.colorbar(im2, ax=axes[1])
    
    # DIFF
    # Diff 的 range 单独设，以便看清微小误差
    diff_max = np.max(np.abs(sl_diff))
    im3 = axes[2].imshow(sl_diff, cmap='coolwarm', vmin=-diff_max, vmax=diff_max) 
    axes[2].set_title(f"Difference (Max Diff={diff_max:.2f})")
    plt.colorbar(im3, ax=axes[2]) 
    
    plt.tight_layout()
    save_p2 = out_dir / "raw_slice_debug.png"
    plt.savefig(save_p2)
    print(f"📸 Saved Slice Plot: {save_p2}")
    
    # 判断逻辑
    avg_diff = np.mean(np.abs(w_pred.numpy() - w_gt.numpy()))
    print(f"\n🧠 Diagnostics:")
    print(f"   Avg Weight Difference: {avg_diff:.4f}")
    if avg_diff < 1e-5:
        print("   -> 🔴 WARNING: Identical. Check data split leakage.")
    else:
        print("   -> 🟢 SUCCESS: Distinct but similar. Model works.")

if __name__ == "__main__":
    main()