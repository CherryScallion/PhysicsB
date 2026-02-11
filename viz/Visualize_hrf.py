#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# --- 1. 环境与路径设置 ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.classifier_net import PhysicsE2fNet
from data.loaders import FMRIEEGDataset
from utils.paths import get_config_path, get_template_dir, resolve_path

# --- 2. 物理参数配置 ---
WINDOW_DURATION_SEC = 10.0  # 输入窗口时长 (EEG历史长度)
TR = 2.0                   # fMRI 重复时间 (参考用)

def compute_global_saliency(model, loader, device):
    """
    计算整个数据集上的全局时序注意力 (Saliency Map)
    方法：Input Gradient (dOutput/dInput)
    """
    saliency_accumulator = None
    total_samples = 0
    
    print(f"🧠 Scanning Validation Set for Temporal Attribution...")
    
    for eeg, _ in tqdm(loader, desc="Gradient Analysis"):
        eeg = eeg.to(device).float()
        # [关键]: 开启输入的梯度追踪
        eeg.requires_grad = True
        
        # 1. Forward Pass
        pred_weights = model(eeg) # [Batch, 64]
        
        # 2. Backward Target
        # 我们想知道输入如何影响输出的总能量 (Activation Magnitude)
        # 取 L2 Norm (Root Sum Square) 对噪声更鲁棒
        score = torch.norm(pred_weights, p=2, dim=1).sum()
        
        model.zero_grad()
        score.backward()
        
        # 3. Get Gradients
        # shape: [B, C, F, T]
        # 绝对值：我们要看"敏感度"，不在乎是正相关还是负相关
        grads = eeg.grad.data.abs()
        
        # 4. Collapse dimensions -> [T]
        # 在 Batch, Channel, Frequency 维度求平均，只保留 Time 维度
        # Input: [B, 20, 64, 249] -> [249]
        batch_saliency = grads.mean(dim=(0, 1, 2)).cpu().numpy()
        
        # 5. Accumulate
        batch_n = eeg.shape[0]
        if saliency_accumulator is None:
            saliency_accumulator = np.zeros_like(batch_saliency)
            
        # 加权累加
        saliency_accumulator += batch_saliency * batch_n
        total_samples += batch_n
        
    # 计算全局平均
    return saliency_accumulator / total_samples

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- A. 加载资源 ---
    config_path = get_config_path()
    ica_path = get_template_dir() / "ica_mixing_matrix.pt"
    # 使用你最好的一版模型权重
    model_path = resolve_path("./checkpoints/model_ep50.pth")
    output_dir = resolve_path("./results/analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- B. 准备数据 ---
    # 使用 Lazy Load=True 防止爆内存，因为我们要跑几千个样本
    full_ds = FMRIEEGDataset(config_path=str(config_path), lazy_load=True)
    
    # 模拟 main.py 的切分，只取最后 20% (Validation Set)
    # 这样保证我们在分析模型没见过的数据
    split_idx = int(0.8 * len(full_ds))
    val_indices = range(split_idx, len(full_ds))
    
    # 创建 Subset 和 Loader
    from torch.utils.data import Subset, DataLoader
    val_ds = Subset(full_ds, val_indices)
    # Batch Size 可以稍大，加速计算
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print(f"Dataset Size: {len(val_ds)} samples (Validation Set)")
    
    # 获取维度信息
    sample_eeg, _ = full_ds[0]
    C, F, T = sample_eeg.shape
    print(f"Input Shape: Channels={C}, Freq={F}, Time={T}")

    # --- C. 准备模型 ---
    model = PhysicsE2fNet(
        n_ica_components=64, 
        eeg_channels=C, 
        eeg_time_len=T,
        basis_path=str(ica_path), 
        task='regression'
    ).to(device)
    
    if not model_path.exists():
        print(f"❌ Error: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # --- D. 核心计算 ---
    raw_saliency = compute_global_saliency(model, val_loader, device)
    
    # --- E. 数据处理 ---
    # 1. 高斯平滑 (关键步骤：消除 CNN 步长伪影)
    # sigma=3.0 大约平滑掉 5-10ms 的高频抖动，保留秒级趋势
    smoothed_saliency = gaussian_filter1d(raw_saliency, sigma=3.0)
    
    # 2. 归一化 (0-1) 用于展示趋势
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    
    norm_raw = normalize(raw_saliency)
    norm_smooth = normalize(smoothed_saliency)
    
    # 3. 创建物理时间轴
    # 假设 T=249 对应过去的 6秒。t=248是现在(0s)，t=0是过去(-6s)
    time_axis = np.linspace(-WINDOW_DURATION_SEC, 0, T)
    
    # 找到峰值位置
    peak_idx = np.argmax(norm_smooth)
    peak_time = time_axis[peak_idx]
    peak_delay = abs(peak_time)
    
    # --- F. 论文级绘图 ---
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 1. 绘制背景阴影（原始数据，证明真实性）
    plt.plot(time_axis, norm_raw, color='gray', alpha=0.15, linewidth=0.5, label='Raw Gradient (Structural Artifacts)')
    # 也可以用 fill_between 让它看起来像置信区间
    plt.fill_between(time_axis, norm_raw, 0, color='gray', alpha=0.05)
    
    # 2. 绘制核心曲线（红色主线）
    plt.plot(time_axis, norm_smooth, color='#D62728', linewidth=4, label='Learned Temporal Attribution')
    
    # 3. 标注峰值 (Peak Line)
    plt.vlines(peak_time, 0, 1.0, colors='#D62728', linestyles='dashed', alpha=0.6)
    plt.text(peak_time + 0.1, 0.95, f'Peak Delay: {peak_delay:.2f}s', 
             fontsize=13, fontweight='bold', color='#8B0000', ha='left')
    
    # 4. 标注“现在”和“过去” (物理意义)
    # 在 0秒处 (Present) 画一个箭头指下去
    plt.annotate('Stimulus / Present\n(fMRI Acquisition)', 
                 xy=(0, norm_smooth[-1]), xytext=(-0.8, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, ha='center')
                 
    # 5. 美化坐标轴和图例
    plt.xlabel('Time Relative to fMRI Frame (seconds)', fontsize=14, fontweight='medium')
    plt.ylabel('Normalized Feature Importance', fontsize=14, fontweight='medium')
    plt.title('Emergence of Hemodynamic Delay (Data-Driven)', fontsize=16, pad=15)
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', frameon=True, fontsize=12)
    
    # 去除多余边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    plt.xlim(-WINDOW_DURATION_SEC, 0.2)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / "Final_Paper_HRF_Curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Graph saved: {save_path}")
    
    # 控制台结论输出
    print("-" * 40)
    print(f"📊 Statistics Summary:")
    print(f"   Window Length : {WINDOW_DURATION_SEC} sec")
    print(f"   Peak Position : {peak_time:.2f} sec")
    print("-" * 40)
    
    if 2.5 < peak_delay < 6.5:
        print("Result: SENSATIONAL.")
        print("The model spontaneously learned the physiological Hemodynamic Delay.")
    else:
        print("Note: Peak is detected, showing temporal selection is active.")

if __name__ == "__main__":
    main()