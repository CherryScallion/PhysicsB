# main.py
import torch
import yaml
from torch.utils.data import DataLoader
from data.loaders import FMRIEEGDataset
from models.classifier_net import PhysicsB
from models.loss import WeightsRegressionLoss
from training.trainer import Trainer
from utils.paths import get_config_path, resolve_path

def main():
    # Config
    config_path = get_config_path()
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Training Parameters
    train_cfg = cfg.get('training', {})
    num_epochs = train_cfg.get('num_epochs', 50)
    batch_size = train_cfg.get('batch_size', 4)
    num_workers = train_cfg.get('num_workers', 0)
    lazy_load = train_cfg.get('lazy_load', True)
    device_mode = train_cfg.get('device', 'auto')
    
    # Device Setup
    if device_mode == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("⚠️ WARNING: CUDA requested but not available! Falling back to CPU.")
    elif device_mode == 'cpu':
        device = torch.device('cpu')
    else:  # 'auto'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[Device] Device: {device}")
    print(f"[Device] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"         GPU: {torch.cuda.get_device_name(0)}")
        print(f"         VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    print("\n" + "="*60)
    print("[Config] Training Config")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {train_cfg.get('learning_rate', 1e-4)}")
    print(f"Lazy Load: {lazy_load}")
    print("="*60 + "\n")
        
    print("Initializing DataLoaders...")
    dataset = FMRIEEGDataset(
        config_path=str(config_path),
        lazy_load=lazy_load
    )
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # aquire sample shape
    sample_eeg, _ = dataset[0] 
    eeg_c, eeg_f, eeg_t = sample_eeg.shape
    
    print(f"EEG shape: C={eeg_c}, F={eeg_f}, T={eeg_t}")
    
    print("Building PhysicsB Model...")
    model = PhysicsB(
        n_ica_components=cfg['data_specs']['n_ica_components'],
        eeg_channels=eeg_c,
        eeg_time_len=eeg_t
    )
    
    print("Setting up Loss & Trainer...")
    loss_fn = WeightsRegressionLoss(
        lambda_cos=train_cfg.get('cosine_loss_weight', 0.5)
    )
    
    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        loss_fn, 
        cfg,
        device=device
    )
    trainer.run(num_epochs=num_epochs)

if __name__ == "__main__":
    main()