# training/trainer.py
import torch
from tqdm import tqdm
import torch.optim as optim
import os
from utils.paths import get_checkpoint_dir

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, config, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.cfg = config

        train_cfg = config.get('training', {})
        lr = train_cfg.get('learning_rate', 1e-4)
        weight_decay = train_cfg.get('weight_decay', 1e-5)
        optimizer_type = train_cfg.get('optimizer', 'AdamW')
        
        # optimizer
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, 
                betas=(train_cfg.get('beta1', 0.9), train_cfg.get('beta2', 0.999)),
                eps=train_cfg.get('epsilon', 1e-8),
                weight_decay=weight_decay
            )
        else:  # AdamW (default)
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, 
                weight_decay=weight_decay
            )
        
        # Device settings
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        
        self.epoch = 0
        self.gradient_clip_norm = train_cfg.get('gradient_clip_norm', 1.0)

    def run(self, num_epochs=50):
        print(f"\n[Training] Training started on {self.device}")
        print(f"[Training] Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"[Training] Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            self._update_scheduler(epoch)
            
            # Train Loop
            train_loss = self._train_one_epoch()
            
            # Validation Loop
            val_loss = self._validate()
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save Checkpoint
            if (epoch + 1) % self.cfg.get('training', {}).get('checkpoint_interval', 5) == 0:
                self._save_checkpoint(epoch)

    def _update_scheduler(self, epoch):
        """Learning rate and loss scheduling (optional for regression)."""
        # For regression tasks, we typically don't need dice loss weighting
        pass

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Train Ep {self.epoch+1}")
        
        # Create CUDA stream for asynchronous data loading
        if self.device.type == 'cuda':
            stream = torch.cuda.Stream()
        
        for batch_idx, (eeg, weights) in enumerate(loop):
            # Async GPU transfer using stream
            if self.device.type == 'cuda':
                with torch.cuda.stream(stream):
                    eeg = eeg.to(self.device, non_blocking=True).float()           # [B, C, F, T]
                    weights = weights.to(self.device, non_blocking=True).float()   # [B, 64]
                torch.cuda.current_stream().wait_stream(stream)
            else:
                eeg = eeg.to(self.device).float()
                weights = weights.to(self.device).float()
            
            # Forward: EEG -> predicted weights
            pred_weights = self.model(eeg) # [B, 64]
            
            # Loss: regression loss
            loss, details = self.loss_fn(pred_weights, weights)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), **details)
            
        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for eeg, weights in self.val_loader:  # Use validation loader
                eeg = eeg.to(self.device).float()
                weights = weights.to(self.device).float()
                
                pred_weights = self.model(eeg)
                loss, _ = self.loss_fn(pred_weights, weights)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def _save_checkpoint(self, epoch):
        checkpoint_dir = get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = checkpoint_dir / f"model_ep{epoch+1}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"checkpoint saved: {path}")