import os
import glob
import torch
import torch.nn.functional as F 
import h5py
import yaml
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class FMRIEEGDataset(Dataset):
    def __init__(self, config_path, lazy_load=False):
        """
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
            
        data_dir = ''
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"Fil Not Found")

        self.samples = []

        self.TARGET_FREQ = 64
        self.TARGET_TIME = 249
        
        print(f"{len(self.file_list)} H5 files.")
        
        for fpath in tqdm(self.file_list, desc="Caching & Resizing"):
            try:
                with h5py.File(fpath, 'r') as hf:
                    # Shape: [N, 20, F, 249]
                    eeg_numpy = hf['eeg'][:]
                    weights_numpy = hf['weights'][:] 
                    
                    eeg_tensor = torch.from_numpy(eeg_numpy).float()
                    weights_tensor = torch.from_numpy(weights_numpy).float()
                    
                    # eeg_tensor [N, 20, F, 249]
                    curr_freq = eeg_tensor.shape[2]
                    curr_time = eeg_tensor.shape[3]
                    
                    if curr_freq != self.TARGET_FREQ or curr_time != self.TARGET_TIME:
                        eeg_tensor = F.interpolate(
                            eeg_tensor, 
                            size=(self.TARGET_FREQ, self.TARGET_TIME),
                            mode='bilinear', 
                            align_corners=False
                        )
                        # print(f"Fixed {os.path.basename(fpath)}: {curr_freq} -> {self.TARGET_FREQ}")

                    # 3. save
                    for i in range(len(eeg_tensor)):
                        self.samples.append((eeg_tensor[i], weights_tensor[i]))
                        
            except Exception as e:
                print(f"Error loading {fpath}: {e}")

        print(f"[Dataset] Ready. Total Cached Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg, weights = self.samples[idx]
        return eeg, weights