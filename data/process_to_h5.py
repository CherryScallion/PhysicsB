
import os
import sys
import glob
from pathlib import Path
import argparse

import numpy as np
import h5py
from scipy.stats import zscore
from scipy.io import loadmat
from nilearn import image as niimage
import mne
from tqdm import tqdm

# ============================================================================
# 配置参数
# ============================================================================

EEG_WINDOW_SECONDS = 10.0
FMRI_DELAY_SECONDS = 0.0

PROCESSING_CFG = {
    'bold_shift': 0,
    'eeg_limit': True,
    'eeg_f_limit': 250,
    'eeg_window_seconds': EEG_WINDOW_SECONDS,
    'fmri_delay_seconds': FMRI_DELAY_SECONDS,
    'interval_eeg': None,
    
    "NODDI": {
        'n_volumes': 294,
        'f_resample': 2.160,
    },
    
    "Oddball": {
        'n_volumes': 164,
        'f_resample': 2.0,
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = Path(r"") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def correct_vhdr_path(vhdr_path: Path):
    vmrk_path = vhdr_path.parent / f"{vhdr_path.stem}.vmrk"
    eeg_path = vhdr_path.parent / f"{vhdr_path.stem}.eeg"
    if not eeg_path.exists():
        eeg_path = vhdr_path.parent / f"{vhdr_path.stem}.dat"
    if not eeg_path.exists() or not vmrk_path.exists(): return
    
    data_filename = eeg_path.name
    vmrk_filename = vmrk_path.name
    
    with open(vhdr_path, 'r', encoding='utf-8') as f: lines = f.readlines()
    is_modify = False
    for i, line in enumerate(lines):
        if line.startswith('DataFile=') and data_filename not in line:
            lines[i] = f'DataFile={data_filename}\n'; is_modify = True
        elif line.startswith('MarkerFile=') and vmrk_filename not in line:
            lines[i] = f'MarkerFile={vmrk_filename}\n'; is_modify = True
    if is_modify:
        with open(vhdr_path, 'w', encoding='utf-8') as f: f.writelines(lines)

def compute_fft(signal_1d: np.ndarray, limit: bool = True, f_limit: int = 250):
    fft_vals = np.fft.fft(signal_1d)
    if limit:
        clipped = np.zeros((f_limit,), dtype=fft_vals.dtype)
        length = min(f_limit, fft_vals.shape[0])
        clipped[:length] = fft_vals[:length]
        fft_vals = clipped
    return np.abs(fft_vals[1:]) 

def stft(eeg, channel: int = 0, window_size: int = 2, fs: int = 250, 
         limit=True, f_limit: int = 250, start_time: int = None, stop_time: int = None):
    if hasattr(eeg, 'ch_names'): signal = eeg.get_data(channel)[0]
    else: signal = eeg[channel, :].reshape(-1)
    
    if start_time is None: start_time = 0
    if stop_time is None: stop_time = len(signal)
    signal = signal[start_time:stop_time]
    
    t = []; Z = []; seconds = 0
    fs_window_size = int(window_size * fs)
    sample_range = list(range(start_time, stop_time, fs_window_size))
    if (stop_time - start_time) % fs_window_size != 0: sample_range = sample_range[:-1]
    
    for time in sample_range:
        segment = signal[time:time + fs_window_size]
        if segment.shape[0] < fs_window_size:
            segment = np.pad(segment, (0, fs_window_size - segment.shape[0]), mode='constant')
        fft1 = compute_fft(segment, limit=limit, f_limit=f_limit)
        Z.append(list(abs(fft1)))
        t.append(seconds); seconds += window_size
    
    if not Z:
        fft1 = compute_fft(signal, limit=limit, f_limit=f_limit)
        Z = [list(abs(fft1))]; t = [0]
    return None, np.transpose(np.array(Z)), t

def create_eeg_bold_pairs(eeg_data: np.ndarray, fmri_data: np.ndarray, 
                          interval_eeg: int, n_volumes: int):
    x_eeg = np.empty((n_volumes - interval_eeg, ) + eeg_data.shape[1:] + (interval_eeg, ))
    x_bold = np.empty((n_volumes - interval_eeg, ) + fmri_data.shape[1:])
    
    for index_volume in range(interval_eeg, n_volumes):
        eeg_slice = eeg_data[index_volume - interval_eeg : index_volume]
        if (np.transpose(eeg_slice, (1, 2, 0)).shape[-1] != interval_eeg): continue
        x_eeg[index_volume - interval_eeg] = np.transpose(eeg_slice, (1, 2, 0))
        x_bold[index_volume - interval_eeg] = fmri_data[index_volume]
    return x_eeg, x_bold

def process_noddi_subject(subject_dir: Path, output_dir: Path):
    subject_id = subject_dir.name
    fmri_files = list(subject_dir.glob("*_cross.nii.gz"))
    if not fmri_files: return None
    
    export_dir = subject_dir / "export"
    if not export_dir.exists(): return None
    vhdr_files = list(export_dir.glob("*.vhdr"))
    if not vhdr_files: return None
    vhdr_path = Path(vhdr_files[0])
    
    print(f"Processing NODDI subject {subject_id}...")
    cfg = PROCESSING_CFG["NODDI"]
    bold_shift = cfg.get('bold_shift', 0)
    n_volumes = cfg['n_volumes']
    f_resample = cfg['f_resample']
    interval_eeg = max(1, int(np.round(PROCESSING_CFG['eeg_window_seconds'] / f_resample)))
    
    # 1. Load fMRI
    fmri_data = niimage.load_img(str(fmri_files[0])).get_fdata()
    recording_time = min(n_volumes, fmri_data.shape[-1])
    fmri_data = fmri_data[:, :, :, bold_shift: recording_time + bold_shift]
    
    fmri_data = fmri_data / 1000.0
    # -------------------------
    
    fmri_data = fmri_data.transpose(3, 0, 1, 2) # [T, H, W, D]
    
    # 2. Load EEG (Z-score)
    correct_vhdr_path(vhdr_path)
    try:
        eeg_raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose=0)
    except: return None
    
    fs_sample = eeg_raw.info['sfreq']
    x_instance = []
    for ch in range(len(eeg_raw.ch_names)):
        _, Zxx, _ = stft(eeg_raw, channel=ch, window_size=f_resample, fs=fs_sample, 
                         limit=PROCESSING_CFG['eeg_limit'], f_limit=PROCESSING_CFG['eeg_f_limit'])
        x_instance.append(Zxx)
    eeg_data = zscore(np.array(x_instance))
    eeg_data = eeg_data.transpose(2, 0, 1)[bold_shift: recording_time + bold_shift]
    
    # 3. Save
    eeg_pairs, fmri_pairs = create_eeg_bold_pairs(eeg_data, fmri_data, interval_eeg, n_volumes)
    eeg_pairs = eeg_pairs.transpose(0, 3, 1, 2)
    fmri_pairs = fmri_pairs.transpose(0, 3, 1, 2)
    
    output_path = output_dir / f"NODDI_{subject_id}_1.h5"
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('eeg', data=eeg_pairs.astype(np.float32), compression='gzip')
        hf.create_dataset('fmri', data=fmri_pairs.astype(np.float32), compression='gzip')
    print(f"  ✓ Saved: {output_path}")

def process_oddball_run(subject_name, subject_dir, task, run, run_counter, output_dir):
    task_run = f"task{task:03}_run{run:03}"
    fmri_path = subject_dir / "BOLD" / task_run / "bold.nii.gz"
    eeg_path = subject_dir / "EEG" / task_run / "EEG_noGA.mat"
    if not fmri_path.exists() or not eeg_path.exists(): return
    
    print(f"Processing Oddball {subject_name} {task_run}...")
    cfg = PROCESSING_CFG["Oddball"]
    bold_shift = cfg.get('bold_shift', 0)
    n_volumes = cfg['n_volumes']
    f_resample = cfg['f_resample']
    interval_eeg = max(1, int(np.round(PROCESSING_CFG['eeg_window_seconds'] / f_resample)))
    
    # 1. Load fMRI
    fmri_data = niimage.load_img(str(fmri_path)).get_fdata()
    recording_time = min(n_volumes, fmri_data.shape[-1])
    fmri_data = fmri_data[:, :, :, bold_shift: recording_time + bold_shift]
    
    fmri_data = fmri_data / 1000.0
    
    fmri_data = fmri_data.transpose(3, 0, 1, 2)
    
    # 2. Load EEG
    eeg_mat = loadmat(str(eeg_path))
    eeg_data = eeg_mat['data_noGA'][:43, :]
    x_instance = []
    for ch in range(len(eeg_data)):
        _, Zxx, _ = stft(eeg_data, channel=ch, window_size=f_resample, fs=1000, 
                         limit=PROCESSING_CFG['eeg_limit'], f_limit=PROCESSING_CFG['eeg_f_limit'])
        x_instance.append(Zxx)
    eeg_data = zscore(np.array(x_instance))
    eeg_data = eeg_data.transpose(2, 0, 1)[bold_shift: recording_time + bold_shift]
    
    # 3. Save
    eeg_pairs, fmri_pairs = create_eeg_bold_pairs(eeg_data, fmri_data, interval_eeg, n_volumes)
    eeg_pairs = eeg_pairs.transpose(0, 3, 1, 2)
    fmri_pairs = fmri_pairs.transpose(0, 3, 1, 2)
    
    output_path = output_dir / f"Oddball_{subject_name}_{run_counter}.h5"
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('eeg', data=eeg_pairs.astype(np.float32), compression='gzip')
        hf.create_dataset('fmri', data=fmri_pairs.astype(np.float32), compression='gzip')
    print(f"  ✓ Saved: {output_path}")

def process_noddi_dataset(data_dir, output_dir):
    noddi_dir = data_dir / "noddi"
    if not noddi_dir.exists(): return
    for subject_dir in tqdm(sorted([d for d in noddi_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))):
        try: process_noddi_subject(subject_dir, output_dir)
        except Exception as e: print(f"Error {subject_dir.name}: {e}")

def process_oddball_dataset(data_dir, output_dir):
    oddball_dir = data_dir / "oddball"
    if not oddball_dir.exists(): return
    for subject_dir in tqdm(sorted([d for d in oddball_dir.iterdir() if d.is_dir() and d.name.startswith("sub")])):
        try:
            subject_name = subject_dir.name
            if not (subject_dir / "BOLD").exists(): continue
            run_counter = 1
            for task in [1, 2]:
                for run in [1, 2, 3, 4, 5, 6]:
                    run_path = subject_dir / "BOLD" / f"task{task:03}_run{run:03}"
                    if run_path.exists():
                        process_oddball_run(subject_name, subject_dir, task, run, run_counter, output_dir)
                        run_counter += 1
        except Exception as e: print(f"Error {subject_dir.name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["NODDI", "Oddball", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    print(f"Processing Raw Scaled Data to: {args.output_dir}")
    if args.dataset in ["NODDI", "all"]: process_noddi_dataset(args.data_dir, args.output_dir)
    if args.dataset in ["Oddball", "all"]: process_oddball_dataset(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()

def process_oddball_subject(subject_dir: Path, output_dir: Path):
    subject_name = subject_dir.name
    if not (subject_dir / "BOLD").exists():
        return
    
    tasks = [1, 2]
    runs = [1, 2, 3, 4, 5, 6] 
    
    run_counter = 1
    for task in tasks:
        for run in runs:
            try:
                run_folder_name = f"task{task:03}_run{run:03}"
                run_path = subject_dir / "BOLD" / run_folder_name
                
                if not run_path.exists(): continue
                
                process_oddball_run(subject_name, subject_dir, task, run, 
                                    run_counter, output_dir)
                run_counter += 1
            except Exception as e:
                print(f"[ERROR] Processing {subject_name} task{task} run{run}: {e}")


def process_oddball_dataset(data_dir: Path, output_dir: Path):
    oddball_dir = data_dir / "oddball"
    if not oddball_dir.exists():
        print(f"[ERROR] Oddball directory not found: {oddball_dir}")
        return
    subject_dirs = [d for d in oddball_dir.iterdir() 
                    if d.is_dir() and d.name.startswith("sub")]
    subject_dirs = sorted(subject_dirs)
    
    for subject_dir in tqdm(subject_dirs, desc="Processing Oddball"):
        process_oddball_subject(subject_dir, output_dir)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["NODDI", "Oddball", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset in ["NODDI", "all"]:
        process_noddi_dataset(args.data_dir, args.output_dir)
    
    if args.dataset in ["Oddball", "all"]:
        process_oddball_dataset(args.data_dir, args.output_dir)
        
    print("\nfinished")

if __name__ == "__main__":
    main()