from pathlib import Path
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
import nibabel as nib
from nilearn.masking import compute_epi_mask
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from scipy.ndimage import zoom
from collections import Counter

# Config
SRC_DIR = Path()
OUT_DIR = Path()
TPL_DIR = Path()
OUT_DIR.mkdir(parents=True, exist_ok=True)
TPL_DIR.mkdir(parents=True, exist_ok=True)

# Force depth to 30 as requested for alignment
TARGET_DEPTH = 30

# Candidate keys inside original H5 that may contain fmri volumes
FMRI_KEYS = ["fmri", "bold", "img", "data", "volume"]

def detect_and_standardize_fmri_array(arr):
    """
    Make sure returned array has shape [N, D, H, W] and dtype float32.
    Accepts inputs like [D,H,W], [N,D,H,W], [D,H,W,N] etc.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[None, ...]  # single sample
    elif arr.ndim == 4:
        # Heuristic: if first axis small and others typical ints, treat as [N,D,H,W]
        # Prefer first axis as sample axis when that axis length is >1 and other dims >= 8
        first_is_samples = (arr.shape[0] > 1 and arr.shape[1] >= 8 and arr.shape[2] >= 8)
        last_is_samples = (arr.shape[-1] > 1 and arr.shape[0] >= 8 and arr.shape[1] >= 8)
        if last_is_samples and not first_is_samples:
            arr = np.moveaxis(arr, -1, 0)  # bring sample axis to front
        # else keep as-is (first axis is samples)
    else:
        raise ValueError("Unsupported" % arr.ndim)
    return arr.astype(np.float32)

def resample_volume_to_shape(vol, target_shape, order=3):
    """
    Resample a 3D volume to target_shape (D,H,W) using scipy.ndimage.zoom (interpolation).
    If zoom produces slight off-by-one sizes, do center-crop or zero-pad to match exactly.
    """
    vol = np.asarray(vol, dtype=np.float32)
    D, H, W = vol.shape
    tD, tH, tW = target_shape
    if (D, H, W) == (tD, tH, tW):
        return vol

    zf = (tD / D, tH / H, tW / W)
    vol_z = zoom(vol, zf, order=order, mode='constant', cval=0.0)

    # crop/pad to exact shape
    out = np.zeros(target_shape, dtype=np.float32)
    sd = min(tD, vol_z.shape[0])
    sh = min(tH, vol_z.shape[1])
    sw = min(tW, vol_z.shape[2])
    # center placement
    td_off = (tD - sd) // 2
    th_off = (tH - sh) // 2
    tw_off = (tW - sw) // 2
    out[td_off:td_off + sd, th_off:th_off + sh, tw_off:tw_off + sw] = vol_z[:sd, :sh, :sw]
    return out

def determine_target_hw(volumes):
	"""Pick the most common (H,W) among volumes."""
	counter = Counter((v.shape[1], v.shape[2]) for v in volumes)
	(h, w), _ = counter.most_common(1)[0]
	return h, w

def _resample_mask_if_needed(mask_bool, target_shape):
    """If mask_bool.shape != target_shape, resample mask (nearest neighbor) and threshold."""
    if mask_bool.shape == target_shape:
        return mask_bool
    # zoom factors
    zf = (target_shape[0] / mask_bool.shape[0],
          target_shape[1] / mask_bool.shape[1],
          target_shape[2] / mask_bool.shape[2])
    mask_z = zoom(mask_bool.astype(np.float32), zf, order=0, mode='constant', cval=0.0)
    return (mask_z >= 0.5)
 
def gather_all_samples(src_dir, max_samples=None):
    """
    Iterate H5 in src_dir and collect a list of 3D volumes and mapping metadata:
     - volumes: list of numpy arrays shape (D,H,W)
     - mapping: list of (h5_path, sample_idx_in_file)
    """
    volumes = []
    mapping = []
    files = sorted(SRC_DIR.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No H5 files found")
    for fpath in files:
        with h5py.File(fpath, 'r') as hf:
            # find fmri key
            fmri_key = next((k for k in FMRI_KEYS if k in hf.keys()), None)
            if fmri_key is None:
                print(f" {fpath.name} has no known fmri key")
                continue
            arr = detect_and_standardize_fmri_array(hf[fmri_key][:])
            for i in range(arr.shape[0]):
                volumes.append(arr[i])
                mapping.append((fpath, i))
                if max_samples and len(volumes) >= max_samples:
                    return volumes, mapping
    return volumes, mapping

def make_mask_from_volumes(volumes, n_ref=50):
    """
    Compute a binary brain mask using the mean image over a subset of volumes.
    Returns mask_img (nib.Nifti1Image) and mask_bool (np.bool_ array shape D,H,W)
    """
    n_ref = min(n_ref, len(volumes))
    # build a 4D image from first n_ref volumes
    sample_stack = np.stack(volumes[:n_ref], axis=0)  # [N, D, H, W]
    mean_vol = np.mean(sample_stack, axis=0)
    img = nib.Nifti1Image(mean_vol, np.eye(4))
    mask_img = compute_epi_mask(img)
    mask_bool = mask_img.get_fdata().astype(bool)
    # safety: ensure mask matches sample volume shape; if not, resample mask (nearest)
    target_shape = mean_vol.shape
    if mask_bool.shape != target_shape:
        mask_bool = _resample_mask_if_needed(mask_bool, target_shape)
    return mask_img, mask_bool

def build_data_matrix(volumes, mask_bool):
    """
    Apply boolean mask_bool to each 3D volume directly (no nibabel/nilearn overhead).
    mask_bool shape must equal volume shape (D,H,W).
    Returns X shape [n_samples, n_voxels]
    """
    X = []
    n_vox = int(mask_bool.sum())
    for vol in volumes:
        if vol.shape != mask_bool.shape:
            raise ValueError(f"Volume shape {vol.shape} does not match mask shape {mask_bool.shape}")
        vec = vol[mask_bool]  # direct indexing yields 1D vector
        X.append(vec)
    X = np.vstack(X).astype(np.float32)
    # sanity checks
    if np.isnan(X).any() or np.isinf(X).any():
        print("⚠️ Warning: NaN or Inf detected in X, replacing with zeros for safety.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def run_ica_and_save(X, mask_bool, mapping, n_components=64, whiten_pca=128):
    """
    Run PCA (optional) then FastICA. Save templates and optimized H5 files.
    """
    n_samples, n_vox = X.shape
    n_components = min(n_components, max(1, n_samples - 1))  # ensure feasible
    print(f"Samples: {n_samples}, Voxels: {n_vox}, ICA components: {n_components}")

    # Optional PCA whitening to reduce dimensionality before ICA for speed/stability
    if whiten_pca is not None and whiten_pca < n_vox and whiten_pca < n_samples:
        pca = PCA(n_components=min(whiten_pca, n_samples - 1), random_state=0, svd_solver='auto')
        X_red = pca.fit_transform(X)
        print(f"PCA reduced: {X.shape} -> {X_red.shape}")
    else:
        pca = None
        X_red = X

    ica = FastICA(n_components=n_components, max_iter=1000, random_state=0)
    S = ica.fit_transform(X_red)  # [n_samples, n_components]
    # mixing_ shape: (n_features_after_pca, n_components) when PCA used; need to map back to full voxel space
    if pca is not None:
        # mixing_ is in PCA space; backproject mixing to voxel space:
        # data approx X ≈ S @ ica.mixing_.T
        mixing_pca = ica.mixing_  # shape (n_pca_comp, n_components)
        # backproject: voxel_mixing = pca.components_.T @ mixing_pca   if pca.components_ shape (n_pca_comp, n_voxels)
        # Note: sklearn PCA has components_ shape (n_components, n_features)
        voxel_mixing = pca.components_.T @ mixing_pca  # shape (n_voxels, n_components)
    else:
        voxel_mixing = ica.mixing_  # shape (n_voxels, n_components)

    # basis must be [K, V] so take transpose
    basis = voxel_mixing.T.astype(np.float32)  # [K, V]
    weights_all = S.astype(np.float32)         # [n_samples, K]

    # Save templates
    basis_torch = torch.from_numpy(basis)  # [K, V]
    torch.save(basis_torch, TPL_DIR / "ica_mixing_matrix.pt")
    print(f"Saved: {TPL_DIR / 'ica_mixing_matrix.pt'} (shape {tuple(basis_torch.shape)})")

    # reconstruct volumetric basis [K, D, H, W]
    D, H, W = mask_bool.shape
    basis_vol = np.zeros((basis.shape[0], D, H, W), dtype=np.float32)
    mask_flat_indices = np.where(mask_bool.flatten())[0]
    for k in range(basis.shape[0]):
        vol_flat = np.zeros(mask_bool.size, dtype=np.float32)
        vol_flat[mask_flat_indices] = basis[k]
        basis_vol[k] = vol_flat.reshape(D, H, W)
    torch.save(torch.from_numpy(basis_vol), TPL_DIR / "ica_basis.pt")
    print(f"Saved: {TPL_DIR / 'ica_basis.pt'} (shape {basis_vol.shape})")

    # Save mask
    torch.save(torch.from_numpy(mask_bool.astype(np.bool_)), TPL_DIR / "mask_dhw.pt")
    print(f"Saved: {TPL_DIR / 'mask_dhw.pt'} (shape {mask_bool.shape})")

    # Write optimized H5s: group weights per original file
    # mapping: list of (orig_path, sample_idx)
    per_file = {}
    for idx, (fpath, sample_idx) in enumerate(mapping):
        per_file.setdefault(fpath, []).append((sample_idx, idx))

    for orig_path, idx_list in per_file.items():
        idx_list_sorted = sorted(idx_list, key=lambda x: x[0])  # sort by sample_idx
        weights_for_file = np.stack([weights_all[global_idx] for (_, global_idx) in idx_list_sorted], axis=0)
        # create output filename
        out_name = f"opt_{orig_path.name}"
        out_path = OUT_DIR / out_name
        # copy eeg if present and any other datasets we want to keep
        with h5py.File(orig_path, 'r') as hf_in, h5py.File(out_path, 'w') as hf_out:
            # copy 'eeg' if exists
            if 'eeg' in hf_in:
                hf_out.create_dataset('eeg', data=hf_in['eeg'][:], compression="gzip")
            # copy other non-fmri datasets (weights will be overwritten)
            for k in hf_in.keys():
                if k in FMRI_KEYS:
                    hf_out.create_dataset('fmri', data=hf_in[k][:], compression="gzip")
                elif k != 'eeg':
                    # copy other meta datasets (e.g., 'meta')
                    hf_out.create_dataset(k, data=hf_in[k][:], compression="gzip")
            # write weights
            hf_out.create_dataset('weights', data=weights_for_file, compression="gzip")
        print(f"Saved optimized H5: {out_path} (weights shape: {weights_for_file.shape})")

    return basis, weights_all

def main():
    print("Starting")
    try:
        volumes, mapping = gather_all_samples(SRC_DIR)
        print(f"Gathered {len(volumes)} volumes from {len(set(m[0] for m in mapping))} files")
        target_h, target_w = determine_target_hw(volumes)
        print(f"Target (H,W): ({target_h}, {target_w}), depth: {TARGET_DEPTH}")
        volumes = [resample_volume_to_shape(v, (TARGET_DEPTH, target_h, target_w)) for v in volumes]
        print(f"Resampled all volumes to ({TARGET_DEPTH}, {target_h}, {target_w})")
        mask_img, mask_bool = make_mask_from_volumes(volumes)
        print(f"Computed mask shape: {mask_bool.shape}, voxels in mask: {mask_bool.sum()}")
        X = build_data_matrix(volumes, mask_bool)
        print(f"Data matrix shape: {X.shape}")
        basis, weights = run_ica_and_save(X, mask_bool, mapping, n_components=64, whiten_pca=128)
    except Exception as e:
        raise

if __name__ == "__main__":
    main()
    