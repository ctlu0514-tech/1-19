#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export 3D-ResNet deep features (embedding) to a CSV for downstream CDGAFS + RFE.

What this script does:
1) Read your label CSV (e.g., qianliexian_clinical_isup.csv) which contains columns: id and label (e.g., isup2).
2) For each id, locate modality images in:
   - mpMRI_nii/<id>/  (ADC, DWI, T2 or T2FS etc.)
   - PETCT_nii/<id>/  (CT and PET)
3) Load a trained 3D-ResNet checkpoint (.pth), run inference, and export:
   id, label, prob0, prob1, deep_0 ... deep_(F-1)

Important:
- The exported features are meaningful ONLY if you provide a trained checkpoint.
- The checkpoint's expected input channels must match the number of modalities you pass.

Default modalities: ADC, DWI, T2, CT, PET  (5 channels)

Example (GPU):
python extract_deep_features_prostate.py \
  --mpmri_nii /path/to/mpMRI_nii \
  --petct_nii /path/to/PETCT_nii \
  --clinical_csv qianliexian_clinical_isup.csv \
  --label_col isup2 \
  --ckpt best_model.pth \
  --arch resnet50 \
  --out prostate_deep_features.csv

Dry-run to only check availability (no model):
python extract_deep_features_prostate.py ... --dry_run --out availability_report.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F


# --------------------------
# File discovery helpers
# --------------------------
_EXCLUDE_KEYWORDS = ("mask", "seg", "label", "roi", "contour", "merge")

def _best_candidate(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    # Prefer .nii.gz over .nii, and shorter names
    files.sort(key=lambda p: (0 if p.name.lower().endswith(".nii.gz") else 1, len(p.name)))
    return files[0]

def _find_modality_file(root: Path, pid: str, modality: str) -> Optional[Path]:
    """
    Search for a NIfTI image under root/<pid>/ where filename contains `modality`
    (case-insensitive), excluding mask-like files.
    """
    pdir = root / str(pid)
    if not pdir.exists():
        return None
    mod = modality.lower()
    cand: List[Path] = []
    for p in pdir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            continue
        if mod not in name:
            continue
        if any(k in name for k in _EXCLUDE_KEYWORDS):
            continue
        cand.append(p)
    return _best_candidate(cand)

def _collect_paths(mpmri_root: Path, petct_root: Path, pid: str, modalities: List[str]) -> Dict[str, Optional[Path]]:
    """
    For each modality, decide which root to search:
      - ADC/DWI/T2/T2FS/... -> mpMRI_nii
      - CT/PET -> PETCT_nii
    """
    out: Dict[str, Optional[Path]] = {}
    for m in modalities:
        m_upper = m.upper()
        if m_upper in ("CT", "PET"):
            out[m] = _find_modality_file(petct_root, pid, m)
        else:
            out[m] = _find_modality_file(mpmri_root, pid, m)
    return out


# --------------------------
# Image loading + preprocess
# --------------------------
def _safe_percentile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))

def load_nifti_as_float(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(), dtype=np.float32)
    # Ensure (D,H,W)
    if arr.ndim == 4:
        # if (H,W,D,C) or similar, take first channel
        arr = arr[..., 0]
    # Nibabel returns typically (X,Y,Z). We interpret as (H,W,D) then transpose to (D,H,W)
    if arr.shape[2] != arr.shape[-1]:
        # keep generic; we'll transpose assuming last axis is depth
        pass
    arr = np.transpose(arr, (2, 0, 1)).copy()  # (D,H,W)
    return arr

def preprocess_volume(vol: np.ndarray, out_shape: Tuple[int,int,int]) -> torch.Tensor:
    """
    - Robust clip (1st-99th percentile on non-zero voxels if possible)
    - Min-max normalization to [0,1]
    - Resize to out_shape (D,H,W) with trilinear interpolation
    Returns torch tensor shape (1, D, H, W)
    """
    v = vol.astype(np.float32)
    # choose nonzero voxels for statistics if available
    nz = v[np.abs(v) > 1e-8]
    src = nz if nz.size > 100 else v.flatten()
    lo = _safe_percentile(src, 1.0)
    hi = _safe_percentile(src, 99.0)
    if hi <= lo:
        lo, hi = float(np.min(src)), float(np.max(src))
    if hi <= lo:
        # constant image
        v = np.zeros_like(v, dtype=np.float32)
    else:
        v = np.clip(v, lo, hi)
        v = (v - lo) / (hi - lo + 1e-8)

    t = torch.from_numpy(v)[None, None, ...]  # (1,1,D,H,W)
    t = F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)
    t = t[0]  # (1,D,H,W)
    return t


# --------------------------
# Model loading + feature hook
# --------------------------
def build_model(arch: str, in_ch: int, num_classes: int = 2) -> torch.nn.Module:
    # resnet_3d.py must be in the same folder
    import CT67.dl_work.resnet_3d as r3d
    if not hasattr(r3d, arch):
        raise ValueError(f"resnet_3d.py has no function named '{arch}'. Available: resnet10/resnet18/resnet34/resnet50/resnet101/seresnet50/resnest50...")
    fn = getattr(r3d, arch)
    # Our resnet_3d factory uses signature (input_channels=1, num_classes=2) for resnet10/18/34/50/101
    model = fn(input_channels=in_ch, num_classes=num_classes)
    return model

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # Support different checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    # Strip 'module.' if present
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    # We don't print huge lists; just summary.
    if missing:
        print(f"[CKPT] Missing keys: {len(missing)} (loaded with strict=False)")
    if unexpected:
        print(f"[CKPT] Unexpected keys: {len(unexpected)} (loaded with strict=False)")

class FCInputHook:
    """
    Captures the input to model.fc (the embedding before the final linear layer).
    """
    def __init__(self):
        self.last: Optional[torch.Tensor] = None

    def __call__(self, module, module_in, module_out):
        # module_in is a tuple; first element is (B, F)
        self.last = module_in[0].detach()

def infer_one(model: torch.nn.Module, hook: FCInputHook, x: torch.Tensor, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: torch tensor (1,C,D,H,W)
    Returns: (prob (2,), embedding (F,))
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device, non_blocking=True)
        prob = model(x)  # softmax output in this implementation
        prob_np = prob[0].detach().cpu().numpy().astype(np.float32)
        if hook.last is None:
            raise RuntimeError("Failed to capture embedding: hook.last is None. Is model.fc present?")
        emb_np = hook.last[0].detach().cpu().numpy().astype(np.float32)
    return prob_np, emb_np


# --------------------------
# Main
# --------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpmri_nii", type=str, required=True, help="Path to mpMRI_nii folder (images).")
    ap.add_argument("--petct_nii", type=str, required=True, help="Path to PETCT_nii folder (images).")
    ap.add_argument("--clinical_csv", type=str, required=True, help="CSV with at least columns: id and label.")
    ap.add_argument("--id_col", type=str, default="id", help="ID column name in clinical CSV. Default: id")
    ap.add_argument("--label_col", type=str, required=True, help="Label column name in clinical CSV, e.g., isup2.")
    ap.add_argument("--ckpt", type=str, required=True, help="Trained checkpoint .pth")
    ap.add_argument("--arch", type=str, default="resnet50", help="Model architecture in resnet_3d.py (e.g., resnet50)")
    ap.add_argument("--modalities", type=str, default="ADC,DWI,T2,CT,PET",
                    help="Comma-separated modalities. Default: ADC,DWI,T2,CT,PET")
    ap.add_argument("--out_shape", type=str, default="8,64,64", help="Resize target (D,H,W). Default: 8,64,64")
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size. Default 1 (safe).")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu. Default cuda.")
    ap.add_argument("--out", type=str, default="prostate_deep_features.csv")
    ap.add_argument("--allow_missing", action="store_true",
                    help="If a modality is missing, fill that channel with zeros instead of skipping the case.")
    ap.add_argument("--dry_run", action="store_true", help="Only check availability, do not run the model.")
    return ap.parse_args()

def main():
    args = parse_args()
    mpmri_root = Path(args.mpmri_nii)
    petct_root = Path(args.petct_nii)
    ckpt_path = Path(args.ckpt)
    out_csv = Path(args.out)

    if not Path("resnet_3d.py").exists():
        raise FileNotFoundError("resnet_3d.py not found in current working directory. Put resnet_3d.py next to this script, or run from that folder.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if len(modalities) == 0:
        raise ValueError("modalities is empty.")

    out_shape = tuple(int(x) for x in args.out_shape.split(","))
    if len(out_shape) != 3:
        raise ValueError("out_shape must be 'D,H,W', e.g., 8,64,64")

    df = pd.read_csv(args.clinical_csv)
    if args.id_col not in df.columns:
        raise ValueError(f"id_col '{args.id_col}' not in clinical_csv columns: {list(df.columns)}")
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' not in clinical_csv columns: {list(df.columns)}")

    # Build availability report first
    rows_avail = []
    ok_all = 0
    for _, r in df.iterrows():
        pid = str(r[args.id_col])
        paths = _collect_paths(mpmri_root, petct_root, pid, modalities)
        missing = [m for m,p in paths.items() if p is None]
        has_all = (len(missing) == 0)
        ok_all += int(has_all)
        row = {"id": pid, "label": int(r[args.label_col]), "has_all": int(has_all), "missing": ",".join(missing)}
        for m,p in paths.items():
            row[f"path_{m}"] = str(p) if p is not None else ""
        rows_avail.append(row)

    avail_df = pd.DataFrame(rows_avail)
    if args.dry_run:
        avail_df.to_csv(out_csv, index=False)
        print(f"[DRY_RUN] wrote availability report: {out_csv}")
        print(f"[DRY_RUN] ok(all modalities) = {ok_all}/{len(df)}")
        return

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, switching to CPU.")
        device = "cpu"

    # Build model (input channels = number of modalities)
    model = build_model(args.arch, in_ch=len(modalities), num_classes=2)
    load_checkpoint(model, ckpt_path)
    model.to(device)

    # Hook to capture embedding
    if not hasattr(model, "fc"):
        raise RuntimeError("Model has no attribute 'fc' to hook for embeddings.")
    hook = FCInputHook()
    h = model.fc.register_forward_hook(hook)

    out_rows = []
    miss_rows = []

    try:
        for _, r in df.iterrows():
            pid = str(r[args.id_col])
            y = int(r[args.label_col])

            paths = _collect_paths(mpmri_root, petct_root, pid, modalities)
            missing = [m for m,p in paths.items() if p is None]
            if missing and not args.allow_missing:
                miss_rows.append({"id": pid, "label": y, "missing": ",".join(missing)})
                continue

            # Load each modality volume; if missing allowed, fill zeros
            ch_tensors: List[torch.Tensor] = []
            for m in modalities:
                p = paths.get(m)
                if p is None:
                    vol = np.zeros(out_shape, dtype=np.float32)
                    t = torch.from_numpy(vol)[None, ...]  # (1,D,H,W)
                else:
                    vol = load_nifti_as_float(p)
                    t = preprocess_volume(vol, out_shape=out_shape)  # (1,D,H,W)
                ch_tensors.append(t)

            x = torch.cat(ch_tensors, dim=0)[None, ...]  # (1,C,D,H,W)
            prob, emb = infer_one(model, hook, x, device=device)

            row = {"id": pid, "label": y, "prob0": float(prob[0]), "prob1": float(prob[1])}
            for j, v in enumerate(emb.tolist()):
                row[f"deep_{j}"] = float(v)
            out_rows.append(row)

        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(out_csv, index=False)
        print(f"[OK] wrote features CSV: {out_csv}  (n={len(out_df)})")

        if miss_rows:
            miss_df = pd.DataFrame(miss_rows)
            miss_path = out_csv.with_suffix(".missing.csv")
            miss_df.to_csv(miss_path, index=False)
            print(f"[WARN] some cases missing modalities, wrote: {miss_path}  (n={len(miss_df)})")

        # also write availability for reference
        avail_path = out_csv.with_suffix(".availability.csv")
        avail_df.to_csv(avail_path, index=False)
        print(f"[INFO] wrote availability detail: {avail_path}")

    finally:
        h.remove()

if __name__ == "__main__":
    main()
