#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-modal radiomics feature extraction for prostate datasets (MRI + CT)
with extremely detailed logging and "no full registration" alignment strategies.

What this script DOES (alignment):
  - Primary: resample mask -> image grid (nearest neighbor) + binarize (>0)
  - If image size is exactly 2x mask size AND mask disappears after resample:
      => resample image -> mask grid (linear) ("take half") and extract on that aligned pair
  - Optional: translation-only refinement for small offsets (no rotation/scale/deform)

What this script DOES NOT DO:
  - No rigid/affine/bspline registration between images (moving->fixed) across modalities.
    Only translation-only refinement is optional.

Outputs:
  - OUTPUT_CSV: wide-format features
  - SUMMARY_CSV: per-case per-modality summary (status, strategy, metrics)
  - LOG_FILE: extremely detailed log
  - DEBUG_DUMP_DIR: optional debug NIfTI dumps (only failures/refined by default)
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


# ================= Configuration =================
BASE_PATH = Path("/nfs/zc1/qianliexian/dataset")

MRI_IMAGE_DIR = BASE_PATH / "mpMri_nii"   # MRI images (NIfTI)
MRI_MASK_DIR  = BASE_PATH / "mpMRI"       # MRI masks or related files
CT_IMAGE_DIR  = BASE_PATH / "PETCT_nii"   # CT images (NIfTI)
CT_MASK_DIR   = BASE_PATH / "PETCT"       # CT masks (NIfTI)

OUTPUT_CSV  = BASE_PATH / "prostate_radiomics_features_wide_format.csv"
SUMMARY_CSV = BASE_PATH / "extraction_case_summary.csv"
LOG_FILE    = BASE_PATH / "extraction_debug_log.txt"

DEBUG_DUMP_DIR = BASE_PATH / "_debug_alignment_dumps"
DUMP_DEBUG_NIFTI = True
DUMP_ONLY_FAILURES_OR_REFINED = True  # if False, dumps for all cases/modalities (will be huge)

MRI_MODALITIES = ["T2", "ADC", "DWI"]
CT_MODALITY = "CT"

# Logging
CONSOLE_LEVEL = logging.INFO
FILE_LEVEL    = logging.DEBUG

# Alignment strategy knobs
ENABLE_TRANSLATION_REFINE = True       # for 2061-like small offset
MAX_TRANSLATION_MM = 15.0              # reject translation larger than this (safety)
TRANSLATION_OPT_ITERS = 120

# QC / sanity thresholds
MIN_MASK_NONZERO = 20                  # below this, consider empty/unreliable
MAX_MASK_FRACTION_OF_IMAGE = 0.50      # mask occupying >50% of image voxels is suspicious
VOLUME_RATIO_MIN = 0.10                # aligned mask volume / original mask volume
VOLUME_RATIO_MAX = 10.0                # aligned mask volume / original mask volume

# PyRadiomics settings (similar to your CT3)
DEFAULT_BINWIDTH = 25
DEFAULT_SIGMA = [3, 5]
DEFAULT_RESAMPLED_SPACING = [1, 1, 1]


# ================= Logging Setup =================
logger = logging.getLogger("radiomics_extraction")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.propagate = False

_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(CONSOLE_LEVEL)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_fh.setLevel(FILE_LEVEL)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


# ================= Helper Functions =================
def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name

def _ensure_3d(img: sitk.Image) -> sitk.Image:
    """If image is 4D, take the first volume. If 2D, keep as-is."""
    dim = img.GetDimension()
    if dim == 3:
        return img
    if dim == 4:
        # Extract first time/frame
        size = list(img.GetSize())
        idx = [0, 0, 0, 0]
        size[3] = 0
        return sitk.Extract(img, size=size[:3], index=idx[:3])
    return img

def _img_info(img: sitk.Image) -> Dict[str, Any]:
    img = _ensure_3d(img)
    return {
        "dim": img.GetDimension(),
        "size": img.GetSize(),
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "pixelID": img.GetPixelIDTypeAsString(),
    }

def _format_tuple(x, n=6) -> Tuple:
    return tuple(round(float(v), n) for v in x)

def _log_geometry(tag: str, img: sitk.Image) -> None:
    info = _img_info(img)
    logger.debug(
        f"    - [{tag}] dim={info['dim']} size={info['size']} spacing={_format_tuple(info['spacing'])} "
        f"origin={_format_tuple(info['origin'])} pixel={info['pixelID']}"
    )
    logger.debug(f"    - [{tag}] direction={_format_tuple(info['direction'])}")

def _mask_stats(mask: sitk.Image, max_uniques: int = 20) -> Dict[str, Any]:
    mask = _ensure_3d(mask)
    arr = sitk.GetArrayFromImage(mask)
    nonzero = int(np.count_nonzero(arr))
    uniques = np.unique(arr)
    unique_count = int(uniques.size)
    preview = uniques[:max_uniques].tolist()
    maxv = float(uniques.max()) if unique_count else 0.0
    minv = float(uniques.min()) if unique_count else 0.0
    binary_like = (unique_count <= 5) and (maxv <= 5.0) and (minv >= 0.0)
    return {
        "nonzero": nonzero,
        "unique_count": unique_count,
        "unique_preview": preview,
        "min": minv,
        "max": maxv,
        "binary_like": bool(binary_like),
    }

def _binarize_mask(mask: sitk.Image) -> sitk.Image:
    mask = _ensure_3d(mask)
    return sitk.Cast(mask > 0, sitk.sitkUInt8)

def _voxel_volume_mm3(img: sitk.Image) -> float:
    sp = img.GetSpacing()
    return float(sp[0] * sp[1] * sp[2]) if img.GetDimension() == 3 else float(sp[0] * sp[1])

def _mask_volume_mm3(mask: sitk.Image) -> float:
    mask = _ensure_3d(mask)
    arr = sitk.GetArrayFromImage(mask)
    nz = int(np.count_nonzero(arr))
    return nz * _voxel_volume_mm3(mask)

def _mask_fraction_of_image(mask: sitk.Image, image: sitk.Image) -> float:
    m = _ensure_3d(mask)
    i = _ensure_3d(image)
    mnz = int(np.count_nonzero(sitk.GetArrayFromImage(m)))
    isz = int(np.prod(i.GetSize()))
    return float(mnz / isz) if isz > 0 else 0.0

def _centroid_physical(mask: sitk.Image) -> Optional[Tuple[float, float, float]]:
    """Compute centroid in physical coordinates using label shape statistics."""
    mask = _ensure_3d(mask)
    if int(np.count_nonzero(sitk.GetArrayFromImage(mask))) == 0:
        return None
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask)
    # mask is binary with label=1
    if not ls.HasLabel(1):
        return None
    c = ls.GetCentroid(1)  # physical
    return (float(c[0]), float(c[1]), float(c[2])) if len(c) == 3 else None

def _image_center_physical(img: sitk.Image) -> Tuple[float, float, float]:
    img = _ensure_3d(img)
    size = img.GetSize()
    center_index = [(s - 1) / 2.0 for s in size]
    c = img.TransformContinuousIndexToPhysicalPoint(center_index)
    return (float(c[0]), float(c[1]), float(c[2]))

def _distance_mm(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def _find_first_match(directory: Path, include_globs: List[str], exclude_substrings: List[str] = None) -> Optional[Path]:
    exclude_substrings = exclude_substrings or []
    candidates: List[Path] = []
    for g in include_globs:
        candidates.extend(directory.glob(g))
    uniq = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        fname = p.name.lower()
        if any(ex in fname for ex in exclude_substrings):
            continue
        uniq.append(p)
    uniq.sort()
    return uniq[0] if uniq else None



def _is_nii_file(p: Path) -> bool:
    """Return True if path looks like a NIfTI file (.nii / .nii.gz)."""
    n = p.name.lower()
    return p.is_file() and (n.endswith(".nii") or n.endswith(".nii.gz"))
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _dump_debug(case_dir: Path, tag: str, img: sitk.Image, msk: sitk.Image, suffix: str) -> None:
    if not DUMP_DEBUG_NIFTI:
        return
    _safe_mkdir(case_dir)
    img_p = case_dir / f"{tag}_{suffix}_img.nii.gz"
    msk_p = case_dir / f"{tag}_{suffix}_msk.nii.gz"
    try:
        sitk.WriteImage(_ensure_3d(img), str(img_p))
        sitk.WriteImage(_ensure_3d(msk), str(msk_p))
    except Exception as e:
        logger.debug(f"[DUMP] failed to write debug nifti for {tag}: {e}")


def _translation_refine(mask_on_grid: sitk.Image, image_on_grid: sitk.Image) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Translation-only refinement. Returns (refined_mask, meta).
    Safety:
      - if optimizer finds translation > MAX_TRANSLATION_MM => reject
    """
    meta: Dict[str, Any] = {
        "applied": False,
        "offset_mm": None,
        "offset_norm_mm": None,
        "rejected_reason": None,
        "exception": None,
    }

    m = _ensure_3d(mask_on_grid)
    i = _ensure_3d(image_on_grid)

    if int(np.count_nonzero(sitk.GetArrayFromImage(m))) == 0:
        meta["rejected_reason"] = "empty_mask"
        return m, meta

    # fixed: gradient magnitude (edge-like)
    fixed = sitk.GradientMagnitude(i)
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)

    # moving: distance map from mask (smooth-ish)
    moving = sitk.SignedMaurerDistanceMap(m, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    tx = sitk.TranslationTransform(fixed.GetDimension())
    tx.SetOffset((0.0, 0.0, 0.0))

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.02)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.01,
        numberOfIterations=TRANSLATION_OPT_ITERS,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(tx, inPlace=False)

    try:
        out_tx = reg.Execute(fixed, moving)
        offset = np.array(list(out_tx.GetOffset()), dtype=float)
        norm = float(np.linalg.norm(offset))
        meta["offset_mm"] = tuple(float(x) for x in offset.tolist())
        meta["offset_norm_mm"] = norm

        if norm > MAX_TRANSLATION_MM:
            meta["rejected_reason"] = f"translation_too_large>{MAX_TRANSLATION_MM}mm"
            return m, meta

        refined = sitk.Resample(m, i, out_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        refined = sitk.Cast(refined > 0, sitk.sitkUInt8)
        meta["applied"] = True
        return refined, meta

    except Exception as e:
        meta["exception"] = str(e)
        meta["rejected_reason"] = "registration_exception"
        return m, meta


def _prepare_aligned_pair(
    image: sitk.Image,
    mask: sitk.Image,
    enable_translation_refine: bool = ENABLE_TRANSLATION_REFINE,
) -> Tuple[sitk.Image, sitk.Image, Dict[str, Any]]:
    """
    Core alignment logic (NO full registration):
      Strategy A: resample mask -> image grid (NN)
      If ROI disappears AND image size == 2x mask size => Strategy B: resample image -> mask grid ("take half")
      Optional: translation-only refine after a non-empty alignment.
    """
    meta: Dict[str, Any] = {
        "strategy": None,
        "mask_nonzero_before": None,
        "mask_nonzero_after": None,
        "volume_before_mm3": None,
        "volume_after_mm3": None,
        "volume_ratio": None,
        "mask_fraction_of_image": None,
        "translation_refine": None,
        "warnings": [],
    }

    img = _ensure_3d(image)
    msk0 = _binarize_mask(mask)

    nz_before = int(np.count_nonzero(sitk.GetArrayFromImage(msk0)))
    meta["mask_nonzero_before"] = nz_before
    meta["volume_before_mm3"] = _mask_volume_mm3(msk0)

    # Strategy A: mask -> image grid
    aligned = sitk.Resample(msk0, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    aligned = sitk.Cast(aligned > 0, sitk.sitkUInt8)
    nz_after = int(np.count_nonzero(sitk.GetArrayFromImage(aligned)))

    meta["strategy"] = "A_mask_to_image_nn"
    meta["mask_nonzero_after"] = nz_after
    meta["volume_after_mm3"] = _mask_volume_mm3(aligned)
    meta["volume_ratio"] = (meta["volume_after_mm3"] / meta["volume_before_mm3"]) if meta["volume_before_mm3"] > 0 else 0.0
    meta["mask_fraction_of_image"] = _mask_fraction_of_image(aligned, img)

    if nz_before > 0 and nz_after == 0:
        meta["warnings"].append("roi_vanished_after_mask_to_image_resample")

        ms = np.array(msk0.GetSize(), dtype=np.int64)
        im = np.array(img.GetSize(), dtype=np.int64)

        # 只处理：x/y 相同，z 轴 2 倍
        is_z2x = (im[0] == ms[0]) and (im[1] == ms[1]) and (im[2] == 2 * ms[2])

        if is_z2x:
            # Strategy B: image -> mask grid（实际上是“把图像重采样到mask的网格”）
            img_half = sitk.Resample(img, msk0, sitk.Transform(), sitk.sitkLinear, 0.0, img.GetPixelID())
            aligned2 = msk0

            meta["strategy"] = "B_image_to_mask_linear_take_half"
            meta["mask_nonzero_after"] = int(np.count_nonzero(sitk.GetArrayFromImage(aligned2)))
            meta["volume_after_mm3"] = _mask_volume_mm3(aligned2)
            meta["volume_ratio"] = (meta["volume_after_mm3"] / meta["volume_before_mm3"]) if meta["volume_before_mm3"] > 0 else 0.0
            meta["mask_fraction_of_image"] = _mask_fraction_of_image(aligned2, img_half)

            if enable_translation_refine and meta["mask_nonzero_after"] > 0:
                refined, tmeta = _translation_refine(aligned2, img_half)
                meta["translation_refine"] = tmeta
                if tmeta.get("applied"):
                    aligned2 = refined
                    meta["strategy"] += "+T_translation_refine"
                else:
                    meta["strategy"] += "+T_skipped_or_rejected"

            return img_half, aligned2, meta

    # Optional translation refine for Strategy A when mask exists
    if enable_translation_refine and nz_after > 0:
        refined, tmeta = _translation_refine(aligned, img)
        meta["translation_refine"] = tmeta
        if tmeta.get("applied"):
            aligned = refined
            meta["strategy"] += "+T_translation_refine"
        else:
            meta["strategy"] += "+T_skipped_or_rejected"

    return img, aligned, meta


# ================= Extractor Class =================
class ProstateRadiomicsExtractor:
    def __init__(self):
        self.mri_image_dir = MRI_IMAGE_DIR
        self.mri_mask_dir = MRI_MASK_DIR
        self.ct_image_dir = CT_IMAGE_DIR
        self.ct_mask_dir = CT_MASK_DIR

        self.ct_extractor = self._initialize_ct_extractor()
        self.mri_extractor = self._initialize_mri_extractor()

        # Build a fast lookup for CT masks to avoid per-case directory scans
        self._ct_mask_index = self._build_ct_mask_index()

        if DUMP_DEBUG_NIFTI:
            _safe_mkdir(DEBUG_DUMP_DIR)

    def _initialize_ct_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        settings = {
            "geometryTolerance": 1e-3,
            "binWidth": DEFAULT_BINWIDTH,
            "sigma": DEFAULT_SIGMA,
            "interpolator": sitk.sitkBSpline,
            "resampledPixelSpacing": DEFAULT_RESAMPLED_SPACING,
            "voxelArrayShift": 1000,
            "normalize": True,
            "normalizeScale": 100,
        }
        ext = featureextractor.RadiomicsFeatureExtractor(**settings)
        ext.enableAllImageTypes()
        ext.enableAllFeatures()
        logger.info("[INIT] CT extractor: AllImageTypes + AllFeatures; sigma=%s; resample=%s", DEFAULT_SIGMA, DEFAULT_RESAMPLED_SPACING)
        logger.debug(f"[INIT] CT enabled image types: {list(ext.enabledImagetypes.keys())}")
        logger.debug(f"[INIT] CT enabled feature classes: {list(ext.enabledFeatures.keys())}")
        return ext

    def _initialize_mri_extractor(self) -> featureextractor.RadiomicsFeatureExtractor:
        settings = {
            "geometryTolerance": 1e-3,
            "binWidth": DEFAULT_BINWIDTH,
            "sigma": DEFAULT_SIGMA,
            "interpolator": sitk.sitkBSpline,
            "resampledPixelSpacing": DEFAULT_RESAMPLED_SPACING,
            "normalize": True,
            "normalizeScale": 100,
        }
        ext = featureextractor.RadiomicsFeatureExtractor(**settings)
        ext.enableAllImageTypes()
        ext.enableAllFeatures()
        logger.info("[INIT] MRI extractor: AllImageTypes + AllFeatures; sigma=%s; resample=%s", DEFAULT_SIGMA, DEFAULT_RESAMPLED_SPACING)
        logger.debug(f"[INIT] MRI enabled image types: {list(ext.enabledImagetypes.keys())}")
        logger.debug(f"[INIT] MRI enabled feature classes: {list(ext.enabledFeatures.keys())}")
        return ext

    def _infer_patient_name(self, folder: Path, patient_id: str) -> str:
        nii = sorted(folder.glob("*.nii*"))
        if not nii:
            return ""
        stem = _strip_nii_suffix(nii[0].name)
        if stem.startswith(patient_id):
            stem = stem[len(patient_id):]
        for t in ["_CT", "_ct", "CT", "T2", "ADC", "DWI", "Merge", "_", "-"]:
            stem = stem.replace(t, "")
        return stem.strip()


    def _build_ct_mask_index(self) -> Dict[str, Path]:
        """Index CT masks under PETCT to avoid per-case directory scans.

        Supported layouts:
          - PETCT/<PatientID>/... (folder layout; handled directly in _find_ct_mask)
          - PETCT/<PatientID><Name>.nii.gz (flat layout; indexed here)
        """
        idx: Dict[str, Path] = {}
        if not self.ct_mask_dir.exists():
            return idx

        for p in self.ct_mask_dir.iterdir():
            if not _is_nii_file(p):
                continue
            name_l = p.name.lower()
            if "pet" in name_l:
                continue

            # Extract leading digits as patient_id (e.g., "1002xxx.nii.gz" -> "1002")
            pid = ""
            for ch in p.name:
                if ch.isdigit():
                    pid += ch
                else:
                    break
            if not pid:
                continue

            # Keep the first occurrence deterministically
            if pid not in idx:
                idx[pid] = p

        return idx

    def _find_mri_image(self, folder: Path, mod: str) -> Optional[Path]:
        """Fast deterministic lookup inside mpMri_nii/<PatientID>/.

        The dataset layout implies each patient folder contains only the modality images.
        If missing, return None immediately (do not fall back to searching elsewhere).
        """
        if folder is None or (not folder.exists()):
            return None

        mod_l = mod.lower()
        files = [p for p in folder.iterdir() if _is_nii_file(p)]
        # Defensive: avoid accidentally picking masks if they appear here
        files = [p for p in files if not any(x in p.name.lower() for x in ("mask", "seg", "label", "roi"))]

        hits = [p for p in files if mod_l in p.name.lower()]
        hits.sort()
        if not hits:
            return None
        if len(hits) > 1:
            logger.warning(f"[{folder.name}:MRI-{mod}] Multiple candidate images found, using first: {[h.name for h in hits]}")
        return hits[0]

    def _find_mri_mask(self, patient_id: str, mod: str) -> Optional[Path]:
        """Fast deterministic lookup inside mpMRI/<PatientID>/.

        IMPORTANT: Do NOT scan the mpMRI root as a fallback. If a patient's mask folder or
        modality mask is missing, we record the issue and move on.
        """
        d = self.mri_mask_dir / patient_id
        if not d.exists():
            return None

        mod_l = mod.lower()
        files = [p for p in d.iterdir() if _is_nii_file(p)]
        hits = [p for p in files if mod_l in p.name.lower()]
        hits.sort()
        if not hits:
            return None
        if len(hits) > 1:
            logger.warning(f"[{patient_id}:MRI-{mod}] Multiple candidate masks found, using first: {[h.name for h in hits]}")
        return hits[0]

    def _find_ct_image(self, patient_id: str) -> Optional[Path]:
        """Locate CT image inside PETCT_nii/<PatientID>/ (ignore PET)."""
        d = self.ct_image_dir / patient_id
        if not d.exists():
            return None

        files = [p for p in d.iterdir() if _is_nii_file(p)]
        files = [p for p in files if "pet" not in p.name.lower()]

        # Prefer filenames that clearly indicate CT
        hits = [p for p in files if "ct" in p.name.lower()]
        if not hits:
            hits = files  # if the folder only contains CT, just use it

        hits.sort()
        if not hits:
            return None
        if len(hits) > 1:
            logger.warning(f"[{patient_id}:CT] Multiple candidate CT images found, using first: {[h.name for h in hits]}")
        return hits[0]

    def _find_ct_mask(self, patient_id: str) -> Optional[Path]:
        """Locate CT mask inside PETCT.

        Preferred: PETCT/<PatientID>/... (folder layout).
        Fallback: PETCT root contains files like <PatientID><Name>.nii.gz (flat layout).
        """
        d = self.ct_mask_dir / patient_id
        if d.is_dir():
            files = [p for p in d.iterdir() if _is_nii_file(p)]
            files = [p for p in files if "pet" not in p.name.lower()]
            files.sort()
            if not files:
                return None
            if len(files) > 1:
                logger.warning(f"[{patient_id}:CT] Multiple candidate CT masks found in folder, using first: {[h.name for h in files]}")
            return files[0]

        # Flat layout lookup (pre-indexed)
        p = (self._ct_mask_index or {}).get(patient_id)
        return p

    def _extract_features(self, img: sitk.Image, msk: sitk.Image, prefix: str) -> Dict[str, Any]:
        extractor = self.ct_extractor if prefix.upper() == "CT" else self.mri_extractor
        feats = extractor.execute(img, msk, label=1)
        out: Dict[str, Any] = {}
        for k, v in feats.items():
            if k.startswith("diagnostics_"):
                continue
            key = f"{prefix}_{k}"
            if isinstance(v, (int, float, np.number)):
                out[key] = float(v)
            else:
                out[key] = v
        return out

    def _read_and_prepare(self, image_path: Path, mask_path: Path, tag: str) -> Tuple[Optional[sitk.Image], Optional[sitk.Image], Dict[str, Any], str]:
        meta: Dict[str, Any] = {
            "tag": tag,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "status": None,
            "reason": None,
            "alignment": None,
            "qc_warnings": [],
            "mask_stats_before": None,
            "mask_stats_after": None,
            "centroid_dist_to_img_center_mm": None,
        }

        try:
            img = sitk.ReadImage(str(image_path))
            img = _ensure_3d(img)
        except Exception as e:
            logger.exception(f"[{tag}] FAILED to read image: {image_path} | {e}")
            meta["status"] = "READ_IMAGE_FAIL"
            meta["reason"] = str(e)
            return None, None, meta, "READ_IMAGE_FAIL"

        try:
            msk = sitk.ReadImage(str(mask_path))
            msk = _ensure_3d(msk)
        except Exception as e:
            logger.exception(f"[{tag}] FAILED to read mask: {mask_path} | {e}")
            meta["status"] = "READ_MASK_FAIL"
            meta["reason"] = str(e)
            return img, None, meta, "READ_MASK_FAIL"

        logger.debug(f"[{tag}] image={image_path}")
        _log_geometry(f"{tag}:IMG", img)

        logger.debug(f"[{tag}] mask={mask_path}")
        _log_geometry(f"{tag}:MSK", msk)

        st_before = _mask_stats(msk)
        meta["mask_stats_before"] = st_before
        logger.debug(
            f"[{tag}] mask BEFORE: nonzero={st_before['nonzero']} uniques={st_before['unique_count']} "
            f"min={st_before['min']} max={st_before['max']} binary_like={st_before['binary_like']} "
            f"preview={st_before['unique_preview']}"
        )
        if not st_before["binary_like"]:
            meta["qc_warnings"].append("mask_not_binary_like_possible_wrong_file")

        # Align
        img_aligned, msk_aligned, al_meta = _prepare_aligned_pair(img, msk, enable_translation_refine=ENABLE_TRANSLATION_REFINE)
        meta["alignment"] = al_meta

        st_after = _mask_stats(msk_aligned)
        meta["mask_stats_after"] = st_after
        logger.debug(
            f"[{tag}] mask AFTER: nonzero={st_after['nonzero']} uniques={st_after['unique_count']} "
            f"min={st_after['min']} max={st_after['max']} binary_like={st_after['binary_like']} "
            f"preview={st_after['unique_preview']}"
        )

        # QC warnings
        if st_after["nonzero"] == 0:
            meta["status"] = "EMPTY_MASK"
            meta["qc_warnings"].append("empty_after_alignment")
            return img_aligned, msk_aligned, meta, "EMPTY_MASK"

        if st_after["nonzero"] < MIN_MASK_NONZERO:
            meta["qc_warnings"].append(f"mask_too_small<{MIN_MASK_NONZERO}_voxels")

        frac = al_meta.get("mask_fraction_of_image", 0.0)
        if frac is not None and frac > MAX_MASK_FRACTION_OF_IMAGE:
            meta["qc_warnings"].append(f"mask_fraction_too_large>{MAX_MASK_FRACTION_OF_IMAGE}")

        vr = al_meta.get("volume_ratio", None)
        if vr is not None and (vr < VOLUME_RATIO_MIN or vr > VOLUME_RATIO_MAX):
            meta["qc_warnings"].append(f"volume_ratio_out_of_range[{VOLUME_RATIO_MIN},{VOLUME_RATIO_MAX}]")

        # centroid distance to image center (rough sanity; not organ-aware)
        c = _centroid_physical(msk_aligned)
        if c is not None:
            ic = _image_center_physical(img_aligned)
            dmm = _distance_mm(c, ic)
            meta["centroid_dist_to_img_center_mm"] = dmm
            # don't hard-fail, just warn if extremely off-center
            if dmm > 120:
                meta["qc_warnings"].append("centroid_far_from_image_center>120mm_possible_mismatch")

        meta["status"] = "OK"
        return img_aligned, msk_aligned, meta, "OK"

    def run(self) -> None:
        logger.info("=== Radiomics extraction started ===")
        logger.info(f"BASE_PATH={BASE_PATH}")
        logger.info(f"MRI_IMAGE_DIR={self.mri_image_dir}")
        logger.info(f"MRI_MASK_DIR={self.mri_mask_dir}")
        logger.info(f"CT_IMAGE_DIR={self.ct_image_dir}")
        logger.info(f"CT_MASK_DIR={self.ct_mask_dir}")
        logger.info(f"LOG_FILE={LOG_FILE}")
        logger.info(f"SUMMARY_CSV={SUMMARY_CSV}")
        logger.info(f"OUTPUT_CSV={OUTPUT_CSV}")
        logger.info(f"DEBUG_DUMP_DIR={DEBUG_DUMP_DIR} (enabled={DUMP_DEBUG_NIFTI})")

        if not self.mri_image_dir.exists():
            logger.error(f"MRI_IMAGE_DIR not found: {self.mri_image_dir}")
            return

        patient_folders = sorted([d for d in self.mri_image_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(patient_folders)} patient folders under MRI_IMAGE_DIR")

        all_records: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        counters = {m: {"OK": 0, "MISSING": 0, "EMPTY_MASK": 0, "READ_FAIL": 0, "EXTRACT_FAIL": 0} for m in ["CT"] + MRI_MODALITIES}

        for idx, folder in enumerate(patient_folders, start=1):
            patient_id = folder.name
            patient_name = self._infer_patient_name(folder, patient_id)

            logger.info("=" * 90)
            logger.info(f"[CASE {idx}/{len(patient_folders)}] PatientID={patient_id} PatientName={patient_name}")

            record: Dict[str, Any] = {"PatientID": patient_id, "PatientName": patient_name}
            case_sum: Dict[str, Any] = {"PatientID": patient_id, "PatientName": patient_name}

            # ---------- MRI ----------
            for mod in MRI_MODALITIES:
                tag = f"{patient_id}:MRI-{mod}"
                img_path = self._find_mri_image(folder, mod)
                msk_path = self._find_mri_mask(patient_id, mod)

                if img_path is None or msk_path is None:
                    reason = []
                    if img_path is None:
                        reason.append("image_missing")
                    if msk_path is None:
                        reason.append("mask_missing")
                    logger.warning(f"[{tag}] MISSING ({', '.join(reason)}). img={img_path} mask={msk_path}")
                    case_sum[f"MRI_{mod}_status"] = "MISSING"
                    case_sum[f"MRI_{mod}_reason"] = ",".join(reason)
                    counters[mod]["MISSING"] += 1
                    continue

                img, aligned_msk, meta, status = self._read_and_prepare(img_path, msk_path, tag)

                # dump debug if needed
                if DUMP_DEBUG_NIFTI:
                    refined_applied = bool(((meta.get("alignment") or {}).get("translation_refine") or {}).get("applied", False))
                    should_dump = (not DUMP_ONLY_FAILURES_OR_REFINED) or (status != "OK") or refined_applied
                    if should_dump and img is not None and aligned_msk is not None:
                        case_dir = DEBUG_DUMP_DIR / patient_id
                        _dump_debug(case_dir, tag.replace(":", "_"), img, aligned_msk, suffix=status)

                # write summary meta
                case_sum[f"MRI_{mod}_status"] = status
                case_sum[f"MRI_{mod}_strategy"] = (meta.get("alignment", {}) or {}).get("strategy")
                case_sum[f"MRI_{mod}_mask_nz_before"] = (meta.get("alignment", {}) or {}).get("mask_nonzero_before")
                case_sum[f"MRI_{mod}_mask_nz_after"]  = (meta.get("alignment", {}) or {}).get("mask_nonzero_after")
                case_sum[f"MRI_{mod}_vol_ratio"] = (meta.get("alignment", {}) or {}).get("volume_ratio")
                case_sum[f"MRI_{mod}_mask_frac_img"] = (meta.get("alignment", {}) or {}).get("mask_fraction_of_image")
                case_sum[f"MRI_{mod}_centroid_dist_mm"] = meta.get("centroid_dist_to_img_center_mm")
                case_sum[f"MRI_{mod}_qc_warnings"] = "|".join(meta.get("qc_warnings", []))

                tmeta = (meta.get("alignment", {}) or {}).get("translation_refine")
                if isinstance(tmeta, dict):
                    case_sum[f"MRI_{mod}_trans_applied"] = tmeta.get("applied")
                    case_sum[f"MRI_{mod}_trans_offset_mm"] = str(tmeta.get("offset_mm"))
                    case_sum[f"MRI_{mod}_trans_norm_mm"] = tmeta.get("offset_norm_mm")
                    case_sum[f"MRI_{mod}_trans_reject"] = tmeta.get("rejected_reason")

                if status in ("READ_IMAGE_FAIL", "READ_MASK_FAIL"):
                    counters[mod]["READ_FAIL"] += 1
                    continue
                if status == "EMPTY_MASK":
                    counters[mod]["EMPTY_MASK"] += 1
                    continue

                # Extract features
                try:
                    feats = self._extract_features(img, aligned_msk, f"MRI{mod}")
                    record.update(feats)
                    nfeats = len(feats)
                    logger.info(f"[{tag}] EXTRACT_OK features={nfeats} strategy={case_sum[f'MRI_{mod}_strategy']}")
                    case_sum[f"MRI_{mod}_n_features"] = nfeats
                    counters[mod]["OK"] += 1
                except Exception as e:
                    logger.exception(f"[{tag}] EXTRACT_FAIL: {e}")
                    case_sum[f"MRI_{mod}_status"] = "EXTRACT_FAIL"
                    case_sum[f"MRI_{mod}_reason"] = str(e)
                    counters[mod]["EXTRACT_FAIL"] += 1

            # ---------- CT ----------
            tag = f"{patient_id}:CT"
            ct_img_path = self._find_ct_image(patient_id)
            ct_msk_path = self._find_ct_mask(patient_id)

            if ct_img_path is None or ct_msk_path is None:
                reason = []
                if ct_img_path is None:
                    reason.append("image_missing")
                if ct_msk_path is None:
                    reason.append("mask_missing")
                logger.warning(f"[{tag}] MISSING ({', '.join(reason)}). img={ct_img_path} mask={ct_msk_path}")
                case_sum["CT_status"] = "MISSING"
                case_sum["CT_reason"] = ",".join(reason)
                counters["CT"]["MISSING"] += 1
            else:
                img, aligned_msk, meta, status = self._read_and_prepare(ct_img_path, ct_msk_path, tag)

                # dump debug if needed
                if DUMP_DEBUG_NIFTI:
                    refined_applied = bool(((meta.get("alignment") or {}).get("translation_refine") or {}).get("applied", False))
                    should_dump = (not DUMP_ONLY_FAILURES_OR_REFINED) or (status != "OK") or refined_applied
                    if should_dump and img is not None and aligned_msk is not None:
                        case_dir = DEBUG_DUMP_DIR / patient_id
                        _dump_debug(case_dir, tag.replace(":", "_"), img, aligned_msk, suffix=status)

                case_sum["CT_status"] = status
                case_sum["CT_strategy"] = (meta.get("alignment", {}) or {}).get("strategy")
                case_sum["CT_mask_nz_before"] = (meta.get("alignment", {}) or {}).get("mask_nonzero_before")
                case_sum["CT_mask_nz_after"]  = (meta.get("alignment", {}) or {}).get("mask_nonzero_after")
                case_sum["CT_vol_ratio"] = (meta.get("alignment", {}) or {}).get("volume_ratio")
                case_sum["CT_mask_frac_img"] = (meta.get("alignment", {}) or {}).get("mask_fraction_of_image")
                case_sum["CT_centroid_dist_mm"] = meta.get("centroid_dist_to_img_center_mm")
                case_sum["CT_qc_warnings"] = "|".join(meta.get("qc_warnings", []))

                tmeta = (meta.get("alignment", {}) or {}).get("translation_refine")
                if isinstance(tmeta, dict):
                    case_sum["CT_trans_applied"] = tmeta.get("applied")
                    case_sum["CT_trans_offset_mm"] = str(tmeta.get("offset_mm"))
                    case_sum["CT_trans_norm_mm"] = tmeta.get("offset_norm_mm")
                    case_sum["CT_trans_reject"] = tmeta.get("rejected_reason")

                if status in ("READ_IMAGE_FAIL", "READ_MASK_FAIL"):
                    counters["CT"]["READ_FAIL"] += 1
                elif status == "EMPTY_MASK":
                    counters["CT"]["EMPTY_MASK"] += 1
                else:
                    try:
                        feats = self._extract_features(img, aligned_msk, "CT")
                        record.update(feats)
                        nfeats = len(feats)
                        logger.info(f"[{tag}] EXTRACT_OK features={nfeats} strategy={case_sum['CT_strategy']}")
                        case_sum["CT_n_features"] = nfeats
                        counters["CT"]["OK"] += 1
                    except Exception as e:
                        logger.exception(f"[{tag}] EXTRACT_FAIL: {e}")
                        case_sum["CT_status"] = "EXTRACT_FAIL"
                        case_sum["CT_reason"] = str(e)
                        counters["CT"]["EXTRACT_FAIL"] += 1

            all_records.append(record)
            summary_rows.append(case_sum)

        # ---------- Save outputs ----------
        if all_records:
            df = pd.DataFrame(all_records)
            base_cols = ["PatientID", "PatientName"]
            other_cols = [c for c in df.columns if c not in base_cols]
            df = df[base_cols + other_cols]
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            logger.info("=" * 90)
            logger.info(f"[DONE] Feature CSV saved: {OUTPUT_CSV} | shape={df.shape}")
        else:
            logger.warning("[DONE] No feature records generated. Nothing to save.")

        if summary_rows:
            s = pd.DataFrame(summary_rows)
            s.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
            logger.info(f"[DONE] Case summary CSV saved: {SUMMARY_CSV} | shape={s.shape}")

        logger.info("=" * 90)
        logger.info("[SUMMARY] modality-level counters (OK / MISSING / EMPTY_MASK / READ_FAIL / EXTRACT_FAIL)")
        for mod, d in counters.items():
            logger.info(f"  - {mod}: {d}")
        logger.info(f"[SUMMARY] Full logs: {LOG_FILE}")
        logger.info("=" * 90)


if __name__ == "__main__":
    ProstateRadiomicsExtractor().run()