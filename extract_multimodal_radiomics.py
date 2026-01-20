#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prostate Multi-modal Radiomics Feature Extraction Script (Parallel Version)

Extracts radiomics features from:
- mpMRI: ADC, DWI, T2 modalities
- PET-CT: PET, CT modalities

Output: CSV with ID, label (isup2), and features prefixed by modality

Features:
- Multiprocessing support for parallel extraction
- Configurable number of workers
"""

import os
import re
import glob
import argparse
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress radiomics debug messages
logging.getLogger('radiomics').setLevel(logging.WARNING)


def setup_extractor():
    """Setup PyRadiomics feature extractor with extensive settings.
    
    With wavelet (8 decompositions) + LoG (5 sigma values) + original,
    we get approximately: (1 + 8 + 5) * 93 texture features + shape features
    = ~1300+ features per modality, 6500+ total features across 5 modalities.
    """
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,  # Use original spacing
        'interpolator': sitk.sitkBSpline,
        'enableCExtensions': True,
        'normalize': True,
        'normalizeScale': 100,
        'geometryTolerance': 1e-3,  # Increased tolerance for geometry mismatch
        'correctMask': True,  # Resample mask to image space if needed
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # Enable all feature classes
    extractor.enableAllFeatures()
    
    # Enable image filters for more features:
    # 1. Wavelet filter: 8 decompositions (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
    extractor.enableImageTypeByName('Wavelet')
    
    # 2. LoG (Laplacian of Gaussian) filter: multiple sigma values
    # Sigma typically ranges from 1-5mm for medical imaging
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]})
    
    # 3. Square filter
    extractor.enableImageTypeByName('Square')
    
    # 4. SquareRoot filter
    extractor.enableImageTypeByName('SquareRoot')
    
    # 5. Logarithm filter
    extractor.enableImageTypeByName('Logarithm')
    
    # 6. Exponential filter
    extractor.enableImageTypeByName('Exponential')
    
    # 7. Gradient filter
    extractor.enableImageTypeByName('Gradient')
    
    # 8. LBP2D (Local Binary Pattern) - for 2D texture
    extractor.enableImageTypeByName('LBP2D')
    
    # 9. LBP3D (Local Binary Pattern 3D) - for 3D texture
    extractor.enableImageTypeByName('LBP3D')
    
    return extractor


def find_file_pattern(directory, pattern):
    """Find file matching pattern in directory."""
    matches = glob.glob(os.path.join(directory, pattern))
    if matches:
        return matches[0]
    return None


def extract_features_for_modality(image_path, mask_path, modality_prefix):
    """Extract radiomics features for a single image-mask pair.
    
    This function creates its own extractor to be process-safe.
    """
    features = {}
    
    if not os.path.exists(image_path):
        return features
    
    if not os.path.exists(mask_path):
        return features
    
    # Create extractor in this process
    extractor = setup_extractor()
    
    try:
        # Read image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        img_size = image.GetSize()
        mask_size = mask.GetSize()
        
        # Check if sizes match, if not try axis permutation or resampling
        if img_size != mask_size:
            # Check if axes are permuted (e.g., (Z,X,Y) vs (X,Y,Z))
            if set(img_size) == set(mask_size) and len(set(img_size)) > 1:
                mask_array = sitk.GetArrayFromImage(mask)
                
                # Find the permutation that matches sizes
                target_np_shape = (img_size[2], img_size[1], img_size[0])
                current_np_shape = mask_array.shape
                
                # Try different permutations
                permuted = False
                for axes in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                    test_shape = tuple(current_np_shape[i] for i in axes)
                    if test_shape == target_np_shape:
                        mask_array = np.transpose(mask_array, axes)
                        permuted = True
                        break
                
                if permuted:
                    mask = sitk.GetImageFromArray(mask_array)
                    mask.CopyInformation(image)
                else:
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(image)
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    mask = resampler.Execute(mask)
            else:
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                mask = resampler.Execute(mask)
        
        # Execute extraction with SimpleITK objects
        result = extractor.execute(image, mask)
        
        for key, value in result.items():
            if not key.startswith('diagnostics_'):
                feature_name = f"{modality_prefix}_{key}"
                if isinstance(value, np.ndarray):
                    features[feature_name] = float(value.flatten()[0])
                else:
                    features[feature_name] = float(value)
                    
    except Exception as e:
        # Log error but don't crash
        pass
    
    return features


def get_patient_files_mpmri(base_dir, patient_id):
    """Get mpMRI image and mask files for a patient."""
    image_dir = os.path.join(base_dir, 'mpMri_nii', patient_id)
    mask_dir = os.path.join(base_dir, 'mpMRI', patient_id)
    
    files = {}
    
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        return files
    
    for modality in ['ADC', 'DWI', 'T2']:
        image_pattern = f"{patient_id}*{modality}.nii.gz"
        image_file = find_file_pattern(image_dir, image_pattern)
        
        mask_pattern = f"{patient_id}*{modality}_Merge.nii"
        mask_file = find_file_pattern(mask_dir, mask_pattern)
        
        if image_file and mask_file:
            files[modality] = {'image': image_file, 'mask': mask_file}
    
    return files


def get_patient_files_petct(base_dir, patient_id):
    """Get PET-CT image and mask files for a patient."""
    image_dir = os.path.join(base_dir, 'PETCT_nii', patient_id)
    mask_dir = os.path.join(base_dir, 'PETCT', patient_id)
    
    files = {}
    
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        return files
    
    # PET modality
    pet_image_pattern = f"{patient_id}*_PET.nii.gz"
    pet_image_file = find_file_pattern(image_dir, pet_image_pattern)
    
    pet_mask_pattern = f"{patient_id}pet.nii"
    pet_mask_file = find_file_pattern(mask_dir, pet_mask_pattern)
    
    if pet_image_file and pet_mask_file:
        files['PET'] = {'image': pet_image_file, 'mask': pet_mask_file}
    
    # CT modality
    ct_image_pattern = f"{patient_id}*_CT.nii.gz"
    ct_image_file = find_file_pattern(image_dir, ct_image_pattern)
    
    ct_mask_file = None
    if os.path.isdir(mask_dir):
        for f in os.listdir(mask_dir):
            if f.endswith('.nii.gz') and f.startswith(patient_id) and 'pet' not in f.lower():
                ct_mask_file = os.path.join(mask_dir, f)
                break
    
    if ct_image_file and ct_mask_file:
        files['CT'] = {'image': ct_image_file, 'mask': ct_mask_file}
    
    return files


def process_patient_wrapper(args):
    """Wrapper function for multiprocessing - processes a single patient.
    
    Args:
        args: tuple of (base_dir, patient_id, label)
    
    Returns:
        dict: Patient features including ID and label
    """
    base_dir, patient_id, label = args
    
    patient_features = {'ID': patient_id, 'label': label}
    
    # Get all modality files
    mpmri_files = get_patient_files_mpmri(base_dir, patient_id)
    petct_files = get_patient_files_petct(base_dir, patient_id)
    
    all_modality_files = {}
    all_modality_files.update(mpmri_files)
    all_modality_files.update(petct_files)
    
    # Extract features for each modality
    for modality, paths in all_modality_files.items():
        features = extract_features_for_modality(
            paths['image'], paths['mask'], modality
        )
        patient_features.update(features)
    
    return patient_features


def main():
    parser = argparse.ArgumentParser(
        description='Extract radiomics features from prostate multi-modal images (Parallel Version)'
    )
    parser.add_argument(
        '--base-dir', type=str,
        default='/nfs/zc1/qianliexian/dataset',
        help='Base directory containing image folders'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/nfs/zc1/qianliexian/dataset/radiomics_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--label-file', type=str,
        default='/nfs/zc1/qianliexian/dataset/qianliexian_clinical_isup.csv',
        help='CSV file containing patient labels'
    )
    parser.add_argument(
        '--test-mode', action='store_true',
        help='Run on first 3 patients only for testing'
    )
    parser.add_argument(
        '--n-workers', type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 2, min 1)'
    )
    parser.add_argument(
        '--serial', action='store_true',
        help='Run in serial mode (no parallelization)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine number of workers
    if args.serial:
        n_workers = 1
    elif args.n_workers:
        n_workers = args.n_workers
    else:
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    
    logger.info(f"Using {n_workers} parallel workers")
    
    # Load labels
    logger.info(f"Loading labels from {args.label_file}")
    labels_df = pd.read_csv(args.label_file)
    labels_dict = dict(zip(labels_df['id'].astype(str), labels_df['isup2']))
    
    # Get patient IDs
    patient_ids = list(labels_dict.keys())
    
    if args.test_mode:
        patient_ids = patient_ids[:3]
        logger.info(f"Test mode: Processing {len(patient_ids)} patients only")
    else:
        logger.info(f"Processing {len(patient_ids)} patients")
    
    # Prepare arguments for parallel processing
    patient_args = [
        (args.base_dir, pid, labels_dict.get(pid, -1))
        for pid in patient_ids
    ]
    
    start_time = time.time()
    all_features = []
    
    if n_workers == 1:
        # Serial processing
        for i, patient_arg in enumerate(patient_args):
            logger.info(f"Processing patient {patient_arg[1]} ({i+1}/{len(patient_ids)})")
            result = process_patient_wrapper(patient_arg)
            all_features.append(result)
    else:
        # Parallel processing
        logger.info("Starting parallel extraction...")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_patient_wrapper, arg): arg[1] 
                      for arg in patient_args}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                patient_id = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    all_features.append(result)
                    
                    if completed % 10 == 0 or completed == len(patient_ids):
                        elapsed = time.time() - start_time
                        eta = (elapsed / completed) * (len(patient_ids) - completed)
                        logger.info(f"Completed {completed}/{len(patient_ids)} patients "
                                   f"(ETA: {eta/60:.1f} min)")
                        
                except Exception as e:
                    logger.error(f"Error processing patient {patient_id}: {e}")
                    all_features.append({'ID': patient_id, 'label': labels_dict.get(patient_id, -1)})
    
    elapsed_time = time.time() - start_time
    logger.info(f"Extraction completed in {elapsed_time/60:.1f} minutes")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns: ID, label first, then sorted feature columns
    feature_cols = [c for c in df.columns if c not in ['ID', 'label']]
    feature_cols.sort()
    df = df[['ID', 'label'] + feature_cols]
    
    # Save to CSV
    output_file = os.path.join(args.output_dir, 'radiomics_features_with_label.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Saved features to {output_file}")
    logger.info(f"Output shape: {df.shape[0]} patients x {df.shape[1]} columns")
    
    # Print summary
    logger.info("Feature count by modality:")
    for prefix in ['ADC', 'DWI', 'T2', 'PET', 'CT']:
        count = sum(1 for c in feature_cols if c.startswith(prefix + '_'))
        logger.info(f"  {prefix}: {count} features")


if __name__ == '__main__':
    main()
