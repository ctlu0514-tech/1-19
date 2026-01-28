import os
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd

# 配置您的数据路径 (参考了您原本脚本的路径)
BASE_DIR = '/nfs/zc1/qianliexian/dataset'
LABEL_FILE = '/nfs/zc1/qianliexian/dataset/qianliexian_clinical_isup.csv'

def get_spacing(image_path):
    """读取影像的分辨率 (Spacing)"""
    if not os.path.exists(image_path):
        return None
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        reader.ReadImageInformation()
        return reader.GetSpacing() # 返回 (x, y, z) 的物理间距
    except:
        return None

def check_data_consistency():
    print(f"正在检查数据: {BASE_DIR} ...")
    
    # 1. 获取病人列表
    if os.path.exists(LABEL_FILE):
        df = pd.read_csv(LABEL_FILE)
        # 为了节省时间，我们只随机抽查 10 个病人，足够说明问题了
        patient_ids = df['id'].astype(str).sample(n=min(10, len(df)), random_state=42).tolist()
    else:
        print("没找到标签文件，将尝试自动搜索文件夹...")
        # 备选方案
        patient_ids = os.listdir(os.path.join(BASE_DIR, 'mpMri_nii'))[:10]

    print(f"抽查病人ID: {patient_ids}")
    print("-" * 60)
    
    # 存储所有分辨率数据
    spacings = {'ADC': [], 'DWI': [], 'T2': [], 'PET': [], 'CT': []}
    
    for pid in patient_ids:
        # 构建路径 (逻辑参考您的 extract_multimodal_radiomics.py)
        # mpMRI
        for mod in ['ADC', 'DWI', 'T2']:
            search_path = os.path.join(BASE_DIR, 'mpMri_nii', pid, f"{pid}*{mod}.nii.gz")
            files = glob.glob(search_path)
            if files:
                sp = get_spacing(files[0])
                if sp: spacings[mod].append(sp)
        
        # PET/CT
        pet_files = glob.glob(os.path.join(BASE_DIR, 'PETCT_nii', pid, f"{pid}*_PET.nii.gz"))
        if pet_files:
            sp = get_spacing(pet_files[0])
            if sp: spacings['PET'].append(sp)
            
        ct_files = glob.glob(os.path.join(BASE_DIR, 'PETCT_nii', pid, f"{pid}*_CT.nii.gz"))
        if ct_files:
            sp = get_spacing(ct_files[0])
            if sp: spacings['CT'].append(sp)

    # 2. 汇报结果
    print(f"{'模态':<10} | {'是否统一?':<10} | {'典型分辨率 (X, Y, Z)':<30} | {'Z轴层厚范围'}")
    print("-" * 80)
    
    for mod, sp_list in spacings.items():
        if not sp_list:
            print(f"{mod:<10} | {'无数据':<10}")
            continue
            
        sp_arr = np.array(sp_list)
        # 检查是否所有行都一样 (允许 0.001 的微小误差)
        is_uniform = np.allclose(sp_arr, sp_arr[0], atol=1e-3)
        
        # 典型值 (取中位数)
        median_sp = np.median(sp_arr, axis=0)
        median_str = f"({median_sp[0]:.2f}, {median_sp[1]:.2f}, {median_sp[2]:.2f})"
        
        # Z轴范围
        z_min, z_max = np.min(sp_arr[:, 2]), np.max(sp_arr[:, 2])
        z_range = f"{z_min:.2f} - {z_max:.2f}"
        
        status = "✅ 统一" if is_uniform else "⚠️ 不统一"
        print(f"{mod:<10} | {status:<10} | {median_str:<30} | {z_range}")

    print("-" * 80)

    
    # 简单逻辑判断
    all_uniform = all(np.allclose(np.array(spacings[m]), np.array(spacings[m])[0], atol=1e-3) for m in spacings if spacings[m])
    pet_res = np.median(spacings['PET'], axis=0) if spacings['PET'] else [0,0,0]
    mri_res = np.median(spacings['T2'], axis=0) if spacings['T2'] else [0,0,0]

if __name__ == "__main__":
    check_data_consistency()