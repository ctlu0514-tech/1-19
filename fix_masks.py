import SimpleITK as sitk
import os
import glob

# --- 配置您的路径 ---
# BASE_PATH 是指向 'dataset' 文件夹的 *父* 文件夹
BASE_PATH = '/nfs/zc1/qianliexian'
# -------------------------

def fix_mask_header(image_path, mask_path, output_path):
    """
    将图像的头文件信息复制给掩码，并保存为新的文件
    """
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # 关键一步：将image的几何信息（原点，间距，方向）复制给mask
        mask.CopyInformation(image)
        
        # 保存修复后的mask
        sitk.WriteImage(mask, output_path)
        print(f"  [成功] 已修复并保存: {output_path}")
        
    except Exception as e:
        print(f"  [失败] 处理失败. 图像: {image_path}, 掩码: {mask_path}. 错误: {e}")


if __name__ == '__main__':
    dataset_path = os.path.join(BASE_PATH, 'dataset')
    
    # 找到所有患者ID (以PETCT_nii目录为准)
    pet_image_dir = os.path.join(dataset_path, 'PETCT_nii')
    patient_ids = sorted([d for d in os.listdir(pet_image_dir) if os.path.isdir(os.path.join(pet_image_dir, d))])

    print(f"找到 {len(patient_ids)} 个患者. 开始修复PET掩码...")
    
    for patient_id in patient_ids:
        print(f"- 正在处理患者: {patient_id}")
        
        # 构建路径
        image_file_path = glob.glob(os.path.join(pet_image_dir, patient_id, f"{patient_id}*_PET.nii.gz"))
        mask_file_path = glob.glob(os.path.join(dataset_path, 'PETCT', patient_id, f"{patient_id}pet.nii"))
        
        if not image_file_path or not mask_file_path:
            print(f"  [警告] 找不到图像或掩码文件，跳过。")
            continue
            
        image_path = image_file_path[0]
        mask_path = mask_file_path[0]
        
        # 定义修复后的掩码保存路径
        output_dir = os.path.join(dataset_path, 'PETCT', patient_id)
        output_path = os.path.join(output_dir, f"{patient_id}pet_corrected.nii.gz")
        
        # 执行修复
        fix_mask_header(image_path, mask_path, output_path)
        
    print("\n修复完成！")