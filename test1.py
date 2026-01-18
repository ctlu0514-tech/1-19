import os
import csv

# ================= 配置区域 =================
# 1. 你的数据根目录
DATA_ROOT = "/nfs/zc1/qianliexian/dataset/"

# 2. 你的 CSV 文件路径
CSV_PATH = os.path.join(DATA_ROOT, "qianliexian_clinical_isup.csv")

# 3. 结果保存到哪个文件
OUTPUT_FILE = "missing_samples_report.txt"

# 4. 定义模态和对应的文件夹
# 根据你之前的报错，你的 PET/CT 在 PETCT_nii，MRI 在 mpMri_nii
MODALITIES = {
    'ADC': 'mpMri_nii',
    'DWI': 'mpMri_nii',
    'T2':  'mpMri_nii',  # 你的文件名应该是 T2 (之前改过 T2FS -> T2)
    'CT':  'PETCT_nii',
    'PET': 'PETCT_nii'
}
# ===========================================

def check_file_exists(folder_path, keyword):
    """模拟 dataloader 的查找逻辑: 包含关键字且以 .nii.gz 结尾"""
    if not os.path.exists(folder_path):
        return False
    
    files = os.listdir(folder_path)
    for f in files:
        if keyword in f and f.endswith('.nii.gz'):
            return True
    return False

def main():
    print(f"正在读取 CSV: {CSV_PATH} ...")
    
    missing_records = []
    total_patients = 0
    
    # 读取 CSV
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['id'] # 确保这里列名是 id
            total_patients += 1
            
            patient_missing_mods = []
            
            # 检查 5 个模态
            for mod_name, sub_folder in MODALITIES.items():
                # 拼接完整路径: /root/sub_folder/patient_id
                target_dir = os.path.join(DATA_ROOT, sub_folder, patient_id)
                
                # 检查该路径下是否有对应的 nii 文件
                if not check_file_exists(target_dir, mod_name):
                    patient_missing_mods.append(mod_name)
            
            # 如果有缺失，记录下来
            if patient_missing_mods:
                print(f"病人 {patient_id} 缺失: {patient_missing_mods}")
                missing_records.append(f"ID: {patient_id} | 缺失: {', '.join(patient_missing_mods)}")

    # 保存报告
    print("-" * 30)
    print(f"检查完成！共扫描 {total_patients} 个病人。")
    print(f"发现 {len(missing_records)} 个病人存在缺失。")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Total Patients Checked: {total_patients}\n")
        f.write(f"Patients with Missing Modalities: {len(missing_records)}\n")
        f.write("-" * 30 + "\n")
        for line in missing_records:
            f.write(line + "\n")
            
    print(f"详细名单已保存至当前目录下的: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()