import pandas as pd
import os

def merge_datasets_auto_id():
    # 1. 定义文件路径
    base_dir = '/data/qh_20T_share_file/lct/CT67/' 
    
    file_list = [
        {'path': os.path.join(base_dir, '附一.csv'),  'name': 'FuYi'},
        {'path': os.path.join(base_dir, '附二.csv'),  'name': 'FuEr'},
        {'path': os.path.join(base_dir, '市中心.csv'), 'name': 'ShiZhongXin'},
        {'path': os.path.join(base_dir, '宁波.csv'), 'name': 'NingBo'}
    ]

    dfs = []
    
    print(f"{'='*40}")
    print("开始合并多中心数据 (自动生成 ID)")
    print(f"{'='*40}")

    for entry in file_list:
        file_path = entry['path']
        center_prefix = entry['name'] # 用于 ID 前缀
        
        if not os.path.exists(file_path):
            print(f"!!! 警告: 文件不存在，跳过: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # --- 步骤 1: 添加中心来源列 (放在最后一列) ---
            df['Center_Source'] = center_prefix
            
            # --- 步骤 2: 自动生成唯一 ID (放在第一列) ---
            # 格式: FuYi_0, FuYi_1, FuYi_2 ...
            new_ids = [f"{center_prefix}_{i}" for i in range(len(df))]
            
            # insert(位置, 列名, 数据) -> 0 表示插入到最前面
            df.insert(0, 'Sample_ID', new_ids)
            
            dfs.append(df)
            print(f"已加载 [{center_prefix}]: {df.shape[0]} 例")
            
        except Exception as e:
            print(f"读取错误 {file_path}: {e}")

    # 3. 合并
    if not dfs:
        print("没有数据被合并。")
        return

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # 4. 检查列对齐情况
    # 如果某个文件少了一列特征，Pandas会填NaN。这里检查一下。
    if merged_df.isna().any().any():
        print(f"\n!!! 警告: 合并后发现 NaN 值。")
        print("可能是三个文件的特征列数量或名称不一致。")
        # 简单策略：删除含有 NaN 的列 (即删除不是三者共有的特征)
        # merged_df = merged_df.dropna(axis=1)
        # print("已自动剔除不匹配的特征列。")
    
    # 5. 保存
    output_path = os.path.join(base_dir, 'Merged_All_Centers.csv')
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*40}")
    print(f"合并成功!")
    print(f"新文件: {output_path}")
    print(f"总形状: {merged_df.shape} (行数, 列数)")
    print(f"包含新列: 'Sample_ID' 和 'Center_Source'")
    print(f"{'='*40}")

if __name__ == "__main__":
    merge_datasets_auto_id()