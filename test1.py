import pandas as pd
import os

# 1. 定义文件路径
input_file = '/data/qh_20T_share_file/lct/CT67/localdata/ovarian_two_centers_with_label.csv'
output_file = '/data/qh_20T_share_file/lct/CT67/localdata/ovarian_with_label.csv'

try:
    print(f"正在读取文件: {input_file} ...")
    df = pd.read_csv(input_file)
    
    # 打印原始数据形状
    print(f"原始数据大小: {df.shape[0]} 行, {df.shape[1]} 列")

    # 检查 'Center' 列是否存在
    if 'Center' in df.columns:
        
        # 2. 删除 Center 列中值为 'Ningbo' 的样本 (保留不是 Ningbo 的行)
        # 注意：这里假设 'Ningbo' 拼写和大小写完全匹配
        df_filtered = df[df['Center'] != 'Ningbo'].copy()
        
        removed_count = len(df) - len(df_filtered)
        print(f"已剔除 Center 为 'Ningbo' 的样本数: {removed_count}")
        
        # 3. 剔除 'Center' 列
        df_final = df_filtered.drop(columns=['Center'])
        print("已移除 'Center' 列")
        
        # 4. 保存文件
        df_final.to_csv(output_file, index=False)
        print("-" * 30)
        print(f"处理完成！")
        print(f"新文件保存至: {output_file}")
        print(f"最终数据大小: {df_final.shape[0]} 行, {df_final.shape[1]} 列")
        
    else:
        print("错误: 文件中未找到名为 'Center' 的列，无法进行过滤和删除操作。")

except FileNotFoundError:
    print(f"错误: 找不到文件 {input_file}")
except Exception as e:
    print(f"发生未知错误: {e}")