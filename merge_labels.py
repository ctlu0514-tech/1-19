import pandas as pd
import os
import sys

# --- 1. 配置区域 ---
FEATURES_FILE_PATH = r'/data/qh_20T_share_file/lct/CT67/dl_work/new_code/features_output/all_features.csv'
LABEL_FILE_PATH = r'/data/qh_20T_share_file/lct/CT67/qianliexian_clinical_isup.csv'
FEATURES_ID_COLUMN_NAME = 'patient_id'
LABEL_ID_COLUMN_NAME = 'id'
LABEL_COLUMN_NAME = 'isup2'
OUTPUT_FILE_PATH = r'/data/qh_20T_share_file/lct/CT67/dl_work/new_code/features_output/all_features_with_label.csv'
CENTER_VALUE = None 

# --- 2. 主程序 ---
print("--- 脚本开始 ---")

# --- 加载数据并立即检查 ---
try:
    print(f"正在加载特征文件: {FEATURES_FILE_PATH}")
    # low_memory=False 防止大文件列混合警告
    df_features = pd.read_csv(FEATURES_FILE_PATH, low_memory=False) 
    
    # 【侦探点 1】: 刚读进来是多少列？
    print(f"✅ [检查点1] 特征文件原始大小: {df_features.shape} (行, 列)")
    original_feature_cols = set(df_features.columns) # 记录原始所有列名
    
except FileNotFoundError:
    print(f"!!! 致命错误: 找不到特征文件。")
    sys.exit(1)

try:
    print(f"正在加载标签文件: {LABEL_FILE_PATH}")
    df_labels = pd.read_csv(LABEL_FILE_PATH)
except FileNotFoundError:
    print(f"!!! 致命错误: 找不到标签文件。")
    sys.exit(1)

# --- 准备工作 ---
if FEATURES_ID_COLUMN_NAME not in df_features.columns:
    print(f"!!! 致命错误: 特征文件中找不到ID列 '{FEATURES_ID_COLUMN_NAME}'")
    sys.exit(1)
    
# 统一ID类型
df_features[FEATURES_ID_COLUMN_NAME] = df_features[FEATURES_ID_COLUMN_NAME].astype(str)

# 准备标签数据
if LABEL_ID_COLUMN_NAME not in df_labels.columns or LABEL_COLUMN_NAME not in df_labels.columns:
    print(f"!!! 错误: 标签文件列名不对。")
    sys.exit(1)

df_labels_subset = df_labels[[LABEL_ID_COLUMN_NAME, LABEL_COLUMN_NAME]].copy()
df_labels_subset[LABEL_ID_COLUMN_NAME] = df_labels_subset[LABEL_ID_COLUMN_NAME].astype(str)

# --- 合并 ---
print("\n--- 正在合并 ---")
df_merged = pd.merge(
    df_features,
    df_labels_subset,
    left_on=FEATURES_ID_COLUMN_NAME,
    right_on=LABEL_ID_COLUMN_NAME,
    how='left'
)
print(f"✅ [检查点2] 合并后大小: {df_merged.shape}")

# --- 清理与重排 ---

# 1. 检查未匹配ID
missing_count = df_merged[LABEL_COLUMN_NAME].isna().sum()
if missing_count > 0:
    print(f"!!! 警告: {missing_count} 行未匹配到标签。")

# 2. 删除多余ID列
# 注意：这里我们记录一下，我们要删的是 LABEL_ID_COLUMN_NAME
# 如果 LABEL_ID_COLUMN_NAME 和 FEATURES_ID_COLUMN_NAME 名字一样，这里处理会有所不同
# 但通常它是 'id' 和 'PatientID'，所以没问题
if LABEL_ID_COLUMN_NAME in df_merged.columns and LABEL_ID_COLUMN_NAME != FEATURES_ID_COLUMN_NAME:
    df_merged = df_merged.drop([LABEL_ID_COLUMN_NAME], axis=1)

# 3. 处理 Center
center_col_name = 'Center'
use_center = False
if 'CENTER_VALUE' in locals() and CENTER_VALUE is not None:
    use_center = True
    df_merged[center_col_name] = CENTER_VALUE

# 4. 重排顺序
all_cols = df_merged.columns.tolist()
# 排除掉 ID, Label, Center，剩下的全当特征
feature_cols = [c for c in all_cols if c not in [FEATURES_ID_COLUMN_NAME, LABEL_COLUMN_NAME, center_col_name]]

final_order = []
final_order.append(FEATURES_ID_COLUMN_NAME)
if use_center:
    final_order.append(center_col_name)
final_order.append(LABEL_COLUMN_NAME)
final_order.extend(feature_cols)

df_final = df_merged[final_order]

print(f"✅ [检查点3] 最终处理后大小: {df_final.shape}")

# =======================================================
# 【核心修改】: 消失的特征去哪了？对比分析
# =======================================================
print("\n" + "="*40)
print("🔍 丢失列侦探报告")
print("="*40)

final_cols_set = set(df_final.columns)
# 计算差集：原始有但现在没有的列
dropped_cols = original_feature_cols - final_cols_set

# 注意：还要排除掉标签文件那个多余的ID列，那个是我们故意删的
if LABEL_ID_COLUMN_NAME in dropped_cols:
    dropped_cols.remove(LABEL_ID_COLUMN_NAME)

num_dropped = len(dropped_cols)

if num_dropped == 0:
    print("✨ 完美！没有丢失任何原始特征列。")
else:
    print(f"⚠️ 警告：总共有 {num_dropped} 个原始特征列在处理中消失了！")
    print("可能是读取时被解析错误，或者列名包含特殊字符。")
    print("\n👇 丢失列的示例 (前20个):")
    
    # 转换成列表并排序，方便查看
    dropped_list = sorted(list(dropped_cols))
    for col in dropped_list[:20]:
        print(f"   - {col}")
        
    if num_dropped > 20:
        print(f"   ... (以及其他 {num_dropped - 20} 个)")

# =======================================================

# 保存
df_final.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\n文件已保存: {OUTPUT_FILE_PATH}")



# import pandas as pd
# import os
# import sys

# # ================= 1. 文件路径配置 =================
# # 宁波数据
# FILE_NINGBO = r'/data/qh_20T_share_file/lct/CT67/ningbo_ovarian_with_label.csv'

# # 附一数据
# FILE_FUYI = r'/data/qh_20T_share_file/lct/CT67/fuyi_ovarian_with_label.csv'

# # 输出合并后的文件
# FILE_OUTPUT = r'/data/qh_20T_share_file/lct/CT67/ovarian_All_Centers_with_label.csv'

# # ================= 2. 执行逻辑 =================
# print("--- 正在加载两个数据集 ---")

# if not os.path.exists(FILE_NINGBO) or not os.path.exists(FILE_FUYI):
#     print("!!! 错误: 找不到文件，请检查路径。")
#     sys.exit(1)

# # 读取数据
# df_ningbo = pd.read_csv(FILE_NINGBO)
# df_fuyi = pd.read_csv(FILE_FUYI)

# print(f"原始宁波数据: {df_ningbo.shape}")
# print(f"原始附一数据: {df_fuyi.shape}")

# # ==========================================
# # 核心修改区域：标签清洗与映射
# # ==========================================

# # 1. [附一] 统一列名 (label -> type)
# if 'label' in df_fuyi.columns and 'type' not in df_fuyi.columns:
#     print("[处理] 附一数据：将 'label' 重命名为 'type'")
#     df_fuyi.rename(columns={'label': 'type'}, inplace=True)

# # 2. 检查两个表是否都有 type 列
# if 'type' not in df_ningbo.columns or 'type' not in df_fuyi.columns:
#     print("!!! 致命错误: 缺少 'type' (或 label) 标签列，无法继续。")
#     sys.exit(1)

# # 3. [通用] 去除没有标签的样本 (Drop NaN)
# print("[处理] 正在去除无标签的样本...")
# before_n = len(df_ningbo)
# df_ningbo.dropna(subset=['type'], inplace=True)
# print(f"   - 宁波: {before_n} -> {len(df_ningbo)} (剔除 {before_n - len(df_ningbo)})")

# before_f = len(df_fuyi)
# df_fuyi.dropna(subset=['type'], inplace=True)
# print(f"   - 附一: {before_f} -> {len(df_fuyi)} (剔除 {before_f - len(df_fuyi)})")

# # 4. [通用] 强制转换为字符串 (Character)
# #    防止 pandas 自动识别为数值，满足"type应该是字符"的需求
# #    注意：先转为int再转str是为了防止出现 "1.0" 这样的字符串
# try:
#     df_ningbo['type'] = df_ningbo['type'].astype(float).astype(int).astype(str)
# except:
#     df_ningbo['type'] = df_ningbo['type'].astype(str)

# try:
#     df_fuyi['type'] = df_fuyi['type'].astype(float).astype(int).astype(str)
# except:
#     df_fuyi['type'] = df_fuyi['type'].astype(str)

# print(f"[检查] 宁波 type 列类型: {df_ningbo['type'].dtype}")
# print(f"[检查] 附一 type 列类型: {df_fuyi['type'].dtype}")

# # 5. [宁波特有] 标签数值修改 (0->1, 1->2)
# #    因为上面已经强制转为字符串了，这里替换字符串 '0' 和 '1'
# print("[处理] 正在修改宁波标签: 0->1, 1->2")
# mapping = {'0': '1', '1': '2'}
# # 检查一下替换前的值分布
# print(f"   - 替换前分布: {df_ningbo['type'].value_counts().to_dict()}")

# # 执行替换
# df_ningbo['type'] = df_ningbo['type'].replace(mapping)

# # 检查替换后的值分布
# print(f"   - 替换后分布: {df_ningbo['type'].value_counts().to_dict()}")

# # ==========================================
# # 后续合并逻辑 (保持原有对齐逻辑)
# # ==========================================

# # --- 检查列差异 ---
# cols_ningbo = set(df_ningbo.columns)
# cols_fuyi = set(df_fuyi.columns)
# common_cols = cols_ningbo.intersection(cols_fuyi)

# print(f"\n--- 列名匹配检查 ---")
# print(f"1. 公共特征列数: {len(common_cols)} (这些将被合并)")

# if 'Center' not in common_cols:
#     print("!!! 警告: 两个表没有共同的 'Center' 列，建议合并后检查来源。")

# # --- 合并 ---
# print("\n--- 正在合并 ---")
# # join='inner' 只保留公共列
# df_merged = pd.concat([df_ningbo, df_fuyi], axis=0, join='inner', ignore_index=True)

# # 调整列顺序
# first_cols = ['ID', 'Center', 'type']
# first_cols = [c for c in first_cols if c in df_merged.columns]
# other_cols = [c for c in df_merged.columns if c not in first_cols]
# df_final = df_merged[first_cols + other_cols]

# # 最终保存
# df_final.to_csv(FILE_OUTPUT, index=False)

# print(f"{'='*40}")
# print(f"合并完成！")
# print(f"最终文件: {FILE_OUTPUT}")
# print(f"最终形状: {df_final.shape}")
# if 'Center' in df_final.columns:
#     print(f"包含中心及样本量: \n{df_final['Center'].value_counts()}")
# print(f"标签分布 (type): \n{df_final['type'].value_counts()}")
# print(f"{'='*40}")