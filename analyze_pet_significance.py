
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load data
csv_path = '/data/qh_20T_share_file/lct/CT67/localdata/prostate_features_with_label.csv'
df = pd.read_csv(csv_path)

# Infer label column
label_col = 'label'
if label_col not in df.columns:
    for c in df.columns:
        if df[c].nunique() == 2 and 'label' in c.lower():
            label_col = c
            break

# 1. Filter only subjects with PET data
pet_cols = [c for c in df.columns if 'PET' in c]
# We assume if the first PET col is available, the rest are too
df_pet = df[df[pet_cols[0]].notna()].copy()
df_pet = df_pet.reset_index(drop=True)

print(f"Subjects with PET: {len(df_pet)}")
n_pos = df_pet[label_col].sum()
n_neg = len(df_pet) - n_pos
print(f"  Positive (Label=1): {n_pos}")
print(f"  Negative (Label=0): {n_neg}")

# 2. Analyze significant features
# We can't analyze all 2000 features manually, so let's find the most discriminative ones via T-test
print("\nCalculating T-tests for all PET features...")

results = []
for col in pet_cols:
    # Skip non-numeric
    if not np.issubdtype(df_pet[col].dtype, np.number):
        continue
        
    pos_vals = df_pet[df_pet[label_col] == 1][col]
    neg_vals = df_pet[df_pet[label_col] == 0][col]
    
    # Simple t-test (ignoring variance equality for speed, roughly indicative)
    if len(pos_vals) > 5 and len(neg_vals) > 5:
        # Check for constant
        if pos_vals.var() == 0 and neg_vals.var() == 0:
            continue
            
        t_stat, p_val = ttest_ind(pos_vals, neg_vals, nan_policy='omit')
        
        # Calculate AUC for single feature
        # If t > 0, higher value => class 1. AUC = U / (n1*n0)
        # Using a simpler proxy: Cohen's d or just difference in means normalized
        diff = pos_vals.mean() - neg_vals.mean()
        
        results.append({
            'feature': col,
            'p_value': p_val,
            't_stat': abs(t_stat),
            'mean_pos': pos_vals.mean(),
            'mean_neg': neg_vals.mean(),
            'diff': diff
        })

# Sort by significance (p-value / t-stat)
res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by='p_value', ascending=True)

print(f"\nTop 10 Most Discriminative PET Features (by P-value):")
print(res_df[['feature', 'p_value', 'mean_pos', 'mean_neg']].head(10).to_string(index=False))

# 3. Analyze Feature Types in Top 50 significant ones
top_50 = res_df.head(50)
print("\nFeature Type Distribution in Top 50 Significant PET features:")
# Extract feature types (e.g. log-sigma, wavelet, glcm)
# Assuming name format like "PET_wavelet-LLH_glcm_ClusterShade"
def get_types(name):
    parts = name.split('_')
    # Filter types usually in position 1 or 2
    types = []
    for p in parts:
        if 'wavelet' in p: types.append('wavelet')
        elif 'log' in p: types.append('log')
        elif 'glcm' in p: types.append('glcm')
        elif 'glrlm' in p: types.append('glrlm')
        elif 'glszm' in p: types.append('glszm')
        elif 'firstorder' in p: types.append('firstorder')
        elif 'shape' in p: types.append('shape')
    return ",".join(types)

top_50['types'] = top_50['feature'].apply(get_types)
print(top_50['types'].value_counts().head(10))

# 4. Check previously selected PET features (from user log)
# User log had e.g.:
# 'PET_log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity'
# 'PET_wavelet-HHH_gldm_SmallDependenceEmphasis'
# ...
print("\nChecking significance of GA-selected PET features from previous log:")
selected_examples = [
    'PET_log-sigma-2-0-mm-3D_glszm_GrayLevelNonUniformity',
    'PET_wavelet-HHH_gldm_SmallDependenceEmphasis',
    'PET_wavelet-HHL_firstorder_Mean',
    'PET_wavelet-HLH_firstorder_Entropy',
    'PET_wavelet-HLL_firstorder_Skewness',
    'PET_wavelet-LHH_glszm_SmallAreaLowGrayLevelEmphasis',
    'PET_wavelet-LHL_glszm_GrayLevelVariance',
    'PET_wavelet-LLH_glcm_Id'
]

subset = res_df[res_df['feature'].isin(selected_examples)]
print(subset[['feature', 'p_value', 'mean_pos', 'mean_neg']].to_string(index=False))
