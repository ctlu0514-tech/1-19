
import pandas as pd
import numpy as np
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

# Identify modalities
# Assuming feature names start with Modality_...
# e.g. CT_..., T2_..., DWI_..., ADC_...
all_cols = [c for c in df.columns if c not in [label_col, 'ID', 'PatientID', 'Center']]
feature_cols = [c for c in all_cols if np.issubdtype(df[c].dtype, np.number)]

modalities = set()
for c in feature_cols:
    parts = c.split('_')
    if len(parts) > 0:
        modalities.add(parts[0])

print(f"Detected modalities: {sorted(list(modalities))}")

# Analyze each modality
for mod in sorted(list(modalities)):
    if mod == 'PET': continue # Already analyzed

    print(f"\n{'='*40}")
    print(f"Analyzing Modality: {mod}")
    print(f"{'='*40}")

    mod_cols = [c for c in feature_cols if c.startswith(mod + '_')]
    if not mod_cols:
        continue
    
    # Use full dataset (check missingness first)
    # Check if this modality is sparse like PET
    missing_count = df[mod_cols[0]].isna().sum()
    total = len(df)
    print(f"Total Features: {len(mod_cols)}")
    print(f"Missing Samples: {missing_count} / {total} ({missing_count/total*100:.1f}%)")
    
    # Filter valid samples
    df_mod = df[df[mod_cols[0]].notna()].copy()
    
    # T-tests
    results = []
    for col in mod_cols:
        pos_vals = df_mod[df_mod[label_col] == 1][col]
        neg_vals = df_mod[df_mod[label_col] == 0][col]
        
        # Simple stats
        if len(pos_vals) < 5 or len(neg_vals) < 5: continue
        if pos_vals.var() == 0 and neg_vals.var() == 0: continue
            
        t_stat, p_val = ttest_ind(pos_vals, neg_vals, nan_policy='omit')
        
        results.append({
            'feature': col,
            'p_value': p_val,
            'mean_pos': pos_vals.mean(),
            'mean_neg': neg_vals.mean()
        })

    if not results:
        print("No valid features to analyze.")
        continue

    res_df = pd.DataFrame(results).sort_values(by='p_value')
    
    # Show Top 10
    print(f"Top 10 Most Significant Features ({mod}):")
    print(res_df.head(10).to_string(index=False))
    
    # Feature Type Analysis
    def get_type_broad(name):
        if 'wavelet' in name: return 'Wavelet'
        if 'log' in name: return 'LoG'
        if 'original' in name: return 'Original'
        if 'exponential' in name: return 'Exp'
        if 'gradient' in name: return 'Gradient'
        if 'square' in name: return 'Square'
        if 'lbp' in name: return 'LBP'
        return 'Other'

    top_50 = res_df.head(50)
    top_50['ftype'] = top_50['feature'].apply(get_type_broad)
    print(f"\nFeature Type Distribution (Top 50 - {mod}):")
    print(top_50['ftype'].value_counts())
    
    # Compare P-value range with PET
    min_p = res_df['p_value'].min()
    print(f"\nBest P-value for {mod}: {min_p:.2e}")
