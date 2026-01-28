
import pandas as pd
import numpy as np

csv_path = '/data/qh_20T_share_file/lct/CT67/localdata/prostate_features_with_label.csv'
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
label_col = 'label'
if label_col not in df.columns:
    print(f"Warning: {label_col} not found. Available columns: {df.columns.tolist()[:10]}")
    # Try to find a label column
    for c in df.columns:
        if df[c].nunique() == 2 and 'label' in c.lower():
            label_col = c
            print(f"Found candidate label column: {label_col}")
            break

pet_cols = [c for c in df.columns if 'PET' in c]
print(f"Number of PET columns: {len(pet_cols)}")

if len(pet_cols) > 0:
    # Use the first PET column to check missingness pattern (assuming consistent missingness across PET cols for a subject)
    sample_pet = pet_cols[0]
    df['has_pet'] = df[sample_pet].notna()
    
    n_with = df['has_pet'].sum()
    n_without = len(df) - n_with
    print(f"Subjects with PET: {n_with}")
    print(f"Subjects without PET: {n_without}")
    
    print("\nCrosstab (Has PET vs Label):")
    ct = pd.crosstab(df['has_pet'], df[label_col])
    print(ct)
    
    # Calculate percentages
    ct_norm = pd.crosstab(df['has_pet'], df[label_col], normalize='index')
    print("\nProportions (Row-wise normalize):")
    print(ct_norm)
    
else:
    print("No PET columns found.")
