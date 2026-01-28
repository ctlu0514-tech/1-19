import pandas as pd
import os

input_file = '/data/qh_20T_share_file/lct/CT67/localdata/prostate_features_with_label.csv'
output_file = '/data/qh_20T_share_file/lct/CT67/localdata/prostate_features_no_pet_with_label.csv'

if not os.path.exists(input_file):
    print(f"Error: File {input_file} not found.")
    exit(1)

print(f"Reading {input_file}...")
df = pd.read_csv(input_file)
print(f"Original shape: {df.shape}")

# Identify PET columns
# Based on previous context, PET features likely start with "PET" or contain "PET_"
pet_cols = [col for col in df.columns if col.startswith('PET')]

print(f"Found {len(pet_cols)} PET feature columns to remove.")

# Drop PET columns
df_dropped = df.drop(columns=pet_cols)
print(f"New shape: {df_dropped.shape}")

# Save to new file
print(f"Saving to {output_file}...")
df_dropped.to_csv(output_file, index=False)
print("Done.")
