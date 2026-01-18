import pandas as pd
import os
import glob

base_dir = "/data/qh_20T_share_file/lct/CT67"

datasets = {
    "LGG": "results_LGG.csv",
    "UPENN": "results_UPENN.csv",
    "GIST": "results_GIST.csv",
    "Desmoid": "results_Desmoid.csv",
    "Prostate": "results_Prostate.csv"
}

all_dfs = []

for ds_name, filename in datasets.items():
    path = os.path.join(base_dir, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.insert(0, "Dataset", ds_name)
        all_dfs.append(df)
    else:
        print(f"Warning: {filename} not found.")

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined
    final_path = os.path.join(base_dir, "final_validation_summary.csv")
    final_df.to_csv(final_path, index=False)
    
    print("Combined Summary:")
    # Display columns nicely
    cols_to_show = [c for c in final_df.columns if "Selected" not in c]
    print(final_df[cols_to_show].to_string(index=False))
else:
    print("No results found.")
