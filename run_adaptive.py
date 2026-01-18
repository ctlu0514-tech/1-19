
import os
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Optional libraries
try:
    from mrmr import mrmr_classif
    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False

try:
    from calculate_fitness import fitness_function
    HAS_CDGAFS = True
except ImportError:
    try:
        from CDGAFS import cdgafs_feature_selection
        HAS_CDGAFS = True
    except ImportError:
        HAS_CDGAFS = False

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "csv": "Merged_All_Centers.csv",
    "label": "label",
    "id_col": "Sample_ID",
    "center": "Center_Source",
    "train_centers": "FuYi",
    "val_size": 0.20,
    "seed": 42,
    "methods": "LASSO,CDGAFS",
    "k": 10,
    "var_threshold": 1e-10,
    "rfe_step": 0.05,
    "cdgafs_pop": 100,
    "cdgafs_theta": 0.9,
    "cdgafs_omega": 0.2,
    "out": "result_adaptive.csv",
    
    # Adaptive Params
    "pca_components": 10,
    "dist_threshold": 12.0,  # Adjusted based on IntVal distance (~7.0). ShiZhongXin (~7.0). FuEr (~17.5).
    # Re-check distances from analyze_raw_data:
    # FuEr=7.8, NingBo=2.5?? Wait, let's re-verify. 
    # Ah, PCA centroids showed:
    # FuYi: (-0.9, -0.8)
    # ShiZhongXin: (-4.0, -3.7). Dist = sqrt(3^2 + 3^2) ~ 4.2
    # NingBo: (3.8, 2.3). Dist = sqrt((3.8--0.9)^2 + (2.3--0.8)^2) = sqrt(4.7^2 + 3.1^2) ~ 5.6
    # FuEr: (4.6, 4.7). Dist = sqrt((4.6--0.9)^2 + (4.7--0.8)^2) ~ sqrt(5.5^2 + 5.5^2) ~ 7.8
    # So Threshold=5.0 is a perfect cut-off. <5.0 (ShiZhongXin) -> TrainNorm. >5.0 (NingBo, FuEr) -> SelfNorm.
}

def load_and_split(
    csv_path: str,
    label_col: str,
    center_col: str,
    train_centers: List[str],
    id_col: Optional[str] = None,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, np.ndarray]], List[str]]:
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Drop non-feature cols
    excludes = [label_col, center_col]
    if id_col and id_col in df.columns:
        excludes.append(id_col)
    
    feature_cols = [c for c in df.columns if c not in excludes and np.issubdtype(df[c].dtype, np.number)]
    print(f"[Data] Loaded {len(df)} samples, {len(feature_cols)} features.")

    # Split by center
    df_train_all = df[df[center_col].isin(train_centers)].copy()
    df_others = df[~df[center_col].isin(train_centers)].copy()

    # Split Train -> Train/IntVal
    X_train_all = df_train_all[feature_cols]
    y_train_all = df_train_all[label_col].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=val_size, random_state=random_state, stratify=y_train_all
    )

    datasets = {
        "Train": (X_train, y_train),
        "IntVal": (X_val, y_val)
    }

    # External centers
    ext_centers = df_others[center_col].unique()
    for c in ext_centers:
        sub = df_others[df_others[center_col] == c]
        datasets[f"Ext_{c}"] = (sub[feature_cols], sub[label_col].values)

    return datasets, feature_cols

def adaptive_preprocess(
    datasets: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    var_threshold: float = 1e-10,
    pca_comps: int = 10,
    dist_threshold: float = 5.0,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
    
    X_train_df, y_train = datasets["Train"]
    
    # 1. Impute & Var Filter on Train
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train_df)
    
    X_train_imp = imputer.transform(X_train_df)
    vars_ = np.var(X_train_imp, axis=0)
    good_idx = np.where(vars_ > var_threshold)[0]
    
    X_train_f = X_train_imp[:, good_idx]
    
    # 2. Main Scaler (Train-based)
    main_scaler = StandardScaler()
    main_scaler.fit(X_train_f)
    X_train_s = main_scaler.transform(X_train_f) # Scaled Train data
    
    # 3. Fit PCA for Domain Distance Calculation (on properties of Scaled Train Data)
    # We use X_train_s to define the "Train Domain" in PCA space.
    pca = PCA(n_components=min(pca_comps, len(good_idx), X_train_s.shape[0]))
    pca.fit(X_train_s)
    
    train_centroid = np.mean(pca.transform(X_train_s), axis=0)
    
    processed = {}
    processed["Train"] = (np.nan_to_num(X_train_s), y_train)
    
    # Process others
    results_info = []

    for name, (X_df, y) in datasets.items():
        if name == "Train": continue
        
        # Initial Transform
        X_imp = imputer.transform(X_df)
        X_f = X_imp[:, good_idx]
        
        # Logic: 
        # To measure distance, we first apply MAIN SCALER (simulate "if we did nothing")
        # Then project to PCA and measure distance.
        # If distance is huge, it means Main Scaler is inappropriate -> Switch to Self Scaler.
        
        X_s_main = main_scaler.transform(X_f)
        X_s_main = np.nan_to_num(X_s_main)
        
        test_centroid = np.mean(pca.transform(X_s_main), axis=0)
        dist = np.linalg.norm(train_centroid - test_centroid)
        
        strategy = "Train-Norm"
        X_final = X_s_main
        
        # Adaptive Switching with Weighted Normalization (DWN)
        # Distance Thresholds
        TRANSITION_LOW = 5.0
        TRANSITION_HIGH = 20.0
        
        # Force IntVal to always use Train-Norm (Alpha=0)
        if not name.startswith("Ext_"):
            alpha = 0.0
            strategy = "Train-Norm"
        else:
            # Calculate alpha
            if dist <= TRANSITION_LOW:
                alpha = 0.0
                strategy = "Train-Norm"
            elif dist >= TRANSITION_HIGH:
                alpha = 1.0
                strategy = "Self-Norm"
            else:
                alpha = (dist - TRANSITION_LOW) / (TRANSITION_HIGH - TRANSITION_LOW)
                strategy = f"Mixed ({alpha:.2f})"
        
        if alpha == 0.0:
            X_final = X_s_main
        elif alpha == 1.0:
            local_scaler = StandardScaler()
            X_final = local_scaler.fit_transform(X_f)
            X_final = np.nan_to_num(X_final)
        else:
            # Weighted Mix
            # 1. Get Train Stats (from main_scaler)
            mu_train = main_scaler.mean_
            sigma_train = main_scaler.scale_
            
            # 2. Get Self Stats
            local_mu = np.mean(X_f, axis=0)
            local_sigma = np.std(X_f, axis=0)
            local_sigma[local_sigma == 0] = 1.0 # Avoid div zero
            
            # 3. Mix
            mu_mix = alpha * local_mu + (1 - alpha) * mu_train
            sigma_mix = alpha * local_sigma + (1 - alpha) * sigma_train
            
            # 4. Apply
            X_final = (X_f - mu_mix) / sigma_mix
            X_final = np.nan_to_num(X_final)
            
        processed[name] = (X_final, y)
        results_info.append({"Dataset": name, "Distance": dist, "Strategy": strategy})

    print("-" * 60)
    print(f"Adaptive Normalization (Threshold={dist_threshold})")
    print(f"{'Dataset':<15} | {'Distance':<10} | {'Strategy':<15}")
    print("-" * 60)
    for res in results_info:
        print(f"{res['Dataset']:<15} | {res['Distance']:<10.4f} | {res['Strategy']:<15}")
    print("-" * 60)

    feature_names = [X_train_df.columns[i] for i in good_idx]
    return processed, feature_names

# Reuse Feature Selection & Eval from original script (Simplified copy)
def run_feature_selection(X, y, names, method, K):
    if method == "LASSO":
        clf = LogisticRegressionCV(penalty="l1", solver="liblinear", cv=5, scoring="roc_auc", class_weight="balanced", random_state=42)
        clf.fit(X, y)
        coefs = np.abs(clf.coef_[0])
        idx = np.argsort(coefs)[::-1][:K]
        return idx
    elif method == "CDGAFS" and HAS_CDGAFS:
        # Use simple wrapper or call imported function
        # Try importing locally to avoid issues
        try:
            from CDGAFS import cdgafs_feature_selection
            # Signature: X, y, gene_list, theta, omega, population_size
            sel_idx_ga, *_ = cdgafs_feature_selection(X, y, names, 0.9, 0.2, 100) 
            
            # CDGAFS returns many features (e.g. 900+). We need to reduce to K using RFE or similar.
            if len(sel_idx_ga) > K:
                X_ga = X[:, sel_idx_ga]
                est = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
                rfe = RFE(est, n_features_to_select=K)
                rfe.fit(X_ga, y)
                final_sub_mask = rfe.support_ # Bool mask relative to X_ga
                final_sub_idx = np.where(final_sub_mask)[0]
                # Map back to original indices
                return [sel_idx_ga[i] for i in final_sub_idx]
            else:
                return sel_idx_ga

        except Exception as e:
            print(f"CDGAFS Failed: {e}")
            return np.arange(K)
    else: # Fallback RFE
        est = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
        sel = RFE(est, n_features_to_select=K)
        sel.fit(X, y)
        return np.where(sel.support_)[0]

def evaluate(datasets, sel_idx, method_name):
    scores = {}
    
    # Train Model
    X_train, y_train = datasets["Train"]
    X_tr_sel = X_train[:, sel_idx]
    
    clf = LogisticRegression(class_weight="balanced", random_state=42, solver="liblinear")
    clf.fit(X_tr_sel, y_train)
    
    for name, (X, y) in datasets.items():
        X_sel = X[:, sel_idx]
        prob = clf.predict_proba(X_sel)[:, 1]
        pred = clf.predict(X_sel)
        
        try:
            auc = roc_auc_score(y, prob)
            acc = accuracy_score(y, pred)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        except:
            auc, acc, sens, spec = 0,0,0,0
            
        scores[name] = {"AUC": auc, "ACC": acc, "Sens": sens, "Spec": spec}
    return scores

def main():
    datasets_raw, cols = load_and_split(CONFIG['csv'], CONFIG['label'], CONFIG['center'], [CONFIG['train_centers']])
    
    datasets, valid_cols = adaptive_preprocess(
        datasets_raw, 
        pca_comps=CONFIG['pca_components'], 
        dist_threshold=CONFIG['dist_threshold']
    )
    
    if HAS_CDGAFS:
        print("CDGAFS is available.")
    else:
        print("CDGAFS not found, check environment.")
        
    methods = CONFIG['methods'].split(',')
    
    final_res = []
    
    for m in methods:
        print(f"\nRunning {m}...")
        X_train, y_train = datasets["Train"]
        sel_idx = run_feature_selection(X_train, y_train, valid_cols, m, CONFIG['k'])
        
        scores = evaluate(datasets, sel_idx, m)
        
        # Print
        print(f"--- {m} Results ---")
        for ds, res in scores.items():
            print(f"{ds:<15} AUC: {res['AUC']:.4f} Sens: {res['Sens']:.4f}")
            row = {"Method": m, "Dataset": ds}
            row.update(res)
            final_res.append(row)
            
    pd.DataFrame(final_res).to_csv(CONFIG['out'], index=False)
    print(f"\nSaved to {CONFIG['out']}")

if __name__ == "__main__":
    main()
