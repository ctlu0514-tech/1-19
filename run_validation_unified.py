#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_validation_unified.py

统一验证脚本：同一套代码同时支持
- 单中心随机拆分（Train / IntVal）
- 多中心：指定训练中心（Train / IntVal）+ 其他中心作为外部验证集（Ext_<Center>）

核心原则：Split -> Impute(Fit on Train) -> Variance Filter(Fit on Train) -> Scale(Fit on Train) -> Select(Fit on Train) -> Train(Fit on Train) -> Eval

用法示例：
1) 单中心（自动判断：没有中心列或只有一个中心）：
   直接运行脚本即可；在脚本顶部的 CONFIG 区域里修改 csv/label/mode 等参数

2) 多中心（指定训练中心）：
   python run_validation_unified.py --csv your.csv --label label --center Center_Source --train_centers FuYi --mode multi

3) 自定义方法与K：
   python run_validation_unified.py --csv your.csv --label label --methods CDGAFS,LASSO,RFE,mRMR --k 12
"""

import os
import sys
import time
import warnings
import argparse 
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE

warnings.filterwarnings("ignore")

# --- Optional dependencies ---
try:
    from CDGAFS import cdgafs_feature_selection  # type: ignore
    HAS_CDGAFS = True
except Exception:
    HAS_CDGAFS = False

try:
    from mrmr import mrmr_classif  # type: ignore
    HAS_MRMR = True
except Exception:
    HAS_MRMR = False


# ============================================================
# Utilities
# ============================================================
def binarize_labels(y: np.ndarray) -> np.ndarray:
    """Robustly map a 2-class label array to {0,1}."""
    y = np.asarray(y)
    # Drop NaN-like; caller should have removed NaN rows already, but keep defensive
    if y.dtype.kind in {"f"}:
        if np.isnan(y).any():
            raise ValueError("y contains NaN; please drop NaN labels before binarize_labels().")

    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"Label must have exactly 2 unique classes; got {uniq}.")
    # map smaller class label -> 0, larger -> 1 (consistent with 'min->0' approach)
    lo, hi = uniq[0], uniq[1]
    return np.where(y == lo, 0, 1).astype(int)


def infer_feature_columns(df: pd.DataFrame, label_col: str, center_col: Optional[str], id_col: Optional[str]) -> List[str]:
    """
    推断特征列，排除 Label/ID/Center 以及 diagnostics 诊断列。
    [修改注] 增加了打印被排除列名的逻辑
    """
    # 1. 明确要排除的元数据列 (ID, Label, Center)
    # 注意：只排除那些确实存在于 dataframe 中的列
    meta_cols = [c for c in [label_col, center_col, id_col, "_label_bin_"] if c and c in df.columns]
    
    # 2. 排除 PyRadiomics 自动生成的 diagnostics_ 开头的列 (通常不是影像组学特征)
    diag_cols = [c for c in df.columns if str(c).startswith("diagnostics_")]
    
    # 3. 打印排除信息 (这是你要求的修改)
    if meta_cols:
        print(f"[Data] Excluding Metadata columns (ID/Label/Center): {meta_cols}")
    
    if diag_cols:
        # 如果列太多，只打印前几个
        msg = f"{diag_cols[:5]}..." if len(diag_cols) > 5 else str(diag_cols)
        print(f"[Data] Excluding Diagnostics columns ({len(diag_cols)}): {msg}")

    # 4. 生成最终特征列表
    exclude_set = set(meta_cols + diag_cols)
    feature_cols = [c for c in df.columns if c not in exclude_set]
    
    return feature_cols


# ============================================================
# 1) Load + Split (single or multi)
# ============================================================

def load_and_split(
    csv_path: str,
    label_col: str,
    mode: str = "auto",
    center_col: Optional[str] = None,
    train_centers: Optional[List[str]] = None,
    id_col: Optional[str] = None,
    val_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, np.ndarray]], List[str], str]:
    """
    Returns:
      datasets_raw: dict name -> (X_df, y_array) where X_df is *raw* dataframe (no impute/scale yet)
      feature_cols: list of feature column names (after coercion / dropping unusable cols)
      resolved_mode: 'single' or 'multi'

    Important:
      - Prevent label leakage by ALWAYS excluding label_col from features.
      - Build a global {raw_label -> 0/1} mapping from the FULL dataset so that per-center subsets
        with only one class will still be mapped consistently.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, compression="gzip" if csv_path.endswith(".gz") else "infer")

    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")

    # Drop rows with missing labels; reset index to keep alignment stable
    df = df.dropna(subset=[label_col]).copy().reset_index(drop=True)

    # Auto-detect a center column only when user didn't provide one
    if center_col is None:
        for cand in ["Center_Source", "center", "Center", "site", "Site"]:
            if cand in df.columns:
                center_col = cand
                break

    # Resolve mode
    resolved_mode = mode.lower()
    if resolved_mode not in {"auto", "single", "multi"}:
        raise ValueError("mode must be one of: auto|single|multi")

    if resolved_mode == "auto":
        if center_col is None or center_col not in df.columns:
            resolved_mode = "single"
        else:
            n_centers = df[center_col].nunique(dropna=True)
            resolved_mode = "multi" if n_centers >= 2 else "single"

    # ---- Global binary label mapping from FULL dataset ----
    uniq = np.unique(df[label_col].values)
    if len(uniq) != 2:
        raise ValueError(f"Label must have exactly 2 unique classes in the full dataset; got {uniq}.")
    lo, hi = np.sort(uniq)

    def map_y(vals: np.ndarray) -> np.ndarray:
        vals = np.asarray(vals)
        return np.where(vals == lo, 0, 1).astype(int)

    y_all = map_y(df[label_col].values)

    # ---- Infer feature columns (exclude raw label/id/center) ----
    feature_cols = infer_feature_columns(df, label_col=label_col, center_col=center_col, id_col=id_col)
    if label_col in feature_cols:
        raise RuntimeError(f"Label leakage detected: '{label_col}' is included in feature columns.")

    # Build X and coerce to numeric. Drop columns that become entirely NaN after coercion.
    X_all = df[feature_cols].copy()
    X_all = X_all.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = X_all.columns[X_all.isna().all(axis=0)].tolist()
    if all_nan_cols:
        X_all = X_all.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
        print(f"[Info] Dropped non-numeric/all-NaN feature columns after coercion: {all_nan_cols}")

    if len(feature_cols) == 0:
        raise ValueError("No usable (numeric) feature columns detected after dropping non-feature columns.")

    datasets: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}

    if resolved_mode == "single":
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=val_size, random_state=random_state, stratify=y_all
        )
        datasets["Train"] = (X_train, y_train)
        datasets["IntVal"] = (X_val, y_val)
        print(f"[Split] 模式=单中心(single) | 总数={len(df)} | 训练集={len(X_train)} | 内部验证集={len(X_val)}")
    else:
        # multi-center
        if center_col is None or center_col not in df.columns:
            raise ValueError("mode=multi requires a valid center column in the CSV (center_col).")

        centers = df[center_col].dropna().astype(str).unique().tolist()
        if not centers:
            raise ValueError(f"Center column '{center_col}' exists but has no valid values.")

        # If train_centers not specified, default to the largest center
        if not train_centers:
            vc = df[center_col].astype(str).value_counts()
            train_centers = [str(vc.index[0])]
            print(f"[Info] train_centers not provided. Defaulting to largest center: {train_centers}")

        train_centers = [str(c) for c in train_centers]
        missing = [c for c in train_centers if c not in centers]
        if missing:
            raise ValueError(f"Train centers not found in data: {missing}. Available: {centers}")

        train_idx = df.index[df[center_col].astype(str).isin(train_centers)].to_numpy()
        if len(train_idx) == 0:
            raise ValueError("No samples found for the specified train_centers.")

        X_train_full = X_all.loc[train_idx]
        y_train_full = y_all[train_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
        )
        datasets["Train"] = (X_train, y_train)
        datasets["IntVal"] = (X_val, y_val)

        # External centers
        ext_centers = [c for c in centers if c not in train_centers]
        for c in ext_centers:
            ext_idx = df.index[df[center_col].astype(str) == str(c)].to_numpy()
            datasets[f"Ext_{c}"] = (X_all.loc[ext_idx], y_all[ext_idx])

        print(
            f"[Split] 模式=多中心(multi) | 总数={len(df)} | 训练中心={train_centers} (n={len(train_idx)}) | "
            f"训练集={len(X_train)} | 内部验证集={len(X_val)} | 外部中心={ext_centers}"
        )

    return datasets, feature_cols, resolved_mode

# ============================================================
# 2) Secure preprocessing (fit on Train only)
# ============================================================

def preprocess_securely(
    datasets: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    var_threshold: float = 1e-10,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
    """
    Fit preprocessing on Train ONLY, then transform IntVal and any Ext_* splits.
    Steps:
      - Drop features that are entirely missing in Train (mean-imputer cannot fit them)
      - Impute(mean) on Train
      - Variance filter on Train
      - Standardize on Train
    """
    X_train_df, y_train = datasets["Train"]

    # Defensive: drop columns that are all-NaN in the TRAIN split
    all_missing_cols = X_train_df.columns[X_train_df.isna().all(axis=0)].tolist()
    if all_missing_cols:
        print(f"[Preprocess] Dropping all-missing-in-train columns: {all_missing_cols}")
        for name in list(datasets.keys()):
            X_df, y = datasets[name]
            datasets[name] = (X_df.drop(columns=all_missing_cols), y)
        X_train_df, y_train = datasets["Train"]

    # Ensure numeric dtype (should already be numeric after load_and_split coercion)
    non_numeric = [c for c in X_train_df.columns if not pd.api.types.is_numeric_dtype(X_train_df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns remain after coercion: {non_numeric}. Please drop them.")

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    imputer.fit(X_train_df)
    X_train_imp = imputer.transform(X_train_df)

    # Variance filter fit on Train only
    vars_ = np.var(X_train_imp, axis=0)
    good_idx = np.where(vars_ > var_threshold)[0]
    if len(good_idx) == 0:
        raise ValueError("All features are constant (or below variance threshold) on training set.")

    X_train_f = X_train_imp[:, good_idx]
    scaler.fit(X_train_f)

    processed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (X_df, y) in datasets.items():
        X_imp = imputer.transform(X_df)
        X_f = X_imp[:, good_idx]
        X_s = scaler.transform(X_f)
        X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
        processed[name] = (X_s, np.asarray(y, dtype=int))

    all_cols = X_train_df.columns.tolist()
    final_features = [all_cols[i] for i in good_idx]

    # Hard guard against label leakage (common column names)
    bad = [c for c in final_features if str(c).lower() in {"target", "label", "_label_bin_"}]
    if bad:
        raise RuntimeError(f"Label leakage detected in final feature list: {bad}")

    print(f"[Preprocess] kept_features={len(final_features)} (removed {len(all_cols)-len(final_features)})")
    return processed, final_features

# ============================================================
# 3) Feature selection
# ============================================================
def run_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    method: str,
    K: int,
    rfe_step: float = 0.05,
    cdgafs_pop: int = 100,
    cdgafs_theta: float = 0.9,
    cdgafs_omega: float = 0.5,
    use_semantic: bool = False,
    max_cluster_size: int = 200,
    use_quality_quota: bool = False,
    top_cluster_ratio: float = 0.5,
    temperature: float = 10.0,
) -> np.ndarray:
    method = method.upper()
    start_t = time.time()
    selected_idx: List[int] = []

    if K <= 0:
        raise ValueError("K must be > 0")

    if method == "LASSO":
        # LogisticRegressionCV supports l1 + liblinear for binary
        clf = LogisticRegressionCV(
            cv=5, penalty="l1", solver="liblinear",
            scoring="roc_auc", class_weight="balanced",
            random_state=42, max_iter=3000
        )
        clf.fit(X_train, y_train)
        coefs = np.abs(clf.coef_[0])
        # Prefer non-zeros; if insufficient, fill by magnitude
        nonzero = np.where(coefs > 1e-8)[0]
        if len(nonzero) >= K:
            order = nonzero[np.argsort(coefs[nonzero])[::-1]]
            selected_idx = order[:K].tolist()
        else:
            order = np.argsort(coefs)[::-1]
            selected_idx = order[:K].tolist()

    elif method == "RFE":
        estimator = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
        n_feats = X_train.shape[1]
        # If user passes integer step via argparse, we keep it; else float proportion for big d
        step_val = rfe_step
        if isinstance(step_val, float) and (step_val <= 0 or step_val >= 1):
            # fallback to 1 (one by one)
            step_val = 1
        # If low-dimensional, make it exact
        if n_feats <= 500 and isinstance(step_val, float):
            step_val = 1
        selector = RFE(estimator, n_features_to_select=K, step=step_val)
        selector.fit(X_train, y_train)
        selected_idx = np.where(selector.support_)[0].tolist()

    elif method == "MRMR":
        if not HAS_MRMR:
            print("  [Skip] mRMR requested but mrmr-selection is not installed.")
            return np.array([], dtype=int)
        X_df = pd.DataFrame(X_train, columns=feature_names)
        y_s = pd.Series(y_train)
        selected_features = mrmr_classif(X=X_df, y=y_s, K=K, show_progress=False)
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        selected_idx = [name_to_idx[n] for n in selected_features if n in name_to_idx]

    elif method == "CDGAFS":
        if not HAS_CDGAFS:
            print("  [Skip] CDGAFS requested but CDGAFS is not installed.")
            return np.array([], dtype=int)
        try:
            sel_idx, *_ = cdgafs_feature_selection(
                X=X_train, y=y_train, feature_list=feature_names,
                theta=cdgafs_theta, omega=cdgafs_omega, population_size=cdgafs_pop,
                use_semantic_clustering=use_semantic, max_cluster_size=max_cluster_size,
                use_quality_quota=use_quality_quota, target_k=K, 
                top_cluster_ratio=top_cluster_ratio, temperature=temperature
            )
            selected_idx = list(sel_idx) if sel_idx is not None else []
            if len(selected_idx) > K:
                # prune to K by RFE on the subspace
                est = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
                rfe = RFE(est, n_features_to_select=K, step=1)
                rfe.fit(X_train[:, selected_idx], y_train)
                selected_idx = np.array(selected_idx)[rfe.support_].tolist()
            else:
                selected_idx = selected_idx
        except Exception as e:
            print(f"  [CDGAFS] failed: {e}")
            selected_idx = []

    else:
        raise ValueError(f"Unknown method: {method}")

    # Fallback
    if len(selected_idx) == 0:
        vars_ = np.var(X_train, axis=0)
        selected_idx = np.argsort(vars_)[::-1][:K].tolist()

    elapsed = time.time() - start_t
    selected_idx_arr = np.array(selected_idx, dtype=int)

    # Safety bounds
    selected_idx_arr = selected_idx_arr[(selected_idx_arr >= 0) & (selected_idx_arr < X_train.shape[1])]
    # if len(selected_idx_arr) > K:
    #     selected_idx_arr = selected_idx_arr[:K]

    print(f"[Select] {method} | selected={len(selected_idx_arr)} | time={elapsed:.2f}s")
    return selected_idx_arr


# ============================================================
# 4) Train + Evaluate
# ============================================================
def evaluate_model(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    selected_idx: np.ndarray
) -> Dict[str, Dict[str, float]]:
    X_train, y_train = datasets["Train"]
    X_train_sel = X_train[:, selected_idx]

    clf = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
    clf.fit(X_train_sel, y_train)

    results: Dict[str, Dict[str, float]] = {}
    for name, (X, y_true) in datasets.items():
        X_sel = X[:, selected_idx]
        y_pred = clf.predict(X_sel)
        y_prob = clf.predict_proba(X_sel)[:, 1]

        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        acc = float(accuracy_score(y_true, y_pred))

        # y_true/y_pred are {0,1} by construction; safe confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        results[name] = {"AUC": auc, "ACC": acc, "Sens": sens, "Spec": spec}
    return results


# ============================================================
# CLI
# ============================================================
# ============================================================
# USER CONFIG 
# 
# radMLBench 公开数据集统一字段：
#   - ID 列名: 'ID'
#   - 标签列名: 'Target' (二分类 0/1)
#   - 其余列: radiomics 特征
# ============================================================

CONFIG = {
    # 数据路径（支持 .csv 或 .gz 的 gzip 压缩 CSV）
    "csv": "/data/qh_20T_share_file/lct/CT67/localdata/prostate_features_with_label.csv",

    # 列名
    "label": "label",
    "id_col": "ID",

    # auto / single / multi
    # - single: 单中心随机拆分 Train/IntVal
    # - multi : 按中心划分 Train/IntVal + 外部中心 Ext_*
    # - auto  : 如果存在 center 列且中心数>1，则按 multi，否则按 single
    "mode": "single",

    # 多中心相关参数：
    # 注意：当 mode != 'multi' 时，下列参数会被自动忽略（你无需改成 None）。
    "center": "Center_Source",     # 你的数据如果没有该列也没关系，会自动忽略
    "train_centers": "FuYi",       # 例如 "FuYi,Ningbo"

    # 拆分与随机种子
    "val_size": 0.30,
    "seed": 42,

    # 方法与超参
    "methods": "CDGAFS,LASSO,MRMR",
    "k": 50,
    "var_threshold": 1e-10,
    "rfe_step": 1,

    # CDGAFS 超参（仅在安装了 CDGAFS 且 methods 包含 CDGAFS 时生效）
    "cdgafs_pop": 100,
    "cdgafs_theta": 0.9,
    "cdgafs_omega": 0.05,
    "use_semantic": True,       # 是否使用语义预聚类替代 ISCD
    "max_cluster_size": 200,     # 语义聚类单组最大特征数
    "max_per_cluster": 50,       # 每个社区选择的最大特征数
    "use_quality_quota": True,   # 是否使用方案C质量加权配额分配
    "top_cluster_ratio": 0.5,    # 筛选的高质量社区比例 (Top N%)
    "temperature": 10.0,         # Softmax 温度参数，越大权重差异越大

    # 输出 CSV（相对路径：保存到当前工作目录）
    "out": "validation_summary.csv",
}


def main():
    parser = argparse.ArgumentParser(description="Run Unified Validation")
    parser.add_argument("--csv", type=str, default=CONFIG["csv"], help="Path to CSV file")
    parser.add_argument("--label", type=str, default=CONFIG["label"], help="Label column name")
    parser.add_argument("--id_col", type=str, default=CONFIG["id_col"], help="ID column name")
    parser.add_argument("--center", type=str, default=CONFIG["center"], help="Center column name")
    parser.add_argument("--train_centers", type=str, default=CONFIG["train_centers"], help="Training centers (comma-separated)")
    parser.add_argument("--mode", type=str, default=CONFIG["mode"], help="Mode: single, multi, auto")
    parser.add_argument("--methods", type=str, default=CONFIG["methods"], help="Methods (comma-separated)")
    parser.add_argument("--k", type=int, default=CONFIG["k"], help="Number of features to select")
    parser.add_argument("--out", type=str, default=CONFIG["out"], help="Output CSV file path")
    parser.add_argument("--use_semantic", type=str, default=str(CONFIG["use_semantic"]),
                        help="Use semantic pre-clustering ('True' or 'False')")
    parser.add_argument("--max_cluster_size", type=int, default=CONFIG["max_cluster_size"],
                        help="Max features per semantic cluster before sub-division")
    parser.add_argument("--max_per_cluster", type=int, default=CONFIG["max_per_cluster"],
                        help="Max features to select from each cluster")
    parser.add_argument("--cdgafs_omega", type=float, default=CONFIG["cdgafs_omega"],
                        help="CDGAFS omega parameter (proportion of features per cluster)")
    parser.add_argument("--use_quality_quota", type=str, default=str(CONFIG["use_quality_quota"]),
                        help="Use Plan C quality-weighted quota allocation ('True' or 'False')")
    parser.add_argument("--top_cluster_ratio", type=float, default=CONFIG["top_cluster_ratio"],
                        help="Top cluster ratio for Plan C (e.g., 0.5 for Top 50%%)")
    parser.add_argument("--temperature", type=float, default=CONFIG["temperature"],
                        help="Softmax temperature for weighted sampling (higher = more contrast)")
    args = parser.parse_args()

    cfg = dict(CONFIG)  # shallow copy, safe to edit locally
    
    # Update config with args
    cfg['csv'] = args.csv
    cfg['label'] = args.label
    cfg['id_col'] = args.id_col
    cfg['center'] = args.center
    if args.train_centers and args.train_centers.strip():
        cfg['train_centers'] = args.train_centers
    cfg['mode'] = args.mode
    if args.methods and args.methods.strip():
        cfg['methods'] = args.methods
    cfg['k'] = args.k
    cfg['out'] = args.out
    
    # Parse boolean
    cfg['use_semantic'] = (str(args.use_semantic).lower() == 'true')
    cfg['max_cluster_size'] = args.max_cluster_size
    cfg['max_per_cluster'] = args.max_per_cluster
    cfg['cdgafs_omega'] = args.cdgafs_omega
    cfg['use_quality_quota'] = (str(args.use_quality_quota).lower() == 'true')
    cfg['top_cluster_ratio'] = args.top_cluster_ratio
    cfg['temperature'] = args.temperature

    # Normalize lists
    train_centers = None
    if isinstance(cfg.get("train_centers"), str) and cfg.get("train_centers").strip():
        train_centers = [c.strip() for c in cfg['train_centers'].split(",") if c.strip()]

    methods = [m.strip() for m in str(cfg.get("methods", "")).split(",") if m.strip()]

    # Load + split
    datasets_raw, feature_cols, resolved_mode = load_and_split(
        csv_path=cfg['csv'],
        label_col=cfg['label'],
        mode=cfg['mode'],
        center_col=cfg.get("center"),
        train_centers=train_centers,
        id_col=cfg.get("id_col"),
        val_size=cfg['val_size'],
        random_state=cfg['seed'],
    )

    # Preprocess
    datasets, feature_names = preprocess_securely(datasets_raw, var_threshold=cfg['var_threshold'])
    X_train, y_train = datasets["Train"]

    print("=" * 72)
    print(f"Validation start | mode={resolved_mode} | K={cfg['k']} | omega={cfg['cdgafs_omega']} | methods={methods}")
    print("=" * 72)

    summary_rows = []
    for method in methods:
        sel_idx = run_feature_selection(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            method=method,
            K=cfg['k'],
            rfe_step=cfg['rfe_step'],
            cdgafs_pop=cfg['cdgafs_pop'],
            cdgafs_theta=cfg['cdgafs_theta'],
            cdgafs_omega=cfg['cdgafs_omega'],
            use_semantic=cfg['use_semantic'],
            max_cluster_size=cfg['max_cluster_size'],
            use_quality_quota=cfg['use_quality_quota'],
            top_cluster_ratio=cfg['top_cluster_ratio'],
            temperature=cfg['temperature'],
        )

        selected_names = [feature_names[i] for i in sel_idx]
        # print(f"  Features({len(selected_names)}): {selected_names}")

        scores = evaluate_model(datasets, sel_idx)

        row = {"Method": method.upper(), "Features": len(sel_idx), "Selected": "|".join(selected_names)}
        for ds_name, res in scores.items():
            row[f"{ds_name}_AUC"] = res["AUC"]
            row[f"{ds_name}_ACC"] = res["ACC"]
            row[f"{ds_name}_Sens"] = res["Sens"]
            row[f"{ds_name}_Spec"] = res["Spec"]
        summary_rows.append(row)

        # Print compact table
        header = f"{'Dataset':<18} | {'AUC':<8} | {'ACC':<8} | {'Sens':<8} | {'Spec':<8}"
        print("\n" + header)
        print("-" * len(header))
        for ds_name, res in scores.items():
            print(f"{ds_name:<18} | {res['AUC']:<8.4f} | {res['ACC']:<8.4f} | {res['Sens']:<8.4f} | {res['Spec']:<8.4f}")
        print("-" * len(header))

    out_path = cfg['out']
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\nDone. Saved to: {out_path}")


if __name__ == "__main__":
    main()
