# -*- coding: utf-8 -*-
"""
run_multicenter_validation_brain_robust_nocli.py

目标：在“不改命令行参数、只改代码配置”的前提下，提升脑血肿多中心外部泛化的鲁棒性。
核心改动：
1) 支持按 cluster 前缀过滤特征（默认丢弃 cluster2，保留 cluster1+cluster3）
2) 训练阶段做“中心敏感特征”与“方向不稳定特征”预筛（仅用训练数据计算，避免测试泄露）
3) 评估时用内部验证集(IntVal)自动校准阈值（避免 0.5 阈值导致敏感度/特异度崩溃）
4) 可选：替代 CovBat 的轻量 Harmonization（分位数匹配/均值方差对齐；均为无标签），用于降低中心分布差异

依赖：pandas, numpy, scikit-learn；可选 CDGAFS、mrmr-selection。
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE

warnings.filterwarnings("ignore")

# =========================
# 0) 你只需要改这里
# =========================
CSV_FILE = "/data/qh_20T_share_file/lct/CT67/Merged_All_Centers.csv"  # 改成你的路径
LABEL_COL = "label"
CENTER_COL = "Center_Source"
ID_COL = "Sample_ID"

# 评估模式：
# - "fixed": 固定 TRAIN_CENTERS 训练，其他中心外部测试（你现在的常用模式）
# - "loco": 逐中心留一(Leave-One-Center-Out)，更符合“多中心泛化”定义
EVAL_MODE = "fixed"

TRAIN_CENTERS = ["FuYi"]          # EVAL_MODE="fixed" 时使用
TEST_CENTERS = ["ShiZhongXin", "NingBo"]              # None -> 自动用除训练中心外的所有中心

# ===== 泛化增强开关（建议先按默认跑一轮）=====
# 只保留哪些 cluster 前缀（强烈建议丢弃 cluster2；它在你的数据里最容易“方向反转”）
KEEP_CLUSTER_PREFIX = ("cluster1_", "cluster3_")  # 可改为 ("cluster3_",) 或 ("cluster1_","cluster2_","cluster3_")

# 训练数据上的“中心敏感特征”剔除（ANOVA-F 近似；值越大越像“中心指纹”）
ENABLE_SITE_SENS_FILTER = True
SITE_SENS_DROP_FRAC = 0.40       # 丢弃最“中心敏感”的前 30% 特征（0~0.8 之间调）
MIN_FEATURES_AFTER_FILTER = 500  # 过滤后至少保留这么多特征（避免过度过滤）


# 训练中心 vs 其他中心 的“分布漂移”过滤（不使用 label，只看中心间分布差异）
# 计算每个特征在各中心相对参考训练中心的标准化均值差 (SMD)，丢弃漂移最大的若干特征。
ENABLE_SHIFT_FILTER = True
SHIFT_DROP_FRAC = 0.40          # 丢弃漂移最大的前 30%（0~0.8），建议与 SITE_SENS_DROP_FRAC 配合调整
SHIFT_MAX_ABS_SMD = None        # 若设置为浮点数（例如 0.8），则丢弃 max|SMD| > 阈值 的特征（优先于 DROP_FRAC）
SHIFT_REF_CENTER = None         # None 表示使用 TRAIN_CENTERS[0] 作为参考中心



# 训练数据上的“方向稳定性”筛选：要求特征在训练中心内 d 的符号一致比例 >= 阈值
ENABLE_DIRECTION_STABILITY = True
MIN_SIGN_AGREE = 0.75            # 0.75 表示 4 个中心里至少 3 个中心符号一致（在训练集中心内评估）

# 阈值校准（用 IntVal 找最佳阈值，然后用于所有外部中心）
CALIBRATE_THRESHOLD_ON_INTVAL = True

# 可选：无标签域适配（外部中心用其自身均值/方差对齐到训练分布；不使用 label）
ENABLE_UNLABELED_MOMENT_MATCH = False

# Harmonization（替代 CovBat 的轻量方案）：
# - "none": 不做对齐
# - "moment": 外部中心均值/方差对齐到训练分布（线性、稳健）
# - "qmatch": 外部中心逐特征分位数匹配到训练分布（非线性，适合强度链不一致/分布形状差异）
# - "qmatch+moment": 先分位数匹配，再做均值/方差微调
HARMONIZATION = "qmatch+moment"  # 建议先用 qmatch；如果过拟合/不稳定再回退 moment

# 分位数匹配参数
QMATCH_N_QUANTILES = 101
QMATCH_SHRINK_K = 50.0    # 越大 -> 越保守（对小中心更稳）

# 是否也对“训练中心”内部做对齐（仅在你使用多训练中心时建议开启）
HARMONIZE_TRAIN_CENTERS = False


# 特征选择
METHODS = ["CDGAFS", "LASSO", "RFE", "mRMR"]
K_FEATURES = 12

# CDGAFS 参数（如你的 CDGAFS 实现可调）
CDGAFS_THETA = 0.9
CDGAFS_OMEGA = 0.5
CDGAFS_POP = 100

RANDOM_STATE = 42


# =========================
# 1) 可选依赖
# =========================
try:
    from CDGAFS import cdgafs_feature_selection
    HAS_CDGAFS = True
except Exception:
    HAS_CDGAFS = False

try:
    from mrmr import mrmr_classif
    HAS_MRMR = True
except Exception:
    HAS_MRMR = False


# =========================
# 2) 工具函数
# =========================
def _cohen_d(x1: np.ndarray, x0: np.ndarray) -> float:
    """Cohen's d（用在方向稳定性判断）"""
    x1 = np.asarray(x1, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return np.nan
    s1 = np.var(x1, ddof=1)
    s0 = np.var(x0, ddof=1)
    sp = ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)
    if not np.isfinite(sp) or sp <= 1e-12:
        return np.nan
    return (np.mean(x1) - np.mean(x0)) / np.sqrt(sp)

def _compute_quantile_table(X: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """返回形状 (len(qs), p) 的分位数表。"""
    return np.quantile(X, qs, axis=0)


def _quantile_match_to_ref(X_t: np.ndarray, ref_q: np.ndarray, qs: np.ndarray, shrink_k: float = 50.0) -> np.ndarray:
    """
    按特征做分位数匹配(Quantile Matching)：把目标中心 X_t 的每个特征分布映射到参考分布(ref_q)。
    - 只使用无标签特征
    - 带 shrinkage：n/(n+shrink_k)，避免小中心过拟合
    """
    X_t = np.asarray(X_t, dtype=float)
    n, p = X_t.shape
    if n < 5:
        # 太小就别折腾，直接返回
        return X_t

    tgt_q = np.quantile(X_t, qs, axis=0)
    alpha = float(n / (n + shrink_k)) if shrink_k is not None else 1.0

    X_m = np.empty_like(X_t)
    for j in range(p):
        tq = tgt_q[:, j]
        rq = ref_q[:, j]
        # 处理常数/退化分布
        if np.allclose(tq, tq[0]):
            mapped = np.full(n, rq[len(rq)//2], dtype=float)
        else:
            # np.interp 要求 xp 单调递增；分位数应单调但可能有重复值
            # 用微小扰动保证严格递增（不改变顺序）
            tq2 = tq.copy()
            for k in range(1, len(tq2)):
                if tq2[k] <= tq2[k-1]:
                    tq2[k] = tq2[k-1] + 1e-9
            mapped = np.interp(X_t[:, j], tq2, rq, left=rq[0], right=rq[-1])
        # shrinkage：在“原值”和“匹配值”之间插值
        X_m[:, j] = alpha * mapped + (1.0 - alpha) * X_t[:, j]
    return X_m


def select_feature_cols(df: pd.DataFrame) -> list:
    """按规则选特征列：去掉 id/center/label/diagnostics_，并按 cluster 前缀过滤。"""
    base_drop = {ID_COL, CENTER_COL, LABEL_COL}
    cols = []
    for c in df.columns:
        if c in base_drop:
            continue
        if c.startswith("diagnostics_"):
            continue
        if KEEP_CLUSTER_PREFIX is not None:
            ok = any(c.startswith(p) for p in KEEP_CLUSTER_PREFIX)
            if not ok:
                continue
        cols.append(c)
    # 强制转数值，非数值 -> NaN（避免 mean imputer 报错）
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    # 删除全 NaN 特征
    all_nan = [c for c in cols if df[c].isna().all()]
    if all_nan:
        df.drop(columns=all_nan, inplace=True)
        cols = [c for c in cols if c not in all_nan]
    return cols


def anova_f_center_sensitivity(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    近似的 one-way ANOVA F：衡量“特征随中心变化”的强度。
    X: (n, p) 数值矩阵
    centers: (n,) center 标签
    返回: (p,) F 值（越大越中心敏感）
    """
    centers = np.asarray(centers)
    uniq = np.unique(centers)
    n, p = X.shape
    grand_mean = np.nanmean(X, axis=0)
    ss_between = np.zeros(p)
    ss_within = np.zeros(p)
    for c in uniq:
        idx = (centers == c)
        Xc = X[idx]
        mc = np.nanmean(Xc, axis=0)
        nc = Xc.shape[0]
        ss_between += nc * (mc - grand_mean) ** 2
        ss_within += np.nansum((Xc - mc) ** 2, axis=0)
    dfb = max(len(uniq) - 1, 1)
    dfw = max(n - len(uniq), 1)
    msb = ss_between / dfb
    msw = ss_within / dfw
    F = msb / (msw + 1e-12)
    F[~np.isfinite(F)] = 0.0
    return F



def max_abs_smd_vs_ref(X: np.ndarray, centers: np.ndarray, ref_center: str) -> np.ndarray:
    """计算每个特征相对 ref_center 的最大 |SMD|（标准化均值差）。

    SMD 这里用 (mu_c - mu_ref) / sd_ref，sd_ref 来自参考中心，避免小中心方差不稳。
    仅依赖中心标签与特征，不依赖 y。
    """
    centers = np.asarray(centers)
    ref_mask = (centers == ref_center)
    if ref_mask.sum() < 3:
        # 参考中心太小就退化为全体
        ref_mask = np.ones_like(ref_mask, dtype=bool)

    X_ref = X[ref_mask]
    mu_ref = np.nanmean(X_ref, axis=0)
    sd_ref = np.nanstd(X_ref, axis=0, ddof=1)
    sd_ref = np.where(np.isfinite(sd_ref) & (sd_ref > 1e-8), sd_ref, 1e-8)

    max_abs = np.zeros(X.shape[1], dtype=float)
    for c in np.unique(centers):
        if c == ref_center:
            continue
        m = (centers == c)
        if m.sum() < 3:
            continue
        mu_c = np.nanmean(X[m], axis=0)
        smd = (mu_c - mu_ref) / sd_ref
        max_abs = np.maximum(max_abs, np.abs(smd))
    max_abs[~np.isfinite(max_abs)] = 0.0
    return max_abs


def direction_stability_mask(df_train: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    方向稳定性：对每个特征，计算各训练中心内 label1-label0 的 Cohen d，
    看符号与“多数符号”一致的比例是否 >= MIN_SIGN_AGREE。
    返回 bool mask（长度=特征数）
    """
    centers = df_train[CENTER_COL].values
    y = df_train[LABEL_COL].values.astype(int)
    uniq = np.unique(centers)
    mask = np.zeros(len(feature_cols), dtype=bool)

    for j, f in enumerate(feature_cols):
        signs = []
        for c in uniq:
            sub = df_train[centers == c]
            y_c = sub[LABEL_COL].values.astype(int)
            x = sub[f].values.astype(float)
            x1 = x[y_c == 1]
            x0 = x[y_c == 0]
            d = _cohen_d(x1[~np.isnan(x1)], x0[~np.isnan(x0)])
            if np.isfinite(d) and abs(d) > 1e-12:
                signs.append(np.sign(d))
        if len(signs) < 2:
            # 太少中心有可用 d，默认保留
            mask[j] = True
            continue
        # 多数符号
        maj = 1.0 if (np.sum(np.array(signs) > 0) >= np.sum(np.array(signs) < 0)) else -1.0
        agree = np.mean(np.array(signs) == maj)
        mask[j] = (agree >= MIN_SIGN_AGREE)
    return mask


def calibrate_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在 IntVal 上找使 Youden J = TPR - FPR 最大的阈值。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    # 候选阈值取分位点，避免 O(n^2)
    qs = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, 201)))
    best_t, best_j = 0.5, -1e9
    for t in qs:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


# =========================
# 3) 核心：拆分、预处理、选择、评估
# =========================
def load_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"!!! 错误: 文件不存在: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    for c in [ID_COL, CENTER_COL, LABEL_COL]:
        if c not in df.columns:
            print(f"!!! 错误: CSV 缺失列: {c}")
            sys.exit(1)
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def build_datasets_fixed(df: pd.DataFrame, train_centers: list) -> dict:
    """固定训练中心：Train/IntVal 来自训练中心，Ext_* 来自其余中心。"""
    feat_cols = select_feature_cols(df)
    train_df = df[df[CENTER_COL].isin(train_centers)].copy()
    X_tr, X_iv, y_tr, y_iv = train_test_split(
        train_df[feat_cols], train_df[LABEL_COL],
        test_size=0.30, stratify=train_df[LABEL_COL], random_state=RANDOM_STATE
    )
    datasets = {
        "Train": (X_tr, y_tr.values),
        "IntVal": (X_iv, y_iv.values)
    }
    all_centers = df[CENTER_COL].unique().tolist()
    ext_centers = [c for c in all_centers if c not in train_centers]
    if TEST_CENTERS is not None:
        ext_centers = [c for c in ext_centers if c in TEST_CENTERS]
    for c in ext_centers:
        ext_df = df[df[CENTER_COL] == c]
        datasets[f"Ext_{c}"] = (ext_df[feat_cols], ext_df[LABEL_COL].values)
    meta = {"feature_cols": feat_cols, "train_centers": train_centers, "ext_centers": ext_centers}
    return datasets, meta


def preprocess_train_only(datasets: dict, df_train_for_filters: pd.DataFrame, feature_cols: list):
    """
    严格预处理：只用训练集拟合 imputer/scaler，并可选做：
    - 中心敏感特征剔除
    - 方向稳定性筛选
    - 外部中心 moment matching（无标签）
    """
    X_train_df, y_train = datasets["Train"]

    # 固定“基准特征列”（来自 shift/site 过滤后的候选列）
    # 注意：datasets 内的各 split 可能仍包含更多列；这里统一 reindex 以确保 imputer 的 fit/transform 列一致
    base_cols = list(feature_cols)
    feature_cols = base_cols  # 之后所有过滤/索引都基于 base_cols

    # 确保 df_train_for_filters 拥有 base_cols（缺失则补 NaN），防止后续方向稳定性/中心敏感过滤 KeyError
    for _c in base_cols:
        if _c not in df_train_for_filters.columns:
            df_train_for_filters[_c] = np.nan

    # 1) impute（用 median 更稳健；mean 对异常值敏感）
    imputer = SimpleImputer(strategy="median")

    X_train_df_base = X_train_df.reindex(columns=base_cols)
    imputer.fit(X_train_df_base)
    X_train_imp = imputer.transform(X_train_df_base)

    # 2) 训练数据上的过滤（在 impute 后进行）
    keep_idx = np.ones(len(feature_cols), dtype=bool)

    # 2.1) 方向稳定性（仅用训练中心数据）
    if ENABLE_DIRECTION_STABILITY:
        mask_dir = direction_stability_mask(df_train_for_filters, feature_cols)
        keep_idx &= mask_dir

    # 2.2) 中心敏感特征（ANOVA-F，丢弃最敏感的一部分）
    if ENABLE_SITE_SENS_FILTER:
        # 仅在训练中心数据上计算
        df_sub = df_train_for_filters.copy()
        X_sub = imputer.transform(df_sub[feature_cols])
        F = anova_f_center_sensitivity(X_sub, df_sub[CENTER_COL].values)
        # 只在当前 keep_idx 内排序/丢弃
        cand = np.where(keep_idx)[0]
        if len(cand) > MIN_FEATURES_AFTER_FILTER:
            drop_k = int(len(cand) * SITE_SENS_DROP_FRAC)
            drop_k = min(drop_k, max(0, len(cand) - MIN_FEATURES_AFTER_FILTER))
            if drop_k > 0:
                order = cand[np.argsort(F[cand])[::-1]]  # 从大到小
                drop_idx = order[:drop_k]
                keep_idx[drop_idx] = False

    kept_cols = [feature_cols[i] for i in np.where(keep_idx)[0]]

    # 3) 方差筛选（避免 0 方差）
    X_train_imp = X_train_imp[:, keep_idx]
    stds = np.std(X_train_imp, axis=0)
    good2 = np.where(stds > 1e-6)[0]
    X_train_imp = X_train_imp[:, good2]
    final_cols = [kept_cols[i] for i in good2]

    # 4) 标准化（fit on Train）
    scaler = StandardScaler()
    scaler.fit(X_train_imp)    # 5) Harmonization（只用无标签特征；默认只对外部中心做）
    mu_train = np.mean(X_train_imp, axis=0)
    sd_train = np.std(X_train_imp, axis=0) + 1e-12

    qs = np.linspace(0.0, 1.0, int(QMATCH_N_QUANTILES))
    ref_q = _compute_quantile_table(X_train_imp, qs)

    processed = {}
    for name, (X_df, y) in datasets.items():
        X_df_base = X_df.reindex(columns=base_cols)
        X_imp = imputer.transform(X_df_base)
        X_imp = X_imp[:, keep_idx]
        X_imp = X_imp[:, good2]

        is_ext = name.startswith("Ext_")
        is_train_center_data = name in ("Train", "IntVal")

        # 是否对训练中心也做对齐（多训练中心时可用；固定单中心一般没必要）
        do_train_harmonize = HARMONIZE_TRAIN_CENTERS and is_train_center_data

        if (is_ext or do_train_harmonize) and HARMONIZATION != "none":
            if HARMONIZATION in ("qmatch", "qmatch+moment"):
                X_imp = _quantile_match_to_ref(X_imp, ref_q, qs, shrink_k=QMATCH_SHRINK_K)

            if HARMONIZATION in ("moment", "qmatch+moment"):
                mu = np.mean(X_imp, axis=0)
                sd = np.std(X_imp, axis=0) + 1e-12
                X_imp = (X_imp - mu) / sd * sd_train + mu_train

        X_scaled = scaler.transform(X_imp)
        processed[name] = (X_scaled, y)

    return processed, final_cols


def run_feature_selection(X_train: np.ndarray, y_train: np.ndarray, feature_names: list, method: str, K: int) -> np.ndarray:
    print(f"\n--- [FS] {method} (K={K}) ---")
    t0 = time.time()

    if method == "LASSO":
        clf = LogisticRegressionCV(
            cv=5, penalty="l1", solver="liblinear",
            scoring="roc_auc", class_weight="balanced",
            random_state=RANDOM_STATE, max_iter=4000
        )
        clf.fit(X_train, y_train)
        coefs = np.abs(clf.coef_[0])
        idx = np.argsort(coefs)[::-1][:K]
        sel = idx

    elif method == "RFE":
        est = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000)
        n_feats = X_train.shape[1]
        step_val = 0.05 if n_feats > 500 else 1
        selector = RFE(est, n_features_to_select=K, step=step_val)
        selector.fit(X_train, y_train)
        sel = np.where(selector.support_)[0]

    elif method == "mRMR":
        if not HAS_MRMR:
            print("    [跳过] 缺少 mrmr-selection 库")
            return np.array([], dtype=int)
        X_df = pd.DataFrame(X_train, columns=feature_names)
        y_s = pd.Series(y_train)
        feats = mrmr_classif(X=X_df, y=y_s, K=K, show_progress=False)
        name2i = {n: i for i, n in enumerate(feature_names)}
        sel = np.array([name2i[f] for f in feats if f in name2i], dtype=int)

    elif method == "CDGAFS":
        if not HAS_CDGAFS:
            print("    [跳过] 缺少 CDGAFS 库")
            return np.array([], dtype=int)
        (sel_idx, _, _, _, _) = cdgafs_feature_selection(
            X=X_train, y=y_train, gene_list=feature_names,
            theta=CDGAFS_THETA, omega=CDGAFS_OMEGA, population_size=CDGAFS_POP,
            w_bio_boost=0.0, graph_type="pearson_only"
        )
        sel_idx = np.array(sel_idx, dtype=int)
        if len(sel_idx) > K:
            est = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000)
            rfe = RFE(est, n_features_to_select=K, step=1)
            rfe.fit(X_train[:, sel_idx], y_train)
            sel = sel_idx[rfe.support_]
        else:
            sel = sel_idx

    else:
        raise ValueError(f"Unknown method: {method}")

    # 兜底：如果没选到特征
    sel = np.array(sel, dtype=int)
    if sel.size == 0:
        vars_ = np.var(X_train, axis=0)
        sel = np.argsort(vars_)[::-1][:K]

    print(f"    - selected={len(sel)} | time={time.time()-t0:.1f}s")
    return sel


def evaluate(datasets: dict, selected_idx: np.ndarray):
    X_tr, y_tr = datasets["Train"]
    X_iv, y_iv = datasets["IntVal"]
    X_tr_sel = X_tr[:, selected_idx]
    X_iv_sel = X_iv[:, selected_idx]

    clf = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000)
    clf.fit(X_tr_sel, y_tr)

    # 阈值校准
    th = 0.5
    if CALIBRATE_THRESHOLD_ON_INTVAL:
        p_iv = clf.predict_proba(X_iv_sel)[:, 1]
        th = calibrate_threshold(y_iv, p_iv)

    res = {}
    for name, (X, y) in datasets.items():
        X_sel = X[:, selected_idx]
        p = clf.predict_proba(X_sel)[:, 1]
        y_pred = (p >= th).astype(int)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        acc = accuracy_score(y, y_pred)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        res[name] = {"AUC": float(auc), "ACC": float(acc), "Sens": float(sens), "Spec": float(spec)}
    return res, th


def run_once(df: pd.DataFrame, train_centers: list, tag: str):
    print(f"\n{'='*70}\n[{tag}] TrainCenters={train_centers}\n{'='*70}")
    datasets_raw, meta = build_datasets_fixed(df, train_centers)

    # 过滤器只用训练中心数据（防泄露）
    df_train_for_filters = df[df[CENTER_COL].isin(train_centers)].copy()
    feature_cols = meta["feature_cols"]

    # 训练中心 vs 其他中心 分布漂移过滤（无标签，允许使用外部中心的特征分布做诊断/过滤）
    if ENABLE_SHIFT_FILTER and len(feature_cols) > 0:
        ref_center = SHIFT_REF_CENTER or (train_centers[0] if len(train_centers) > 0 else None)
        if ref_center is not None:
            # 用训练中心拟合一个临时 imputer，仅用于计算 SMD（不影响最终预处理）
            imp_tmp = SimpleImputer(strategy="median")
            imp_tmp.fit(df[df[CENTER_COL].isin(train_centers)][feature_cols])
            X_all_tmp = imp_tmp.transform(df[feature_cols])
            max_abs = max_abs_smd_vs_ref(X_all_tmp, df[CENTER_COL].values, ref_center)

            keep = np.ones(len(feature_cols), dtype=bool)
            if SHIFT_MAX_ABS_SMD is not None:
                keep = max_abs <= float(SHIFT_MAX_ABS_SMD)
            else:
                cand = np.arange(len(feature_cols))
                # 只在候选里丢弃，确保至少保留 MIN_FEATURES_AFTER_FILTER
                if len(cand) > MIN_FEATURES_AFTER_FILTER:
                    drop_k = int(len(cand) * SHIFT_DROP_FRAC)
                    drop_k = min(drop_k, max(0, len(cand) - MIN_FEATURES_AFTER_FILTER))
                    if drop_k > 0:
                        order = cand[np.argsort(max_abs)[::-1]]
                        keep[order[:drop_k]] = False

            new_cols = [c for c, k in zip(feature_cols, keep) if k]
            if len(new_cols) >= MIN_FEATURES_AFTER_FILTER:
                feature_cols = new_cols
                meta["feature_cols"] = feature_cols
                print(f"[ShiftFilter] ref={ref_center} | kept={len(feature_cols)}")
            else:
                print(f"[ShiftFilter] 过滤过猛({len(new_cols)})，已忽略。")

    datasets, final_features = preprocess_train_only(datasets_raw, df_train_for_filters, feature_cols)
    X_train, y_train = datasets["Train"]

    out_rows = []
    for method in METHODS:
        sel_idx = run_feature_selection(X_train, y_train, final_features, method, K_FEATURES)
        sel_names = [final_features[i] for i in sel_idx]
        scores, th = evaluate(datasets, sel_idx)

        print(f"\n>>> {method} | threshold={th:.3f} | feats={len(sel_idx)}")
        print("Selected:", sel_names[:min(20, len(sel_names))], ("..." if len(sel_names) > 20 else ""))

        header = f"{'Dataset':<20} | {'AUC':<8} | {'ACC':<8} | {'Sens':<8} | {'Spec':<8}"
        print(header)
        print("-" * len(header))
        row = {"RunTag": tag, "Method": method, "K": int(len(sel_idx)), "Threshold": th}
        for ds_name, m in scores.items():
            print(f"{ds_name:<20} | {m['AUC']:.4f}   | {m['ACC']:.4f}   | {m['Sens']:.4f}   | {m['Spec']:.4f}")
            row[f"{ds_name}_AUC"] = m["AUC"]
            row[f"{ds_name}_ACC"] = m["ACC"]
        out_rows.append(row)
        print("-" * 65)

    out_df = pd.DataFrame(out_rows)
    out_csv = f"multicenter_validation_brain_robust_{tag}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")
    return out_df


def main():
    df = load_df(CSV_FILE)
    centers = df[CENTER_COL].unique().tolist()
    print(f"[Data] n={len(df)} | centers={centers}")

    if EVAL_MODE == "fixed":
        run_once(df, TRAIN_CENTERS, tag="fixed")

    elif EVAL_MODE == "loco":
        all_rows = []
        for held_out in centers:
            train_centers = [c for c in centers if c != held_out]
            out = run_once(df, train_centers, tag=f"loco_test_{held_out}")
            all_rows.append(out)
        merged = pd.concat(all_rows, ignore_index=True)
        merged.to_csv("multicenter_validation_brain_robust_LOCO_merged.csv", index=False)
        print("\n[Saved] multicenter_validation_brain_robust_LOCO_merged.csv")
    else:
        raise ValueError(f"Unknown EVAL_MODE: {EVAL_MODE}")

if __name__ == "__main__":
    main()