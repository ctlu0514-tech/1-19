# 文件名: run_evaluation.py
# 描述: 专门用于模型评估和结果报告

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix

def evaluate_model_performance(X, y, selected_indices):
    """
    (修改版)
    使用5折交叉验证和L2逻辑回归，评估所选特征的
    Accuracy, AUC, ...
    同时返回 *训练集* 的指标以诊断过拟合/欠拟合。
    """
    
    if len(selected_indices) == 0:
        # ... (这部分不变) ...
        return {"Accuracy": 0, "AUC": 0, "Sensitivity": 0, "Specificity": 0, "F1-Macro": 0, "Train_AUC": 0, "Train_Accuracy": 0}
        
    X_subset = X[:, selected_indices]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # [修改] 增加 "train_" 相关的指标
    metrics = {
        "acc": [], "auc": [], "sens": [], "spec": [], "f1": [],
        "acc_train": [], "auc_train": []  # <--- 新增
    }

    valid_folds = 0 

    for train_idx, test_idx in skf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(np.unique(y_test)) < 2:
            continue
        
        valid_folds += 1
        
        model = LogisticRegressionCV(
            Cs=10, cv=3, penalty='l2', solver='lbfgs', multi_class='auto',
            random_state=42, max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # --- 验证集 (Test) 指标 (原逻辑) ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        metrics["acc"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred, average='macro'))
        
        n_classes_fold = len(model.classes_)
        try:
            if n_classes_fold == 2:
                metrics["auc"].append(roc_auc_score(y_test, y_proba[:, 1]))
            else:
                metrics["auc"].append(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
        except Exception:
            metrics["auc"].append(0.5)

        # Sensitivity / Specificity for multi-class (Macro Average)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        sens_fold = []
        spec_fold = []
        for i in range(len(model.classes_)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp
            sens_fold.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            spec_fold.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        
        metrics["sens"].append(np.mean(sens_fold))
        metrics["spec"].append(np.mean(spec_fold))

        # --- [!!! 核心新增 !!!] ---
        # --- 训练集 (Train) 指标 ---
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)
        
        metrics["acc_train"].append(accuracy_score(y_train, y_pred_train))
        try:
            if n_classes_fold == 2:
                metrics["auc_train"].append(roc_auc_score(y_train, y_proba_train[:, 1]))
            else:
                metrics["auc_train"].append(roc_auc_score(y_train, y_proba_train, multi_class='ovr', average='macro'))
        except Exception:
            metrics["auc_train"].append(0.5)
        # --- [新增结束] ---

    if valid_folds == 0:
        # ... (这部分不变) ...
        return {"Accuracy": 0, "AUC": 0, "Sensitivity": 0, "Specificity": 0, "F1-Macro": 0, "Train_AUC": 0, "Train_Accuracy": 0}

    # [修改] 计算平均值时加入 "train_" 指标
    avg_results = {
        "Accuracy": np.mean(metrics["acc"]),
        "AUC": np.mean(metrics["auc"]),
        "Sensitivity": np.mean(metrics["sens"]),
        "Specificity": np.mean(metrics["spec"]),
        "F1-Macro": np.mean(metrics["f1"]),
        "Train_Accuracy": np.mean(metrics["acc_train"]), # <--- 新增
        "Train_AUC": np.mean(metrics["auc_train"])       # <--- 新增
    }
    
    return avg_results

def print_summary_table(all_results, all_selected_indices, execution_times=None):
    """
    (修改版)
    打印最终的对比表格，加入训练集指标和运行时间。
    """
    if execution_times is None:
        execution_times = {}

    print("\n" + "#"*90)
    print("### 最终实验对比总结 ###")
    print("#"*90)
    
    # [修改] 增加 Time(s) 列
    header = f"{'Method':<12} | {'K':<4} | {'Time(s)':<8} | {'AUC':<10} | {'Train_AUC':<10} | {'Accuracy':<10} | {'Train_Acc':<10} | {'Sensitivity':<11} | {'Specificity':<11} | {'F1-Macro':<10}"
    print(header)
    print("-" * len(header))
    
    sorted_methods = sorted(all_results.items(), key=lambda item: item[1].get('Accuracy', 0), reverse=True)
    
    for method_name, metrics in sorted_methods:
        k = len(all_selected_indices.get(method_name, []))
        time_taken = execution_times.get(method_name, 0.0) # 获取时间
        
        # [修改] 打印时间列
        print(f"{method_name:<12} | {k:<4} | "
              f"{time_taken:<8.2f} | "                   # <--- 新增：保留2位小数的时间
              f"{metrics.get('AUC', 0):<10.4f} | "
              f"{metrics.get('Train_AUC', 0):<10.4f} | "       
              f"{metrics.get('Accuracy', 0):<10.4f} | "
              f"{metrics.get('Train_Accuracy', 0):<10.4f} | "  
              f"{metrics.get('Sensitivity', 0):<11.4f} | "
              f"{metrics.get('Specificity', 0):<11.4f} | "
              f"{metrics.get('F1-Macro', 0):<10.4f}")
              
    print("#"*90)

# --- 新增：支持多模型评估 ---

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def _build_model_pool():
    """返回统一的模型池字典，供 evaluate_single_fold 和 evaluate_multiple_models 共用。"""
    # Use 'lbfgs' or 'saga' for multinomial support in LogisticRegressionCV
    return {
        'LR': LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', multi_class='auto', random_state=42, class_weight='balanced', max_iter=2000),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'XGB': GradientBoostingClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }


def _compute_metrics(model, X, y_true):
    """用已训练好的 model 在 (X, y_true) 上计算 AUC/ACC/Sens/Spec。"""
    y_pred = model.predict(X)
    classes = np.unique(y_true)
    n_classes = len(classes)
    total_classes = len(model.classes_)

    # 获取概率
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception:
            y_prob = None
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
    else:
        y_prob = None

    # AUC calculation
    try:
        if y_prob is not None:
            if total_classes == 2:
                # Binary case: y_prob might be (n, 2) or (n,)
                prob_to_use = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc = float(roc_auc_score(y_true, prob_to_use))
            else:
                # Multi-class case
                auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro Sensitivity (Recall) and Specificity
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    
    sens_list = []
    spec_list = []
    for i in range(total_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        # Sensitivity (Recall)
        sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Specificity
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sens_list.append(sens_i)
        spec_list.append(spec_i)
        
    sens = float(np.mean(sens_list))
    spec = float(np.mean(spec_list))

    return {"AUC": auc, "ACC": acc, "Sens": sens, "Spec": spec}


def evaluate_single_fold(X_train, y_train, X_test, y_test, selected_idx):
    """
    五折交叉验证中单折的多模型评估。
    在 train 上 fit，在 test 上 predict，同时返回 train 和 test 的指标。

    Returns:
        dict: {
            'LR': {
                'test': {'AUC':…, 'ACC':…, 'Sens':…, 'Spec':…},
                'train': {'AUC':…, 'ACC':…, 'Sens':…, 'Spec':…}
            },
            ...
        }
    """
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]

    models = _build_model_pool()
    fold_results = {}

    for model_name, model in models.items():
        model.fit(X_train_sel, y_train)
        fold_results[model_name] = {
            'test': _compute_metrics(model, X_test_sel, y_test),
            'train': _compute_metrics(model, X_train_sel, y_train)
        }

    return fold_results


def evaluate_multiple_models(datasets, selected_idx):
    """
    使用选定的特征子集，在多个模型上进行训练和评估。
    返回一个嵌套字典结果。
    """
    X_train_full, y_train_full = datasets["Train"]
    
    # 仅使用选定特征
    X_train_sel = X_train_full[:, selected_idx]
    
    # 定义模型池
    models = _build_model_pool()

    all_model_results = {}

    for model_name, model in models.items():
        # 训练
        model.fit(X_train_sel, y_train_full)
        
        model_res = {}
        for ds_name, (X, y_true) in datasets.items():
            model_res[ds_name] = _compute_metrics(model, X[:, selected_idx], y_true)
        
        all_model_results[model_name] = model_res

    return all_model_results
