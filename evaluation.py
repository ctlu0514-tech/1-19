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
            Cs=10, cv=3, penalty='l2', solver='liblinear', 
            random_state=42, max_iter=1000,
            # class_weight='balanced'  # <--- 保持这个（来自上次对话的修改）
        )
        model.fit(X_train, y_train)
        
        # --- 验证集 (Test) 指标 (原逻辑) ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics["acc"].append(accuracy_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred, average='macro'))
        metrics["auc"].append(roc_auc_score(y_test, y_proba))
        metrics["sens"].append(recall_score(y_test, y_pred, pos_label=1, average='binary'))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP)
        metrics["spec"].append(specificity)

        # --- [!!! 核心新增 !!!] ---
        # --- 训练集 (Train) 指标 ---
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        
        metrics["acc_train"].append(accuracy_score(y_train, y_pred_train))
        metrics["auc_train"].append(roc_auc_score(y_train, y_proba_train))
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
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import RidgeClassifier

# (复用 test13.py 中的 LiquidNeuralNetwork 类)
class LiquidNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=500, activation='tanh', alpha=1.0, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        self.input_weights_ = np.random.normal(size=[n_features, self.n_hidden])
        self.bias_ = np.random.normal(size=[self.n_hidden])
        
        H = np.dot(X, self.input_weights_) + self.bias_
        if self.activation == 'tanh':
            H = np.tanh(H)
        elif self.activation == 'relu':
            H = np.maximum(H, 0)
            
        self.output_model_ = RidgeClassifier(alpha=self.alpha, class_weight='balanced')
        self.output_model_.fit(H, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        H = np.dot(X, self.input_weights_) + self.bias_
        if self.activation == 'tanh':
            H = np.tanh(H)
        elif self.activation == 'relu':
            H = np.maximum(H, 0)
        return self.output_model_.predict(H)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        H = np.dot(X, self.input_weights_) + self.bias_
        if self.activation == 'tanh':
            H = np.tanh(H)
        elif self.activation == 'relu':
            H = np.maximum(H, 0)
        d = self.output_model_.decision_function(H)
        if len(d.shape) == 1:
            prob = 1 / (1 + np.exp(-d))
            return np.vstack([1-prob, prob]).T
        else:
            exp_d = np.exp(d)
            return exp_d / exp_d.sum(axis=1, keepdims=True)

def evaluate_multiple_models(datasets, selected_idx):
    """
    使用选定的特征子集，在多个模型上进行训练和评估。
    返回一个嵌套字典结果。
    """
    X_train_full, y_train_full = datasets["Train"]
    
    # 仅使用选定特征
    X_train_sel = X_train_full[:, selected_idx]
    
    # 定义模型池 (均开启 class_weight='balanced' 以解决 Spec 低的问题)
    models = {
        'LR': LogisticRegressionCV(cv=5, penalty='l2', solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'XGB': GradientBoostingClassifier(random_state=42), # XGBoost原生库可能没装，用Sklearn的GBDT近似，注意GBDT没有简单的class_weight参数，通常需要sample_weight
        'NN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), # MLP也没有class_weight
        'LNN': LiquidNeuralNetwork(n_hidden=500, random_state=42)
    }

    all_model_results = {}

    for model_name, model in models.items():
        # 训练
        model.fit(X_train_sel, y_train_full)
        
        model_res = {}
        for ds_name, (X, y_true) in datasets.items():
            X_sel = X[:, selected_idx]
            
            y_pred = model.predict(X_sel)
            
            # 获取概率
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_sel)[:, 1]
                except:
                    # Fallback if probability fails
                    y_prob = model.decision_function(X_sel) if hasattr(model, "decision_function") else y_pred
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_sel)
            else:
                y_prob = y_pred

            # 计算指标
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except:
                auc = 0.5
            
            acc = float(accuracy_score(y_true, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            model_res[ds_name] = {"AUC": auc, "ACC": acc, "Sens": sens, "Spec": spec}
        
        all_model_results[model_name] = model_res

    return all_model_results