import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# --- 定义液态神经网络 (Liquid Neural Network) 的近似实现 ---
# 这里使用储备池计算 (Reservoir Computing) / 极限学习机 (ELM) 的思路
# 即：固定随机权重的隐藏层 + 可训练的线性输出层
class LiquidNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden=500, activation='tanh', alpha=1.0, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # 随机初始化输入权重 (模拟液态储备池)
        self.input_weights_ = np.random.normal(size=[n_features, self.n_hidden])
        self.bias_ = np.random.normal(size=[self.n_hidden])
        
        # 投影到隐藏层
        H = np.dot(X, self.input_weights_) + self.bias_
        if self.activation == 'tanh':
            H = np.tanh(H)
        elif self.activation == 'relu':
            H = np.maximum(H, 0)
            
        # 使用 Ridge 回归训练输出层
        self.output_model_ = RidgeClassifier(alpha=self.alpha)
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
        # 使用决策函数估算概率
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

# --- 主程序 ---

# 1. 读取数据
file_path = '/data/qh_20T_share_file/lct/CT67/localdata/prostate_features_with_label.csv' # 请确保文件在当前目录下
df = pd.read_csv(file_path)

# 2. 数据清洗
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

X = df.drop(columns=['label'])
y = df['label']

# 填充缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 3. 特征筛选 (关键步骤：从10000+特征中选出50个，防止过拟合和运行缓慢)
fs = SelectKBest(score_func=f_classif, k=50)
X_selected = fs.fit_transform(X_imputed, y)

# 获取被选中的特征索引
selected_indices = fs.get_support(indices=True)

# 根据索引获取原始特征名称
selected_features = X.columns[selected_indices].tolist()

print("\n选出的 50 个特征如下：")
for i, feat in enumerate(selected_features):
    print(f"{i+1}. {feat}")

# 4. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 5. 定义模型
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': GradientBoostingClassifier(random_state=42), # 使用 GradientBoosting 近似
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    'Liquid Neural Network': LiquidNeuralNetwork(n_hidden=500, random_state=42)
}

# 6. 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print(f"{'Model':<25} {'ACC (Tr/Te)':<15} {'AUC (Tr/Te)':<15} {'Sens (Tr/Te)':<15} {'Spec (Tr/Te)':<15}")
print("-" * 85)

for name, model in models.items():
    acc_scores, train_acc_scores = [], []
    auc_scores, train_auc_scores = [], []
    sen_scores, train_sen_scores = [], []
    spe_scores, train_spe_scores = [], []
    
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        
        # 1. 训练集指标
        y_train_pred = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            y_train_prob = model.predict_proba(X_train)[:, 1]
        else:
            y_train_prob = model.decision_function(X_train)
        
        tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred).ravel()
        acc_tr = accuracy_score(y_train, y_train_pred)
        auc_tr = roc_auc_score(y_train, y_train_prob)
        sen_tr = tp_tr / (tp_tr + fn_tr) if (tp_tr + fn_tr) > 0 else 0
        spe_tr = tn_tr / (tn_tr + fp_tr) if (tn_tr + fp_tr) > 0 else 0
        
        train_acc_scores.append(acc_tr)
        train_auc_scores.append(auc_tr)
        train_sen_scores.append(sen_tr)
        train_spe_scores.append(spe_tr)

        # 2. 测试集指标
        y_test_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_test_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = model.decision_function(X_test)
            
        tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred).ravel()
        acc_te = accuracy_score(y_test, y_test_pred)
        auc_te = roc_auc_score(y_test, y_test_prob)
        sen_te = tp_te / (tp_te + fn_te) if (tp_te + fn_te) > 0 else 0
        spe_te = tn_te / (tn_te + fp_te) if (tn_te + fp_te) > 0 else 0
        
        acc_scores.append(acc_te)
        auc_scores.append(auc_te)
        sen_scores.append(sen_te)
        spe_scores.append(spe_te)
    
    # 汇总结果
    res = {
        'Model': name,
        'Train_ACC': np.mean(train_acc_scores),
        'Test_ACC': np.mean(acc_scores),
        'Train_AUC': np.mean(train_auc_scores),
        'Test_AUC': np.mean(auc_scores),
        'Train_Sens': np.mean(train_sen_scores),
        'Test_Sens': np.mean(sen_scores),
        'Train_Spec': np.mean(train_spe_scores),
        'Test_Spec': np.mean(spe_scores)
    }
    results.append(res)
    print(f"{name:<25} {res['Train_ACC']:.3f}/{res['Test_ACC']:.3f}   {res['Train_AUC']:.3f}/{res['Test_AUC']:.3f}   {res['Train_Sens']:.3f}/{res['Test_Sens']:.3f}   {res['Train_Spec']:.3f}/{res['Test_Spec']:.3f}")

# 将结果转换为DataFrame方便查看
df_results = pd.DataFrame(results)