#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
前列腺特征数据集全特征分类性能测试脚本
使用多种机器学习和深度学习分类器进行评估

包含分类器:
- XGBoost
- K近邻 (KNN)
- 随机森林 (Random Forest)
- 支持向量机 (SVM)
- 逻辑回归 (Logistic Regression)
- 梯度提升 (Gradient Boosting)
- 多层感知机 (MLP - 深度学习)
- 自定义深度神经网络 (PyTorch)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.pipeline import Pipeline

# 机器学习分类器
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 尝试导入XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: XGBoost未安装，将跳过XGBoost分类器")

# 深度学习
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time
from datetime import datetime


def load_data(filepath):
    """加载数据集"""
    print(f"正在加载数据: {filepath}")
    df = pd.read_csv(filepath)
    
    # 假设最后一列是标签列，或者名为'label'的列
    if 'label' in df.columns:
        X = df.drop('label', axis=1)
        y = df['label']
    elif 'Label' in df.columns:
        X = df.drop('Label', axis=1)
        y = df['Label']
    else:
        # 尝试找第一个非数值列作为标签
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            label_col = non_numeric_cols[0]
            X = df.drop(label_col, axis=1)
            y = df[label_col]
        else:
            # 假设第一列是标签
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]
    
    # 去掉ID列（避免把ID当作特征）
    dropped_id_cols = []
    for id_col in ['ID', 'id']:
        if id_col in X.columns:
            X = X.drop(id_col, axis=1)
            dropped_id_cols.append(id_col)
    if dropped_id_cols:
        print(f"已移除ID列: {dropped_id_cols}")
    
    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    # 替换无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    print(f"数据集形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    print(f"类别标签: {le.classes_}")
    
    return X.values, y, le.classes_


class DeepClassifier(nn.Module):
    """自定义深度神经网络分类器"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], num_classes=2, dropout=0.3):
        super(DeepClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_deep_network(X_train, y_train, X_test, y_test, input_dim, num_classes=2,
                       epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
    """训练深度神经网络"""
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = DeepClassifier(input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # 训练
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    
    # 评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
        
        # 获取概率(用于AUC计算)
        probs = torch.softmax(outputs, dim=1)
        if num_classes == 2:
            y_prob = probs[:, 1].cpu().numpy()
        else:
            y_prob = probs.cpu().numpy()
    
    return y_pred, y_prob


def evaluate_classifier(y_true, y_pred, y_prob=None, num_classes=2):
    """评估分类器性能"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # 计算AUC
    if y_prob is not None:
        try:
            if num_classes == 2:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = np.nan
    else:
        metrics['AUC'] = np.nan
    
    return metrics


def run_experiments(X, y, n_splits=5, random_state=42):
    """运行所有分类器实验"""
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    num_classes = len(np.unique(y))
    print(f"类别数量: {num_classes}")
    
    # 定义分类器
    classifiers = {}
    
    # 条件添加XGBoost
    if HAS_XGBOOST:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=random_state, n_jobs=-1
        )
    
    # 其他分类器
    classifiers.update({
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2,
            random_state=random_state, n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5, weights='distance', n_jobs=-1
        ),
        # SVM已移除（高维数据运行太慢）
        'Logistic Regression': LogisticRegression(
            max_iter=1000, solver='lbfgs',
            random_state=random_state, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=random_state
        ),
        'MLP (Sklearn)': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            solver='adam', max_iter=500, early_stopping=True,
            random_state=random_state
        ),
    })
    
    # 存储结果
    results = {name: {'Accuracy': [], 'Precision': [], 'Recall': [], 
                      'F1-Score': [], 'AUC': [], 'Time': []} 
               for name in list(classifiers.keys()) + ['Deep Neural Network']}
    
    # 交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    print("\n" + "="*80)
    print("开始交叉验证实验")
    print("="*80)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 测试每个机器学习分类器
        for name, clf in classifiers.items():
            print(f"  训练 {name}...", end=' ')
            start_time = time.time()
            
            try:
                clf_copy = clf.__class__(**clf.get_params())
                clf_copy.fit(X_train_scaled, y_train)
                y_pred = clf_copy.predict(X_test_scaled)
                
                # 获取概率预测
                if hasattr(clf_copy, 'predict_proba'):
                    y_prob = clf_copy.predict_proba(X_test_scaled)
                    if num_classes == 2:
                        y_prob = y_prob[:, 1]
                else:
                    y_prob = None
                
                elapsed_time = time.time() - start_time
                metrics = evaluate_classifier(y_test, y_pred, y_prob, num_classes)
                
                for metric_name, value in metrics.items():
                    results[name][metric_name].append(value)
                results[name]['Time'].append(elapsed_time)
                
                print(f"完成 (Acc: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}, Time: {elapsed_time:.2f}s)")
                
            except Exception as e:
                print(f"失败: {e}")
                for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                    results[name][metric_name].append(np.nan)
                results[name]['Time'].append(np.nan)
        
        # 测试深度神经网络
        print(f"  训练 Deep Neural Network...", end=' ')
        start_time = time.time()
        
        try:
            y_pred, y_prob = train_deep_network(
                X_train_scaled, y_train, X_test_scaled, y_test,
                input_dim=X_train_scaled.shape[1], num_classes=num_classes,
                epochs=100, batch_size=32, device=device
            )
            
            elapsed_time = time.time() - start_time
            metrics = evaluate_classifier(y_test, y_pred, y_prob, num_classes)
            
            for metric_name, value in metrics.items():
                results['Deep Neural Network'][metric_name].append(value)
            results['Deep Neural Network']['Time'].append(elapsed_time)
            
            print(f"完成 (Acc: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}, Time: {elapsed_time:.2f}s)")
            
        except Exception as e:
            print(f"失败: {e}")
            for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                results['Deep Neural Network'][metric_name].append(np.nan)
            results['Deep Neural Network']['Time'].append(np.nan)
    
    return results


def print_results(results):
    """打印结果摘要"""
    print("\n" + "="*100)
    print("分类性能结果摘要 (均值 ± 标准差)")
    print("="*100)
    
    # 创建结果DataFrame
    summary_data = []
    
    for name, metrics in results.items():
        row = {'Classifier': name}
        for metric_name, values in metrics.items():
            values = [v for v in values if not np.isnan(v)]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[f'{metric_name}_mean'] = mean_val
                row[f'{metric_name}_std'] = std_val
                row[metric_name] = f"{mean_val:.4f} ± {std_val:.4f}"
            else:
                row[f'{metric_name}_mean'] = np.nan
                row[f'{metric_name}_std'] = np.nan
                row[metric_name] = "N/A"
        summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # 按F1-Score排序
    df_summary = df_summary.sort_values('F1-Score_mean', ascending=False)
    
    # 打印表格
    print(f"\n{'Classifier':<25} {'Accuracy':<18} {'Precision':<18} {'Recall':<18} {'F1-Score':<18} {'AUC':<18} {'Time(s)':<15}")
    print("-"*130)
    
    for _, row in df_summary.iterrows():
        print(f"{row['Classifier']:<25} {row['Accuracy']:<18} {row['Precision']:<18} {row['Recall']:<18} {row['F1-Score']:<18} {row['AUC']:<18} {row['Time']:<15}")
    
    print("\n" + "="*100)
    print("最佳分类器 (按F1-Score):")
    best_row = df_summary.iloc[0]
    print(f"  {best_row['Classifier']}: F1-Score = {best_row['F1-Score']}")
    print("="*100)
    
    return df_summary


def save_results(results, output_path):
    """保存结果到CSV文件"""
    summary_data = []
    
    for name, metrics in results.items():
        for fold_idx in range(len(metrics['Accuracy'])):
            row = {
                'Classifier': name,
                'Fold': fold_idx + 1
            }
            for metric_name, values in metrics.items():
                row[metric_name] = values[fold_idx] if fold_idx < len(values) else np.nan
            summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"\n详细结果已保存到: {output_path}")
    
    return df


def main():
    """主函数"""
    print("="*80)
    print("前列腺特征数据集分类性能测试")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 数据路径
    data_path = '/data/qh_20T_share_file/lct/CT67/localdata/radiomics_features_with_label_without_pet.csv'
    
    # 加载数据
    X, y, class_names = load_data(data_path)
    
    # 运行实验
    results = run_experiments(X, y, n_splits=5, random_state=42)
    
    # 打印结果
    df_summary = print_results(results)
    
    # 保存结果
    output_path = 'localdata/classification_results.csv'
    save_results(results, output_path)
    
    # 保存摘要
    summary_output_path = 'localdata/classification_summary.csv'
    
    # 创建摘要数据
    summary_rows = []
    for name, metrics in results.items():
        row = {'Classifier': name}
        for metric_name, values in metrics.items():
            values = [v for v in values if not np.isnan(v)]
            if values:
                row[f'{metric_name}_mean'] = np.mean(values)
                row[f'{metric_name}_std'] = np.std(values)
        summary_rows.append(row)
    
    df_summary_save = pd.DataFrame(summary_rows)
    df_summary_save = df_summary_save.sort_values('F1-Score_mean', ascending=False)
    df_summary_save.to_csv(summary_output_path, index=False)
    print(f"摘要结果已保存到: {summary_output_path}")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
