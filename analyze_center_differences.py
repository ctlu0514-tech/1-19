
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os

# CONFIG
csv_path = 'Merged_All_Centers.csv'
train_center = 'FuYi'

def analyze():
    df = pd.read_csv(csv_path)
    
    # Features
    meta = ['Sample_ID', 'Center_Source', 'label', 'Center']
    feats = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    
    # Preprocess (Impute + Scale based on ALL data for visualization fairness, 
    # or arguably Scale on Train and transform others to see the shift. 
    # To see the "Shift", we should scale based on Train parameters.)
    
    X = df[feats].values
    y = df['label'].values
    centers = df['Center_Source'].values
    
    # Train Mask
    train_mask = (centers == train_center)
    
    # Impute
    imp = SimpleImputer()
    imp.fit(X[train_mask])
    X_imp = imp.transform(X)
    
    # Scale (Fit on Train)
    scaler = StandardScaler()
    scaler.fit(X_imp[train_mask])
    X_scaled = scaler.transform(X_imp)
    
    # 1. Covariance Shift Analysis
    print("=== Covariance Matrix Difference (Frobenius Norm) vs FuYi ===")
    
    # Cov of Train
    cov_train = np.cov(X_scaled[train_mask], rowvar=False)
    
    diffs = {}
    for c in np.unique(centers):
        if c == train_center: continue
        mask = (centers == c)
        if np.sum(mask) < 2: continue
        
        # Cov of Test Center
        cov_test = np.cov(X_scaled[mask], rowvar=False)
        
        # Difference
        diff = np.linalg.norm(cov_train - cov_test, ord='fro')
        diffs[c] = diff
        print(f"{c}: {diff:.4f}")
        
    print("\nInterpretation: Larger val => Different Feature Correlations. ComBat/Self-Norm can't fix this.")

    # 2. t-SNE Visualization
    print("\nRunning t-SNE (this may take a moment)...")
    # Reduce dim first with PCA to speed up
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=centers, style=centers, alpha=0.7)
    plt.title(f't-SNE of Hematoma Features (Train Norm: {train_center})')
    plt.savefig('tsne_distribution.png')
    print("Saved t-SNE plot to tsne_distribution.png")
    
    # Calculate Centroid Distances in t-SNE space (just for intuition)
    tsne_df = pd.DataFrame(X_tsne, columns=['D1', 'D2'])
    tsne_df['Center'] = centers
    centroids = tsne_df.groupby('Center').mean()
    print("\nt-SNE Centroids:")
    print(centroids)

if __name__ == "__main__":
    analyze()
