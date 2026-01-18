import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
csv_path = 'Merged_All_Centers.csv'
df = pd.read_csv(csv_path)

# Centers and Labels
center_col = 'Center_Source'
label_col = 'label'

print("=== 1. Class Distribution per Center ===")
print(df.groupby([center_col, label_col]).size().unstack(fill_value=0))

# Prepare features
# Exclude non-feature columns
meta_cols = ['Sample_ID', 'Center_Source', 'label', 'Center']
feature_cols = [c for c in df.columns if c not in meta_cols and not c.startswith('diagnostics_')]

X = df[feature_cols]
y = df[label_col]
centers = df[center_col]

# Impute and Scale
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

# PCA
print("\n=== 2. PCA Analysis (Top 2 Components) ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Center'] = centers.values

# Calculate centroids of each center in PC space
centroids = pca_df.groupby('Center').mean()
print("Centroids in PC1/PC2 space:")
print(centroids)

# Calculate average distance between centers
from scipy.spatial.distance import pdist, squareform
dists = squareform(pdist(centroids))
dist_df = pd.DataFrame(dists, index=centroids.index, columns=centroids.index)
print("\nEuclidean Distance between Center Centroids (PC1/PC2):")
print(dist_df)

# Explain variance
print(f"\nExplained Variance Ratio: {pca.explained_variance_ratio_}")
