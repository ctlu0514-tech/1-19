
import os
import sys
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple, Optional

# Add current path to sys.path
sys.path.append(os.getcwd())

from run_validation_unified import load_and_split, preprocess_securely, evaluate_model, CONFIG, run_feature_selection

# Import GA utilities
from CDGAFS import compute_fisher_score, normalize_scores, semantic_pre_clustering, compute_similarity_matrix, initialize_population
from genetic_algorithm_utils import set_random_seed, allocate_quota_by_quality, perform_crossover_mutation, calculate_fitness, select_top_individuals

def main():
    # Use config from original script
    cfg = CONFIG
    print(f"Using Config: CSV={cfg['csv']}, Label={cfg['label']}, ID={cfg['id_col']}")
    
    # --- Load + split (Same logic as run_validation_unified) ---
    datasets_raw, feature_cols, resolved_mode = load_and_split(
        csv_path=cfg['csv'],
        label_col=cfg['label'],
        mode=cfg['mode'],
        center_col=cfg.get("center"),
        train_centers=None,
        id_col=cfg.get("id_col"),
        val_size=cfg['val_size'],
        random_state=cfg['seed'],
    )

    # Preprocess
    datasets, feature_names = preprocess_securely(datasets_raw, var_threshold=cfg['var_threshold'])
    X_train, y_train = datasets["Train"]
    
    print(f"Data Prep: X_train shape {X_train.shape}, Total Features {len(feature_names)}")
    
    # --- CDGAFS Setup ---
    set_random_seed(42)  # Consistency
    
    print("Step 1: Calculating Fisher Scores...")
    fisher_scores = compute_fisher_score(X_train, y_train)
    normalized_fisher_scores = normalize_scores(fisher_scores)
    
    print("Step 2: Semantic Clustering...")
    clusters, cluster_names = semantic_pre_clustering(feature_names)
    
    print("Step 3: Similarity Matrix...")
    similarity_matrix = compute_similarity_matrix(X_train)
    
    target_k = cfg['k']
    top_cluster_ratio = cfg['top_cluster_ratio']
    temperature = cfg['temperature']
    omega = cfg['cdgafs_omega']
    population_size = cfg['cdgafs_pop']
    
    print("Step 4: Allocating Quota...")
    cluster_quotas = allocate_quota_by_quality(
        clusters, normalized_fisher_scores, target_k, 
        top_cluster_ratio=top_cluster_ratio, temperature=temperature,
        cluster_names=cluster_names, similarity_matrix=similarity_matrix
    )
    
    print("Step 5: Initializing Population...")
    population = initialize_population(
        X_train.shape[1], clusters, omega, population_size, 
        cluster_quotas=cluster_quotas
    )
    
    # Fitness calculation for Gen 0
    fitness_values, ca_values, den_values = calculate_fitness(population, X_train, y_train, similarity_matrix, n_jobs=10)
    
    # --- Generation 1 ---
    print("\n[GA] Running Generation 1...")
    new_population, _, _ = perform_crossover_mutation(
        population, clusters, omega, X_train.shape[1], normalized_fisher_scores,
        cluster_quotas=cluster_quotas
    )
    new_fitness_values, new_ca_values, new_den_values = calculate_fitness(
        new_population, X_train, y_train, similarity_matrix, n_jobs=10
    )
    
    # Selection
    combined_population = population + new_population
    combined_fitness = fitness_values + new_fitness_values
    combined_ca = ca_values + new_ca_values
    combined_den = den_values + new_den_values

    population, fitness_values, ca_values, den_values = select_top_individuals(
        combined_population, combined_fitness, population_size,
        ca_values=combined_ca, den_values=combined_den
    )
    
    # Record Best of Gen 1
    best_idx_g1 = np.argmax(fitness_values)
    best_chrom_g1 = population[best_idx_g1]
    selected_idx_g1 = np.where(best_chrom_g1 == 1)[0]
    
    print(f"Gen 1 Best Found: Fitness={fitness_values[best_idx_g1]:.4f}, CA={ca_values[best_idx_g1]:.4f}")
    
    # --- Progress to Gen 100 ---
    print("\n[GA] Progressing to Generation 100...")
    for gen in range(1, 100):
        new_population, _, _ = perform_crossover_mutation(
            population, clusters, omega, X_train.shape[1], normalized_fisher_scores,
            cluster_quotas=cluster_quotas
        )
        new_fitness_values, new_ca_values, new_den_values = calculate_fitness(
            new_population, X_train, y_train, similarity_matrix, n_jobs=10
        )
        combined_population = population + new_population
        combined_fitness = fitness_values + new_fitness_values
        combined_ca = ca_values + new_ca_values
        combined_den = den_values + new_den_values
        population, fitness_values, ca_values, den_values = select_top_individuals(
            combined_population, combined_fitness, population_size,
            ca_values=combined_ca, den_values=combined_den
        )
        if (gen + 1) % 20 == 0:
            fit_max = max(fitness_values)
            best_i = np.argmax(fitness_values)
            print(f"Gen {gen + 1}: Best Fitness = {fit_max:.4f} (CA={ca_values[best_i]:.4f})")

    # Record Best of Gen 100
    best_idx_g100 = np.argmax(fitness_values)
    best_chrom_g100 = population[best_idx_g100]
    selected_idx_g100 = np.where(best_chrom_g100 == 1)[0]
    
    print(f"Gen 100 Best Found: Fitness={fitness_values[best_idx_g100]:.4f}, CA={ca_values[best_idx_g100]:.4f}")
    
    # --- Final Evaluation ---
    print("\n" + "="*60)
    print("FINAL PERFORMANCE COMPARISON (Logistic Regression)")
    print("="*60)
    
    print("\n[Gen 1 Best Results]")
    scores_g1 = evaluate_model(datasets, selected_idx_g1)
    print_scores(scores_g1)
    
    print("\n[Gen 100 Best Results]")
    scores_g100 = evaluate_model(datasets, selected_idx_g100)
    print_scores(scores_g100)
    print("="*60)

def print_scores(scores):
    header = f"{'Dataset':<18} | {'AUC':<8} | {'ACC':<8} | {'Sens':<8} | {'Spec':<8}"
    print(header)
    print("-" * len(header))
    for ds_name, res in scores.items():
        print(f"{ds_name:<18} | {res['AUC']:<8.4f} | {res['ACC']:<8.4f} | {res['Sens']:<8.4f} | {res['Spec']:<8.4f}")
    print("-" * len(header))

if __name__ == "__main__":
    main()
