import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, chisquare, spearmanr, skew, kurtosis, wasserstein_distance as scipy_wasserstein
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_regression

N_BINS = 10


def do_all_statistic_tests(real_samples: np.ndarray, gen_samples: np.ndarray, real_labels: np.ndarray = None, gen_labels: np.ndarray = None):
    #run all statistic tests I can think of
    real_samples = np.asarray(real_samples, dtype=np.float64)
    gen_samples = np.asarray(gen_samples, dtype=np.float64)

    results = {}
    results["ks_complement"] = kolmogorov_smirnov_test(real_samples, gen_samples)
    results["wasserstein_dist"] = wasserstein_distance(real_samples, gen_samples)
    results["jensenshannon_dist"] = jensenshannon_distance(real_samples, gen_samples)
    results["corr_matrix_diff"] = feature_corr_diff(real_samples, gen_samples)
    results["mmd_rbf"] = max_mean_discrepancy(real_samples, gen_samples)

    # Advanced Manifold & Coverage Tests
    pr_res = precision_recall_coverage(real_samples, gen_samples)
    results["precision"] = pr_res["precision"]
    results["recall_coverage"] = pr_res["recall_coverage"]

    svd_res = svd_spectrum_similarity(real_samples, gen_samples)
    results["svd_spectrum_similarity"] = svd_res["spectrum_similarity"]
    results["real_effective_rank"] = svd_res["real_effective_rank"]
    results["gen_effective_rank"] = svd_res["gen_effective_rank"]

    results["mi_matrix_diff"] = mutual_information_matrix_diff(real_samples, gen_samples)
    results["feature_importance_rank_corr"] = feature_importance_fidelity(real_samples, gen_samples)

    # Tail, Moments & Copulas
    tail_res = tail_and_moment_similarity(real_samples, gen_samples)
    results["skewness_diff"] = tail_res["mean_skewness_diff"]
    results["kurtosis_diff"] = tail_res["mean_kurtosis_diff"]
    results["q99_relative_error"] = tail_res["q99_relative_error"]

    results["tail_copula_diff"] = joint_tail_dependence_diff(real_samples, gen_samples)

    # Outliers & Tree Boundary Boxiness
    outlier_res = outlier_concordance_test(real_samples, gen_samples)
    results["outlier_concordance"] = outlier_res["outlier_concordance_score"]

    box_res = tree_boundary_cliff_test(real_samples, gen_samples)
    results["boxiness_cliff_score"] = box_res["smoothness_score"]

    # Clustering Alignment
    cluster_res = clustering_structure_alignment(real_samples, gen_samples)
    results["silhouette_gap"] = cluster_res["silhouette_gap"]
    results["centroid_distance"] = cluster_res["centroid_distance"]

    # Conditional Subgroup Fidelity (if labels present)
    if real_labels is not None:
        sub_res = conditional_subgroup_fidelity(real_samples, gen_samples, real_labels, gen_labels)
        results["subgroup_fidelity"] = sub_res["mean_subgroup_ks"]

    # Discriminator Detection AUCs
    results["detection_logistic_auc"] = detection_linear(real_samples, gen_samples)
    results["detection_rf_auc"] = detection_rf(real_samples, gen_samples)

    return results


def kolmogorov_smirnov_test(real_samples: np.ndarray, gen_samples: np.ndarray) -> float:
    """Average (1 - KS statistic) across features. 1.0 = identical distributions."""
    n_feats = real_samples.shape[1]
    complements = []
    for j in range(n_feats):
        stat, _ = ks_2samp(real_samples[:, j], gen_samples[:, j])
        complements.append(1.0 - stat)
    return float(np.mean(complements))


def wasserstein_distance(real_samples: np.ndarray, gen_samples: np.ndarray, normalized: bool = True) -> float:
    """Average Wasserstein distance across features."""
    n_feats = real_samples.shape[1]
    dists = []
    for j in range(n_feats):
        w_d = scipy_wasserstein(real_samples[:, j], gen_samples[:, j])
        if normalized:
            std = np.std(real_samples[:, j])
            scale = std if std > 1e-6 else 1.0
            w_d /= scale
        dists.append(w_d)
    return float(np.mean(dists))


def jensenshannon_distance(real_samples: np.ndarray, gen_samples: np.ndarray) -> float:
    """Average Jensen-Shannon distance across features (0 = identical, 1 = disjoint)."""
    real_feat = real_samples.T
    gen_feat = gen_samples.T
    res = []
    for i in range(real_feat.shape[0]):
        bins = min(N_BINS, max(2, len(np.unique(real_feat[i]))))
        all_vals = np.concatenate([real_feat[i], gen_feat[i]])
        _, bin_edges = np.histogram(all_vals, bins=bins)
        r_counts, _ = np.histogram(real_feat[i], bins=bin_edges)
        g_counts, _ = np.histogram(gen_feat[i], bins=bin_edges)

        r_prob = r_counts / (np.sum(r_counts) + 1e-10)
        g_prob = g_counts / (np.sum(g_counts) + 1e-10)
        js_dist = jensenshannon(r_prob, g_prob)
        if np.isnan(js_dist):
            js_dist = 0.0
        res.append(js_dist)
    return float(np.mean(res))


def feature_corr_diff(real_samples: np.ndarray, gen_samples: np.ndarray) -> float:
    """Normalized Frobenius norm of the difference between feature correlation matrices."""
    if real_samples.shape[1] < 2:
        return 0.0

    corr_real = np.nan_to_num(np.corrcoef(real_samples, rowvar=False), nan=0.0)
    corr_gen = np.nan_to_num(np.corrcoef(gen_samples, rowvar=False), nan=0.0)

    diff = corr_real - corr_gen
    n = real_samples.shape[1]
    max_norm = np.sqrt(2 * n * (n - 1)) if n > 1 else 1.0
    return float(np.linalg.norm(diff, 'fro') / max_norm)


def precision_recall_coverage(real: np.ndarray, gen: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Precision & Recall for Generative Distributions."""
    k = min(k, min(real.shape[0] - 1, gen.shape[0] - 1))
    if k < 1:
        return {"precision": 1.0, "recall_coverage": 1.0}

    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real)
    nn_gen = NearestNeighbors(n_neighbors=k + 1).fit(gen)

    real_radii = nn_real.kneighbors(real)[0][:, -1]
    gen_radii = nn_gen.kneighbors(gen)[0][:, -1]

    dists_gen_to_real, idx_real = nn_real.kneighbors(gen, n_neighbors=1)
    precision = np.mean(dists_gen_to_real[:, 0] <= real_radii[idx_real[:, 0]])

    dists_real_to_gen, idx_gen = nn_gen.kneighbors(real, n_neighbors=1)
    recall = np.mean(dists_real_to_gen[:, 0] <= gen_radii[idx_gen[:, 0]])

    return {
        "precision": float(precision),
        "recall_coverage": float(recall)
    }


def svd_spectrum_similarity(real: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    """Evaluates intrinsic dimensionality and manifold topology via SVD spectrum & Effective Rank."""
    real_c = real - np.mean(real, axis=0)
    gen_c = gen - np.mean(gen, axis=0)

    _, s_real, _ = np.linalg.svd(real_c, full_matrices=False)
    _, s_gen, _ = np.linalg.svd(gen_c, full_matrices=False)

    p_real = s_real / (np.sum(s_real) + 1e-12)
    p_gen = s_gen / (np.sum(s_gen) + 1e-12)

    norm_r = np.linalg.norm(p_real)
    norm_g = np.linalg.norm(p_gen)
    spectrum_sim = float(np.dot(p_real, p_gen) / (norm_r * norm_g + 1e-12))

    ent_r = -np.sum(p_real * np.log(p_real + 1e-12))
    ent_g = -np.sum(p_gen * np.log(p_gen + 1e-12))
    eff_rank_real = float(np.exp(ent_r))
    eff_rank_gen = float(np.exp(ent_g))

    return {
        "spectrum_similarity": float(np.clip(spectrum_sim, 0.0, 1.0)),
        "real_effective_rank": eff_rank_real,
        "gen_effective_rank": eff_rank_gen
    }


def tail_and_moment_similarity(real: np.ndarray, gen: np.ndarray) -> Dict[str, float]:
    """Evaluates higher-order moments (Skewness, Kurtosis) and extreme tail quantiles (1st & 99th percentiles)."""
    skew_real = skew(real, axis=0, nan_policy='omit')
    skew_gen = skew(gen, axis=0, nan_policy='omit')
    skew_diff = float(np.mean(np.abs(np.nan_to_num(skew_real - skew_gen))))

    kurt_real = kurtosis(real, axis=0, nan_policy='omit')
    kurt_gen = kurtosis(gen, axis=0, nan_policy='omit')
    kurt_diff = float(np.mean(np.abs(np.nan_to_num(kurt_real - kurt_gen))))

    q99_r = np.percentile(real, 99, axis=0)
    q99_g = np.percentile(gen, 99, axis=0)
    scale_99 = np.maximum(1e-6, np.abs(q99_r))
    q99_err = float(np.mean(np.abs(q99_r - q99_g) / scale_99))

    q01_r = np.percentile(real, 1, axis=0)
    q01_g = np.percentile(gen, 1, axis=0)
    scale_01 = np.maximum(1e-6, np.abs(q01_r))
    q01_err = float(np.mean(np.abs(q01_r - q01_g) / scale_01))

    return {
        "mean_skewness_diff": skew_diff,
        "mean_kurtosis_diff": kurt_diff,
        "q99_relative_error": q99_err,
        "q01_relative_error": q01_err
    }


# -----------------------------------------------------------------------------
# New Advanced Metrics: Outliers, Tail Copulas, Boxiness Cliff, Fairness
# -----------------------------------------------------------------------------

def outlier_concordance_test(real: np.ndarray, gen: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates whether synthetic data contains realistic anomalies without hallucinating
    unphysical outliers or dropping real outlier modes.
    Fits Isolation Forest on real data and compares anomaly score distributions.
    """
    n_samples = min(1000, min(real.shape[0], gen.shape[0]))
    X_r = real[:n_samples]
    X_g = gen[:n_samples]

    iso = IsolationForest(n_estimators=60, random_state=42, n_jobs=-1).fit(X_r)
    scores_r = iso.score_samples(X_r)
    scores_g = iso.score_samples(X_g)

    # 1.0 - KS statistic of anomaly scores: 1.0 = identical outlier profiles
    stat, _ = ks_2samp(scores_r, scores_g)
    concordance = float(max(0.0, 1.0 - stat))

    # Anomaly rate comparison at 5th percentile threshold
    thresh = np.percentile(scores_r, 5)
    anom_rate_r = float(np.mean(scores_r < thresh))
    anom_rate_g = float(np.mean(scores_g < thresh))

    return {
        "outlier_concordance_score": concordance,
        "real_anomaly_rate": anom_rate_r,
        "gen_anomaly_rate": anom_rate_g,
        "anomaly_rate_gap": float(abs(anom_rate_r - anom_rate_g))
    }


def joint_tail_dependence_diff(real: np.ndarray, gen: np.ndarray, quantile: float = 0.05, max_dim: int = 10) -> float:
    """
    Measures joint tail copula dependence (co-extreme events).
    Evaluates whether features peak or crash together at the 5th and 95th percentiles.
    0.0 = perfect preservation of joint extreme risks.
    """
    d = min(real.shape[1], max_dim)
    if d < 2:
        return 0.0

    X_r = real[:, :d]
    X_g = gen[:, :d]

    q_low_r = np.percentile(X_r, quantile * 100.0, axis=0)
    q_low_g = np.percentile(X_g, quantile * 100.0, axis=0)

    q_high_r = np.percentile(X_r, (1.0 - quantile) * 100.0, axis=0)
    q_high_g = np.percentile(X_g, (1.0 - quantile) * 100.0, axis=0)

    diffs = []
    for j in range(d):
        for k in range(j + 1, d):
            # Lower tail joint probability
            p_low_r = np.mean((X_r[:, j] <= q_low_r[j]) & (X_r[:, k] <= q_low_r[k])) / quantile
            p_low_g = np.mean((X_g[:, j] <= q_low_g[j]) & (X_g[:, k] <= q_low_g[k])) / quantile

            # Upper tail joint probability
            p_high_r = np.mean((X_r[:, j] >= q_high_r[j]) & (X_r[:, k] >= q_high_r[k])) / quantile
            p_high_g = np.mean((X_g[:, j] >= q_high_g[j]) & (X_g[:, k] >= q_high_g[k])) / quantile

            diffs.append(abs(p_low_r - p_low_g) + abs(p_high_r - p_high_g))

    return float(np.mean(diffs)) if diffs else 0.0


def tree_boundary_cliff_test(real: np.ndarray, gen: np.ndarray, n_bins: int = 25) -> Dict[str, float]:
    """
    Detects rectangular grid / boxiness artifacts specific to decision tree generative models.
    Measures whether synthetic data has unnatural abrupt density cliffs or step functions
    compared to the smooth distribution of real data.
    """
    n_feats = min(real.shape[1], 10)
    gradient_ratios = []

    for j in range(n_feats):
        all_vals = np.concatenate([real[:, j], gen[:, j]])
        _, edges = np.histogram(all_vals, bins=n_bins)

        h_r, _ = np.histogram(real[:, j], bins=edges, density=True)
        h_g, _ = np.histogram(gen[:, j], bins=edges, density=True)

        # Compute 1st order differences (density gradients)
        grad_r = np.abs(np.diff(h_r))
        grad_g = np.abs(np.diff(h_g))

        max_grad_r = np.max(grad_r) if np.max(grad_r) > 1e-6 else 1.0
        max_grad_g = np.max(grad_g)

        # Peak cliff ratio: if generator has sharp step function jumps, max_grad_g >> max_grad_r
        cliff_ratio = max_grad_g / max_grad_r
        gradient_ratios.append(cliff_ratio)

    avg_cliff = float(np.mean(gradient_ratios))
    # Smoothness score: 1.0 when cliff ratio is close to 1.0 (smooth, matching real gradients)
    smoothness = float(np.clip(1.0 / (1.0 + abs(avg_cliff - 1.0)), 0.0, 1.0))

    return {
        "smoothness_score": smoothness,
        "mean_cliff_ratio": avg_cliff
    }


def clustering_structure_alignment(real: np.ndarray, gen: np.ndarray, n_clusters: int = 3) -> Dict[str, float]:
    """Evaluates unsupervised clustering geometry, Silhouette scores, and centroid alignments."""
    n_clusters = min(n_clusters, min(real.shape[0] - 1, gen.shape[0] - 1))
    if n_clusters < 2:
        return {"silhouette_gap": 0.0, "centroid_distance": 0.0}

    try:
        km_real = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(real)
        km_gen = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(gen)

        sil_real = float(silhouette_score(real, km_real.labels_))
        sil_gen = float(silhouette_score(gen, km_gen.labels_))
        sil_gap = float(abs(sil_real - sil_gen))

        c_r = km_real.cluster_centers_
        c_g = km_gen.cluster_centers_
        order_r = np.argsort(c_r[:, 0])
        order_g = np.argsort(c_g[:, 0])
        c_diff = np.linalg.norm(c_r[order_r] - c_g[order_g], 'fro') / float(n_clusters)

        return {
            "silhouette_real": sil_real,
            "silhouette_gen": sil_gen,
            "silhouette_gap": sil_gap,
            "centroid_distance": float(c_diff)
        }
    except Exception:
        return {"silhouette_gap": 0.0, "centroid_distance": 0.0}


def conditional_subgroup_fidelity(
    real: np.ndarray,
    gen: np.ndarray,
    real_labels: np.ndarray,
    gen_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Evaluates fidelity within subgroups or minority classes."""
    classes = np.unique(real_labels)
    subgroup_ks = []

    if gen_labels is None or len(gen_labels) != gen.shape[0]:
        clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42).fit(real, real_labels)
        gen_labels = clf.predict(gen)

    for cls in classes:
        mask_r = (real_labels == cls)
        mask_g = (gen_labels == cls)

        if np.sum(mask_r) >= 5 and np.sum(mask_g) >= 5:
            ks_sub = kolmogorov_smirnov_test(real[mask_r], gen[mask_g])
            subgroup_ks.append(ks_sub)
        else:
            subgroup_ks.append(0.0)

    mean_ks = float(np.mean(subgroup_ks)) if subgroup_ks else 1.0

    return {
        "mean_subgroup_ks": mean_ks,
        "classes_evaluated": [int(c) for c in classes],
        "subgroup_ks_scores": [float(s) for s in subgroup_ks]
    }


def mutual_information_matrix_diff(real: np.ndarray, gen: np.ndarray, max_dim: int = 15) -> float:
    """Normalized Frobenius difference of pairwise Mutual Information matrices."""
    d = min(real.shape[1], max_dim)
    n_samples = min(500, min(real.shape[0], gen.shape[0]))

    X_r = real[:n_samples, :d]
    X_g = gen[:n_samples, :d]

    mi_real = np.zeros((d, d))
    mi_gen = np.zeros((d, d))

    for i in range(d):
        try:
            mi_real[i, :] = mutual_info_regression(X_r, X_r[:, i], random_state=42)
            mi_gen[i, :] = mutual_info_regression(X_g, X_g[:, i], random_state=42)
        except Exception:
            pass

    max_val = max(np.max(mi_real), np.max(mi_gen), 1e-5)
    diff = np.linalg.norm((mi_real - mi_gen) / max_val, 'fro') / (d * d)
    return float(diff)


def feature_importance_fidelity(real: np.ndarray, gen: np.ndarray, target_col: int = -1) -> float:
    """Verifies that predictive models learn the same feature hierarchies on real vs synthetic data."""
    if real.shape[1] < 3:
        return 1.0

    X_r = np.delete(real, target_col, axis=1)
    y_r = real[:, target_col]
    X_g = np.delete(gen, target_col, axis=1)
    y_g = gen[:, target_col]

    try:
        rf_real = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_r, y_r)
        rf_gen = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X_g, y_g)

        corr, _ = spearmanr(rf_real.feature_importances_, rf_gen.feature_importances_)
        return float(corr) if not np.isnan(corr) else 1.0
    except Exception:
        return 1.0


def max_mean_discrepancy(real_samples: np.ndarray, gen_samples: np.ndarray, max_samples: int = 1000) -> float:
    """MMD with RBF kernel (0.0 = identical distributions)."""
    n_r = min(real_samples.shape[0], max_samples)
    n_g = min(gen_samples.shape[0], max_samples)

    X = real_samples[np.random.choice(real_samples.shape[0], size=n_r, replace=False)]
    Y = gen_samples[np.random.choice(gen_samples.shape[0], size=n_g, replace=False)]

    gamma = 1.0 / float(X.shape[1])
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)

    mmd = XX.mean() + YY.mean() - 2.0 * XY.mean()
    return float(max(0.0, mmd))


def _eval_classifier_auc(clf_factory, real_samples: np.ndarray, gen_samples: np.ndarray, cv: int = 5) -> float:
    """Helper to evaluate cross-validated discriminator ROC-AUC."""
    n_r = real_samples.shape[0]
    n_g = gen_samples.shape[0]

    X = np.vstack([real_samples, gen_samples])
    y = np.hstack([np.zeros(n_r), np.ones(n_g)])

    skf = StratifiedKFold(n_splits=min(cv, min(n_r, n_g)), shuffle=True, random_state=42)
    aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clf_factory()
        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            y_pred = clf.predict_proba(X_test)[:, 1]
        else:
            y_pred = clf.predict(X_test)

        try:
            auc = roc_auc_score(y_test, y_pred)
            aucs.append(auc)
        except Exception:
            pass

    return float(np.mean(aucs)) if len(aucs) > 0 else 0.5


def detection_linear(real_samples: np.ndarray, gen_samples: np.ndarray) -> float:
    """Logistic Regression Discriminator AUC (0.50 = perfectly indistinguishable)."""
    return _eval_classifier_auc(lambda: LogisticRegression(max_iter=500, random_state=42), real_samples, gen_samples)


def detection_rf(real_samples: np.ndarray, gen_samples: np.ndarray) -> float:
    """Random Forest Discriminator AUC (0.50 = perfectly indistinguishable)."""
    return _eval_classifier_auc(lambda: RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1), real_samples, gen_samples)