import warnings
import numpy as np
from typing import Dict, Any, Optional, Union, List, Callable

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import CONFIG

from src.compare.sanity_checks import do_all_sanity_checks
from src.compare.statistic_sim import (
    kolmogorov_smirnov_test,
    wasserstein_distance,
    jensenshannon_distance,
    feature_corr_diff,
    max_mean_discrepancy,
    precision_recall_coverage,
    svd_spectrum_similarity,
    mutual_information_matrix_diff,
    feature_importance_fidelity,
    tail_and_moment_similarity,
    joint_tail_dependence_diff,
    outlier_concordance_test,
    tree_boundary_cliff_test,
    clustering_structure_alignment,
    conditional_subgroup_fidelity,
    detection_linear,
    detection_rf
)

warnings.filterwarnings("ignore")


class GEN_VS_REAL_Evaluator:
    def evaluate(self, real_data: np.ndarray, real_labels: np.ndarray = None, gen_data: np.ndarray = None,
                 gen_labels: np.ndarray = None, test_data: np.ndarray = None, sample_obj = None) -> Dict[str, Any]:

        real_data = np.asarray(real_data, dtype=np.float64)
        gen_data = np.asarray(gen_data, dtype=np.float64)

        report = {}

        # Tier 1
        sanity_res = do_all_sanity_checks(real_samples=real_data, gen_samples=gen_data, test_samples=test_data)
        ks_comp = kolmogorov_smirnov_test(real_data, gen_data)
        w_dist = wasserstein_distance(real_data, gen_data, normalized=True)
        js_dist = jensenshannon_distance(real_data, gen_data)

        report["tier_1_gatekeepers"] = {

        }

        # ---------------------------------------------------------
        # Tier 2: Core Generative Quality
        # ---------------------------------------------------------
        pr_res = precision_recall_coverage(real_data, gen_data, k=5)
        corr_diff = feature_corr_diff(real_data, gen_data)
        mi_diff = mutual_information_matrix_diff(real_data, gen_data)

        # Heterogeneous Multi-Model TSTR Utility & Augmentation Gain
        if real_labels is not None and len(np.unique(real_labels)) > 1:
            multi_tstr = self._eval_heterogeneous_ml_utility(real_data, gen_data, real_labels, gen_labels)
            aug_util = self._eval_data_augmentation_utility(real_data, gen_data, real_labels, gen_labels)
        else:
            multi_tstr = None
            aug_util = None

        report["tier_2_core_quality"] = {
            "alpha_precision": pr_res["precision"],
            "beta_recall": pr_res["recall_coverage"],
            "corr_matrix_diff": corr_diff,
            "mutual_info_diff": mi_diff,
            "heterogeneous_tstr": multi_tstr,
            "data_augmentation_utility": aug_util
        }

        # ---------------------------------------------------------
        # Tier 3: Structural & Geometric Integrity
        # ---------------------------------------------------------
        disc_log_auc = detection_linear(real_data, gen_data)
        disc_rf_auc = detection_rf(real_data, gen_data)
        indist_log = max(0.0, 1.0 - 2.0 * abs(disc_log_auc - 0.5))
        indist_rf = max(0.0, 1.0 - 2.0 * abs(disc_rf_auc - 0.5))
        mean_indist = 0.5 * (indist_log + indist_rf)

        svd_res = svd_spectrum_similarity(real_data, gen_data)
        tail_res = tail_and_moment_similarity(real_data, gen_data)
        tail_copula_d = joint_tail_dependence_diff(real_data, gen_data)
        box_res = tree_boundary_cliff_test(real_data, gen_data)
        mmd_val = max_mean_discrepancy(real_data, gen_data)
        feat_rank_corr = feature_importance_fidelity(real_data, gen_data)

        report["tier_3_structural_integrity"] = {
            "discriminator_auc_rf": disc_rf_auc,
            "discriminator_auc_logistic": disc_log_auc,
            "indistinguishability_score": mean_indist,
            "svd_spectrum_similarity": svd_res["spectrum_similarity"],
            "real_effective_rank": svd_res["real_effective_rank"],
            "gen_effective_rank": svd_res["gen_effective_rank"],
            "tails": tail_res,
            "tail_copula_diff": tail_copula_d,
            "boxiness_smoothness": box_res["smoothness_score"],
            "mmd_rbf": mmd_val,
            "feature_importance_rank_corr": feat_rank_corr
        }

        # ---------------------------------------------------------
        # Tier 4: Contextual & Specialized Diagnostics
        # ---------------------------------------------------------
        outlier_res = outlier_concordance_test(real_data, gen_data)
        cluster_res = clustering_structure_alignment(real_data, gen_data)

        if real_labels is not None and len(np.unique(real_labels)) > 1:
            sub_res = conditional_subgroup_fidelity(real_data, gen_data, real_labels, gen_labels)
        else:
            sub_res = None

        chunk_info = CONFIG.GLOBAL_CHUNK_INFO

        if chunk_info is not None and chunk_info.n_chunks > 1 and sample_obj is not None:
            chunk_gen = {"cross_chunk_ks": [], "cross_chunk_wasserstein": [], "overfitting_gap": []}
            for chunk in range(chunk_info.n_chunks):
                sample_obj.load_chunk(chunk)
                chunk_samples = sample_obj.samples
                cross_ks = kolmogorov_smirnov_test(chunk_samples, gen_data)
                cross_w = wasserstein_distance(chunk_samples, gen_data, normalized=True)
                chunk_gen["cross_chunk_ks"].append(cross_ks)
                chunk_gen["cross_chunk_wasserstein"].append(cross_w)
                chunk_gen["overfitting_gap"].append(float(max(0.0, ks_comp - cross_ks)))
        else:
            chunk_gen = None

        report["tier_4_contextual_diagnostics"] = {
            "outlier_concordance": outlier_res["outlier_concordance_score"],
            "clustering": cluster_res,
            "subgroup_fidelity": sub_res,
            "cross_chunk_generalization": chunk_gen
        }

        # ---------------------------------------------------------
        # Composite Quality Score (0 to 100)
        # ---------------------------------------------------------
        s_ks = ks_comp
        s_corr = max(0.0, 1.0 - corr_diff)
        s_prec = pr_res["precision"]
        s_rec = pr_res["recall_coverage"]
        s_svd = svd_res["spectrum_similarity"]
        s_tail = max(0.0, 1.0 - 0.5 * (tail_res["q99_relative_error"] + tail_res["q01_relative_error"]))
        s_out = outlier_res["outlier_concordance_score"]
        s_box = box_res["smoothness_score"]
        s_indist = mean_indist
        s_priv = 1.0

        if multi_tstr is not None:
            s_tstr = multi_tstr["mean_utility_ratio"]
        else:
            s_tstr = 1.0

        overall_score = (
            0.15 * s_ks +
            0.15 * s_tstr +
            0.12 * s_corr +
            0.10 * s_rec +
            0.10 * s_prec +
            0.10 * s_svd +
            0.08 * s_tail +
            0.07 * s_out +
            0.06 * s_box +
            0.07 * s_priv
        ) * 100.0

        report["overall_quality_score"] = float(np.clip(overall_score, 0.0, 100.0))

        self._print_executive_summary(report)

        return report

    def _eval_heterogeneous_ml_utility(
        self,
        X_real: np.ndarray,
        X_gen: np.ndarray,
        y_real: np.ndarray,
        y_gen: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates TSTR (Train on Synthetic, Test on Real) across a heterogeneous suite:
        Random Forest (Tree), Logistic Regression (Linear), MLP (Neural), Gradient Boosting.
        Prevents architecture-specific inductive bias from masking synthetic defects.
        """
        X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
        )

        models = {
            "RandomForest": lambda: RandomForestClassifier(n_estimators=40, max_depth=5, random_state=42),
            "Logistic": lambda: LogisticRegression(max_iter=500, random_state=42),
            "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42),
            "MLP_Neural": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
        }

        # If synthetic labels missing, use real RF to label them
        if y_gen is None or len(y_gen) != X_gen.shape[0]:
            labeler = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_real, y_real)
            y_gen = labeler.predict(X_gen)

        model_results = {}
        ratios = []

        for name, factory in models.items():
            try:
                # 1. TRTR (Baseline)
                clf_trtr = factory().fit(X_r_train, y_r_train)
                acc_trtr = float(accuracy_score(y_r_test, clf_trtr.predict(X_r_test)))

                # 2. TSTR
                clf_tstr = factory().fit(X_gen, y_gen)
                acc_tstr = float(accuracy_score(y_r_test, clf_tstr.predict(X_r_test)))

                ratio = (acc_tstr / acc_trtr) if acc_trtr > 0 else 0.0
                ratios.append(ratio)

                model_results[name] = {
                    "trtr_acc": acc_trtr,
                    "tstr_acc": acc_tstr,
                    "utility_ratio": float(ratio)
                }
            except Exception:
                pass

        mean_ratio = float(np.mean(ratios)) if ratios else 0.0

        return {
            "models": model_results,
            "mean_utility_ratio": mean_ratio
        }

    def _eval_data_augmentation_utility(
        self,
        X_real: np.ndarray,
        X_gen: np.ndarray,
        y_real: np.ndarray,
        y_gen: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Data Augmentation Test:
        Compares model trained on Real Train alone vs. Real Train + Synthetic.
        Measures whether synthetic samples boost downstream test performance on real data.
        """
        # Small real subset to simulate data scarcity
        X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
            X_real, y_real, test_size=0.4, random_state=42, stratify=y_real
        )

        if y_gen is None or len(y_gen) != X_gen.shape[0]:
            labeler = RandomForestClassifier(n_estimators=40, random_state=42).fit(X_real, y_real)
            y_gen = labeler.predict(X_gen)

        # 1. Real Only Baseline
        clf_real = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X_r_train, y_r_train)
        acc_real_only = float(accuracy_score(y_r_test, clf_real.predict(X_r_test)))

        # 2. Augmented (Real + Synthetic)
        X_aug = np.vstack([X_r_train, X_gen])
        y_aug = np.hstack([y_r_train, y_gen])

        clf_aug = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X_aug, y_aug)
        acc_augmented = float(accuracy_score(y_r_test, clf_aug.predict(X_r_test)))

        gain = float(acc_augmented - acc_real_only)

        return {
            "real_only_acc": acc_real_only,
            "augmented_acc": acc_augmented,
            "augmentation_gain": gain,
            "augmentation_beneficial": (gain >= 0.0)
        }

    def _print_executive_summary(self, report: Dict[str, Any]):
        score = report["overall_quality_score"]
        if score >= 85.0:
            tier = "EXCELLENT (Production Ready)"
        elif score >= 70.0:
            tier = "GOOD (High Fidelity)"
        elif score >= 50.0:
            tier = "MODERATE (Acceptable for Data Augmentation)"
        else:
            tier = "POOR (Underfitting / Mode Collapse)"

        print("=========================================================================")
        print(f"       RESEARCH-GRADE SYNTHETIC DATA EVALUATION REPORT")
        print(f"       COMPOSITE QUALITY SCORE: {score:.1f} / 100  -->  [{tier}]")
        print("=========================================================================")
        print(f"{'Evaluation Dimension':<42} | {'Measured Value':<15} | {'Ideal Target'}")
        print("=" * 80)
        """
        # 🥇 TIER 1
        print("🥇 TIER 1: NON-NEGOTIABLE GATEKEEPERS (Dealbreakers)")
        print("-" * 80)
        t1 = report["tier_1_gatekeepers"]
        p = t1["exact_duplicates"]
        print(f"{'  * Exact Memorization Rate':<42} | {p['memorized_pct']:<14.2f}% | 0.0000%")
        mia = t1["membership_inference"]
        print(f"{'  * Membership Inference (MIA Ratio)':<42} | {mia['mia_ratio']:<15.4f} | ~1.0000 (No Leakage)")
        aia = t1["attribute_inference"]
        print(f"{'  * Attribute Inference (AIA Score)':<42} | {aia['aia_score']:<15.4f} | ~0.0000 (No Leakage)")
        print(f"{'  * KS-Complement (1D Marginals)':<42} | {t1['ks_complement']:<15.4f} | 1.0000")
        print(f"{'  * Normalized Wasserstein Distance':<42} | {t1['normalized_wasserstein']:<15.4f} | 0.0000")

        # 🥈 TIER 2
        print("\n🥈 TIER 2: CORE GENERATIVE QUALITY (The Triad of Fidelity)")
        print("-" * 80)
        t2 = report["tier_2_core_quality"]
        print(f"{'  * Alpha-Precision (Data Fidelity)':<42} | {t2['alpha_precision']:<15.4f} | 1.0000 (No Hallucination)")
        print(f"{'  * Beta-Recall (Mode Coverage)':<42} | {t2['beta_recall']:<15.4f} | 1.0000 (No Mode Collapse)")
        print(f"{'  * Correlation Matrix Frobenius Diff':<42} | {t2['corr_matrix_diff']:<15.4f} | 0.0000")
        print(f"{'  * Mutual Information Matrix Diff':<42} | {t2['mutual_info_diff']:<15.4f} | 0.0000 (Non-Linear)")

        if t2["heterogeneous_tstr"] is not None:
            ht = t2["heterogeneous_tstr"]
            print(f"{'  * Heterogeneous Multi-Model TSTR Ratio':<42} | {ht['mean_utility_ratio']:<15.4f} | 1.0000 (Cross-Model)")
            for m_name, m_info in ht["models"].items():
                print(f"     > {m_name:<18} TSTR / TRTR: {m_info['tstr_acc']:.3f} / {m_info['trtr_acc']:.3f} (Ratio: {m_info['utility_ratio']:.2f})")

        if t2["data_augmentation_utility"] is not None:
            aug = t2["data_augmentation_utility"]
            print(f"{'  * Augmentation Gain (Real+Synth vs Real)':<42} | {aug['augmentation_gain']:<+14.4f} | > 0.0000 (Boosts Acc)")

        # 🥉 TIER 3
        print("\n🥉 TIER 3: STRUCTURAL & GEOMETRIC INTEGRITY")
        print("-" * 80)
        t3 = report["tier_3_structural_integrity"]
        print(f"{'  * Random Forest Discriminator AUC':<42} | {t3['discriminator_auc_rf']:<15.4f} | 0.5000 (Random Guess)")
        print(f"{'  * SVD Spectrum Cosine Similarity':<42} | {t3['svd_spectrum_similarity']:<15.4f} | 1.0000 (Same Manifold)")
        print(f"{'  * Effective Rank (Real vs Gen)':<42} | {t3['real_effective_rank']:.2f} vs {t3['gen_effective_rank']:.2f} | Equal Rank")
        t = t3["tails"]
        print(f"{'  * Skewness / Kurtosis Abs Diff':<42} | {t['mean_skewness_diff']:.2f} / {t['mean_kurtosis_diff']:.2f} | 0.00 / 0.00")
        print(f"{'  * 99th / 1st Quantile Rel. Error':<42} | {t['q99_relative_error']:.3f} / {t['q01_relative_error']:.3f} | 0.000 / 0.000")
        print(f"{'  * Joint Tail Copula Dependence Diff':<42} | {t3['tail_copula_diff']:<15.4f} | 0.0000 (Co-Extremes)")
        print(f"{'  * Tree Boundary Smoothness (No Boxiness)':<42} | {t3['boxiness_smoothness']:<15.4f} | 1.0000 (No Grid Cliffs)")
        print(f"{'  * Feature Importance Rank Correlation':<42} | {t3['feature_importance_rank_corr']:<15.4f} | 1.0000 (Same Signals)")

        # 🎖️ TIER 4
        print("\n🎖️ TIER 4: CONTEXTUAL & SPECIALIZED DIAGNOSTICS")
        print("-" * 80)
        t4 = report["tier_4_contextual_diagnostics"]
        print(f"{'  * Outlier Concordance (IsoForest)':<42} | {t4['outlier_concordance']:<15.4f} | 1.0000 (Same Outliers)")
        cl = t4["clustering"]
        print(f"{'  * Cluster Silhouette Score Gap':<42} | {cl['silhouette_gap']:<15.4f} | 0.0000")
        if t4["subgroup_fidelity"] is not None:
            sub = t4["subgroup_fidelity"]
            print(f"{'  * Subgroup / Minority Class KS':<42} | {sub['mean_subgroup_ks']:<15.4f} | 1.0000 (All Classes)")
        if t4["cross_chunk_generalization"] is not None:
            cc = t4["cross_chunk_generalization"]
            print(f"{'  * Cross-Chunk Out-of-Sample KS':<42} | {cc['cross_chunk_ks']} | Overfit Gap: {cc['overfitting_gap']}")"""

        print("=========================================================================\n")


def test_generator_stability(generator, n: int = 500, conf: float = 0.8) -> Dict[str, float]:
    """
    Measures stochastic stability / inter-batch variance across two independent draws
    from the generative model.
    """
    batch_1 = generator.gen_new_data(n=n, conf=conf)
    batch_2 = generator.gen_new_data(n=n, conf=conf)

    w_inter = wasserstein_distance(batch_1, batch_2, normalized=True)
    ks_inter = kolmogorov_smirnov_test(batch_1, batch_2)

    return {
        "inter_batch_wasserstein": float(w_inter),
        "inter_batch_ks_stability": float(ks_inter)
    }

def evaluate(real_samples: np.ndarray, real_labels, gen_samples: np.ndarray, gen_labels: np.ndarray, test_samples=None, sample_obj=None):
    evaluator = GEN_VS_REAL_Evaluator()
    evaluator.evaluate(real_data=real_samples,real_labels=real_labels,gen_data=gen_samples,gen_labels=gen_labels,test_data=test_samples,sample_obj=sample_obj)

if __name__ == "__main__":
    print("Running self-test on SyntheticDataEvaluator...")
    np.random.seed(42)
    real_sample = np.random.normal(loc=0.0, scale=1.0, size=(300, 5))
    gen_sample = np.random.normal(loc=0.05, scale=0.98, size=(300, 5))
    real_l = (real_sample[:, 0] + real_sample[:, 1] > 0).astype(int)
    gen_l = (gen_sample[:, 0] + gen_sample[:, 1] > 0).astype(int)
    evaluator = GEN_VS_REAL_Evaluator()
    evaluator.evaluate(real_data=real_sample,real_labels=real_l,gen_data=gen_sample,gen_labels=gen_l)