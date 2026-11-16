"""SHAP-based explainability utilities."""
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance as sk_permutation_importance


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

TREE_MODELS = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
               "DecisionTreeClassifier", "GradientBoostingClassifier",
               "ExtraTreesClassifier"}
LINEAR_MODELS = {"LogisticRegression", "SGDClassifier", "LinearSVC",
                 "RidgeClassifier"}
MLP_MODELS    = {"MLPClassifier"}


def _extract_final_estimator(pipeline):
    """Return (preprocessor_steps, final_estimator) from a Pipeline."""
    from sklearn.pipeline import Pipeline
    if isinstance(pipeline, Pipeline):
        return pipeline[:-1], pipeline[-1]
    return None, pipeline


def _transform_X(preprocessor, X: np.ndarray) -> np.ndarray:
    """Apply preprocessor to X; return X unchanged if no preprocessor."""
    if preprocessor is None:
        return X
    try:
        return preprocessor.transform(X)
    except Exception:
        return X


def get_shap_explainer(pipeline, X_background: np.ndarray, model_name: str):
    """
    Create appropriate SHAP explainer for the model type.

    - TreeExplainer  for RF, XGBoost, LightGBM, DT, GBT, ExtraTrees
    - LinearExplainer for LR
    - KernelExplainer (100-sample background) for MLP / unknown
    """
    preprocessor, estimator = _extract_final_estimator(pipeline)
    estimator_type = type(estimator).__name__

    X_bg_transformed = _transform_X(preprocessor, X_background)

    if estimator_type in TREE_MODELS:
        try:
            explainer = shap.TreeExplainer(estimator)
            return explainer
        except Exception:
            pass  # fall through to Kernel

    if estimator_type in LINEAR_MODELS:
        try:
            explainer = shap.LinearExplainer(estimator, X_bg_transformed)
            return explainer
        except Exception:
            pass

    # Fallback: KernelExplainer with 100-sample background
    n_bg = min(100, X_bg_transformed.shape[0])
    summary_bg = shap.sample(X_bg_transformed, n_bg, random_state=42)

    def predict_proba_wrapper(X_arr):
        return pipeline.predict_proba(X_arr)[:, 1]

    # Use preprocessor to build predict function that takes original space
    if preprocessor is not None:
        def _predict_fn(X_arr):
            return estimator.predict_proba(X_arr)[:, 1]
        explainer = shap.KernelExplainer(_predict_fn, summary_bg)
    else:
        def _predict_fn(X_arr):
            return estimator.predict_proba(X_arr)[:, 1]
        explainer = shap.KernelExplainer(_predict_fn, summary_bg)

    return explainer


# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(explainer, X: np.ndarray, model_name: str) -> np.ndarray:
    """
    Compute SHAP values.  Handles TreeExplainer shape differences
    (binary classification returns list of two arrays).
    """
    shap_vals = explainer.shap_values(X)

    # TreeExplainer for binary classification returns [neg_class, pos_class]
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            return shap_vals[1]        # positive class (phishing)
        return shap_vals[0]

    # Some versions return a 3-D array (n_samples, n_features, n_classes)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        return shap_vals[:, :, 1]

    return shap_vals


# ---------------------------------------------------------------------------
# Global SHAP summary plots
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_shap_summary(shap_values, X, feature_names, save_path: str,
                      plot_type: str = 'beeswarm') -> None:
    """
    Plot SHAP summary (beeswarm or bar).
    Saves to artifacts/plots/shap_summary_{plot_type}.png.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 8))

    if plot_type == 'bar':
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type='bar', show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                          plot_type='dot', show=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_shap_dependence(shap_values, X, feature_names, top_n: int = 6,
                         save_dir: str = None) -> None:
    """
    Dependence plots for top-N features by mean |SHAP|.
    Saves each to artifacts/plots/shap_dep_{feature_name}.png.
    """
    if save_dir is None:
        save_dir = "artifacts/plots"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    for idx in top_indices:
        fname = feature_names[idx].replace(" ", "_")
        save_path = str(Path(save_dir) / f"shap_dep_{fname}.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, X,
                             feature_names=feature_names,
                             ax=ax, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close('all')


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def compute_permutation_importance(pipeline, X_test, y_test, feature_names,
                                   n_repeats=10, seed=42) -> pd.DataFrame:
    """
    sklearn permutation_importance on test set.
    Returns DataFrame sorted by importance descending.
    """
    result = sk_permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats, random_state=seed,
        scoring='roc_auc', n_jobs=-1
    )
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return df


def plot_permutation_importance(importance_df: pd.DataFrame,
                                save_path: str) -> None:
    """Bar plot of permutation importance."""
    _ensure_dir(save_path)
    top = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#d73027' if v > 0 else '#4575b4'
              for v in top['importance_mean']]
    ax.barh(top['feature'][::-1], top['importance_mean'][::-1],
            xerr=top['importance_std'][::-1],
            color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Permutation Importance (ROC-AUC drop)")
    ax.set_title("Feature Permutation Importance (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ---------------------------------------------------------------------------
# Local (single-URL) SHAP
# ---------------------------------------------------------------------------

def compute_shap_local(explainer, url_features: np.ndarray,
                       feature_names: list, model_name: str) -> dict:
    """
    Compute SHAP values for a single URL.

    Returns:
        {
            'shap_values': np.ndarray (n_features,),
            'base_value': float,
            'feature_values': np.ndarray (n_features,)
        }
    """
    if url_features.ndim == 1:
        url_features = url_features.reshape(1, -1)

    shap_vals = explainer.shap_values(url_features)

    # Handle list output (binary TreeExplainer)
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1][0] if len(shap_vals) == 2 else shap_vals[0][0]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_arr = shap_vals[0, :, 1]
    else:
        shap_arr = shap_vals[0]

    # Base value
    try:
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1]) if len(base_val) == 2 else float(base_val[0])
        else:
            base_val = float(base_val)
    except Exception:
        base_val = 0.0

    return {
        'shap_values': shap_arr,
        'base_value': base_val,
        'feature_values': url_features[0],
    }


def plot_shap_waterfall(shap_values_single, base_value, feature_names,
                        feature_values, prediction_prob: float,
                        save_path: str = None) -> str:
    """
    Waterfall plot for a single URL.
    Returns base64-encoded PNG string.
    If save_path given, also saves to file.
    """
    # Build shap Explanation object for waterfall
    exp = shap.Explanation(
        values=np.array(shap_values_single),
        base_values=float(base_value),
        data=np.array(feature_values),
        feature_names=list(feature_names),
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.waterfall_plot(exp, show=False, max_display=15)
    plt.title(f"Prediction probability: {prediction_prob:.3f}")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')

    if save_path:
        _ensure_dir(save_path)
        Path(save_path).write_bytes(base64.b64decode(encoded))

    return encoded


# ---------------------------------------------------------------------------
# Compare importance methods
# ---------------------------------------------------------------------------

def compare_importances(shap_df, perm_df, model_native_df,
                        feature_names) -> pd.DataFrame:
    """
    Compare SHAP vs permutation vs model-native importance rankings.
    Computes Spearman rank correlation between pairs.

    Parameters
    ----------
    shap_df : DataFrame with 'feature' and 'importance' columns (mean |SHAP|)
    perm_df : DataFrame with 'feature' and 'importance_mean' columns
    model_native_df : DataFrame with 'feature' and 'importance' columns
                      (may be None for models without native importance)

    Returns DataFrame with feature, rank_shap, rank_perm, rank_native columns.
    """
    # Build a master list in canonical order
    master = pd.DataFrame({'feature': feature_names})

    def _rank_df(df, feat_col, imp_col):
        tmp = df[[feat_col, imp_col]].copy()
        tmp = tmp.rename(columns={feat_col: 'feature', imp_col: 'imp'})
        tmp['rank'] = tmp['imp'].rank(ascending=False, method='min').astype(int)
        return tmp[['feature', 'rank']]

    shap_rank = _rank_df(shap_df, 'feature', 'importance')
    perm_rank  = _rank_df(perm_df, 'feature', 'importance_mean')

    merged = master.merge(shap_rank, on='feature', how='left') \
                   .rename(columns={'rank': 'rank_shap'})
    merged = merged.merge(perm_rank, on='feature', how='left') \
                   .rename(columns={'rank': 'rank_perm'})

    if model_native_df is not None:
        nat_rank = _rank_df(model_native_df, 'feature', 'importance')
        merged = merged.merge(nat_rank, on='feature', how='left') \
                       .rename(columns={'rank': 'rank_native'})
    else:
        merged['rank_native'] = np.nan

    # Fill missing ranks with worst rank
    n = len(feature_names)
    merged = merged.fillna(n)

    # Compute Spearman correlations and store as metadata attribute
    valid = merged.dropna(subset=['rank_shap', 'rank_perm'])
    rho_sp, p_sp = spearmanr(valid['rank_shap'], valid['rank_perm'])

    merged.attrs['spearman_shap_perm'] = round(float(rho_sp), 4)
    merged.attrs['p_shap_perm'] = round(float(p_sp), 4)

    if model_native_df is not None:
        valid_n = merged.dropna(subset=['rank_shap', 'rank_native'])
        rho_sn, p_sn = spearmanr(valid_n['rank_shap'], valid_n['rank_native'])
        merged.attrs['spearman_shap_native'] = round(float(rho_sn), 4)
        merged.attrs['p_shap_native'] = round(float(p_sn), 4)

    return merged.sort_values('rank_shap').reset_index(drop=True)
