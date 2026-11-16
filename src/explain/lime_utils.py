"""LIME-based explainability utilities."""
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------

def get_lime_explainer(X_train: np.ndarray, feature_names: list,
                       class_names: list = None) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Create LimeTabularExplainer.

    Parameters
    ----------
    X_train : np.ndarray  — background training data for statistics
    feature_names : list of str
    class_names : list of str, defaults to ['legitimate', 'phishing']
    """
    if class_names is None:
        class_names = ['legitimate', 'phishing']

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True,
        random_state=42,
    )
    return explainer


# ---------------------------------------------------------------------------
# Single-URL explanation
# ---------------------------------------------------------------------------

def explain_url_lime(
    lime_explainer,
    url_features: np.ndarray,
    predict_fn,                   # pipeline.predict_proba
    num_features: int = 15,
    num_samples: int = 500,
) -> dict:
    """
    Explain a single URL with LIME.

    Parameters
    ----------
    lime_explainer : LimeTabularExplainer
    url_features   : 1-D or 2-D np.ndarray (single sample)
    predict_fn     : callable(X) -> (n_samples, n_classes) probabilities
    num_features   : number of features to include in explanation
    num_samples    : number of perturbed samples for LIME

    Returns
    -------
    {
        'explanation'    : lime Explanation object,
        'feature_weights': [(feature_name, weight), ...] sorted by |weight| desc,
        'prediction'     : [prob_legit, prob_phish],
        'intercept'      : float
    }
    """
    if url_features.ndim == 2:
        url_features = url_features[0]

    exp = lime_explainer.explain_instance(
        data_row=url_features,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),              # explain the phishing class
    )

    # Feature weights sorted by absolute weight (descending)
    raw_weights = exp.as_list(label=1)
    sorted_weights = sorted(raw_weights, key=lambda x: abs(x[1]), reverse=True)

    # Prediction probabilities [legit, phish]
    prediction = exp.predict_proba.tolist()

    # Intercept for label=1
    try:
        intercept = float(exp.intercept[1])
    except Exception:
        intercept = 0.0

    return {
        'explanation':     exp,
        'feature_weights': sorted_weights,
        'prediction':      prediction,
        'intercept':       intercept,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_lime_explanation(lime_result: dict, save_path: str = None) -> str:
    """
    Bar plot of LIME feature weights.

    Green bars = reduces phishing risk (negative weight toward phishing class)
    Red bars   = increases phishing risk (positive weight toward phishing class)

    Returns base64-encoded PNG string.
    Saves to file if save_path given.
    """
    weights = lime_result['feature_weights']
    if not weights:
        return ""

    # Unpack
    names, vals = zip(*weights)
    vals  = list(vals)
    names = list(names)

    colors = ['#d73027' if v > 0 else '#4dac26' for v in vals]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4 + 1)))
    y_pos = range(len(names))
    ax.barh(y_pos, vals[::-1] if False else vals, color=colors,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Weight (positive = more phishing risk)")
    ax.set_title(
        f"LIME Explanation  |  P(phishing)={lime_result['prediction'][1]:.3f}"
    )
    ax.invert_yaxis()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(base64.b64decode(encoded))

    return encoded


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

def lime_to_dataframe(lime_result: dict) -> pd.DataFrame:
    """
    Convert LIME explanation to DataFrame.

    Columns: feature_name, weight, direction
    direction: 'increases_risk' | 'reduces_risk'
    """
    rows = []
    for feat, weight in lime_result['feature_weights']:
        rows.append({
            'feature_name': feat,
            'weight':       weight,
            'direction':    'increases_risk' if weight > 0 else 'reduces_risk',
        })
    return pd.DataFrame(rows)
