"""
src/data/split.py
-----------------
Leak-free train / val / test splitting for the Phishing-XAI project.

Key design choice
-----------------
URLs from the same registered domain must stay in the same split.
This prevents the model from learning domain-identity signals and forces
it to generalise to unseen domains — the realistic deployment scenario.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    import tldextract
    _HAS_TLDEXTRACT = True
except ImportError:
    _HAS_TLDEXTRACT = False
    print(
        "[split] WARNING: tldextract not installed. "
        "Falling back to naive domain extraction. "
        "Install with: pip install tldextract"
    )


# ── Project root ──────────────────────────────────────────────────────────────
def _get_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# ── Domain extraction ─────────────────────────────────────────────────────────

def extract_registered_domain(url: str) -> str:
    """
    Extract the registered domain (eTLD+1) from a URL.

    Examples
    --------
    >>> extract_registered_domain("https://www.google.com/search?q=test")
    'google.com'
    >>> extract_registered_domain("http://evil-login.tk/phish")
    'evil-login.tk'
    >>> extract_registered_domain("http://192.168.1.1/login.php")
    '192.168.1.1'
    """
    if not isinstance(url, str) or not url.strip():
        return "unknown"

    if _HAS_TLDEXTRACT:
        try:
            extracted = tldextract.extract(url)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}"
            elif extracted.domain:
                return extracted.domain
            else:
                # IP address or bare host
                from urllib.parse import urlparse
                return urlparse(url).hostname or "unknown"
        except Exception:
            pass

    # Naive fallback
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or "unknown"
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    except Exception:
        return "unknown"


# ── Leakage analysis ──────────────────────────────────────────────────────────

def analyze_split_leakage(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    group_col: str = "registered_domain",
) -> dict:
    """
    Check for domain overlap between splits.

    Returns
    -------
    dict
        {
            'train_val_overlap' : int,
            'train_test_overlap': int,
            'val_test_overlap'  : int,
        }
    """
    result = {}

    for a_name, a_df, b_name, b_df in [
        ("train", X_train, "val",  X_val),
        ("train", X_train, "test", X_test),
        ("val",   X_val,  "test",  X_test),
    ]:
        key = f"{a_name}_{b_name}_overlap"
        if group_col in a_df.columns and group_col in b_df.columns:
            overlap = len(
                set(a_df[group_col].unique()) & set(b_df[group_col].unique())
            )
        else:
            overlap = -1  # column not present
        result[key] = overlap

    return result


# ── Save helpers ──────────────────────────────────────────────────────────────

def _save_splits(
    root: Path,
    processed_dir: Path,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    suffix: str = "",
) -> None:
    """Persist feature DataFrames and labels to data/processed/."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        fname = f"{split_name}{suffix}.csv"
        df_out = X.copy()
        df_out["label"] = y.values
        df_out.to_csv(processed_dir / fname, index=False)
        print(f"[split] Saved {fname}  shape={df_out.shape}")


def _save_feature_schema(
    root: Path,
    artifacts_dir: Path,
    X_train: pd.DataFrame,
) -> None:
    """Write artifacts/feature_schema.json with column names and dtypes."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        col: str(dtype)
        for col, dtype in X_train.dtypes.items()
    }
    schema_path = artifacts_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"[split] Feature schema saved to {schema_path}  ({len(schema)} features)")


# ── Group-aware split ─────────────────────────────────────────────────────────

def group_aware_split(
    df: pd.DataFrame,
    config: dict,
    url_col: str = "url",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train / val / test with GroupShuffleSplit on registered domain.

    No domain will appear in more than one split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with at least ``url_col`` and ``label_col`` columns.
    config : dict
        Project configuration dictionary.
    url_col : str
        Name of the URL column.
    label_col : str
        Name of the label column.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    root = _get_root()
    processed_dir = root / config["paths"]["data_processed"]
    artifacts_dir = root / config["paths"]["artifacts"]

    test_size = float(config["data"]["test_size"])
    val_size  = float(config["data"]["val_size"])
    seed      = int(config["project"]["seed"])

    # 1. Add registered_domain column
    print("[split] Extracting registered domains …")
    df = df.copy()
    df["registered_domain"] = df[url_col].apply(extract_registered_domain)

    groups = df["registered_domain"].values
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # 2. First split: (train+val) vs test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    X_trainval, y_trainval = X.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test,     y_test     = X.iloc[test_idx],      y.iloc[test_idx]
    groups_trainval = groups[train_val_idx]

    # 3. Second split: train vs val (val_size relative to train+val)
    effective_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=effective_val, random_state=seed)
    train_idx, val_idx = next(
        gss_val.split(X_trainval, y_trainval, groups=groups_trainval)
    )

    X_train, y_train = X_trainval.iloc[train_idx], y_trainval.iloc[train_idx]
    X_val,   y_val   = X_trainval.iloc[val_idx],   y_trainval.iloc[val_idx]

    # 4. Report sizes
    print(f"[split] Group-aware split  "
          f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # 5. Verify zero domain overlap between train and test
    leakage = analyze_split_leakage(X_train, X_val, X_test)
    print(f"[split] Domain overlap — {leakage}")
    if leakage["train_test_overlap"] > 0:
        print(
            f"[split] WARNING: {leakage['train_test_overlap']} domains appear in "
            "both train and test splits. Consider increasing dataset size or "
            "re-checking the grouping column."
        )
    else:
        print("[split] ✅ Zero domain overlap between train and test.")

    # 6. Save splits
    _save_splits(root, processed_dir, X_train, X_val, X_test, y_train, y_val, y_test)

    # 7. Save feature schema (uses url + registered_domain as "features" at this stage)
    _save_feature_schema(root, artifacts_dir, X_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Random stratified split (baseline / comparison) ───────────────────────────

def random_split(
    df: pd.DataFrame,
    config: dict,
    url_col: str = "url",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Standard random stratified split (ignores domain grouping).

    Used for comparison to quantify leakage bias vs group_aware_split.

    Returns same tuple format as group_aware_split.
    """
    root = _get_root()
    processed_dir = root / config["paths"]["data_processed"]
    artifacts_dir = root / config["paths"]["artifacts"]

    test_size = float(config["data"]["test_size"])
    val_size  = float(config["data"]["val_size"])
    seed      = int(config["project"]["seed"])

    df = df.copy()
    df["registered_domain"] = df[url_col].apply(extract_registered_domain)

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # First split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Second split
    effective_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=effective_val,
        random_state=seed,
        stratify=y_trainval,
    )

    print(f"[split] Random split  "
          f"train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    leakage = analyze_split_leakage(X_train, X_val, X_test)
    print(f"[split] (random) Domain overlap — {leakage}")

    _save_splits(
        root, processed_dir,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        suffix="_random",
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
