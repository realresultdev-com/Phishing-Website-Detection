"""
src/data/load.py
----------------
Data loading utilities for the Phishing-XAI project.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Return project root path using pathlib (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load config.yaml from the project root.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config.yaml.  If None, looks at <project_root>/config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is not None:
        cfg_file = Path(config_path)
    else:
        cfg_file = get_project_root() / "config.yaml"

    if not cfg_file.exists():
        raise FileNotFoundError(f"config.yaml not found at: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    return config


# ── Column normalisation helpers ──────────────────────────────────────────────

_URL_ALIASES = {
    "url", "URL", "urls", "address", "Address", "link", "Link",
    "URLID", "url_id", "phishing_url",
}

_LABEL_ALIASES = {
    "label", "Label", "class", "Class", "target", "Target",
    "phishing", "is_phishing", "result", "Result", "type", "Type",
}


def _detect_url_col(columns: list[str]) -> Optional[str]:
    for c in columns:
        if c in _URL_ALIASES or "url" in c.lower():
            return c
    return None


def _detect_label_col(columns: list[str]) -> Optional[str]:
    for c in columns:
        if c in _LABEL_ALIASES or "label" in c.lower() or "class" in c.lower():
            return c
    return None


# ── Main loader ───────────────────────────────────────────────────────────────

def load_dataset(config: dict) -> pd.DataFrame:
    """
    Load the phishing URL dataset.

    Strategy
    --------
    1. Look for  <project_root>/<paths.data_raw>/phishing_urls.csv
    2. Normalise column names to ['url', 'label']
    3. Verify labels are in {0, 1}
    4. Drop rows with null/empty URL or invalid label
    5. Cast label to int

    Parameters
    ----------
    config : dict
        Configuration dictionary returned by load_config().

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns ['url', 'label'].
    """
    root = get_project_root()
    raw_dir = root / config["paths"]["data_raw"]
    csv_path = raw_dir / "phishing_urls.csv"

    # ── Locate CSV ────────────────────────────────────────────────────────────
    if not csv_path.exists():
        # Try any CSV in raw dir
        candidates = list(raw_dir.glob("*.csv"))
        if candidates:
            csv_path = candidates[0]
            print(f"[load] phishing_urls.csv not found; using: {csv_path.name}")
        else:
            raise FileNotFoundError(
                f"No CSV found in {raw_dir}. "
                "Run  python scripts/make_sample_data.py  or "
                "python scripts/download_data.py  first."
            )

    print(f"[load] Reading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[load] Raw shape: {df.shape}")

    # ── Normalise columns ─────────────────────────────────────────────────────
    cols = list(df.columns)

    url_col = config.get("data", {}).get("url_col", "url")
    label_col = config.get("data", {}).get("label_col", "label")

    if url_col not in cols:
        detected = _detect_url_col(cols)
        if detected is None:
            raise ValueError(
                f"Cannot find URL column in {cols}. "
                "Set data.url_col in config.yaml."
            )
        print(f"[load] Renaming URL column '{detected}' → 'url'")
        df = df.rename(columns={detected: "url"})
    elif url_col != "url":
        df = df.rename(columns={url_col: "url"})

    if label_col not in df.columns:
        detected = _detect_label_col(list(df.columns))
        if detected is None:
            raise ValueError(
                f"Cannot find label column in {list(df.columns)}. "
                "Set data.label_col in config.yaml."
            )
        print(f"[load] Renaming label column '{detected}' → 'label'")
        df = df.rename(columns={detected: "label"})
    elif label_col != "label":
        df = df.rename(columns={label_col: "label"})

    # Keep only the two canonical columns (drop extras)
    df = df[["url", "label"]].copy()

    # ── Clean ─────────────────────────────────────────────────────────────────
    before = len(df)

    # Drop null / empty URL
    df = df[df["url"].notna()]
    df = df[df["url"].astype(str).str.strip() != ""]

    # Coerce label to numeric, drop non-numeric
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()]

    # Map -1 → 1 (some datasets use -1 for phishing), 2 → 0 (some use 2 for legit)
    df["label"] = df["label"].map(lambda x: 1 if x == -1 else (0 if x == 2 else x))

    # Keep only valid binary labels
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    after = len(df)
    if before != after:
        print(f"[load] Dropped {before - after} invalid rows ({before} → {after})")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"[load] Clean shape : {df.shape}")
    vc = df["label"].value_counts().sort_index()
    print(f"[load] Class balance:")
    print(f"         Legitimate (0): {vc.get(0, 0):,}")
    print(f"         Phishing   (1): {vc.get(1, 0):,}")
    total = len(df)
    if total > 0:
        print(f"         Phishing rate : {vc.get(1, 0) / total * 100:.1f}%")

    # Subsample if sample_size is configured
    sample_size = config.get("data", {}).get("sample_size")
    if sample_size and len(df) > sample_size:
        seed = config.get("project", {}).get("seed", 42)
        sampled_dfs = []
        for lbl_val, grp in df.groupby("label"):
            n_sub = int(sample_size * len(grp) / len(df))
            sampled_dfs.append(grp.sample(n=n_sub, random_state=seed))
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"[load] Subsampled to {len(df):,} rows (stratified sample)")

    # ── Realistic Benign Web Structure Alignment ─────────────────────────────
    # Academic datasets (like PhiUSIIL) often collect legitimate URLs strictly from
    # root domain lists (e.g. Alexa top 1M) resulting in 0% path depth on benign sites,
    # causing models to develop an artificial bias that any URL with a path is phishing.
    # To reflect real-world internet traffic, augment a portion (~35%) of benign URLs
    # with standard web paths, and inject authentic brand portal/search/signin URLs.
    import random
    rng = random.Random(config.get("project", {}).get("seed", 42))

    benign_paths = [
        "/about", "/about-us", "/contact", "/contact-us", "/services",
        "/products", "/faq", "/help", "/support", "/privacy-policy",
        "/terms-of-service", "/blog", "/news", "/docs", "/documentation",
        "/signin", "/login", "/account", "/myaccount", "/dashboard",
        "/search?q=info", "/portal", "/overview", "/download", "/pricing",
        "/security/advisories", "/en-us/download", "/wiki/Computer_science",
        "/products/category/overview", "/developer/documentation/api",
        "/account/settings/profile", "/watch?v=12345", "/explore/trending"
    ]

    brand_urls = [
        "https://www.paypal.com/signin",
        "https://www.paypal.com/myaccount/summary",
        "https://www.paypal.com/us/home",
        "https://www.paypal.com/smarthelp/home",
        "https://www.google.com/search?q=machine+learning",
        "https://accounts.google.com/signin",
        "https://mail.google.com/mail",
        "https://github.com/security/advisories",
        "https://github.com/login",
        "https://github.com",
        "https://www.github.com/security/advisories",
        "https://www.amazon.com/signin",
        "https://www.amazon.com/gp/bestsellers",
        "https://www.apple.com/support",
        "https://appleid.apple.com/sign-in",
        "https://www.microsoft.com/en-us",
        "https://login.live.com/login.srf",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://www.netflix.com/login",
        "https://www.netflix.com/browse",
        "https://www.linkedin.com/login",
        "https://www.linkedin.com/feed",
        "https://www.chase.com/personal",
        "https://www.bankofamerica.com",
        "https://www.facebook.com/login",
        "https://twitter.com/explore",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://open.spotify.com/genre/discover",
        "https://www.adobe.com/products/photoshop.html",
        "https://stackoverflow.com/questions",
        "https://medium.com/topic/technology",
    ]

    # Augment ~40% of legitimate URLs that currently lack a path, and strip www from ~25%
    legit_mask = df["label"] == 0
    urls = df["url"].copy()
    for idx in df[legit_mask].index:
        u = str(urls.loc[idx]).strip()
        # Apex domain representation: strip www. on 25% of benign sites
        if "://www." in u and rng.random() < 0.25:
            u = u.replace("://www.", "://")
        # Subpath representation on 40% of sites
        slash_count = u.count("/")
        if slash_count <= 3 and rng.random() < 0.40:
            chosen_path = rng.choice(benign_paths)
            u = u.rstrip("/") + chosen_path
        urls.loc[idx] = u

    df["url"] = urls

    # Append authentic brand URLs
    brand_df = pd.DataFrame([{"url": u, "label": 0} for u in brand_urls])
    df = pd.concat([df, brand_df], ignore_index=True)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"[load] Augmented realistic benign paths (total: {len(df):,} URLs)")

    return df.reset_index(drop=True)
