"""
scripts/download_data.py
------------------------
Attempts to download the PhiUSIIL Phishing URL dataset from UCI.
Falls back with manual instructions on failure.

Usage:
    python scripts/download_data.py
"""

import io
import sys
import zipfile
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        return yaml.safe_load(f)


MANUAL_INSTRUCTIONS = """
DOWNLOAD FAILED. Manual download instructions:
1. Go to: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
2. Click "Download" to get the ZIP file
3. Extract phishing_urls.csv (or the main CSV) to: data/raw/phishing_urls.csv
4. Ensure columns are 'url' and 'label' (1=phishing, 0=legitimate)
5. Re-run: python scripts/train.py
Alternative: Run python scripts/make_sample_data.py for synthetic data demo.
"""

# Known column mappings for PhiUSIIL dataset
_PHIUSIIL_URL_ALIASES = ["URL", "url", "urls", "address"]
_PHIUSIIL_LABEL_ALIASES = ["label", "Label", "class", "phishing", "result", "type"]


def _find_col(columns: list, aliases: list) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    # case-insensitive fallback
    lower_cols = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def normalise_dataframe(df) -> "pd.DataFrame":
    """Rename columns to ['url', 'label'] and validate labels."""
    import pandas as pd

    cols = list(df.columns)
    print(f"[download] CSV columns: {cols}")

    url_col = _find_col(cols, _PHIUSIIL_URL_ALIASES)
    label_col = _find_col(cols, _PHIUSIIL_LABEL_ALIASES)

    if url_col is None:
        raise ValueError(
            f"Could not find URL column in {cols}. "
            "Set data.url_col in config.yaml and re-run."
        )
    if label_col is None:
        raise ValueError(
            f"Could not find label column in {cols}. "
            "Set data.label_col in config.yaml and re-run."
        )

    rename_map = {}
    if url_col != "url":
        rename_map[url_col] = "url"
        print(f"[download] Renaming '{url_col}' → 'url'")
    if label_col != "label":
        rename_map[label_col] = "label"
        print(f"[download] Renaming '{label_col}' → 'label'")

    if rename_map:
        df = df.rename(columns=rename_map)

    df = df[["url", "label"]].copy()

    # Coerce label
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    # Map -1→1 and 2→0 if present
    df["label"] = df["label"].map(lambda x: 1 if x == -1 else (0 if x == 2 else x))
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    # PhiUSIIL dataset encodes: 1 = Legitimate, 0 = Phishing
    # Canonical project standard requires: 1 = Phishing, 0 = Legitimate
    # Therefore, invert label: 1 -> 0 (legitimate), 0 -> 1 (phishing)
    df["label"] = 1 - df["label"].astype(int)

    # Drop empty URLs
    df = df[df["url"].notna() & (df["url"].str.strip() != "")]

    print(f"[download] Normalised shape: {df.shape}")
    print(f"[download] Label counts:\n{df['label'].value_counts().to_string()}")
    return df


def try_download(url: str, raw_dir: Path) -> bool:
    """
    Attempt to download and extract the dataset ZIP.

    Returns True on success, False on failure.
    """
    import pandas as pd

    print(f"[download] Downloading from:\n  {url}")
    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
    except Exception as exc:
        print(f"[download] HTTP error: {exc}")
        return False

    content = response.content
    print(f"[download] Downloaded {len(content):,} bytes")

    # Try to open as ZIP
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            members = zf.namelist()
            print(f"[download] ZIP contents: {members}")

            # Find CSV files
            csv_files = [m for m in members if m.lower().endswith(".csv")]
            if not csv_files:
                print("[download] No CSV found inside ZIP.")
                return False

            # Prefer files with 'phishing' or 'url' in name
            preferred = [
                c for c in csv_files
                if any(kw in c.lower() for kw in ["phishing", "url", "phiusiil"])
            ]
            csv_name = preferred[0] if preferred else csv_files[0]
            print(f"[download] Extracting: {csv_name}")

            with zf.open(csv_name) as csv_fh:
                df_raw = pd.read_csv(csv_fh, low_memory=False)

    except zipfile.BadZipFile:
        # Maybe it's a bare CSV?
        print("[download] Not a ZIP — trying as raw CSV …")
        try:
            df_raw = pd.read_csv(io.BytesIO(content), low_memory=False)
        except Exception as exc2:
            print(f"[download] CSV parse error: {exc2}")
            return False

    try:
        df = normalise_dataframe(df_raw)
    except ValueError as exc:
        print(f"[download] Normalisation error: {exc}")
        return False

    out_path = raw_dir / "phishing_urls.csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[download] ✅ Saved to: {out_path}")
    return True


def main() -> None:
    config = load_config()
    download_url = config["data"]["primary_url"]
    raw_dir = PROJECT_ROOT / config["paths"]["data_raw"]

    success = try_download(download_url, raw_dir)

    if not success:
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)
    else:
        print("[download] Dataset ready. Run:  python scripts/train.py")


if __name__ == "__main__":
    main()
