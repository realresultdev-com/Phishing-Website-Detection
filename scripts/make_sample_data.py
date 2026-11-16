"""
make_sample_data.py
-------------------
Generates 10,000 synthetic phishing/legitimate URLs for demo/testing purposes.

WARNING: SYNTHETIC DATA — FOR DEMO/TESTING ONLY.
Do not report results from this data as research findings.
"""

import random
import string
import csv
import sys
from pathlib import Path
import yaml

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ── Vocabulary ────────────────────────────────────────────────────────────────
BRANDS = [
    "google", "amazon", "microsoft", "facebook", "github",
    "twitter", "linkedin", "apple", "netflix", "spotify",
    "adobe", "dropbox", "slack", "zoom", "salesforce",
]

LEGIT_PATHS = [
    "/products/electronics", "/account/settings", "/help/faq",
    "/shop/clothing", "/news/technology", "/docs/api",
    "/blog/tips", "/about/team", "/services/cloud",
    "/downloads/software", "/support/tickets", "/pricing",
    "/signin", "/careers", "/contact",
]

LEGIT_QUERY_PARAMS = [
    "?ref=homepage", "?utm_source=google&utm_medium=cpc",
    "?page=2&sort=popular", "?lang=en", "?category=featured",
    "?q=search+term", "?id=12345", "",
]

SUBDOMAINS = ["", "www.", "mail.", "shop.", "support.", "docs.", "blog.", "m."]

PHISHING_KEYWORDS = [
    "login", "verify", "secure", "account", "update",
    "confirm", "signin", "banking", "password", "wallet",
    "support", "alert", "suspended", "unlock", "reward",
]

SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club",
    "online", "site", "info", "biz", "click", "link",
]

EVIL_DOMAINS = [
    "phish.net", "evil.tk", "scam.ml", "fake.xyz",
    "malware.top", "hack.club", "steal.online",
]

PHISHING_PATHS = [
    "/login.php", "/verify.php", "/secure/update",
    "/account/confirm", "/signin/index.html",
    "/update/credentials", "/banking/login",
    "/wallet/verify", "/reward/claim",
    "/suspended/unlock",
]


# ── Generators ────────────────────────────────────────────────────────────────
def _rand_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _rand_string(length: int) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_legitimate_url() -> str:
    brand = random.choice(BRANDS)
    subdomain = random.choice(SUBDOMAINS)
    path = random.choice(LEGIT_PATHS)
    query = random.choice(LEGIT_QUERY_PARAMS)
    scheme = "https"
    return f"{scheme}://{subdomain}{brand}.com{path}{query}"


def generate_phishing_url() -> str:
    pattern = random.randint(1, 4)

    if pattern == 1:
        # IP-based
        ip = _rand_ip()
        kw = random.choice(PHISHING_KEYWORDS)
        return f"http://{ip}/{kw}/login.php"

    elif pattern == 2:
        # Typosquatting: brand-keyword.suspicious_tld
        brand = random.choice(BRANDS)
        kw = random.choice(PHISHING_KEYWORDS)
        tld = random.choice(SUSPICIOUS_TLDS)
        path = random.choice(PHISHING_PATHS)
        scheme = random.choice(["http", "https"])
        return f"{scheme}://{brand}-{kw}.{tld}{path}"

    elif pattern == 3:
        # Subdomain trick: www.brand.com.evil_domain/long_path
        brand = random.choice(BRANDS)
        evil = random.choice(EVIL_DOMAINS)
        kw1 = random.choice(PHISHING_KEYWORDS)
        kw2 = random.choice(PHISHING_KEYWORDS)
        return f"http://www.{brand}.com.{evil}/{kw1}/{kw2}/index.php"

    else:
        # Random long domain
        length = random.randint(12, 28)
        domain = _rand_string(length)
        path_kw = random.choice(PHISHING_KEYWORDS)
        return f"http://{domain}.com/secure/{path_kw}"


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    config = load_config()
    n_each = 5_000
    seed = config.get("project", {}).get("seed", 42)
    random.seed(seed)

    rows = []

    # Legitimate URLs (label=0)
    for _ in range(n_each):
        rows.append({"url": generate_legitimate_url(), "label": 0})

    # Phishing URLs (label=1)
    for _ in range(n_each):
        rows.append({"url": generate_phishing_url(), "label": 1})

    # Shuffle
    random.shuffle(rows)

    # Save
    out_dir = PROJECT_ROOT / config["paths"]["data_raw"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phishing_urls.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    n_legit = sum(1 for r in rows if r["label"] == 0)
    n_phish = sum(1 for r in rows if r["label"] == 1)

    print("\n" + "=" * 70)
    print("⚠️  WARNING: SYNTHETIC DATA — FOR DEMO/TESTING ONLY.")
    print("   Do not report results from this data as research findings.")
    print("=" * 70)
    print(f"\n✅ Dataset saved to: {out_path}")
    print(f"   Total rows  : {len(rows):,}")
    print(f"   Legitimate  : {n_legit:,}  (label=0)")
    print(f"   Phishing    : {n_phish:,}  (label=1)")

    legit_examples = [r["url"] for r in rows if r["label"] == 0][:3]
    phish_examples = [r["url"] for r in rows if r["label"] == 1][:3]

    print("\nExample legitimate URLs:")
    for u in legit_examples:
        print(f"  {u}")

    print("\nExample phishing URLs:")
    for u in phish_examples:
        print(f"  {u}")
    print()


if __name__ == "__main__":
    main()
