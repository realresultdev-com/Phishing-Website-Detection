"""
URL Feature Extractor for Phishing Detection.
Extracts lexical/structural features from raw URL strings only.
No network requests, no page visits.
"""

import re
import math
import string
import difflib
from urllib.parse import urlparse, parse_qs, unquote
from typing import Union

import tldextract
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS = {
    "tk", "xyz", "top", "click", "ml", "ga", "cf", "gq", "pw",
    "cc", "ws", "online", "site", "tech", "info",
}

SENSITIVE_WORDS = {
    "login", "secure", "account", "update", "verify", "verification", "bank", "banking",
    "signin", "sign-in", "log-in", "confirm", "password", "wallet", "credential",
    "authenticate", "auth", "recover", "recovery", "billing", "payment", "checkout",
    "token", "session",
}

BRAND_NAMES = {
    "google", "facebook", "amazon", "microsoft", "apple", "paypal",
    "netflix", "instagram", "twitter", "youtube", "linkedin", "dropbox",
    "github", "gmail", "outlook", "yahoo", "ebay", "walmart", "chase",
    "bankofamerica", "wellsfargo", "citi", "spotify", "adobe",
}

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "adf.ly", "bit.do", "tiny.cc", "is.gd", "rb.gy", "cutt.ly", "short.io",
}

SUSPICIOUS_EXTENSIONS = {
    ".exe", ".zip", ".rar", ".gz", ".tar", ".bat", ".cmd",
    ".scr", ".vbs", ".js",
}

FILE_EXTENSIONS = {".php", ".exe", ".html", ".asp", ".jsp"}

# IPv4 patterns
_IPV4_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)
_HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")

VOWELS = set("aeiouAEIOU")
CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")

# tldextract extractor (uses local cache for offline mode)
_EXTRACTOR = tldextract.TLDExtract(cache_dir=".tld_cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Return Shannon entropy of string s (bits per character)."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


def _is_ipv4(hostname: str) -> int:
    """Return 1 if hostname looks like a dotted-decimal IPv4 address."""
    if _IPV4_RE.match(hostname):
        parts = hostname.split(".")
        if all(0 <= int(p) <= 255 for p in parts):
            return 1
    # hex/decimal IP heuristic: starts with 0x or is a large integer
    if hostname.startswith("0x") or hostname.isdigit():
        return 1
    return 0


def _longest_consonant_run(hostname: str) -> int:
    """Return length of the longest consecutive consonant sequence."""
    max_run = cur = 0
    for ch in hostname.lower():
        if ch in CONSONANTS:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _has_long_random_token(url: str) -> int:
    """Return 1 if any path/query token is >20 chars with no vowels or entropy>4.5."""
    # split on common delimiters
    tokens = re.split(r"[/\?&=\.\-_#]", url)
    for tok in tokens:
        if len(tok) > 20:
            has_vowel = any(c in VOWELS for c in tok)
            if not has_vowel:
                return 1
            if _shannon_entropy(tok) > 4.5:
                return 1
    return 0


def _typosquatting_score(domain: str) -> float:
    """
    Return similarity score (0-1) of domain vs nearest brand name.
    A score near 1.0 indicates a domain that looks suspiciously close to a brand
    WITHOUT being the actual brand (e.g. 'paypa1', 'gooogle').
    If the domain is an EXACT match for the brand (e.g. 'paypal', 'google'),
    it is the authentic brand, NOT a typosquat, so the score is 0.0.
    """
    if not domain:
        return 0.0
    domain_clean = domain.lower()
    if domain_clean in BRAND_NAMES:
        return 0.0

    best = 0.0
    for brand in BRAND_NAMES:
        ratio = difflib.SequenceMatcher(None, domain_clean, brand).ratio()
        if ratio > best:
            best = ratio
    # Ignore coincidental weak similarity (< 0.65)
    if best < 0.65:
        return 0.0
    return round(best, 6)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Extract lexical/structural features from a raw URL string.
    No network requests are made.

    Returns a dict with numeric features sorted alphabetically.
    On any error, returns a zero-valued dict with the same keys.
    """
    zero_dict = {k: 0 for k in get_feature_names()}
    if not isinstance(url, str) or not url.strip():
        return dict(sorted(zero_dict.items()))

    try:
        # ------------------------------------------------------------------ #
        # Normalise
        # ------------------------------------------------------------------ #

        # rationale: add scheme so urlparse works correctly
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
            url = "https://" + url

        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        query    = parsed.query or ""
        fragment = parsed.fragment or ""
        scheme   = parsed.scheme or ""

        # tldextract for domain decomposition
        ext = _EXTRACTOR(url)
        subdomain = ext.subdomain or ""
        domain    = ext.domain or ""
        tld_str   = ext.suffix or ""

        # ------------------------------------------------------------------ #
        # GROUP 1 — Length features
        # ------------------------------------------------------------------ #
        url_length      = len(url)        # rationale: phishing URLs tend to be longer
        hostname_length = len(hostname)   # rationale: long hostnames may hide the true domain
        path_length     = len(path)       # rationale: deep paths can mask structure
        query_length    = len(query)      # rationale: long query strings hide parameters
        fragment_length = len(fragment)   # rationale: unusual fragment use indicates evasion

        # ------------------------------------------------------------------ #
        # GROUP 2 — Character count features
        # ------------------------------------------------------------------ #
        count_dots        = url.count(".")   # rationale: many dots = many subdomains
        count_hyphens     = url.count("-")   # rationale: hyphens used to spoof brand names
        count_underscores = url.count("_")   # rationale: underscores unusual in legit URLs
        count_slashes     = url.count("/")   # rationale: deep paths increase slash count
        count_at          = url.count("@")   # rationale: @ redirects to real host after it
        count_question    = url.count("?")   # rationale: multiple ? indicates obfuscation
        count_equals      = url.count("=")   # rationale: many = signs = many parameters
        count_ampersand   = url.count("&")   # rationale: many & = many query parameters
        count_percent     = url.count("%")   # rationale: hex-encoding hides malicious words
        count_hash        = url.count("#")   # rationale: unusual # may hide redirect target
        count_tilde       = url.count("~")   # rationale: tilde is unusual in legit URLs
        count_digits      = sum(c.isdigit() for c in url)   # rationale: digits common in IPs/random tokens
        count_letters     = sum(c.isalpha() for c in url)   # rationale: low letter ratio = suspicious
        count_special     = sum(
            c not in string.ascii_letters and c not in string.digits for c in url
        )  # rationale: high special-char ratio = obfuscation

        digit_ratio      = count_digits  / max(url_length, 1)   # rationale: IP/random URLs have high digit ratio
        letter_ratio     = count_letters / max(url_length, 1)   # rationale: low letter ratio is suspicious
        special_char_ratio = count_special / max(url_length, 1) # rationale: high special ratio = obfuscation

        # ------------------------------------------------------------------ #
        # GROUP 3 — Host/domain features
        # ------------------------------------------------------------------ #
        subdomain_parts  = [p for p in subdomain.split(".") if p]
        num_subdomains   = len(subdomain_parts)       # rationale: many subdomains = suspicious
        subdomain_length = len(subdomain)              # rationale: long subdomain hides real host
        domain_length    = len(domain)                 # rationale: very short/long domain is suspicious
        # tld_str is kept for internal use only (string, not a model feature)
        suspicious_tld   = int(tld_str.lower() in SUSPICIOUS_TLDS)  # rationale: free/disposable TLDs favoured by phishers
        is_ip_address    = _is_ipv4(hostname)          # rationale: IP addresses avoid DNS trail
        has_port         = int(bool(parsed.port))      # rationale: non-standard ports hide malicious servers
        has_punycode     = int("xn--" in hostname.lower())  # rationale: IDN homoglyph attacks
        hyphens_in_domain = domain.count("-")          # rationale: hyphens used to fake brand domains

        # ------------------------------------------------------------------ #
        # GROUP 4 — Protocol features
        # ------------------------------------------------------------------ #
        uses_https        = int(scheme == "https")                           # rationale: phishing sites often skip HTTPS
        https_in_hostname = int("https" in hostname.lower() or "https" in path.lower())  # rationale: deceptive "https-" in domain/path

        # ------------------------------------------------------------------ #
        # GROUP 5 — Path/query features
        # ------------------------------------------------------------------ #
        path_depth = path.count("/")                                     # rationale: very deep paths hide destination
        num_query_params = len(parse_qs(query))                          # rationale: many params = tracking/obfuscation
        has_double_slash = int("//" in path)                             # rationale: double slash = redirect trick
        # file_extension: 1 if path ends in a known suspicious extension
        _path_lower = path.lower()
        file_extension = int(any(_path_lower.endswith(e) for e in FILE_EXTENSIONS))  # rationale: script/binary extensions on URLs
        has_suspicious_extension = int(any(_path_lower.endswith(e) for e in SUSPICIOUS_EXTENSIONS))  # rationale: executable downloads from URLs

        # ------------------------------------------------------------------ #
        # GROUP 6 — Obfuscation features
        # ------------------------------------------------------------------ #
        at_symbol_present   = int("@" in url)                             # rationale: binary flag for @ obfuscation
        hex_patterns        = _HEX_RE.findall(url)
        hex_encoding_ratio  = len(hex_patterns) / max(url_length, 1)     # rationale: heavy hex encoding hides keywords
        has_non_ascii       = int(any(ord(c) > 127 for c in url))        # rationale: non-ASCII can be homoglyphs
        has_long_random_token = _has_long_random_token(url)              # rationale: random-looking tokens are obfuscation

        # ------------------------------------------------------------------ #
        # GROUP 7 — Randomness/entropy features
        # ------------------------------------------------------------------ #
        url_entropy   = _shannon_entropy(url)       # rationale: high entropy = random/obfuscated URL
        host_entropy  = _shannon_entropy(hostname)  # rationale: random hostnames are suspicious
        path_entropy  = _shannon_entropy(path)      # rationale: random paths hide destination
        longest_consonant_run = _longest_consonant_run(hostname)  # rationale: gibberish hostnames have long consonant runs
        vowel_ratio = sum(c in VOWELS for c in url) / (count_letters + 1)  # rationale: low vowel ratio = random tokens

        # ------------------------------------------------------------------ #
        # GROUP 8 — Keyword signals
        # ------------------------------------------------------------------ #
        url_lower = url.lower()
        sensitive_word_count = sum(w in url_lower for w in SENSITIVE_WORDS)   # rationale: phishing pages mimic official pages
        brand_in_subdomain   = int(any(b in subdomain.lower() for b in BRAND_NAMES))  # rationale: brand in subdomain = impersonation
        brand_in_path        = int(any(b in path.lower() for b in BRAND_NAMES))       # rationale: brand in path = impersonation
        keyword_in_domain    = int(any(w in domain.lower() for w in SENSITIVE_WORDS)) # rationale: sensitive word in domain = phishing signal

        # ------------------------------------------------------------------ #
        # GROUP 9 — URL shortener flag
        # ------------------------------------------------------------------ #
        full_domain = f"{domain}.{tld_str}".lower().strip(".")
        is_shortener = int(full_domain in SHORTENER_DOMAINS or domain.lower() in SHORTENER_DOMAINS)  # rationale: shorteners hide malicious destinations

        # ------------------------------------------------------------------ #
        # GROUP 10 — Typosquatting signal
        # ------------------------------------------------------------------ #
        typosquatting_score = _typosquatting_score(domain)  # rationale: similarity to known brands signals typosquatting

        # ------------------------------------------------------------------ #
        # Build output dict (no string fields — tld excluded)
        # ------------------------------------------------------------------ #
        features = {
            "at_symbol_present":       at_symbol_present,
            "brand_in_path":           brand_in_path,
            "brand_in_subdomain":      brand_in_subdomain,
            "count_ampersand":         count_ampersand,
            "count_at":                count_at,
            "count_digits":            count_digits,
            "count_dots":              count_dots,
            "count_equals":            count_equals,
            "count_hash":              count_hash,
            "count_hyphens":           count_hyphens,
            "count_letters":           count_letters,
            "count_percent":           count_percent,
            "count_question":          count_question,
            "count_slashes":           count_slashes,
            "count_special":           count_special,
            "count_tilde":             count_tilde,
            "count_underscores":       count_underscores,
            "digit_ratio":             round(digit_ratio, 6),
            "domain_length":           domain_length,
            "file_extension":          file_extension,
            "fragment_length":         fragment_length,
            "has_double_slash":        has_double_slash,
            "has_long_random_token":   has_long_random_token,
            "has_non_ascii":           has_non_ascii,
            "has_port":                has_port,
            "has_punycode":            has_punycode,
            "has_suspicious_extension": has_suspicious_extension,
            "hex_encoding_ratio":      round(hex_encoding_ratio, 6),
            "hostname_length":         hostname_length,
            "host_entropy":            round(host_entropy, 6),
            "hyphens_in_domain":       hyphens_in_domain,
            "https_in_hostname":       https_in_hostname,
            "is_ip_address":           is_ip_address,
            "is_shortener":            is_shortener,
            "keyword_in_domain":       keyword_in_domain,
            "letter_ratio":            round(letter_ratio, 6),
            "longest_consonant_run":   longest_consonant_run,
            "num_query_params":        num_query_params,
            "num_subdomains":          num_subdomains,
            "path_depth":              path_depth,
            "path_entropy":            round(path_entropy, 6),
            "path_length":             path_length,
            "query_length":            query_length,
            "sensitive_word_count":    sensitive_word_count,
            "special_char_ratio":      round(special_char_ratio, 6),
            "subdomain_length":        subdomain_length,
            "suspicious_tld":          suspicious_tld,
            "typosquatting_score":     typosquatting_score,
            "url_entropy":             round(url_entropy, 6),
            "url_length":              url_length,
            "uses_https":              uses_https,
            "vowel_ratio":             round(vowel_ratio, 6),
        }

        return dict(sorted(features.items()))

    except Exception:
        return dict(sorted(zero_dict.items()))


def get_feature_names() -> list:
    """Return sorted list of feature names produced by extract_features."""
    names = [
        "at_symbol_present",
        "brand_in_path",
        "brand_in_subdomain",
        "count_ampersand",
        "count_at",
        "count_digits",
        "count_dots",
        "count_equals",
        "count_hash",
        "count_hyphens",
        "count_letters",
        "count_percent",
        "count_question",
        "count_slashes",
        "count_special",
        "count_tilde",
        "count_underscores",
        "digit_ratio",
        "domain_length",
        "file_extension",
        "fragment_length",
        "has_double_slash",
        "has_long_random_token",
        "has_non_ascii",
        "has_port",
        "has_punycode",
        "has_suspicious_extension",
        "hex_encoding_ratio",
        "host_entropy",
        "hostname_length",
        "hyphens_in_domain",
        "https_in_hostname",
        "is_ip_address",
        "is_shortener",
        "keyword_in_domain",
        "letter_ratio",
        "longest_consonant_run",
        "num_query_params",
        "num_subdomains",
        "path_depth",
        "path_entropy",
        "path_length",
        "query_length",
        "sensitive_word_count",
        "special_char_ratio",
        "subdomain_length",
        "suspicious_tld",
        "typosquatting_score",
        "url_entropy",
        "url_length",
        "uses_https",
        "vowel_ratio",
    ]
    return sorted(names)


def batch_extract_features(urls: list) -> pd.DataFrame:
    """
    Extract features for a list of URLs.
    Returns a pandas DataFrame with one row per URL and one column per feature.
    """
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=get_feature_names())
