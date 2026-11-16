"""
Tests for src/features/url_features.py

Covers all 20 required test cases.
"""

import sys
import os
import pytest

# Allow import when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.url_features import extract_features, get_feature_names, batch_extract_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_numeric(d: dict) -> bool:
    """All feature values must be numeric."""
    return all(isinstance(v, (int, float)) for v in d.values())


def _has_all_features(d: dict) -> bool:
    return set(d.keys()) == set(get_feature_names())


# ---------------------------------------------------------------------------
# Test 1: Normal legit URL
# ---------------------------------------------------------------------------
def test_01_legit_google():
    url = "https://www.google.com/search?q=test"
    f = extract_features(url)
    assert _has_all_features(f), "Missing feature keys"
    assert _all_numeric(f), "Non-numeric feature value"
    assert f["uses_https"] == 1
    assert f["is_ip_address"] == 0
    assert f["num_subdomains"] >= 1   # 'www'
    assert f["suspicious_tld"] == 0
    assert f["num_query_params"] >= 1


# ---------------------------------------------------------------------------
# Test 2: Normal phishing URL (IP + login.php)
# ---------------------------------------------------------------------------
def test_02_phishing_ip_login():
    url = "http://192.168.1.1/secure/login.php"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["is_ip_address"] == 1
    assert f["uses_https"] == 0
    assert f["file_extension"] == 1   # .php
    assert f["sensitive_word_count"] >= 1  # 'secure' or 'login'


# ---------------------------------------------------------------------------
# Test 3: URL with @ symbol
# ---------------------------------------------------------------------------
def test_03_at_symbol():
    url = "http://user@evil.com/phish"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["at_symbol_present"] == 1
    assert f["count_at"] >= 1


# ---------------------------------------------------------------------------
# Test 4: URL with IP address
# ---------------------------------------------------------------------------
def test_04_ip_address():
    url = "http://123.45.67.89/update"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["is_ip_address"] == 1
    assert f["sensitive_word_count"] >= 1  # 'update'


# ---------------------------------------------------------------------------
# Test 5: Very long URL (2000 chars)
# ---------------------------------------------------------------------------
def test_05_very_long_url():
    url = "https://www.example.com/" + "a" * 1980
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["url_length"] >= 2000
    assert _all_numeric(f)


# ---------------------------------------------------------------------------
# Test 6: Empty string — must not crash, return zeros
# ---------------------------------------------------------------------------
def test_06_empty_string():
    f = extract_features("")
    assert _has_all_features(f), "Missing feature keys on empty input"
    assert _all_numeric(f)
    # All values should be 0 (or at least numeric)
    assert all(v == 0 for v in f.values()), "Expected all zeros for empty string"


# ---------------------------------------------------------------------------
# Test 7: URL with no scheme — must add https://
# ---------------------------------------------------------------------------
def test_07_no_scheme():
    url = "www.example.com/path"
    f = extract_features(url)
    assert _has_all_features(f)
    assert _all_numeric(f)
    assert f["url_length"] > 0   # was processed, not empty


# ---------------------------------------------------------------------------
# Test 8: Punycode URL
# ---------------------------------------------------------------------------
def test_08_punycode():
    url = "http://xn--80ak6aa92e.com"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["has_punycode"] == 1


# ---------------------------------------------------------------------------
# Test 9: URL shortener
# ---------------------------------------------------------------------------
def test_09_shortener():
    url = "https://bit.ly/abc123"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["is_shortener"] == 1


# ---------------------------------------------------------------------------
# Test 10: Suspicious TLD
# ---------------------------------------------------------------------------
def test_10_suspicious_tld():
    url = "http://login-secure.tk/account"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["suspicious_tld"] == 1
    assert f["sensitive_word_count"] >= 1


# ---------------------------------------------------------------------------
# Test 11: URL with hex encoding
# ---------------------------------------------------------------------------
def test_11_hex_encoding():
    url = "http://evil.com/%70%61%79%70%61%6c/login"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["hex_encoding_ratio"] > 0
    assert f["count_percent"] >= 6


# ---------------------------------------------------------------------------
# Test 12: URL with many subdomains
# ---------------------------------------------------------------------------
def test_12_many_subdomains():
    url = "http://a.b.c.d.evil.com"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["num_subdomains"] >= 4


# ---------------------------------------------------------------------------
# Test 13: Unicode / non-ASCII URL
# ---------------------------------------------------------------------------
def test_13_non_ascii():
    url = "http://evil.com/pàge"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["has_non_ascii"] == 1


# ---------------------------------------------------------------------------
# Test 14: Typosquatting URL
# ---------------------------------------------------------------------------
def test_14_typosquatting():
    url = "http://gooogle.com"
    f = extract_features(url)
    assert _has_all_features(f)
    # gooogle is very similar to google → high score
    assert f["typosquatting_score"] > 0.7


# ---------------------------------------------------------------------------
# Test 15: Sensitive keywords in URL
# ---------------------------------------------------------------------------
def test_15_sensitive_keywords():
    url = "https://paypal-secure-login-verify.com/account/update"
    f = extract_features(url)
    assert _has_all_features(f)
    # paypal, secure, login, verify, account, update are all sensitive
    assert f["sensitive_word_count"] >= 4


# ---------------------------------------------------------------------------
# Test 16: Suspicious file extension (.exe)
# ---------------------------------------------------------------------------
def test_16_suspicious_extension():
    url = "http://evil.com/malware.exe"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["has_suspicious_extension"] == 1
    assert f["file_extension"] == 1


# ---------------------------------------------------------------------------
# Test 17: Double slash redirect
# ---------------------------------------------------------------------------
def test_17_double_slash():
    url = "http://evil.com//redirect"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["has_double_slash"] == 1


# ---------------------------------------------------------------------------
# Test 18: URL with port
# ---------------------------------------------------------------------------
def test_18_port():
    url = "http://evil.com:8080/login"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["has_port"] == 1


# ---------------------------------------------------------------------------
# Test 19: Legitimate bank URL sanity check
# ---------------------------------------------------------------------------
def test_19_legit_bank():
    url = "https://www.chase.com/personal/banking"
    f = extract_features(url)
    assert _has_all_features(f)
    assert f["uses_https"] == 1
    assert f["is_ip_address"] == 0
    assert f["suspicious_tld"] == 0
    assert f["at_symbol_present"] == 0


# ---------------------------------------------------------------------------
# Test 20: Malformed URL that would crash naive parsers
# ---------------------------------------------------------------------------
def test_20_malformed_url():
    # Various malformed inputs — must NOT raise
    malformed_urls = [
        "http://",
        "://no-scheme",
        "http:///triple-slash",
        "not a url at all !!!",
        "http://[::1]:invalid-port/path",
        None,
        12345,
        "h" * 5000,
    ]
    for bad in malformed_urls:
        try:
            f = extract_features(bad)
            assert _has_all_features(f), f"Missing keys for input: {bad!r}"
            assert _all_numeric(f), f"Non-numeric value for input: {bad!r}"
        except Exception as e:
            pytest.fail(f"extract_features raised {type(e).__name__} for input {bad!r}: {e}")


# ---------------------------------------------------------------------------
# Extra: batch_extract_features returns correct shape
# ---------------------------------------------------------------------------
def test_batch_extract():
    urls = [
        "https://www.google.com",
        "http://evil.tk/phish",
        "",
    ]
    df = batch_extract_features(urls)
    assert df.shape == (3, len(get_feature_names()))
    assert list(df.columns) == get_feature_names()


# ---------------------------------------------------------------------------
# Extra: get_feature_names is sorted and has no duplicates
# ---------------------------------------------------------------------------
def test_feature_names_sorted_unique():
    names = get_feature_names()
    assert names == sorted(names), "Feature names must be sorted"
    assert len(names) == len(set(names)), "Feature names must be unique"
    assert len(names) > 40, f"Expected >40 features, got {len(names)}"
