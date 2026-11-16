"""
Plain-English Explanation Generator for Phishing URL Detection.
Translates SHAP and LIME feature attribution values into clear, human-understandable
bullet points explaining why a URL was flagged as phishing or classified as legitimate.
"""

from typing import Dict, List, Any, Optional

FEATURE_EXPLANATIONS = {
    # Group 1: Length
    'url_length': 'Overall URL is unusually long ({value} characters)',
    'hostname_length': 'Hostname is unusually long ({value} characters)',
    'path_length': 'URL path is very long ({value} characters)',
    'query_length': 'Query string contains excessive parameters/tokens ({value} characters)',
    'fragment_length': 'URL fragment identifier is unusually long ({value} characters)',

    # Group 2: Character counts & ratios
    'count_dots': 'Contains {value} dots (common indicator of multi-level fake subdomains)',
    'count_hyphens': 'Contains {value} hyphens across the URL',
    'count_underscores': 'Contains {value} underscores in domain or path',
    'count_slashes': 'Path contains {value} directory slash delimiters',
    'count_at': 'Contains {value} @ symbol(s) which can obscure true destination host',
    'count_question': 'Contains query separator (?) markers',
    'count_equals': 'Contains multiple parameter assignment (=) tokens',
    'count_ampersand': 'Contains multiple query parameter concatenations (&)',
    'count_percent': 'Contains {value} percent-encoded tokens (%XX obfuscation)',
    'count_hash': 'Contains fragment anchor (#) characters',
    'count_tilde': 'Contains tilde (~) user-directory indicators',
    'count_digits': 'Contains {value} numeric digits throughout the URL',
    'count_letters': 'Contains {value} alphabetical characters',
    'count_special': 'High density of special/symbol characters ({value})',
    'digit_ratio': 'High ratio of numeric digits to total URL length ({value:.1%})',
    'letter_ratio': 'Alphabetical character ratio is {value:.1%}',
    'special_char_ratio': 'High proportion of special characters ({value:.1%})',

    # Group 3: Host/Domain
    'num_subdomains': 'Employs {value} subdomains (frequent tactic to mimic legitimate brands)',
    'subdomain_length': 'Subdomain prefix is {value} characters long',
    'domain_length': 'Registered domain name is {value} characters long',
    'suspicious_tld': 'Uses a high-risk/abused Top-Level Domain extension (e.g., .tk, .xyz, .top, .click)',
    'is_ip_address': 'Uses a raw IP address instead of a legitimate domain name',
    'has_port': 'Specifies an explicit non-standard network port in the URL',
    'has_punycode': 'Uses Punycode (xn--) internationalized domain encoding (homograph attack risk)',
    'hyphens_in_domain': 'Domain contains {value} hyphens (often used to forge brand-like names)',

    # Group 4: Protocol
    'uses_https': 'Uses secure encrypted HTTPS protocol',
    'https_in_hostname': 'Uses "https" keyword deceptively inside hostname or path',

    # Group 5: Path/Query
    'path_depth': 'Deep directory nesting path depth of {value}',
    'num_query_params': 'Transfers {value} query parameters',
    'has_double_slash': 'Contains double slash (//) redirect pattern in the path',
    'file_extension': 'Links directly to a script or executable file extension',
    'has_suspicious_extension': 'References a dangerous file extension (.exe, .zip, .bat, etc.)',

    # Group 6: Obfuscation
    'at_symbol_present': 'Contains an @ symbol which tricks browsers into ignoring the preceding authority',
    'hex_encoding_ratio': 'Hexadecimal percent-encoding ratio is {value:.1%}',
    'has_non_ascii': 'Contains non-ASCII Unicode characters in URL string',
    'has_long_random_token': 'Contains long unreadable pseudo-random string tokens (>20 chars)',

    # Group 7: Randomness & Entropy
    'url_entropy': 'High Shannon entropy in overall URL ({value:.2f} bits/char)',
    'host_entropy': 'Elevated Shannon entropy in host name ({value:.2f} bits/char)',
    'path_entropy': 'Elevated Shannon entropy in URL path ({value:.2f} bits/char)',
    'longest_consonant_run': 'Long sequence of consecutive consonants ({value} consonants in a row)',
    'vowel_ratio': 'Unnatural vowel ratio ({value:.1%}) indicating machine-generated domain names (DGA)',

    # Group 8: Keywords
    'sensitive_word_count': 'Contains {value} sensitive authentication/banking keywords (e.g. login, verify, account, secure)',
    'brand_in_subdomain': 'Brand trademark name appears suspiciously in the subdomain prefix',
    'brand_in_path': 'Brand name appears in the URL path rather than the registered host',
    'keyword_in_domain': 'Sensitive authentication keyword appears inside the registered domain name',

    # Group 9: Shorteners
    'is_shortener': 'Known URL shortener service domain hiding the actual target destination',

    # Group 10: Typosquatting
    'typosquatting_score': 'Very high string similarity to a reputable global brand domain (typosquatting score: {value:.2f})',
}


def format_risk_factor(feature: str, value: float, attribution: float, is_risk: bool = True) -> str:
    """Format a single feature attribution into human-readable text."""
    template = FEATURE_EXPLANATIONS.get(feature, f"Feature '{feature}' value is {value}")
    
    # Try formatting with value if template has placeholder
    try:
        if isinstance(value, float):
            desc = template.format(value=value)
        elif isinstance(value, (int, bool)):
            desc = template.format(value=int(value))
        else:
            desc = template.format(value=value)
    except Exception:
        desc = template

    sign = "+" if attribution >= 0 else ""
    return f"{desc} ({sign}{attribution:.2f})"


def generate_reasons(
    shap_local: Optional[Dict[str, Any]] = None,
    lime_result: Optional[Dict[str, Any]] = None,
    url_features: Optional[Dict[str, Any]] = None,
    prediction_label: str = "phishing",
    probability: float = 0.5,
    feature_names: Optional[List[str]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Generate plain-English explanation combining SHAP and LIME.

    Returns:
    {
        'verdict': 'Phishing' or 'Legitimate',
        'confidence': '95.3%',
        'summary': 'This URL was flagged as phishing mainly because...',
        'risk_factors': [...],
        'protective_factors': [...],
        'plain_english': 'Flagged as phishing mainly because: ...'
    }
    """
    if url_features is None:
        url_features = {}

    # Extract feature contributions (prefer SHAP, fall back to LIME)
    feature_impacts = []

    if shap_local and "shap_values" in shap_local and feature_names:
        sv = shap_local["shap_values"]
        for feat_name, impact in zip(feature_names, sv):
            val = url_features.get(feat_name, 0)
            feature_impacts.append({
                "feature": feat_name,
                "attribution": float(impact),
                "value": val
            })
    elif lime_result and "feature_weights" in lime_result:
        for feat_name, weight in lime_result["feature_weights"]:
            # Feature name in LIME might have conditions like 'count_dots > 2'
            clean_name = feat_name
            for fn in (feature_names or url_features.keys()):
                if fn in feat_name:
                    clean_name = fn
                    break
            val = url_features.get(clean_name, 0)
            feature_impacts.append({
                "feature": clean_name,
                "attribution": float(weight),
                "value": val
            })
    else:
        # Fallback based on feature values
        for feat, val in url_features.items():
            if val != 0:
                feature_impacts.append({
                    "feature": feat,
                    "attribution": float(val) if val in (0, 1) else 0.1,
                    "value": val
                })

    # Sort into risk factors (positive attribution towards phishing) and protective factors
    risk_list = [f for f in feature_impacts if f["attribution"] > 0]
    prot_list = [f for f in feature_impacts if f["attribution"] < 0]

    risk_list.sort(key=lambda x: abs(x["attribution"]), reverse=True)
    prot_list.sort(key=lambda x: abs(x["attribution"]), reverse=True)

    top_risks = risk_list[:top_k]
    top_protective = prot_list[:top_k]

    formatted_risks = []
    for r in top_risks:
        formatted_risks.append({
            "feature": r["feature"],
            "value": r["value"],
            "shap": r["attribution"],
            "direction": "increases risk",
            "explanation": format_risk_factor(r["feature"], r["value"], r["attribution"], is_risk=True)
        })

    formatted_protective = []
    for p in top_protective:
        formatted_protective.append({
            "feature": p["feature"],
            "value": p["value"],
            "shap": p["attribution"],
            "direction": "reduces risk",
            "explanation": format_risk_factor(p["feature"], p["value"], p["attribution"], is_risk=False)
        })

    is_phish = prediction_label.lower() in ("phishing", "1", "true")
    verdict = "Phishing" if is_phish else "Legitimate"
    confidence_pct = f"{probability * 100:.1f}%" if is_phish else f"{(1 - probability) * 100:.1f}%"

    # Construct plain-English text
    if is_phish:
        if formatted_risks:
            risk_summary = "; ".join([r["explanation"] for r in formatted_risks[:3]])
            plain_text = f"Flagged as phishing ({confidence_pct} confidence) mainly because: {risk_summary}."
            if formatted_protective:
                prot_summary = "; ".join([p["explanation"] for p in formatted_protective[:2]])
                plain_text += f" Mitigating protective factors: {prot_summary}."
        else:
            plain_text = f"Flagged as suspicious with {confidence_pct} predicted probability of phishing."
        summary = f"This URL exhibits structural anomalies associated with phishing attacks (confidence: {confidence_pct})."
    else:
        if formatted_protective:
            prot_summary = "; ".join([p["explanation"] for p in formatted_protective[:3]])
            plain_text = f"Classified as legitimate ({confidence_pct} confidence) primarily due to: {prot_summary}."
            if formatted_risks:
                risk_summary = "; ".join([r["explanation"] for r in formatted_risks[:2]])
                plain_text += f" Minor risk considerations: {risk_summary}."
        else:
            plain_text = f"Classified as legitimate with {confidence_pct} confidence. Features conform to normal URL structures."
        summary = f"This URL shows standard lexical and domain traits characteristic of legitimate websites (confidence: {confidence_pct})."

    return {
        "verdict": verdict,
        "confidence": confidence_pct,
        "probability": float(probability),
        "summary": summary,
        "risk_factors": formatted_risks,
        "protective_factors": formatted_protective,
        "plain_english": plain_text
    }
