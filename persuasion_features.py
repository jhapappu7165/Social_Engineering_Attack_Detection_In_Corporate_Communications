import re
import math
import pandas as pd


# =========================
# PATTERNS
# =========================

URGENCY_PATTERNS = [
    r"\burgent\b", r"\bimmediately\b", r"\bnow\b", r"\basap\b",
    r"\bact now\b", r"\brespond now\b", r"\bverify now\b",
    r"\bdeadline\b", r"\bexpires?\b", r"\blimited time\b",
    r"\bfinal notice\b", r"\blast chance\b"
]

AUTHORITY_PATTERNS = [
    r"\bsecurity team\b", r"\badministrator\b", r"\badmin\b",
    r"\bit support\b", r"\bcustomer service\b", r"\bofficial\b",
    r"\bbank\b", r"\bpaypal\b", r"\bmicrosoft\b", r"\bapple\b",
    r"\bamazon\b", r"\bgoogle\b"
]

REWARD_PATTERNS = [
    r"\breward\b", r"\bprize\b", r"\bbonus\b", r"\bgift\b",
    r"\bcash\b", r"\bcashback\b", r"\bdiscount\b", r"\bfree\b",
    r"\bwon\b", r"\bwinner\b", r"\bcongratulations\b",
    r"\bclaim\b", r"\boffer\b"
]

FEAR_PATTERNS = [
    r"\bsuspended\b", r"\bterminated\b", r"\blocked\b",
    r"\bunauthorized\b", r"\bfraud\b", r"\balert\b",
    r"\bbreach\b", r"\bcompromised\b", r"\brisk\b",
    r"\bthreat\b", r"\bunusual activity\b"
]

CURIOSITY_PATTERNS = [
    r"\bexclusive\b", r"\bsecret\b", r"\bconfidential\b",
    r"\bspecial access\b", r"\byou have been selected\b",
    r"\blearn more\b", r"\bdiscover\b", r"\bunclaimed\b"
]

CTA_PATTERNS = [
    r"\bclick\b", r"\blogin\b", r"\bsign in\b",
    r"\bverify\b", r"\bconfirm\b", r"\bupdate\b",
    r"\bsubmit\b", r"\bopen\b", r"\bdownload\b"
]

URL_PATTERN = r"(https?://\S+|www\.\S+|<url>)"
EMAIL_PATTERN = r"\b\S+@\S+\b|<email>"


# =========================
# HELPERS
# =========================

def safe_text(text):
    return "" if pd.isna(text) else str(text)


def count_matches(text, patterns):
    count = 0
    for p in patterns:
        count += len(re.findall(p, text, flags=re.IGNORECASE))
    return count


def log_scale(x):
    return math.log1p(x)


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def digit_ratio(text):
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


# =========================
# MAIN FEATURE BLOCK
# =========================

def feature_block(text):
    raw = safe_text(text)
    t = raw.lower()

    urgency = count_matches(t, URGENCY_PATTERNS)
    authority = count_matches(t, AUTHORITY_PATTERNS)
    reward = count_matches(t, REWARD_PATTERNS)
    fear = count_matches(t, FEAR_PATTERNS)
    curiosity = count_matches(t, CURIOSITY_PATTERNS)
    cta = count_matches(t, CTA_PATTERNS)

    persuasion_total = urgency + authority + reward + fear + curiosity

    return {
        # -------- RAW COUNTS --------
        "urgency_count": urgency,
        "authority_count": authority,
        "reward_count": reward,
        "fear_count": fear,
        "curiosity_count": curiosity,
        "cta_count": cta,

        # -------- LOG-SCALED --------
        "urgency_score": log_scale(urgency),
        "authority_score": log_scale(authority),
        "reward_score": log_scale(reward),
        "fear_score": log_scale(fear),
        "curiosity_score": log_scale(curiosity),
        "cta_score": log_scale(cta),

        # -------- BINARY PRESENCE --------
        "urgency_present": int(urgency > 0),
        "authority_present": int(authority > 0),
        "reward_present": int(reward > 0),
        "fear_present": int(fear > 0),
        "curiosity_present": int(curiosity > 0),
        "cta_present": int(cta > 0),

        # -------- AGGREGATE --------
        "persuasion_total": persuasion_total,
        "persuasion_present": int(persuasion_total > 0),

        # -------- STRUCTURAL --------
        "url_count": len(re.findall(URL_PATTERN, t)),
        "email_count": len(re.findall(EMAIL_PATTERN, t)),
        "exclamation_count": raw.count("!"),
        "question_count": raw.count("?"),
        "caps_ratio": caps_ratio(raw),
        "digit_ratio": digit_ratio(raw),
        "message_length": len(raw),
    }


def build_persuasion_features(df, text_col="text_cleaned"):
    if text_col not in df.columns:
        raise ValueError(f"{text_col} not found")

    features = df[text_col].apply(feature_block)
    return pd.DataFrame(features.tolist())