"""
Layer 0: Text cleaning, abbreviation expansion, and 33-feature vector construction.
Implements the preprocessing pipeline from Section 4.2 of the paper.
"""
import re
import math
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 340-entry abbreviation lexicon (core subset — expanded)
ABBREVIATION_LEXICON = {
    # E-commerce
    "AMZN": "Amazon", "AMZN MKTPLACE": "Amazon Marketplace",
    "FKRT": "Flipkart", "MNTRA": "Myntra", "AJIO": "AJIO",
    "MKTPLACE": "Marketplace", "SHPFY": "Shopify",
    # Food delivery
    "SWGY": "Swiggy", "ZMTO": "Zomato", "ZOMATO": "Zomato",
    "SWIGGY": "Swiggy", "BBD": "Big Basket",
    # OTT / Streaming
    "NFLX": "Netflix", "SPFY": "Spotify", "HOTSTAR": "Hotstar",
    "PRIMEVID": "Amazon Prime Video", "YTPREM": "YouTube Premium",
    # Ride-hailing
    "UBER": "Uber", "OLA": "Ola Cabs", "RAPIDO": "Rapido",
    # Banks
    "HDFC": "HDFC Bank", "SBI": "SBI Bank", "ICICI": "ICICI Bank",
    "AXIS": "Axis Bank", "KOTAK": "Kotak Bank", "PNB": "Punjab National Bank",
    "BOI": "Bank of India", "BOB": "Bank of Baroda", "CANARA": "Canara Bank",
    # UPI / Wallets
    "PAYTM": "Paytm", "GPAY": "Google Pay", "PHONEPE": "PhonePe",
    "CRED": "CRED", "BHARATPE": "BharatPe",
    # Telecom
    "BSNL": "BSNL", "AIRTEL": "Airtel", "JIO": "Jio", "VODAFONE": "Vodafone",
    # Supermarkets
    "BIGBZR": "Big Bazaar", "DMRT": "D-Mart", "RELIANCE FRESH": "Reliance Fresh",
    "DMART": "D-Mart",
    # Fast food
    "MCG": "McDonald's", "KFC": "KFC", "SBUX": "Starbucks",
    "DOMINOS": "Domino's", "PIZZA HUT": "Pizza Hut",
    # Cinemas
    "PVR": "PVR Cinemas", "INOX": "INOX Cinemas",
    # Utilities / Govt
    "AUTO": "Auto Debit", "UTL": "Utility", "INS": "Insurance",
    "MUT": "Mutual Fund", "SIP": "SIP Investment", "PMTS": "Payments",
    "IRCTC": "IRCTC", "MSEB": "MSEB Electricity", "BESCOM": "BESCOM Electricity",
    # Transport
    "METRO": "Metro Rail",
    # Health
    "PHRMCY": "Pharmacy", "APPOLLO": "Apollo Pharmacy",
    "MEDPLS": "MedPlus", "NETMEDS": "Netmeds",
    # Misc
    "VPC": "VPC", "RFD": "Refund", "CSHBCK": "Cashback",
}

# Pre-built uppercase set for O(1) lookup
_ABBREV_UPPER: set = set(ABBREVIATION_LEXICON.keys())

# Valid direction strings
_VALID_DIRECTIONS = {"debit", "credit"}

# Patterns compiled once at module load
UPI_HANDLE_PATTERN = re.compile(r'@\w+', re.IGNORECASE)
PAYMENT_CODE_PATTERN = re.compile(
    r'\b(UPI|IMPS|NEFT|REF|TXN|AUTH|POS|P2P|VPA|RRN|UPIREF|UTR|YESB|OKAXIS|OKHDFCBANK|OKICICI)'
    r'[-/]?\w*\b', re.IGNORECASE
)
NUMERIC_ONLY_PATTERN = re.compile(r'\b\d{6,}\b')  # long reference numbers

# Detect genuine URLs (http/https/ftp) or email addresses in the *raw* description
_URL_EMAIL_PATTERN = re.compile(
    r'(?:https?://|ftp://|www\.)\S+'        # URLs
    r'|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',  # emails
    re.IGNORECASE,
)

# Tokens that are valid alpha words — reused in cleaning
_ALPHA_TOKEN_PATTERN = re.compile(r'^[a-zA-Z]{2,}$')


@dataclass
class PreparedTransaction:
    raw_description: str
    cleaned_description: str
    merchant_name: str
    is_low_descriptiveness: bool
    token_count: int
    char_length: int
    has_url_or_email: bool
    # Financial
    amount: float
    log_amount: float
    direction: int          # 1=debit, 0=credit
    payment_method_ohe: list  # [UPI, IMPS, NEFT, ATM, OTHER]
    balance: Optional[float]
    # Temporal (sine-cosine encoded)
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    dom_sin: float
    dom_cos: float
    moy_sin: float
    moy_cos: float
    # Behavioural (filled in Layer 2)
    merchant_freq: int = 0
    mean_interval: float = 0.0
    std_interval: float = 0.0
    is_recurring: int = 0
    recurrence_strength: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """Returns a 33-dimensional feature vector consumed by HDBSCAN.

        Indices:
          0-2   : text features (token_count, char_length, has_url_or_email)
          3     : log1p(merchant_freq)
          4-5   : financial (log_amount, direction)
          6-10  : payment_method_ohe [UPI, IMPS, NEFT, ATM, OTHER]
          11-12 : hour  (sin, cos)
          13-14 : day-of-week (sin, cos)
          15-16 : day-of-month (sin, cos)
          17-18 : month-of-year (sin, cos)
          19    : is_recurring
          20    : recurrence_strength
          21    : mean_interval
          22    : std_interval
          23-32 : reserved / zero-padded
        """
        pme = self.payment_method_ohe  # 5 dims
        vec = np.array([
            self.token_count,               # 0
            self.char_length,               # 1
            int(self.has_url_or_email),     # 2
            math.log1p(self.merchant_freq), # 3
            self.log_amount,                # 4
            float(self.direction),          # 5
            *pme,                           # 6-10 (5 dims)
            self.hour_sin, self.hour_cos,   # 11-12
            self.dow_sin,  self.dow_cos,    # 13-14
            self.dom_sin,  self.dom_cos,    # 15-16
            self.moy_sin,  self.moy_cos,    # 17-18
            float(self.is_recurring),       # 19
            self.recurrence_strength,       # 20
            self.mean_interval,             # 21
            self.std_interval,              # 22
        ], dtype=np.float32)               # 23 populated dims
        # Pad to exactly 33 dims (indices 23–32 reserved for future features)
        padded = np.zeros(33, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cyclic_encode(value: float, max_val: float) -> tuple:
    """
    Maps `value` in [0, max_val) to a unit-circle (sin, cos) pair so that
    the distance between the highest and lowest values wraps correctly.

    Callers are responsible for passing the correct max_val:
      - hour      : max_val=24  (values 0–23)
      - weekday   : max_val=7   (values 0–6)
      - day-of-month: max_val=32 (values 1–31, avoid 0==31 collapse)
      - month     : max_val=13  (values 1–12, avoid 0==12 collapse)
    """
    angle = 2 * math.pi * value / max_val
    return math.sin(angle), math.cos(angle)


def _expand_abbreviations(text: str) -> str:
    """
    Uppercases and splits `text`, then greedily replaces 2-gram and 1-gram
    tokens against ABBREVIATION_LEXICON.  Unknown tokens are title-cased only
    when they are longer than 4 characters; short acronyms (≤4 chars that are
    not in the lexicon) are kept uppercase so downstream models can still
    recognise them (e.g. "ATM", "EMI").
    """
    tokens = text.upper().split()
    expanded = []
    i = 0
    while i < len(tokens):
        two_gram = " ".join(tokens[i:i + 2])
        if two_gram in _ABBREV_UPPER:
            expanded.append(ABBREVIATION_LEXICON[two_gram])
            i += 2
        elif tokens[i] in _ABBREV_UPPER:
            expanded.append(ABBREVIATION_LEXICON[tokens[i]])
            i += 1
        else:
            tok = tokens[i]
            # Title-case only multi-char non-acronym words; keep short caps as-is
            expanded.append(tok.title() if len(tok) > 4 else tok)
            i += 1
    return " ".join(expanded)


def _payment_method_ohe(method: str) -> list:
    """One-hot encodes the payment method into a 5-dim list."""
    methods = ['UPI', 'IMPS', 'NEFT', 'ATM', 'OTHER']
    method_upper = (method.strip().upper() if isinstance(method, str) and method.strip()
                    else 'OTHER')
    if method_upper not in methods:
        logger.debug("Unknown payment method %r — mapped to OTHER", method)
        method_upper = 'OTHER'
    return [1 if method_upper == m else 0 for m in methods]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_transaction(
    raw_description: str,
    amount: float,
    direction: str,
    payment_method: str,
    transaction_date: datetime,
    balance: Optional[float] = None,
) -> PreparedTransaction:
    """Full Layer 0 processing pipeline.

    Parameters
    ----------
    raw_description  : Original transaction narration string.
    amount           : Absolute transaction amount (must be > 0).
    direction        : 'debit' or 'credit' (case-insensitive).
    payment_method   : One of UPI / IMPS / NEFT / ATM / OTHER (case-insensitive).
    transaction_date : Timezone-aware or naive datetime of the transaction.
    balance          : Account balance after the transaction, or None if unavailable.

    Returns
    -------
    PreparedTransaction dataclass with all Layer 0 fields populated.

    Raises
    ------
    TypeError   : If required arguments have the wrong type.
    ValueError  : If `amount` is non-positive or `direction` is not recognised.
    """

    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if not isinstance(raw_description, str):
        raise TypeError(f"raw_description must be str, got {type(raw_description).__name__}")

    if not isinstance(amount, (int, float)):
        raise TypeError(f"amount must be numeric, got {type(amount).__name__}")
    amount = float(amount)
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"amount must be a finite positive number, got {amount!r}")

    if not isinstance(direction, str) or direction.strip().lower() not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be 'debit' or 'credit', got {direction!r}"
        )
    direction_clean = direction.strip().lower()

    if not isinstance(transaction_date, datetime):
        raise TypeError(
            f"transaction_date must be a datetime, got {type(transaction_date).__name__}"
        )

    if balance is not None:
        if not isinstance(balance, (int, float)):
            raise TypeError(f"balance must be numeric or None, got {type(balance).__name__}")
        balance = float(balance)
        if not math.isfinite(balance):
            raise ValueError(f"balance must be finite, got {balance!r}")

    # Sanitise raw_description: replace NUL bytes, normalise Unicode whitespace
    raw_description = raw_description.replace('\x00', ' ')
    raw_description = re.sub(r'[^\S\n]+', ' ', raw_description).strip()

    # ------------------------------------------------------------------
    # 1. Detect URL / email BEFORE stripping UPI handles (@ is shared)
    # ------------------------------------------------------------------
    has_url_or_email = bool(_URL_EMAIL_PATTERN.search(raw_description))

    # ------------------------------------------------------------------
    # 2. Strip payment codes, UPI handles, and long reference numbers
    # ------------------------------------------------------------------
    text = UPI_HANDLE_PATTERN.sub(' ', raw_description)
    text = PAYMENT_CODE_PATTERN.sub(' ', text)
    text = NUMERIC_ONLY_PATTERN.sub(' ', text)

    # ------------------------------------------------------------------
    # 3. Expand abbreviations
    # ------------------------------------------------------------------
    text = _expand_abbreviations(text)

    # ------------------------------------------------------------------
    # 4. Clean whitespace; extract alpha tokens (len ≥ 2)
    # ------------------------------------------------------------------
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if _ALPHA_TOKEN_PATTERN.match(t)]

    # ------------------------------------------------------------------
    # 5. Descriptiveness check and merchant name extraction
    # ------------------------------------------------------------------
    is_low = len(tokens) < 2
    if tokens:
        merchant_name = " ".join(tokens[:4])
    else:
        # Fallback: first 30 printable chars of original description
        merchant_name = raw_description[:30].strip() or "Unknown"
    if not merchant_name:
        merchant_name = "Unknown"

    # ------------------------------------------------------------------
    # 6. Temporal cyclic encoding
    #    hour    : [0, 23]  → max_val=24
    #    weekday : [0,  6]  → max_val=7
    #    dom     : [1, 31]  → max_val=32  (prevents day-1 == day-32 wrap)
    #    month   : [1, 12]  → max_val=13  (prevents Jan==Dec wrap)
    # ------------------------------------------------------------------
    h   = transaction_date.hour
    dow = transaction_date.weekday()
    dom = transaction_date.day
    moy = transaction_date.month

    hour_sin, hour_cos = _cyclic_encode(h,   24)
    dow_sin,  dow_cos  = _cyclic_encode(dow,  7)
    dom_sin,  dom_cos  = _cyclic_encode(dom, 32)
    moy_sin,  moy_cos  = _cyclic_encode(moy, 13)

    return PreparedTransaction(
        raw_description=raw_description,
        cleaned_description=text,
        merchant_name=merchant_name,
        is_low_descriptiveness=is_low,
        token_count=len(tokens),
        char_length=len(text),
        has_url_or_email=has_url_or_email,
        amount=amount,
        log_amount=math.log1p(amount),
        direction=1 if direction_clean == 'debit' else 0,
        payment_method_ohe=_payment_method_ohe(payment_method),
        balance=balance,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos,
        dom_sin=dom_sin,
        dom_cos=dom_cos,
        moy_sin=moy_sin,
        moy_cos=moy_cos,
    )