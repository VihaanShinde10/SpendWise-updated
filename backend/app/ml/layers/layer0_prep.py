"""
Layer 0: Text cleaning, abbreviation expansion, and 33-feature vector construction.
Implements the preprocessing pipeline from Section 4.2 of the paper.
"""
import re
import math
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

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
    "RAPIDO": "Rapido", "METRO": "Metro Rail",
    # Health
    "PHRMCY": "Pharmacy", "APPOLLO": "Apollo Pharmacy",
    "MEDPLS": "MedPlus", "NETMEDS": "Netmeds",
    # Misc
    "VPC": "VPC", "RFD": "Refund", "CSHBCK": "Cashback",
}

UPI_HANDLE_PATTERN = re.compile(r'@\w+', re.IGNORECASE)
PAYMENT_CODE_PATTERN = re.compile(
    r'\b(UPI|IMPS|NEFT|REF|TXN|AUTH|POS|P2P|VPA|RRN|UPIREF|UTR|YESB|OKAXIS|OKHDFCBANK|OKICICI)'
    r'[-/]?\w*\b', re.IGNORECASE
)
NUMERIC_ONLY_PATTERN = re.compile(r'\b\d{6,}\b')  # long reference numbers


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
        """Returns a 33-dimensional feature vector consumed by HDBSCAN."""
        pme = self.payment_method_ohe  # 5 dims
        # text features: 3
        # merchant_freq: 1 (log-transformed)
        # financial: 2 (log_amount, direction)
        # payment_method_ohe: 5
        # temporal: 8 (4 pairs sin/cos)
        # behavioural: 4 (is_recurring, recurrence_strength, mean_interval, std_interval)
        # = 3 + 1 + 2 + 5 + 8 + 4 = 23 dims padded to 33 with zeros
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
        ], dtype=np.float32)
        # Pad to exactly 33 dims
        padded = np.zeros(33, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded


def _cyclic_encode(value: float, max_val: float) -> tuple:
    angle = 2 * math.pi * value / max_val
    return math.sin(angle), math.cos(angle)


def _expand_abbreviations(text: str) -> str:
    tokens = text.upper().split()
    expanded = []
    i = 0
    while i < len(tokens):
        two_gram = " ".join(tokens[i:i+2])
        if two_gram in ABBREVIATION_LEXICON:
            expanded.append(ABBREVIATION_LEXICON[two_gram])
            i += 2
        elif tokens[i] in ABBREVIATION_LEXICON:
            expanded.append(ABBREVIATION_LEXICON[tokens[i]])
            i += 1
        else:
            expanded.append(tokens[i].capitalize())
            i += 1
    return " ".join(expanded)


def _payment_method_ohe(method: str) -> list:
    methods = ['UPI', 'IMPS', 'NEFT', 'ATM', 'OTHER']
    method_upper = method.upper() if method else 'OTHER'
    return [1 if method_upper == m else 0 for m in methods]


def prepare_transaction(
    raw_description: str,
    amount: float,
    direction: str,
    payment_method: str,
    transaction_date: datetime,
    balance: Optional[float] = None,
) -> PreparedTransaction:
    """Full Layer 0 processing pipeline."""

    # Step 1: Remove payment codes and UPI handles
    text = UPI_HANDLE_PATTERN.sub(' ', raw_description)
    text = PAYMENT_CODE_PATTERN.sub(' ', text)
    text = NUMERIC_ONLY_PATTERN.sub(' ', text)

    # Step 2: Expand abbreviations
    text = _expand_abbreviations(text)

    # Step 3: Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t.isalpha() and len(t) > 1]

    # Step 4: Descriptiveness check
    is_low = len(tokens) < 2
    merchant_name = " ".join(tokens[:4]) if tokens else raw_description[:30]

    # Temporal encoding
    h = transaction_date.hour
    dow = transaction_date.weekday()
    dom = transaction_date.day
    moy = transaction_date.month

    return PreparedTransaction(
        raw_description=raw_description,
        cleaned_description=text,
        merchant_name=merchant_name,
        is_low_descriptiveness=is_low,
        token_count=len(tokens),
        char_length=len(text),
        has_url_or_email=bool(re.search(r'[@.]', raw_description)),
        amount=amount,
        log_amount=math.log1p(abs(amount)),
        direction=1 if direction.lower() == 'debit' else 0,
        payment_method_ohe=_payment_method_ohe(payment_method),
        balance=balance,
        hour_sin=_cyclic_encode(h, 24)[0],
        hour_cos=_cyclic_encode(h, 24)[1],
        dow_sin=_cyclic_encode(dow, 7)[0],
        dow_cos=_cyclic_encode(dow, 7)[1],
        dom_sin=_cyclic_encode(dom, 31)[0],
        dom_cos=_cyclic_encode(dom, 31)[1],
        moy_sin=_cyclic_encode(moy, 12)[0],
        moy_cos=_cyclic_encode(moy, 12)[1],
    )
