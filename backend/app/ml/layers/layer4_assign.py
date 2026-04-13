"""
Layer 4: Three-stage hierarchical category assignment.
Implements Section 4.7 of the paper.
"""
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from app.config import get_settings

settings = get_settings()

CATEGORY_MAP = {
    "Food & Dining": "Food & Dining (Zomato, Swiggy, EatClub, restaurants, cafes, bars, and dining establishments)",
    "Transport": "Transport (Uber, Ola, Rapido, BluSmart, Namma Metro, DTC, BEST, petrol, and commute services)",
    "Shopping": "Shopping (Amazon, Flipkart, Myntra, Ajio, Nykaa, retail stores, and e-commerce)",
    "Entertainment": "Entertainment (Movies, Netflix, Hotstar, gaming, park, hobbies, and bookmyshow)",
    "Subscriptions": "Subscriptions (Netflix, Prime, gym memberships, and monthly software or app bills)",
    "Utilities": "Utilities (Mobile recharge like Airtel, Jio, VI, electricity, gas, water, and internet bills)",
    "Health & Medical": "Health & Medical (Pharmacy, medicine, Practo, hospital, and labs)",
    "Education": "Education (Tuition, courses, Unacademy, Byjus, books, and university or school fees)",
    "Travel": "Travel (MakeMyTrip, Goibibo, flights, Airbnb, IRCTC, and vacation rentals)",
    "Financial Services": "Financial Services (Insurance, bank charges, taxes, PolicyBazaar, loan EMI, and investments)",
    "Groceries": "Groceries (Zepto, Blinkit, BigBasket, Instamart, supermarkets, and daily milk/essentials)",
    "Peer Transfer": "Peer Transfer (UPI transfer via PhonePe, GPay, Paytm, or personal bank transfers)",
    "Others": "Others (Miscellaneous or uncategorized expenses)"
}

CATEGORY_NAMES = list(CATEGORY_MAP.keys())
ML_LABELS = list(CATEGORY_MAP.values())
REVERSE_MAP = {v: k for k, v in CATEGORY_MAP.items()}

# Singleton zero-shot pipeline
_zero_shot_pipeline = None

def resolve_entity_context(text: str, user_id: str) -> str:
    """
    Autonomous Entity Resolution.
    Finds the most similar manually-verified merchant to resolve aliases/shortforms.
    """
    from app.db.client import get_supabase
    from app.ml.layers.layer1_semantic import get_e5_model
    import numpy as np
    
    try:
        supabase = get_supabase()
        model = get_e5_model()
        # The E5 model encodes text into high-dimensional vectors
        embedding = model.encode(text)
        
        # Search for exactly similar transactions globally that have high-confidence categories
        # This allows the system to learn 'DTC' -> 'Delhi Transport' autonomously
        res = supabase.rpc('match_transactions', {
            'query_embedding': embedding.tolist(),
            'match_threshold': 0.92,
            'match_count': 3
        }).execute()
        
        matches = res.data or []
        if matches:
            # Return the most frequent merchant name among high-confidence matches
            from collections import Counter
            names = [m['merchant_name'] for m in matches if m['merchant_name']]
            if names:
                top_resolved = Counter(names).most_common(1)[0][0]
                return f"Resolved Entity: {top_resolved}. Original: {text}"
    except Exception:
        pass
        
    return text


def get_zero_shot_pipeline():
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _zero_shot_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=settings.BART_MODEL_NAME,
            device=-1  # CPU
        )
    return _zero_shot_pipeline


@dataclass
class AssignmentResult:
    category: str
    confidence: float
    source: str  # 'semantic' | 'behavioural' | 'fused' | 'zero_shot' | 'manual'
    needs_review: bool
    gating_alpha: float
    top3_similar_txns: list = field(default_factory=list)


def assign_category(
    user_id: str,
    cluster_membership_prob: float,
    cluster_category: Optional[str],
    neighbour_categories: list,
    neighbour_distances: list,
    merchant_name: str,
    final_confidence: float,
    source: str,
    gating_alpha: float,
    top3_txn_ids: list,
    raw_description: str = ""
) -> AssignmentResult:
    """
    Stage 1: HDBSCAN cluster membership >= 0.60 → assign cluster category
    Stage 2: Local neighbourhood validation (k=5, >70% agreement, dist<0.35)
    Stage 3: Zero-shot BART NLI (entailment > 0.70)
    Else: Manual review queue
    """
    # Stage 1
    if cluster_membership_prob >= 0.60 and cluster_category:
        return AssignmentResult(
            category=cluster_category,
            confidence=final_confidence,
            source=source,
            needs_review=False,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )

    # Stage 2: Local neighbourhood validation
    if neighbour_categories and len(neighbour_categories) >= 3:
        from collections import Counter
        counts = Counter(neighbour_categories)
        top_cat, top_count = counts.most_common(1)[0]
        agreement = top_count / len(neighbour_categories)
        mean_dist = sum(neighbour_distances) / len(neighbour_distances) if neighbour_distances else 1.0
        if (agreement > settings.STAGE2_AGREEMENT_THRESHOLD
                and mean_dist < settings.STAGE2_DISTANCE_THRESHOLD):
            return AssignmentResult(
                category=top_cat,
                confidence=round(agreement * 0.85, 4),
                source=source,
                needs_review=False,
                gating_alpha=gating_alpha,
                top3_similar_txns=top3_txn_ids
            )

    # Stage 3: Zero-shot NLI — process full context for robustness
    # We prefix with the merchant name to force the model to recognize brand entities
    merchant_hint = f"Merchant: {merchant_name}. " if merchant_name else ""
    process_text = f"{merchant_hint}{raw_description}" if raw_description else merchant_name
    process_text = resolve_entity_context(process_text, user_id)
    
    try:
        zs = get_zero_shot_pipeline()
        
        # Robust Spend-Focused Hypothesis
        # 'spending money on' works better for financial NLI than 'categorized as'
        hypothesis = "This is a bank transaction for spending money on {}."
        
        result = zs(process_text, candidate_labels=ML_LABELS, hypothesis_template=hypothesis)
        top_expanded_label = result['labels'][0]
        top_score = result['scores'][0]
        
        # Map back to clean category name
        top_label = REVERSE_MAP.get(top_expanded_label, "Others")

        if top_score >= settings.BART_ENTAILMENT_THRESHOLD:
            return AssignmentResult(
                category=top_label,
                confidence=round(top_score, 4),
                source='zero_shot',
                needs_review=False,
                gating_alpha=gating_alpha,
                top3_similar_txns=top3_txn_ids
            )

        # Zero-shot result below threshold → still return it but flag for review
        return AssignmentResult(
            category=top_label,
            confidence=round(top_score, 4),
            source='manual',
            needs_review=True,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )
    except Exception as e:
        logger.error(f"Zero-shot failed: {e}")
        return AssignmentResult(
            category="Others",
            confidence=0.0,
            source='manual',
            needs_review=True,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )

    # All stages failed → manual review
    return AssignmentResult(
        category="Others",
        confidence=0.0,
        source='manual',
        needs_review=True,
        gating_alpha=gating_alpha,
        top3_similar_txns=top3_txn_ids
    )
