"""
Layer 4: Three-stage hierarchical category assignment.
Implements Section 4.7 of the paper.
"""
import threading
from collections import Counter
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

# Startup assertion: ensure REVERSE_MAP is a perfect 1-to-1 mapping
assert len(REVERSE_MAP) == len(CATEGORY_MAP), (
    "CATEGORY_MAP has duplicate expanded label values — REVERSE_MAP will silently drop entries. "
    "Each category's description string must be unique."
)

# Singleton zero-shot pipeline + thread lock
_zero_shot_pipeline = None
_pipeline_lock = threading.Lock()


def resolve_entity_context(text: str, user_id: str) -> str:
    """
    Autonomous Entity Resolution.
    Finds the most similar manually-verified merchant to resolve aliases/shortforms.
    """
    from app.db.client import get_supabase
    from app.ml.layers.layer1_semantic import get_e5_model
    import numpy as np

    if not text or not text.strip():
        return text

    try:
        supabase = get_supabase()
        model = get_e5_model()
        # The E5 model encodes text into high-dimensional vectors
        embedding = model.encode(text)

        # Search for similar transactions globally that have high-confidence categories.
        # This allows the system to learn 'DTC' -> 'Delhi Transport' autonomously.
        res = supabase.rpc('match_transactions', {
            'query_embedding': embedding.tolist(),
            'match_threshold': 0.92,
            'match_count': 3
        }).execute()

        matches = res.data or []
        if matches:
            # Return the most frequent merchant name among high-confidence matches
            names = [m['merchant_name'] for m in matches if m.get('merchant_name')]
            if names:
                top_resolved = Counter(names).most_common(1)[0][0]
                logger.debug(
                    f"Entity resolved: '{text}' → '{top_resolved}' "
                    f"(user_id={user_id})"
                )
                return f"Resolved Entity: {top_resolved}. Original: {text}"
    except Exception as e:
        logger.warning(
            f"resolve_entity_context failed for text='{text}', "
            f"user_id={user_id}: {e}"
        )

    return text


def get_zero_shot_pipeline():
    """Thread-safe lazy initialisation of the zero-shot BART pipeline."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        with _pipeline_lock:
            # Double-checked locking: re-check after acquiring lock
            if _zero_shot_pipeline is None:
                from transformers import pipeline as hf_pipeline
                logger.info(
                    f"Loading zero-shot pipeline: {settings.BART_MODEL_NAME}"
                )
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
    # Sanitise mutable defaults passed as None by callers
    neighbour_categories = neighbour_categories or []
    neighbour_distances = neighbour_distances or []
    top3_txn_ids = top3_txn_ids or []

    # ------------------------------------------------------------------ #
    # Stage 1: HDBSCAN cluster membership
    # ------------------------------------------------------------------ #
    if cluster_membership_prob >= 0.60 and cluster_category:
        logger.debug(
            f"Stage 1 hit — cluster_membership_prob={cluster_membership_prob:.3f}, "
            f"category='{cluster_category}', user_id={user_id}"
        )
        return AssignmentResult(
            category=cluster_category,
            confidence=final_confidence,
            source=source,
            needs_review=False,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )

    # ------------------------------------------------------------------ #
    # Stage 2: Local neighbourhood validation
    # ------------------------------------------------------------------ #
    if (
        neighbour_categories
        and len(neighbour_categories) >= 3
        and len(neighbour_distances) == len(neighbour_categories)
    ):
        counts = Counter(neighbour_categories)
        top_cat, top_count = counts.most_common(1)[0]
        agreement = top_count / len(neighbour_categories)

        # Median distance is more robust than mean against outlier neighbours
        sorted_dists = sorted(neighbour_distances)
        mid = len(sorted_dists) // 2
        median_dist = (
            sorted_dists[mid]
            if len(sorted_dists) % 2 == 1
            else (sorted_dists[mid - 1] + sorted_dists[mid]) / 2.0
        )

        if (
            agreement > settings.STAGE2_AGREEMENT_THRESHOLD
            and median_dist < settings.STAGE2_DISTANCE_THRESHOLD
        ):
            # Confidence = agreement score discounted by distance proximity
            # Tighter cluster (lower median_dist) → higher confidence
            proximity_bonus = max(0.0, 1.0 - median_dist / settings.STAGE2_DISTANCE_THRESHOLD)
            stage2_confidence = round(
                agreement * 0.80 + proximity_bonus * 0.10, 4
            )
            logger.debug(
                f"Stage 2 hit — top_cat='{top_cat}', agreement={agreement:.2f}, "
                f"median_dist={median_dist:.4f}, confidence={stage2_confidence}, "
                f"user_id={user_id}"
            )
            return AssignmentResult(
                category=top_cat,
                confidence=stage2_confidence,
                source=source,
                needs_review=False,
                gating_alpha=gating_alpha,
                top3_similar_txns=top3_txn_ids
            )

    # ------------------------------------------------------------------ #
    # Stage 3: Zero-shot BART NLI
    # ------------------------------------------------------------------ #
    # Build the best possible input text for the model.
    # Prefix with merchant name so the model can recognise brand entities.
    merchant_hint = f"Merchant: {merchant_name}. " if merchant_name and merchant_name.strip() else ""
    process_text = (
        f"{merchant_hint}{raw_description}".strip()
        if raw_description and raw_description.strip()
        else merchant_name.strip() if merchant_name and merchant_name.strip()
        else ""
    )

    if not process_text:
        logger.warning(
            f"Stage 3 skipped — empty process_text for user_id={user_id}. "
            f"Routing to manual review."
        )
        return AssignmentResult(
            category="Others",
            confidence=0.0,
            source='manual',
            needs_review=True,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )

    process_text = resolve_entity_context(process_text, user_id)

    try:
        zs = get_zero_shot_pipeline()

        # 'spending money on' works better for financial NLI than 'categorized as'
        hypothesis = "This is a bank transaction for spending money on {}."

        result = zs(
            process_text,
            candidate_labels=ML_LABELS,
            hypothesis_template=hypothesis
        )
        top_expanded_label = result['labels'][0]
        top_score = result['scores'][0]

        # Map back to clean category name
        top_label = REVERSE_MAP.get(top_expanded_label, "Others")

        logger.debug(
            f"Stage 3 zero-shot — label='{top_label}', score={top_score:.4f}, "
            f"user_id={user_id}, text='{process_text[:60]}'"
        )

        if top_score >= settings.BART_ENTAILMENT_THRESHOLD:
            return AssignmentResult(
                category=top_label,
                confidence=round(top_score, 4),
                source='zero_shot',
                needs_review=False,
                gating_alpha=gating_alpha,
                top3_similar_txns=top3_txn_ids
            )

        # Zero-shot result below threshold → return best guess but flag for review
        return AssignmentResult(
            category=top_label,
            confidence=round(top_score, 4),
            source='manual',
            needs_review=True,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )

    except Exception as e:
        logger.error(
            f"Stage 3 zero-shot failed for user_id={user_id}, "
            f"text='{process_text[:60]}': {e}"
        )
        return AssignmentResult(
            category="Others",
            confidence=0.0,
            source='manual',
            needs_review=True,
            gating_alpha=gating_alpha,
            top3_similar_txns=top3_txn_ids
        )