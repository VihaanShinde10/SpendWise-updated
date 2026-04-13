"""
Main pipeline orchestrator. Calls all 5 layers in sequence for a transaction.
"""
import math
import numpy as np
from typing import Optional
from datetime import datetime
from loguru import logger

from app.ml.layers.layer0_prep import prepare_transaction, PreparedTransaction
from app.ml.layers.layer1_semantic import FAISSIndex, encode_text, compute_semantic_confidence
from app.ml.layers.layer2_behavioural import HDBSCANClusterer, detect_recurrence, compute_behavioural_confidence
from app.ml.layers.layer3_gating import GatingNetwork, QualityIndicator
from app.ml.layers.layer4_assign import assign_category, AssignmentResult, CATEGORY_NAMES
from app.config import get_settings

settings = get_settings()


async def run_pipeline(
    user_id: str,
    transaction_data: dict,
    faiss_index: FAISSIndex,
    hdbscan_clusterer: HDBSCANClusterer,
    gating_network: GatingNetwork,
    user_transaction_history: list,
    cluster_label_map: dict,
    lifetime_txn_count: int
) -> AssignmentResult:

    # --- LAYER 0: Data Preparation ---
    logger.info(f"[L0] Preparing: {transaction_data.get('raw_description', '')[:50]}")
    prepped: PreparedTransaction = prepare_transaction(
        raw_description=transaction_data['raw_description'],
        amount=float(transaction_data['amount']),
        direction=transaction_data['direction'],
        payment_method=transaction_data.get('payment_method', 'UPI'),
        transaction_date=transaction_data['transaction_date'],
        balance=transaction_data.get('balance')
    )

    # Compute behavioural features from history
    merchant_history = [
        t for t in user_transaction_history
        if t.get('merchant_name', '').lower() == prepped.merchant_name.lower()
    ]
    merchant_timestamps = [
        t['transaction_date'] for t in merchant_history
        if isinstance(t.get('transaction_date'), datetime)
    ]
    rec_result = detect_recurrence(merchant_timestamps)

    prepped.merchant_freq = len(merchant_history)
    prepped.is_recurring = int(rec_result.is_recurring)
    prepped.recurrence_strength = rec_result.strength
    prepped.mean_interval = rec_result.mean_interval
    prepped.std_interval = rec_result.std_interval

    feature_vec = prepped.to_feature_vector()

    # --- LAYER 1: Semantic ---
    logger.info(f"[L1] Encoding: '{prepped.merchant_name}'")
    embedding = encode_text(prepped.merchant_name)

    sims, neighbour_labels, neighbour_txn_ids = [], [], []
    c_sem = 0.0
    sem_category = None
    sem_reliable = False

    if faiss_index.total_vectors() > 0:
        sims, neighbour_labels, neighbour_txn_ids = faiss_index.search(embedding, k=settings.FAISS_K_NEIGHBOURS)
        sem_category, c_sem = compute_semantic_confidence(sims, neighbour_labels)
        sem_reliable = (
            prepped.token_count > 2
            and c_sem > settings.SEMANTIC_CONF_THRESHOLD
            and (sims[0] if sims else 0) > settings.SEMANTIC_COSINE_THRESHOLD
        )
    else:
        logger.info("[L1] FAISS empty — skipping semantic search phase.")

    # --- LAYER 2: Behavioural ---
    cluster_id, cluster_stability = hdbscan_clusterer.predict_single(feature_vec)
    beh_category = cluster_label_map.get(cluster_id) if cluster_id >= 0 else None
    c_beh = compute_behavioural_confidence(rec_result.strength, cluster_stability)
    beh_reliable = cluster_stability >= 0.60 and beh_category is not None

    # --- LAYER 3: Adaptive Gating ---
    is_new_user = lifetime_txn_count < settings.COLD_START_THRESHOLD
    qi = QualityIndicator(
        token_count=float(prepped.token_count),
        char_length=float(prepped.char_length),
        has_url_flag=float(prepped.has_url_or_email),
        log_merchant_freq=math.log1p(prepped.merchant_freq),
        semantic_confidence=c_sem,
        recurrence_strength=rec_result.strength,
        is_new_user=float(is_new_user)
    )
    alpha = gating_network.forward(qi.to_vector())
    logger.info(f"[L3] Gating Alpha: {alpha:.3f} (Semantic Weight)")

    # Route to processing path
    top3_txn_ids = neighbour_txn_ids[:3] if neighbour_txn_ids else []
    
    if sem_reliable and not beh_reliable:
        source = 'semantic'
        final_confidence = c_sem
        chosen_category = sem_category
    elif beh_reliable and not sem_reliable:
        source = 'behavioural'
        final_confidence = c_beh
        chosen_category = beh_category
    elif sem_reliable and beh_reliable:
        # True fusion
        z_sem = _category_to_vector(sem_category)
        z_beh = _category_to_vector(beh_category)
        z_final = gating_network.fuse(z_sem, z_beh, alpha)
        chosen_category = _vector_to_category(z_final)
        final_confidence = round(alpha * c_sem + (1 - alpha) * c_beh, 4)
        source = 'fused'
    else:
        source = 'zero_shot'
        final_confidence = 0.0
        chosen_category = None

    # --- LAYER 4: Assignment ---
    result = assign_category(
        user_id=user_id,
        cluster_membership_prob=cluster_stability,
        cluster_category=chosen_category,
        neighbour_categories=neighbour_labels[:5] if neighbour_labels else [],
        neighbour_distances=[1 - s for s in sims[:5]] if sims else [],
        merchant_name=prepped.merchant_name,
        final_confidence=final_confidence,
        source=source,
        gating_alpha=alpha,
        top3_txn_ids=top3_txn_ids,
        raw_description=transaction_data['raw_description']
    )

    logger.info(f"[L4] → {result.category} ({result.source}, conf={result.confidence:.3f})")
    return result


def _category_to_vector(category: Optional[str]) -> np.ndarray:
    vec = np.zeros(len(CATEGORY_NAMES), dtype=np.float32)
    if category and category in CATEGORY_NAMES:
        vec[CATEGORY_NAMES.index(category)] = 1.0
    return vec


def _vector_to_category(vec: np.ndarray) -> str:
    return CATEGORY_NAMES[int(np.argmax(vec))]
