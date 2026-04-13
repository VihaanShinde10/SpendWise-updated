"""
Background task: runs the full ML pipeline on a batch of transactions.
Triggered by FastAPI BackgroundTasks after CSV upload.
"""
import asyncio
from datetime import datetime
from loguru import logger

from app.ml.pipeline import run_pipeline
from app.ml.layers.layer1_semantic import FAISSIndex, encode_text
from app.ml.layers.layer2_behavioural import HDBSCANClusterer
from app.ml.layers.layer3_gating import GatingNetwork
from app.db.client import get_supabase


async def _load_user_ml_context(user_id: str) -> tuple:
    """Load per-user FAISS index, clusterer, gating network, history, cluster map, count."""
    supabase = get_supabase()

    # FAISS index (loads from disk if exists)
    faiss_index = FAISSIndex(user_id)

    # HDBSCAN clusterer (needs to be refitted from history)
    hdbscan_clusterer = HDBSCANClusterer()

    # Gating network
    gating_network = GatingNetwork()

    # Load gating model weights from DB if available
    try:
        gw_resp = supabase.table("gating_model_state").select("*").eq("user_id", user_id).execute()
        if gw_resp.data:
            gw_data = gw_resp.data[0]
            gating_network = GatingNetwork.from_dict({
                'W1': gw_data['W1'], 'b1': gw_data['b1'],
                'W2': gw_data['W2'], 'b2': gw_data['b2']
            })
    except Exception as e:
        logger.warning(f"Could not load gating weights for {user_id}: {e}")

    # Load transaction history
    history_resp = supabase.table("transactions").select(
        "id, merchant_name, transaction_date, category_id, processing_status"
    ).eq("user_id", user_id).eq("processing_status", "completed").order(
        "transaction_date", desc=True
    ).limit(500).execute()

    history = []
    for t in (history_resp.data or []):
        try:
            if t.get('transaction_date'):
                t['transaction_date'] = datetime.fromisoformat(
                    t['transaction_date'].replace('Z', '+00:00')
                ).replace(tzinfo=None)
        except Exception:
            pass
        history.append(t)

    # Load cluster label map from DB
    cluster_map = {}
    try:
        cl_resp = supabase.table("user_clusters").select("*").eq("user_id", user_id).execute()
        for c in (cl_resp.data or []):
            cluster_map[c['cluster_id']] = c['label_name']
    except Exception as e:
        logger.warning(f"Could not load cluster map: {e}")

    lifetime_count = len(history)

    # Refit HDBSCAN if we have enough history with feature vectors
    # (simplified: rely on FAISS proximity for now, HDBSCAN refits per batch)

    return faiss_index, hdbscan_clusterer, gating_network, history, cluster_map, lifetime_count


async def process_transactions_batch(user_id: str, transaction_ids: list):
    """
    Runs the 5-layer ML pipeline for each transaction in the batch.
    Called by FastAPI BackgroundTasks after CSV upload.
    """
    supabase = get_supabase()
    logger.info(f"Starting ML processing for {len(transaction_ids)} transactions (user={user_id})")

    faiss_index, hdbscan_clusterer, gating_network, history, cluster_map, count = \
        await _load_user_ml_context(user_id)

    # Resolve category name → UUID map
    categories_resp = supabase.table("categories").select("id, name").execute()
    cat_name_to_id = {c['name']: c['id'] for c in (categories_resp.data or [])}

    new_embeddings = []
    new_labels = []
    new_txn_ids = []

    for txn_id in transaction_ids:
        try:
            txn_resp = supabase.table("transactions").select("*").eq("id", txn_id).single().execute()
            txn = txn_resp.data

            if not txn or txn.get('processing_status') == 'completed':
                continue

            # Convert date string to datetime
            if txn.get('transaction_date'):
                try:
                    txn['transaction_date'] = datetime.fromisoformat(
                        txn['transaction_date'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                except Exception:
                    txn['transaction_date'] = datetime.now()

            # Run pipeline
            result = await run_pipeline(
                user_id=user_id,
                transaction_data=txn,
                faiss_index=faiss_index,
                hdbscan_clusterer=hdbscan_clusterer,
                gating_network=gating_network,
                user_transaction_history=history,
                cluster_label_map=cluster_map,
                lifetime_txn_count=count
            )

            # Resolve category_id
            category_id = cat_name_to_id.get(result.category)

            # Save result
            supabase.table("transactions").update({
                "category_id": category_id,
                "category_source": result.source,
                "confidence_score": round(result.confidence, 4),
                "gating_alpha": round(result.gating_alpha, 4),
                "needs_review": result.needs_review,
                "is_recurring": txn.get('is_recurring', False),
                "processing_status": "completed",
                "processed_at": datetime.utcnow().isoformat(),
            }).eq("id", txn_id).execute()

            # Add to FAISS index for future lookups
            if result.category and result.category != "Others":
                embedding = encode_text(txn.get('merchant_name') or txn.get('raw_description', '')[:50])
                new_embeddings.append(embedding)
                new_labels.append(result.category)
                new_txn_ids.append(txn_id)

            count += 1

        except Exception as e:
            logger.error(f"Failed to process transaction {txn_id}: {e}")
            try:
                supabase.table("transactions").update({
                    "processing_status": "failed"
                }).eq("id", txn_id).execute()
            except Exception:
                pass

    # Batch-add new embeddings to FAISS
    if new_embeddings:
        import numpy as np
        emb_matrix = np.stack(new_embeddings, axis=0)
        faiss_index.add(emb_matrix, new_labels, new_txn_ids)
        logger.info(f"Added {len(new_embeddings)} embeddings to FAISS index for user {user_id}")

    logger.info(f"Batch processing complete for user {user_id}")
