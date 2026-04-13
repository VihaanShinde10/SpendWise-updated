"""
Layer 1: E5 Transformer Embeddings + FAISS Nearest Neighbour Retrieval.
Implements Section 4.3 of the paper (Equations 1).
"""
import numpy as np
import faiss
import os
import pickle
from typing import Optional
from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()

# Singleton model — loaded once at startup
_e5_model: Optional[SentenceTransformer] = None


def get_e5_model() -> SentenceTransformer:
    global _e5_model
    if _e5_model is None:
        _e5_model = SentenceTransformer(
            settings.E5_MODEL_NAME,
            cache_folder=settings.MODEL_CACHE_DIR
        )
    return _e5_model


def encode_text(text: str) -> np.ndarray:
    """Encode a single merchant name using E5 with query prefix."""
    model = get_e5_model()
    prefixed = f"query: {text}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.astype(np.float32)


def encode_batch(texts: list) -> np.ndarray:
    """Batch encode merchant names for indexing."""
    model = get_e5_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, batch_size=32)
    return embeddings.astype(np.float32)


class FAISSIndex:
    """Manages per-user FAISS IVF+PQ index with label storage."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        # Get dimension dynamically from the model
        model = get_e5_model()
        self.dim = model.get_sentence_embedding_dimension()
        
        self.index_path = os.path.join(settings.FAISS_INDEX_DIR, f"{user_id}.index")
        self.labels_path = os.path.join(settings.FAISS_INDEX_DIR, f"{user_id}.labels")
        self.index = None
        self.labels: list = []    # category labels parallel to index vectors
        self.txn_ids: list = []   # transaction IDs parallel to index vectors
        self._load_or_init()

    def _load_or_init(self):
        os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)
        should_init_fresh = True
        
        if os.path.exists(self.index_path):
            try:
                temp_index = faiss.read_index(self.index_path)
                if temp_index.d == self.dim:
                    self.index = temp_index
                    with open(self.labels_path, 'rb') as f:
                        data = pickle.load(f)
                        self.labels = data.get('labels', [])
                        self.txn_ids = data.get('txn_ids', [])
                    should_init_fresh = False
                else:
                    os.remove(self.index_path)
                    if os.path.exists(self.labels_path):
                        os.remove(self.labels_path)
            except Exception:
                pass
                
        if should_init_fresh:
            self.index = faiss.IndexFlatIP(self.dim)
            # Seed with categories to handle cold start
            from app.ml.layers.layer4_assign import CATEGORY_NAMES
            seeds = []
            seed_labels = []
            for cat in CATEGORY_NAMES:
                seeds.append(f"passage: {cat}")
                seed_labels.append(cat)
            
            # Use batch encoder for seeds
            embeddings = encode_batch(seed_labels)
            self.index.add(embeddings)
            self.labels.extend(seed_labels)
            self.txn_ids.extend(["seed-" + c for c in seed_labels])
            self._save()

    def is_trained(self) -> bool:
        return self.index is not None

    def total_vectors(self) -> int:
        return len(self.labels)

    def add(self, embeddings: np.ndarray, labels: list, txn_ids: list):
        """Add embeddings to index. Upgrade to IVF-PQ when we have enough data."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n_existing = self.total_vectors()
        n_new = len(embeddings)

        # Upgrade to IVF-PQ index when we have >= 256 vectors for better scalability
        if n_existing + n_new >= 256 and isinstance(self.index, faiss.IndexFlatIP):
            self._upgrade_to_ivfpq(embeddings, labels, txn_ids)
            return

        self.index.add(embeddings)
        self.labels.extend(labels)
        self.txn_ids.extend(txn_ids)
        self._save()

    def _upgrade_to_ivfpq(self, new_embeddings: np.ndarray, new_labels: list, new_txn_ids: list):
        """Upgrade from flat to IVF-PQ index when we have enough data."""
        # Reconstruct all existing vectors from the flat index
        n = self.index.ntotal
        all_embeddings = faiss.rev_swig_ptr(self.index.get_xb(), n * self.dim).reshape(n, self.dim)
        all_embeddings = np.vstack([all_embeddings, new_embeddings])

        quantiser = faiss.IndexFlatIP(self.dim)
        nlist = min(32, max(4, len(all_embeddings) // 8))
        m = self.dim // 8  # Sub-quantizers must be a divisor of the dimension
        new_index = faiss.IndexIVFPQ(quantiser, self.dim, nlist, m, 8)
        new_index.train(all_embeddings)
        new_index.add(all_embeddings)
        new_index.nprobe = 8

        self.index = new_index
        self.labels.extend(new_labels)
        self.txn_ids.extend(new_txn_ids)
        self._save()

    def search(self, query: np.ndarray, k: int = 10) -> tuple:
        """Return top-k cosine similarities, category labels, and txn_ids."""
        if self.total_vectors() == 0:
            return [], [], []
        k = min(k, self.total_vectors())
        query = query.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query, k)
        valid = [(distances[0][i], indices[0][i]) for i in range(len(indices[0])) if indices[0][i] >= 0]
        sims = [v[0] for v in valid]
        lbls = [self.labels[v[1]] for v in valid]
        tids = [self.txn_ids[v[1]] for v in valid]
        return sims, lbls, tids

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.labels_path, 'wb') as f:
            pickle.dump({'labels': self.labels, 'txn_ids': self.txn_ids}, f)


def compute_semantic_confidence(similarities: list, labels: list) -> tuple:
    """Weighted majority vote → category and confidence (Equation 1)."""
    if not labels:
        return "Others", 0.0

    weight_by_label: dict = {}
    total_weight = sum(max(0, s) for s in similarities)

    for sim, label in zip(similarities, labels):
        w = max(0, sim)
        weight_by_label[label] = weight_by_label.get(label, 0) + w

    majority = max(weight_by_label, key=weight_by_label.get)
    c_sem = weight_by_label[majority] / total_weight if total_weight > 0 else 0.0

    return majority, round(c_sem, 4)
