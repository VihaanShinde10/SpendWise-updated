"""
Layer 1: E5 Transformer Embeddings + FAISS Nearest Neighbour Retrieval.
Implements Section 4.3 of the paper (Equations 1).
"""
import logging
import os
import pickle
import threading
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Singleton model — loaded once at startup, thread-safe
# ---------------------------------------------------------------------------
_e5_model: Optional[SentenceTransformer] = None
_e5_model_lock = threading.Lock()


def get_e5_model() -> SentenceTransformer:
    global _e5_model
    if _e5_model is None:
        with _e5_model_lock:
            # Double-checked locking: re-test after acquiring the lock
            if _e5_model is None:
                logger.info("Loading E5 model: %s", settings.E5_MODEL_NAME)
                _e5_model = SentenceTransformer(
                    settings.E5_MODEL_NAME,
                    cache_folder=settings.MODEL_CACHE_DIR,
                )
    return _e5_model


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_text(text: str) -> np.ndarray:
    """Encode a single merchant name using E5 with query prefix."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"encode_text requires a non-empty string, got {text!r}")
    model = get_e5_model()
    embedding = model.encode(f"query: {text}", normalize_embeddings=True)
    return np.asarray(embedding, dtype=np.float32)


def encode_batch(texts: list) -> np.ndarray:
    """
    Batch-encode merchant names for indexing using the passage prefix.
    Returns float32 array of shape (len(texts), dim).
    """
    if not texts:
        raise ValueError("encode_batch requires a non-empty list")
    model = get_e5_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, batch_size=32)
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

class FAISSIndex:
    """Manages per-user FAISS IVF+PQ index with label storage."""

    def __init__(self, user_id: str):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"user_id must be a non-empty string, got {user_id!r}")
        self.user_id = user_id
        self.dim: int = get_e5_model().get_sentence_embedding_dimension()
        self.index_path  = os.path.join(settings.FAISS_INDEX_DIR, f"{user_id}.index")
        self.labels_path = os.path.join(settings.FAISS_INDEX_DIR, f"{user_id}.labels")
        self.index: Optional[faiss.Index] = None
        self.labels: list = []    # category labels parallel to index vectors
        self.txn_ids: list = []   # transaction IDs parallel to index vectors
        self._load_or_init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_or_init(self):
        os.makedirs(settings.FAISS_INDEX_DIR, exist_ok=True)

        if os.path.exists(self.index_path):
            loaded = self._try_load_from_disk()
            if loaded:
                return

        self._init_fresh()

    def _try_load_from_disk(self) -> bool:
        """
        Attempt to load index + labels from disk.
        Returns True on success, False if anything is wrong (caller will init fresh).
        """
        try:
            temp_index = faiss.read_index(self.index_path)
        except Exception as exc:
            logger.warning(
                "FAISS index for user %r is unreadable (%s) — rebuilding.",
                self.user_id, exc,
            )
            self._remove_index_files()
            return False

        if temp_index.d != self.dim:
            logger.warning(
                "FAISS index dim mismatch for user %r (stored %d, model %d) — rebuilding.",
                self.user_id, temp_index.d, self.dim,
            )
            self._remove_index_files()
            return False

        if not os.path.exists(self.labels_path):
            logger.warning(
                "Labels file missing for user %r — rebuilding.",
                self.user_id,
            )
            self._remove_index_files()
            return False

        try:
            with open(self.labels_path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                raise ValueError("Labels file is not a dict")
            labels  = data.get('labels',  [])
            txn_ids = data.get('txn_ids', [])
            if not isinstance(labels, list) or not isinstance(txn_ids, list):
                raise ValueError("labels/txn_ids must be lists")
            if len(labels) != len(txn_ids):
                raise ValueError(
                    f"labels ({len(labels)}) and txn_ids ({len(txn_ids)}) length mismatch"
                )
        except Exception as exc:
            logger.warning(
                "Labels file for user %r is corrupt (%s) — rebuilding.",
                self.user_id, exc,
            )
            self._remove_index_files()
            return False

        self.index   = temp_index
        self.labels  = labels
        self.txn_ids = txn_ids
        logger.debug(
            "Loaded FAISS index for user %r: %d vectors.", self.user_id, self.total_vectors()
        )
        return True

    def _init_fresh(self):
        """Create a flat inner-product index seeded with canonical category names."""
        logger.info("Initialising fresh FAISS index for user %r.", self.user_id)
        from app.ml.layers.layer4_assign import CATEGORY_NAMES

        self.index   = faiss.IndexFlatIP(self.dim)
        self.labels  = []
        self.txn_ids = []

        if CATEGORY_NAMES:
            embeddings = encode_batch(CATEGORY_NAMES)          # (n_cats, dim)
            self.index.add(embeddings)
            self.labels.extend(CATEGORY_NAMES)
            self.txn_ids.extend([f"seed-{c}" for c in CATEGORY_NAMES])

        self._save()

    def _remove_index_files(self):
        for path in (self.index_path, self.labels_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.error("Could not remove stale file %r: %s", path, exc)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self.index is not None

    def total_vectors(self) -> int:
        return len(self.labels)

    # ------------------------------------------------------------------
    # Add / upgrade
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, labels: list, txn_ids: list):
        """
        Add embeddings to the index.
        Automatically upgrades from IndexFlatIP → IndexIVFPQ when the
        total vector count reaches 256.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.shape[0] != len(labels) or len(labels) != len(txn_ids):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}), labels ({len(labels)}), "
                f"and txn_ids ({len(txn_ids)}) must all have the same length"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} does not match index dim {self.dim}"
            )

        n_after = self.total_vectors() + len(embeddings)

        # Upgrade flat → IVF-PQ once we have enough data for meaningful quantisation
        if n_after >= 256 and isinstance(self.index, faiss.IndexFlatIP):
            self._upgrade_to_ivfpq(embeddings, labels, txn_ids)
            return

        self.index.add(embeddings)
        self.labels.extend(labels)
        self.txn_ids.extend(txn_ids)
        self._save()

    def _upgrade_to_ivfpq(
        self,
        new_embeddings: np.ndarray,
        new_labels: list,
        new_txn_ids: list,
    ):
        """
        Migrate from IndexFlatIP to IndexIVFPQ.
        Uses index.reconstruct_n() — the stable FAISS API for vector retrieval —
        instead of raw pointer arithmetic which is unsafe across FAISS versions.
        """
        n_existing = self.index.ntotal

        # Reconstruct all stored vectors via the official API
        if n_existing > 0:
            existing = np.empty((n_existing, self.dim), dtype=np.float32)
            self.index.reconstruct_n(0, n_existing, existing)
            all_embeddings = np.vstack([existing, new_embeddings])
        else:
            all_embeddings = new_embeddings.copy()

        n_total = len(all_embeddings)

        # IVF cell count: small enough to train with the data we have.
        # FAISS recommends ≥ 39 * nlist training points.
        nlist = min(32, max(4, n_total // 39))

        # PQ sub-quantizers: must divide self.dim evenly.
        # Prefer 8 but clamp to a safe divisor.
        m = 8
        while self.dim % m != 0 and m > 1:
            m -= 1
        if self.dim % m != 0:
            logger.warning(
                "Cannot find valid PQ sub-quantizer count for dim=%d; "
                "keeping IndexFlatIP.", self.dim
            )
            # Fall back: just add to existing flat index
            self.index.add(new_embeddings)
            self.labels.extend(new_labels)
            self.txn_ids.extend(new_txn_ids)
            self._save()
            return

        quantiser = faiss.IndexFlatIP(self.dim)
        new_index = faiss.IndexIVFPQ(quantiser, self.dim, nlist, m, 8)

        logger.info(
            "Upgrading FAISS index for user %r to IVF-PQ "
            "(n=%d, nlist=%d, m=%d).", self.user_id, n_total, nlist, m
        )
        new_index.train(all_embeddings)
        new_index.add(all_embeddings)
        new_index.nprobe = 8

        self.index = new_index
        self.labels.extend(new_labels)
        self.txn_ids.extend(new_txn_ids)
        self._save()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10) -> tuple:
        """
        Return top-k cosine similarities, category labels, and txn_ids.
        Always returns a 3-tuple of lists (never raises on empty index).
        """
        if self.total_vectors() == 0:
            return [], [], []

        k = min(k, self.total_vectors())
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        distances, indices = self.index.search(query, k)

        sims, lbls, tids = [], [], []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                # FAISS returns -1 for unfilled slots (can happen with IVF when
                # nprobe is smaller than available cells)
                continue
            if idx >= len(self.labels):
                logger.warning(
                    "FAISS returned out-of-range index %d (labels len=%d) "
                    "for user %r — skipping.",
                    idx, len(self.labels), self.user_id,
                )
                continue
            sims.append(float(dist))
            lbls.append(self.labels[idx])
            tids.append(self.txn_ids[idx])

        return sims, lbls, tids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.labels_path, 'wb') as f:
                pickle.dump({'labels': self.labels, 'txn_ids': self.txn_ids}, f)
        except Exception as exc:
            logger.error(
                "Failed to persist FAISS index for user %r: %s", self.user_id, exc
            )
            raise


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_semantic_confidence(similarities: list, labels: list) -> tuple:
    """
    Weighted majority vote → (category, confidence).
    Implements Equation 1.

    Only positive-similarity neighbours contribute weight.
    Returns ("Others", 0.0) when all similarities are non-positive or
    the inputs are empty.
    """
    if not labels:
        return "Others", 0.0

    weight_by_label: dict = {}
    total_weight = 0.0

    for sim, label in zip(similarities, labels):
        w = max(0.0, float(sim))
        if w > 0.0:
            weight_by_label[label] = weight_by_label.get(label, 0.0) + w
            total_weight += w

    if total_weight == 0.0 or not weight_by_label:
        # All neighbours had non-positive similarity — no meaningful signal
        return "Others", 0.0

    majority   = max(weight_by_label, key=weight_by_label.get)
    c_sem      = weight_by_label[majority] / total_weight

    return majority, round(c_sem, 4)