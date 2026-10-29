"""Paired latent semantic analysis, learned only from approved transaction pairs.

Each reviewed pair is a bilingual document: bank words and register words have
separate vocabularies. SVD of the paired TF-IDF matrix discovers shared factors.
New records project into that shared space. This is inspired by cross-language
LSA, not a reproduction of Chew's mutual-information alignment algorithm.
"""
from __future__ import annotations

import re
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import Transaction


def namespaced_text(transaction: Transaction, side: str) -> str:
    # Drop arbitrary store numbers; check references are compared separately.
    words = re.findall(r"[a-z]+", transaction.text.lower())
    tokens = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return " ".join(f"{side}__{word}" for word in tokens)


class PairModel:
    def __init__(self, components: int = 16):
        self.components = components
        self.vectorizer = None
        self.svd = None
        self.training_pairs = 0
        self.lag_center = 2.0
        self.lag_scale = 3.0

    def fit(self, pairs: list[tuple[Transaction, Transaction]]) -> PairModel:
        self.training_pairs = len(pairs)
        self.vectorizer = self.svd = None
        self.lag_center, self.lag_scale = 2.0, 3.0
        if pairs:
            lags = np.array([(b.date - r.date).days for b, r in pairs])
            self.lag_center = float(np.median(lags))
            self.lag_scale = max(2.0, float(np.median(np.abs(lags - self.lag_center))) * 1.4826)
        if len(pairs) < 3:
            return self
        docs = [namespaced_text(b, "b") + " " + namespaced_text(r, "r") for b, r in pairs]
        if not any(d.strip() for d in docs):
            return self
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
        matrix = self.vectorizer.fit_transform(docs)
        dimensions = min(self.components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if dimensions < 1:
            self.vectorizer = None
            return self
        self.svd = TruncatedSVD(n_components=dimensions, random_state=42)
        self.svd.fit(matrix)
        return self

    def similarities(self, bank: list[Transaction], register: list[Transaction], mode: str = "learned") -> np.ndarray:
        if not bank or not register:
            return np.zeros((len(bank), len(register)))
        if mode == "numeric":
            return np.zeros((len(bank), len(register)))
        lexical = self._lexical(bank, register)
        if mode == "lexical" or self.svd is None:
            return lexical
        b = self.vectorizer.transform([namespaced_text(t, "b") for t in bank])
        r = self.vectorizer.transform([namespaced_text(t, "r") for t in register])
        bank_vectors, register_vectors = self.svd.transform(b), self.svd.transform(r)
        usable_bank = np.linalg.norm(bank_vectors, axis=1) > 1e-12
        usable_register = np.linalg.norm(register_vectors, axis=1) > 1e-12
        bv, rv = normalize(bank_vectors), normalize(register_vectors)
        learned = np.clip(bv @ rv.T, 0, 1)
        combined = 0.85 * learned + 0.15 * lexical
        # A zero projection means no learned evidence, not evidence of dissimilarity.
        # Preserve the lexical fallback if either side has no usable representation.
        usable_pair = usable_bank[:, None] & usable_register[None, :]
        return np.where(usable_pair, combined, lexical)

    @staticmethod
    def _lexical(bank: list[Transaction], register: list[Transaction]) -> np.ndarray:
        docs = [t.text.lower() for t in bank + register]
        # Unlabelled candidate text is allowed; no pairing information enters this fallback.
        if not any(len(d.strip()) >= 2 for d in docs):
            return np.zeros((len(bank), len(register)))
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        matrix = vectorizer.fit_transform(docs)
        return (matrix[:len(bank)] @ matrix[len(bank):].T).toarray()

    def lag_score(self, gap: int, learned: bool = True) -> float:
        center, scale = (self.lag_center, self.lag_scale) if learned else (2.0, 3.0)
        return float(np.exp(-0.5 * ((gap - center) / scale) ** 2))

