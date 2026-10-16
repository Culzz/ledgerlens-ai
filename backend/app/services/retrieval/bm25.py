"""A dependency-free BM25 retriever.

Why lexical first: it needs no model, no download and no GPU, so retrieval can
be measured before any of that is set up. It is also a real baseline rather than
a placeholder. Financial tables reward keyword matching, because a question and
the row that answers it tend to share literal tokens - a year, a line item name,
a number.

The plan is hybrid retrieval. This is the lexical half, measured on its own, so
that when dense retrieval is added there is something honest to compare it with.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

# Words, and numbers kept whole including decimals. Commas inside numbers are
# removed before tokenising, so "1,234" and "1234" produce the same token and a
# question phrased either way will match.
_NUMBER_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z]+|[0-9]+(?:\.[0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words and numbers, keep numbers intact."""
    lowered = str(text).lower()
    lowered = _NUMBER_COMMA.sub("", lowered)
    return _TOKEN.findall(lowered)


class BM25Index:
    """Okapi BM25 backed by an inverted index.

    Scores are only accumulated for documents named in ``candidates``. Retrieval
    happens inside one document, so scoring the whole corpus for every question
    would be wasted work.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._documents = 0
        self._average_length = 0.0

    def __len__(self) -> int:
        return self._documents

    @property
    def average_length(self) -> float:
        return self._average_length

    def add(self, doc_id: str, text: str) -> None:
        """Index one document. ``doc_id`` must be unique."""
        if doc_id in self._lengths:
            raise ValueError(f"document {doc_id!r} is already indexed")

        tokens = tokenize(text)
        self._lengths[doc_id] = len(tokens)
        for term, frequency in Counter(tokens).items():
            self._postings.setdefault(term, {})[doc_id] = frequency
        self._documents += 1

    def add_many(self, documents: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in documents:
            self.add(doc_id, text)
        self.finalize()

    def finalize(self) -> None:
        """Compute the average document length. Call once, after adding."""
        if not self._documents:
            self._average_length = 0.0
            return
        self._average_length = sum(self._lengths.values()) / self._documents

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency, floored at zero.

        The floor matters: a term that appears in every candidate carries no
        signal, and the raw formula would make it negative and actively push
        documents down.
        """
        document_frequency = len(self._postings.get(term, ()))
        if document_frequency == 0:
            return 0.0
        raw = math.log(
            1 + (self._documents - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        return max(0.0, raw)

    def search(
        self,
        query: str,
        *,
        candidates: set[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rank candidates against a query, best first.

        Documents that match nothing are omitted rather than returned with a
        score of zero, so the caller can tell "not found" from "found, weakly".
        """
        scores: dict[str, float] = defaultdict(float)
        k1, b = self.k1, self.b
        average = self._average_length or 1.0

        for term in tokenize(query):
            postings = self._postings.get(term)
            if not postings:
                continue
            term_idf = self.idf(term)
            if term_idf <= 0.0:
                continue
            for doc_id, frequency in postings.items():
                if candidates is not None and doc_id not in candidates:
                    continue
                length = self._lengths[doc_id]
                norm = 1 - b + b * (length / average)
                scores[doc_id] += term_idf * (frequency * (k1 + 1)) / (
                    frequency + k1 * norm
                )

        # Ties broken by id, so repeated runs produce the same ranking.
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]
