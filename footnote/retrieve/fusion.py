"""Reciprocal Rank Fusion.

Rank-based, so it sidesteps the fact that dense cosine scores and BM25 scores
live on incomparable scales — normalising those is fragile; ranks are not.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1/(k + rank).

    Returns ids with fused scores, best first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
