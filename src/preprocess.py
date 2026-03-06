"""
preprocess.py — Corpus loading and cleaning for 20 Newsgroups.

DESIGN PHILOSOPHY:
The 20NG dataset is deceptively dirty. Naively embedding raw posts would let
any downstream model "cheat" by memorising metadata rather than learning
semantics. Every cleaning decision below is deliberate.
"""

import re
import logging
from typing import List, Tuple

from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)


def load_and_clean(
    subset: str = "all",
    min_words: int = 30,
    max_words: int = 500,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Load 20 Newsgroups and return (texts, labels, target_names).

    DECISION — remove headers/footers/quotes (sklearn built-in):
        Headers contain lines like "Newsgroup: rec.sport.hockey" and
        "From: user@domain" — explicit category/identity signals. A model
        trained on them memorises metadata, not meaning. Footers and
        quoted-reply stacks are similarly non-semantic. We strip all three.

    DECISION — min_words=30:
        Sub-30-word posts are overwhelmingly noise: accidental posts, "+1"
        one-liners, bounced mail, or leftover signatures after stripping.
        They carry no meaningful semantic signal and would corrupt cluster
        centroids.

    DECISION — max_words=500:
        sentence-transformers truncates inputs at 256-512 tokens regardless.
        Feeding 2000-word posts wastes time and only the first ~400 words
        actually influence the embedding. We truncate early so every
        document has roughly equal representation in embedding space.

    DECISION — ASCII-only (soft strip, not drop):
        ~2% of posts contain mojibake or binary attachment fragments.
        We strip non-ASCII bytes but keep the surrounding clean text,
        rather than discarding the post entirely.

    DECISION — remove emails and URLs:
        These are identifiers, not semantics. "user@mit.edu" tells us
        nothing about the post's topic; keeping it would add spurious
        variance to embeddings.
    """
    raw = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    texts: List[str] = []
    labels: List[int] = []

    for text, label in zip(raw.data, raw.target):
        # --- soft ASCII normalisation ---
        text = text.encode("ascii", errors="ignore").decode("ascii")

        # --- remove identifiers ---
        text = re.sub(r"\S+@\S+", " ", text)          # emails
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs

        # --- remove lines that are purely punctuation/numbers (e.g. PGP keys) ---
        lines = [ln for ln in text.splitlines() if re.search(r"[a-zA-Z]{3,}", ln)]
        text = " ".join(lines)

        # --- collapse whitespace ---
        text = re.sub(r"\s+", " ", text).strip()

        word_count = len(text.split())

        if word_count < min_words:
            continue  # too short to carry semantic content

        if word_count > max_words:
            text = " ".join(text.split()[:max_words])  # truncate overlong posts

        texts.append(text)
        labels.append(int(label))

    dropped = len(raw.data) - len(texts)
    logger.info(
        f"Loaded {len(texts)} documents from 20NG "
        f"(dropped {dropped} / {len(raw.data)} as noise, "
        f"{dropped / len(raw.data):.1%})"
    )
    return texts, labels, list(raw.target_names)
