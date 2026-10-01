"""The two local models behind the model-backed components.

`LocalModels` is the interface the components depend on. `HuggingFaceModels`
loads an inference cross-encoder and a sentence encoder on first use and keeps
them; a test passes its own implementation of the two methods instead.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
INFERENCE = "cross-encoder/nli-deberta-v3-base"


class LocalModels(ABC):
    """What the model-backed components need from a model provider."""

    @abstractmethod
    def contradiction(self, premise: str, hypothesis: str) -> float:
        """Probability that `hypothesis` contradicts `premise`."""

    @abstractmethod
    def similarity(self, left: str, right: str) -> float:
        """Cosine similarity of the two texts in the encoder's space."""


class HuggingFaceModels(LocalModels):
    """The encoder and the inference model, loaded lazily from the local model cache."""

    def __init__(
        self,
        encoder: str = ENCODER,
        inference: str = INFERENCE,
        *,
        offline: bool = True,
    ) -> None:
        self.encoder_name = encoder
        self.inference_name = inference
        self.offline = offline
        self._encoder = None
        self._inference = None
        self._contradiction_index: int | None = None

    def _load(self) -> None:
        if self._encoder is not None:
            return
        if self.offline:
            # Loading from the cache only keeps a scoring run from downloading silently.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self._encoder = SentenceTransformer(self.encoder_name)
        self._inference = CrossEncoder(self.inference_name)
        labels = self._inference.model.config.id2label
        matching = [
            index for index, label in labels.items() if str(label).lower().startswith("contrad")
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"{self.inference_name}: expected one contradiction label, got {labels}"
            )
        self._contradiction_index = matching[0]

    def contradiction(self, premise: str, hypothesis: str) -> float:
        self._load()
        import numpy as np

        logits = np.asarray(self._inference.predict([(premise, hypothesis)]), dtype=float)[0]
        exponent = np.exp(logits - logits.max())
        probabilities = exponent / exponent.sum()
        return float(probabilities[self._contradiction_index])

    def similarity(self, left: str, right: str) -> float:
        self._load()
        from sentence_transformers import util

        encoded_left = self._encoder.encode(left, convert_to_tensor=True)
        encoded_right = self._encoder.encode(right, convert_to_tensor=True)
        return float(util.cos_sim(encoded_left, encoded_right)[0][0])
