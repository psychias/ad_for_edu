"""A metric that compares a description with the picture instead of with references.

The image and the text are embedded in one space and their cosine similarity is
scaled to the range the metric is usually reported on. Negative similarity is
taken as no alignment rather than as evidence against.

The model reads only the first tokens of a text. A description longer than that
is scored on its beginning, so the value is not comparable between a short
description and a long one; the number of truncated items is reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .base import METRICS, Metric, ScoringItem

#: The factor the metric is conventionally reported with.
SCALE = 2.5
#: How many tokens of a text the model reads.
TOKEN_LIMIT = 77


@METRICS.register("clipscore")
class ImageTextAlignment(Metric):
    """Alignment of a description with the still of its moment."""

    needs_image = True

    def __init__(self, model: str = "openai/clip-vit-base-patch32", scale: float = SCALE) -> None:
        self.model_name = model
        self.scale = scale
        self.truncated = 0
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model.eval()

    def one(self, text: str, image: Path) -> float | None:
        if not Path(image).is_file():
            return None
        self._load()
        import torch
        from PIL import Image

        with Image.open(image) as opened:
            picture = opened.convert("RGB")
        inputs = self._processor(
            text=[text], images=[picture], return_tensors="pt", padding=True, truncation=True
        )
        if int(inputs["input_ids"].shape[-1]) >= TOKEN_LIMIT:
            self.truncated += 1
        with torch.no_grad():
            output = self._model(**inputs)
        image_embedding = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
        text_embedding = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
        cosine = float((image_embedding * text_embedding).sum())
        return self.scale * max(cosine, 0.0)

    def score(self, items: Sequence[ScoringItem]) -> list[float | None]:
        return [
            self.one(item.text or "", item.image) if self.scorable(item) else None
            for item in items
        ]
