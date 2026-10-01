"""Calling models: requests and replies, providers, the catalogue, pricing, reply parsing."""

from . import providers as _providers  # noqa: F401  (registers the providers)
from .base import (
    PROVIDERS,
    LLMProvider,
    LLMReply,
    LLMRequest,
    ProviderError,
    TerminalProviderError,
    terminal_condition,
)
from .catalog import ROLES, ModelCatalog, ModelEntry, check_providers
from .client import LLMClient
from .pricing import PriceTable, Shape
from .replies import (
    extract_json,
    keep_raw,
    parse_choice,
    parse_scale,
    parse_verdict,
)

__all__ = [
    "PROVIDERS",
    "ROLES",
    "LLMClient",
    "LLMProvider",
    "LLMReply",
    "LLMRequest",
    "ModelCatalog",
    "ModelEntry",
    "PriceTable",
    "ProviderError",
    "Shape",
    "TerminalProviderError",
    "check_providers",
    "extract_json",
    "keep_raw",
    "parse_choice",
    "parse_scale",
    "parse_verdict",
    "terminal_condition",
]
