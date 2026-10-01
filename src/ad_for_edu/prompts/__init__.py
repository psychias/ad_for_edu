"""The prompts of the pipeline: templates, the shared prefix, and input/output contracts."""

from .catalogue import ENTRIES, PromptEntry, check_catalogue, entry, names, template
from .contracts import (
    PromptContract,
    check_inputs,
    check_output,
    quote_is_grounded,
)
from .prefix import shared_prefix
from .templates import PromptTemplate, read_data, read_text

__all__ = [
    "ENTRIES",
    "PromptContract",
    "PromptEntry",
    "PromptTemplate",
    "check_catalogue",
    "check_inputs",
    "check_output",
    "entry",
    "names",
    "quote_is_grounded",
    "read_data",
    "read_text",
    "shared_prefix",
    "template",
]
