"""The six components. Importing the package registers all of them."""

from . import deixis, faithfulness, length, non_redundancy, style, terminology
from .base import COMPONENTS, Component

__all__ = [
    "COMPONENTS",
    "Component",
    "deixis",
    "faithfulness",
    "length",
    "non_redundancy",
    "style",
    "terminology",
]
