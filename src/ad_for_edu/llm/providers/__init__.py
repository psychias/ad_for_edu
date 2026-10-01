"""The providers. Importing the package registers all of them.

The client libraries of the services are imported when a provider first
connects, so importing this package needs none of them.
"""

from . import gemini, huggingface_local, openai_compatible

__all__ = ["gemini", "huggingface_local", "openai_compatible"]
