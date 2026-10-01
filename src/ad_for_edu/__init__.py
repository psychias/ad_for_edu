"""Audio description for slide-based lecture recordings.

The package is organised by pipeline stage. Every stage is a context object that
receives its interchangeable parts (strategies) through its constructor; the
strategies of each family are registered by name in a `Registry` that sits beside
the family's abstract base.
"""

__version__ = "0.1.0"
