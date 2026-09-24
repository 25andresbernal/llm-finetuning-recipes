"""Shared library code behind the scripts and the training recipe.

Everything in this package is plain Python with no GPU dependencies, so it
imports and runs on any machine. The heavy training dependencies (torch,
unsloth) live only in ``recipes/conversational-agent-qlora/train.py`` and are
imported lazily, gated behind the training path so that ``--dry-run`` never
touches them.
"""

__version__ = "0.1.0"
