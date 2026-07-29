"""Canned agent outputs for the offline integration layer.

One module per stage. Every payload is a HAND-WRITTEN Pydantic constructor, not
a captured JSON snapshot: a snapshot rots silently when a field is renamed or a
validator tightens, whereas a constructor fails loudly at import time -- which
is the whole point of running these models through the real pipeline.
"""
