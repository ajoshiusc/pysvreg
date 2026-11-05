from __future__ import annotations

try:
    from .runtime import remove_extn_basename
except ImportError:
    from runtime import remove_extn_basename

__all__ = ["remove_extn_basename"]
