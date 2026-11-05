from __future__ import annotations

try:
    from .runtime import disp1
except ImportError:
    from runtime import disp1

__all__ = ["disp1"]
