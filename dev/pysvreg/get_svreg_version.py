from __future__ import annotations

try:
    from .runtime import get_svreg_version
except ImportError:
    from runtime import get_svreg_version

__all__ = ["get_svreg_version"]
