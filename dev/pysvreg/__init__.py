"""Python entrypoints for the SVReg pipeline, loaded on demand."""

__all__ = ["main", "svreg"]


def __getattr__(name: str):
    if name in __all__:
        from .svreg import main, svreg

        return {"main": main, "svreg": svreg}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
