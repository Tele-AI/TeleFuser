try:
    from ._version import __version__
except ModuleNotFoundError as exc:
    if exc.name != "telefuser._version":
        raise
    __version__ = "0.0.0+unknown"
