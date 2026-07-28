"""AIPerf CLI bootstrap with TeleFuser adapters registered."""

from __future__ import annotations


def main() -> None:
    """Register TeleFuser adapters and delegate to the AIPerf CLI."""

    from telefuser_aiperf import register_adapters

    register_adapters()

    from aiperf.cli import app

    app()


if __name__ == "__main__":
    main()
