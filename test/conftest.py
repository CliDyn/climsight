"""Shared pytest configuration for ClimSight tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked 'destine' unless explicitly selected with -m destine."""
    # Check if user explicitly requested destine marker via -m
    markexpr = config.getoption("-m", default="")
    if "destine" in markexpr:
        return  # user explicitly asked for destine tests, don't skip

    skip_destine = pytest.mark.skip(
        reason="DestinE tests skipped by default. Run with: pytest -m destine"
    )
    for item in items:
        if "destine" in item.keywords:
            item.add_marker(skip_destine)
