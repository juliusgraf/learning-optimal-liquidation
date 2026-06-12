"""Pytest configuration for the Phase-1 audit tests.

Registers the `legacy` marker used by the characterization tests
(see CLAUDE.md, Phase 1 deliverable E).
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "legacy: characterization tests that pin the behavior of the legacy "
        "main.py/data.py emulators (pre-refactor golden values)",
    )
