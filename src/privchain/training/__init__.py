"""Centralized training utilities (Phase 1).

Structural note (CLAUDE.md §2/§9): ``training/`` is an explicit, documented
extension of the mandated ``src/privchain/`` layout — the centralized training
loop, experiment logging, and data-split helpers need a home in ``src`` so they
are unit-testable. See ADR-0002.
"""
