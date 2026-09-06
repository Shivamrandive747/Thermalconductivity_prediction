"""Pluggable extraction backends.

A backend turns a full-text document (PDF or Elsevier XML text) into a JSON dict
matching `schema.PaperExtraction`. Which backend runs is decided by
`cfg.resolve_backend()` (explicit EXTRACTION_BACKEND, else auto-detect from keys).

Interface:
    backend.extract_paper(doi, path, kind, system_prompt, json_schema) -> dict
    backend.estimate_cost_usd(path, kind) -> float

`get_classifier()` returns a cheap title+abstract -> 'keep'/'skip' function used
by s03 --llm, or None if no key is configured.
"""
from __future__ import annotations

from ..config import cfg


class BackendUnavailable(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """Daily/plan quota is used up — won't recover today; the run should stop."""


def get_backend():
    """Return the active extraction backend module, or raise with guidance."""
    name = cfg.resolve_backend()
    if name == "gemini":
        if not cfg.gemini_key and not cfg.gcp_project:
            raise BackendUnavailable(
                "Gemini selected but neither GEMINI_API_KEY nor GCP_PROJECT is set in .env."
            )
        from . import gemini
        return gemini
    if name == "anthropic":
        if not cfg.anthropic_key:
            raise BackendUnavailable(
                "Anthropic selected but ANTHROPIC_API_KEY is not set in .env."
            )
        from . import anthropic_batch
        return anthropic_batch
    raise BackendUnavailable(
        "No extraction backend configured. Set GEMINI_API_KEY (free tier at "
        "aistudio.google.com) or ANTHROPIC_API_KEY in .env, or set "
        "EXTRACTION_BACKEND explicitly."
    )


def get_classifier():
    """Cheap (title, abstract) -> 'keep'/'skip'/'maybe' classifier, or None."""
    try:
        backend = get_backend()
    except BackendUnavailable:
        return None
    return getattr(backend, "classify", None)
