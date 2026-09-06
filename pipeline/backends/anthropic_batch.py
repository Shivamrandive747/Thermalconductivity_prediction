"""Anthropic (Claude) extraction backend — drop-in alternative to Gemini.

Uses `claude-opus-4-8` with:
  - prompt caching on the shared system prompt (cheap repeated prefix), and
  - structured output via output_config.format (schema-constrained JSON).

This module exposes the same `extract_paper` interface as the Gemini backend, so
s04 treats both uniformly. For large corpora you can cut cost ~50% by moving to
the Message Batches API (client.messages.batches) — the request shape below is
batch-ready (same params per request, keyed by DOI as custom_id). See the README
"Backends" section for how to switch. The synchronous path here is the simplest
correct implementation and is what runs by default.
"""
from __future__ import annotations

import base64
import json
import os

from ..config import cfg
from ..schema import json_schema as _paper_schema

_client = None

# Claude Opus 4.8 pricing (USD / 1M tokens) for the spend estimate only.
_PRICE_IN = 5.0
_PRICE_OUT = 25.0


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=cfg.anthropic_key)
    return _client


def estimate_cost_usd(path: str, kind: str) -> float:
    size = os.path.getsize(path) if os.path.exists(path) else 0
    in_tokens = max(2000, size / 4)
    out_tokens = 2000
    return (in_tokens * _PRICE_IN + out_tokens * _PRICE_OUT) / 1_000_000


def _strictify(schema):
    """Recursively add additionalProperties:false to every object node.

    Anthropic's structured output requires it; Pydantic no longer emits it
    (removed so Gemini accepts the same models). Walks $defs too.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        for v in schema.values():
            _strictify(v)
    elif isinstance(schema, list):
        for v in schema:
            _strictify(v)
    return schema


def _document_block(path: str, kind: str) -> dict:
    if kind == "pdf":
        data = base64.standard_b64encode(open(path, "rb").read()).decode("utf-8")
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        }
    # xml / text
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": text},
    }


def extract_paper(doi: str, path: str, kind: str, system_prompt: str,
                  json_schema: dict) -> dict:
    client = _get_client()
    resp = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=16000,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},   # cache the repeated prefix
        }],
        output_config={"format": {"type": "json_schema",
                                   "schema": _strictify(json_schema or _paper_schema())}},
        messages=[{
            "role": "user",
            "content": [
                _document_block(path, kind),
                {"type": "text", "text": f"Extract from this paper. DOI = {doi}."},
            ],
        }],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("refusal")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


def classify(title: str, abstract: str) -> str:
    """Cheap classifier using Haiku (keeps Opus for extraction)."""
    client = _get_client()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": (
                "Classify this paper for a Heusler experimental-data mining "
                "project. Reply with exactly one word: 'keep' (reports "
                "experimental measurements on Heusler samples), 'skip' (review "
                "or purely computational/DFT), or 'maybe' (unclear).\n\n"
                f"Title: {title}\nAbstract: {abstract}"
            ),
        }],
    )
    word = next((b.text for b in resp.content if b.type == "text"), "").strip().lower()
    for label in ("keep", "skip", "maybe"):
        if label in word:
            return label
    return "maybe"
