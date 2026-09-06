"""Central configuration: paths, env loading, rate limits, backend selection.

All secrets come from the local `.env` file (never committed). Import `cfg`
from here everywhere else so paths and settings stay consistent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project layout ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
PDF_INBOX = PDF_DIR / "inbox"          # user drops manually-saved PDFs here
FULLTEXT_DIR = DATA / "fulltext"       # Elsevier TDM XML lands here
BATCH_DIR = DATA / "batch_results"     # raw backend responses
EXPORT_DIR = DATA / "exports"
DB_PATH = DATA / "heusler.sqlite"
DOWNLOAD_QUEUE = DATA / "download_queue.csv"
PROMPT_DIR = ROOT / "prompts"

# Load .env from the project root (if present).
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Config:
    # Free polite-pool identity
    mailto: str = field(default_factory=lambda: _env("OPENALEX_MAILTO"))

    # Full-text keys
    elsevier_key: str = field(default_factory=lambda: _env("ELSEVIER_API_KEY"))

    # Optional structure DB key
    mp_key: str = field(default_factory=lambda: _env("MATERIALS_PROJECT_API_KEY"))

    # Self-serve publisher / OA-aggregator full-text keys (optional)
    springer_key: str = field(default_factory=lambda: _env("SPRINGER_API_KEY"))
    ieee_key: str = field(default_factory=lambda: _env("IEEE_API_KEY"))
    core_key: str = field(default_factory=lambda: _env("CORE_API_KEY") or _env("CORE_API"))
    s2_key: str = field(default_factory=lambda: _env("SEMANTIC_SCHOLAR_API_KEY") or _env("S2_API_KEY")
                        or _env("semantic_scholar") or _env("SEMANTIC_SCHOLAR"))

    # Extraction backends
    backend: str = field(default_factory=lambda: _env("EXTRACTION_BACKEND"))
    gemini_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    anthropic_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-4-8"))
    gcp_project: str = field(default_factory=lambda: _env("GCP_PROJECT"))
    gcp_region: str = field(default_factory=lambda: _env("GCP_REGION", "us-central1"))

    # Spend guard (hard stop inside s04)
    max_papers_per_run: int = field(default_factory=lambda: _env_int("MAX_PAPERS_PER_RUN", 100))
    max_estimated_spend_usd: float = field(default_factory=lambda: _env_float("MAX_ESTIMATED_SPEND_USD", 5.0))

    # Politeness / rate limits (seconds between requests)
    openalex_delay: float = 0.2
    elsevier_delay: float = 1.0
    paywalled_delay: float = 10.0     # very conservative; stop-on-block

    def resolve_backend(self) -> str:
        """Pick the extraction backend: explicit env wins, else auto-detect."""
        if self.backend:
            return self.backend.lower()
        if self.gemini_key:
            return "gemini"
        if self.anthropic_key:
            return "anthropic"
        return ""

    def ensure_dirs(self) -> None:
        for d in (DATA, PDF_DIR, PDF_INBOX, FULLTEXT_DIR, BATCH_DIR, EXPORT_DIR, PROMPT_DIR):
            d.mkdir(parents=True, exist_ok=True)


cfg = Config()
cfg.ensure_dirs()


def doi_slug(doi: str) -> str:
    """Filesystem-safe slug for a DOI (used as PDF/XML filename stem)."""
    return (
        doi.strip()
        .lower()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )
