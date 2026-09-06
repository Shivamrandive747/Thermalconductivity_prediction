"""Gemini extraction backend (free tier at aistudio.google.com, or GCP credits
via Vertex AI). Default backend for this project.

- Uploads the PDF via the Files API (Elsevier XML is sent inline as text).
- Uses structured output: response_schema = the PaperExtraction Pydantic model,
  response_mime_type = application/json.
- Paces requests to stay under free-tier limits and retries on 429 so an
  unattended multi-day run just slows down instead of erroring.
"""
from __future__ import annotations

import json
import os
import time

from ..config import cfg, DATA, doi_slug
from ..schema import PaperExtraction
from . import BackendUnavailable, QuotaExhausted

_PREPROC_DIR = DATA / "preprocessed"
_MAX_FIGS = 16          # cap attached figure images per paper (cost control)
_MIN_MD_CHARS = 1500    # below this the PDF was likely scanned -> send raw PDF

# --- lazy client ---------------------------------------------------------
_client = None
_last_request_ts = 0.0
# Conservative default pacing for the free tier (requests per minute). Raise via
# GEMINI_RPM in the environment when running on paid/Vertex quota.
_MIN_INTERVAL = 60.0 / float(os.getenv("GEMINI_RPM", "10"))

# --- pricing (USD per 1M tokens) -----------------------------------------
# Was hard-coded to Gemini 2.5 FLASH rates while the project runs 2.5 PRO — a 4x
# under-count that happened to be masked by an offsetting 4x over-count in the old
# filesize-based token heuristic. Both are fixed; keep this table model-aware.
_PRICING = {
    "gemini-2.5-pro":   (1.25, 10.00),   # <=200k-token prompts (2.50/15.00 above)
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
}
_PRICE_DEFAULT = (1.25, 10.00)           # assume Pro rates when unknown (conservative)


def _prices(model: str | None = None) -> tuple[float, float]:
    """(input, output) USD per 1M tokens for the configured model."""
    m = (model or cfg.gemini_model or "").strip().lower()
    for key, val in _PRICING.items():
        if m.startswith(key):
            return val
    return _PRICE_DEFAULT


# Back-compat aliases (some callers/tests read these directly).
_PRICE_IN, _PRICE_OUT = _prices()

# Cost is OUTPUT-token dominated: long structured JSON plus 2.5 Pro thinking
# tokens. Input is small — an inline PDF bills ~258 tok/page (~1k tok/MB), so a
# 3.2 MB paper is only ~3.2k input tokens against ~24k output. Calibrated against
# the one real datapoint: $5.34 / 22 papers = $0.2427/paper on 2.5 Pro.
# Measured on the first real run under this config (10.1002/aenm.202401345, 2.0 MB):
# 7,740 in / 26,928 out = $0.2790 billed. Output ran 3.5x input, confirming the
# output-dominated model. _EST_OUT_TOKENS is set slightly ABOVE that observation so
# the guard errs high — under-estimating is the direction that overspends.
_EST_TOKENS_PER_MB = 1_000
_EST_MIN_IN_TOKENS = 2_000
_EST_OUT_TOKENS = 28_000

# Actual usage from the most recent successful call, for s04's spend ledger.
# dict(prompt_tokens, output_tokens, total_tokens, cost_usd, model) or None.
last_usage: dict | None = None


def _get_client():
    global _client
    if _client is None:
        from google import genai  # imported lazily so the pipeline loads without the SDK
        from google.genai import types
        # Generous timeout (ms) so long figure-heavy generations aren't cut short.
        # 300s proved too short: a 2 MB data-rich paper on 2.5 Pro (thinking tokens
        # + up to 65k output) blew past it and came back 504 DEADLINE_EXCEEDED.
        # Override with GEMINI_TIMEOUT_MS if a paper still deadlines. 900s was too
        # generous in the other direction: healthy papers finish in ~210s, so a
        # 15-min wait only ever bought a hang more time. 600s = ~3x the observed
        # worst healthy case, and pairs with GEMINI_PAPER_MAX_S (2 attempts, then stop).
        http_opts = types.HttpOptions(
            timeout=int(os.getenv("GEMINI_TIMEOUT_MS", "600000")))
        if cfg.gcp_project:
            # Vertex + Application Default Credentials. Do NOT set
            # GOOGLE_APPLICATION_CREDENTIALS anywhere: google.auth.default() reads
            # it BEFORE the gcloud ADC file, so a stale service-account key there
            # silently bills the wrong project (this happened — see .env comments).
            _client = genai.Client(vertexai=True, project=cfg.gcp_project,
                                   location=cfg.gcp_region, http_options=http_opts)
        elif os.getenv("ALLOW_AI_STUDIO") == "1":
            _client = genai.Client(api_key=cfg.gemini_key, http_options=http_opts)
        else:
            # Fail fast instead of silently falling back to the AI Studio key,
            # whose India prepaid wallet is depleted -> every paper dies on a 429
            # after burning retry backoff. A blank GCP_PROJECT is always a config
            # mistake here, so surface it loudly.
            raise BackendUnavailable(
                "GCP_PROJECT is not set, so Vertex AI cannot be used.\n"
                "  Expected: GCP_PROJECT=project-8688aa51-0892-4112-99c in .env\n"
                "  Refusing to fall back to the AI Studio key (GEMINI_API_KEY): its "
                "prepaid wallet is empty and every request would 429.\n"
                "  Set ALLOW_AI_STUDIO=1 only if you deliberately want that path."
            )
    return _client


def _throttle() -> None:
    global _last_request_ts
    wait = _MIN_INTERVAL - (time.time() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def estimate_cost_usd(path: str, kind: str) -> float:
    """Pre-flight per-paper cost estimate for the spend guard in s04.

    The old model was `in_tokens = filesize/4` with Flash prices, which scaled with
    PDF size even though real cost barely does. Measured error: 7.7x UNDER on a
    0.3 MB PDF, 3x OVER on an 8 MB one — it only matched reality near 3.2 MB.

    This is output-dominated and model-aware, so it stays ~flat per paper. It is
    still only an estimate; `last_usage` carries the billed truth after each call.
    """
    size = os.path.getsize(path) if os.path.exists(path) else 0
    price_in, price_out = _prices()
    in_tokens = max(_EST_MIN_IN_TOKENS, (size / 1_000_000) * _EST_TOKENS_PER_MB)
    return (in_tokens * price_in + _EST_OUT_TOKENS * price_out) / 1_000_000


def _reset_usage() -> None:
    global last_usage
    last_usage = None


def _record_usage(resp) -> None:
    """ACCUMULATE real token usage from a response into `last_usage`.

    `usage_metadata` was never read before, so actual spend was invisible and a run
    could only report its own estimate back to itself. Thinking tokens bill as
    output, so they fold into the output count when present.

    Accumulates rather than overwrites: a paper that trips the output limit is
    re-sent downsampled, and BOTH calls are billed. Overwriting would silently drop
    the first (usually larger) call from the ledger. `calls` records how many
    round-trips this paper actually cost.
    """
    global last_usage
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return

    def _int(name: str) -> int:
        return int(getattr(um, name, None) or 0)

    prompt = _int("prompt_token_count")
    out = _int("candidates_token_count") + _int("thoughts_token_count")
    total = _int("total_token_count") or (prompt + out)
    price_in, price_out = _prices()
    cost = (prompt * price_in + out * price_out) / 1_000_000

    prev = last_usage or {"prompt_tokens": 0, "output_tokens": 0,
                          "total_tokens": 0, "cost_usd": 0.0, "calls": 0}
    last_usage = {
        "prompt_tokens": prev["prompt_tokens"] + prompt,
        "output_tokens": prev["output_tokens"] + out,
        "total_tokens": prev["total_tokens"] + total,
        "cost_usd": prev["cost_usd"] + cost,
        "calls": prev.get("calls", 0) + 1,
        "model": cfg.gemini_model,
    }


def _preprocessed_contents(doi: str, types) -> list | None:
    """B1: markdown + figure images from s02b, if available and usable.
    Returns None (caller sends the raw PDF) when there is no preprocessed
    output or the text layer is too thin (scanned PDF -> Gemini OCRs better)."""
    d = _PREPROC_DIR / doi_slug(doi)
    md_file = d / "paper.md"
    if not md_file.exists():
        return None
    md = md_file.read_text(encoding="utf-8", errors="ignore")
    if len(md.strip()) < _MIN_MD_CHARS:
        return None
    figs = list((d / "figures").glob("*.png"))
    if len(figs) > _MAX_FIGS:            # keep the largest (most content-dense)
        figs = sorted(figs, key=lambda p: -p.stat().st_size)[:_MAX_FIGS]
    figs.sort(key=lambda p: p.name)      # restore reading order
    parts: list = [
        f"DOI = {doi}. Extract from this paper, given below as markdown with the "
        f"paper's figures attached as images afterwards, in the order referenced "
        f"(![figure](name) markers). Read data points off the attached figures.\n\n{md}"
    ]
    for p in figs:
        parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png"))
    return parts


_DOWNSAMPLE_NOTE = (
    "\n\nIMPORTANT: this paper is very data-rich and a full extraction exceeded "
    "the output limit. Re-extract with AT MOST 20 evenly-spaced (temperature, "
    "value) points per property curve. Keep ALL samples, ALL properties and all "
    "context fields — only thin dense curves."
)


def _finish_reason(resp) -> str:
    try:
        return str(resp.candidates[0].finish_reason or "")
    except Exception:  # noqa: BLE001
        return ""


def _recover_or_raise(text: str) -> dict:
    """A paper too data-rich to fit the output limit even after downsampling:
    instead of discarding the whole thing, structurally repair the truncated JSON
    (drop the incomplete trailing sample; never invent values) and KEEP the
    complete samples the model already returned. Only fail if nothing survives."""
    from ..s05_extract_collect import _repair_json     # lazy: avoids import cycle
    data = _repair_json(text or "")
    if isinstance(data, dict) and data.get("samples"):
        data["notes"] = ((data.get("notes") or "") +
                         " [partial: response truncated at output limit; trailing "
                         "sample(s) dropped, complete samples retained]").strip()
        return data
    raise RuntimeError("truncated: no complete sample survived the output limit")


def extract_paper(doi: str, path: str, kind: str, system_prompt: str,
                  json_schema: dict) -> dict:
    """Return a dict matching PaperExtraction (validated later in s05)."""
    from google.genai import types
    client = _get_client()
    _reset_usage()          # per-paper ledger; retries accumulate into it

    def _config(downsample: bool) -> "types.GenerateContentConfig":
        return types.GenerateContentConfig(
            system_instruction=system_prompt + (_DOWNSAMPLE_NOTE if downsample else ""),
            response_mime_type="application/json",
            response_schema=PaperExtraction,   # google-genai accepts the pydantic model
            temperature=0.0,
            max_output_tokens=65535,           # explicit ceiling; truncation is detected
        )

    config = _config(downsample=False)
    downsampled = False
    last_exc = None
    # Separate retry budgets: network/DNS blips cost $0 (request never sent), so
    # be patient with them (the user's machine has intermittent DNS); service
    # errors (429/503) get a shorter, standard budget; downsample is one-shot and
    # consumes neither. Prevents a brief DNS blip from wasting a paper.
    net_attempt = svc_attempt = dl_attempt = 0
    NET_MAX, SVC_MAX, DL_MAX = 8, 6, 2
    # HARD WALL-CLOCK CEILING on one paper. The three budgets above are counted
    # INDEPENDENTLY, so their worst case is 16 attempts x the per-call timeout —
    # measured at 4h33m and $0.95 on a single paper (2026-08-15), which consumed
    # an entire overnight campaign window while 40 gold papers waited. Attempt
    # counts alone cannot bound this; only wall-clock can. On expiry the paper is
    # abandoned so the queue keeps moving: one lost paper beats a lost night.
    PAPER_MAX_S = float(os.getenv("GEMINI_PAPER_MAX_S", "1500"))   # 25 min
    started = time.monotonic()

    def _over_budget() -> bool:
        return time.monotonic() - started > PAPER_MAX_S

    while True:
        _throttle()
        try:
            # (re)build contents inside the loop so an upload drop is retried too
            # B1 markdown+figures is OPT-IN (USE_PREPROCESSED=1): measured ~4.6x
            # MORE input tokens than native PDF (258 tok/page), but much sharper
            # figures — use it as the quality second-pass on figure-heavy papers.
            if kind == "pdf" and os.getenv("USE_PREPROCESSED") \
                    and (pre := _preprocessed_contents(doi, types)) is not None:
                contents = pre
            elif kind == "pdf":
                file_size = os.path.getsize(path)
                if cfg.gcp_project or file_size < 15 * 1024 * 1024:
                    # Send the PDF inline as bytes directly to avoid Files API upload latency and hangs
                    with open(path, "rb") as fh:
                        pdf_part = types.Part.from_bytes(
                            data=fh.read(), mime_type="application/pdf")
                    contents = [pdf_part, f"Extract from this paper. DOI = {doi}."]
                else:  # AI Studio key for very large files — upload via Files API
                    uploaded = client.files.upload(file=path)
                    contents = [uploaded, f"Extract from this paper. DOI = {doi}."]
            else:  # xml / text
                text = _read_text(path)
                contents = [f"DOI = {doi}. Full text follows:\n\n{text}"]

            resp = client.models.generate_content(
                model=cfg.gemini_model, contents=contents, config=config,
            )
            _record_usage(resp)   # billed truth, regardless of how we branch below
            if "MAX_TOKENS" in _finish_reason(resp).upper():
                if downsampled:                # already thinned once
                    return _recover_or_raise(resp.text)   # keep the samples we got
                downsampled = True
                config = _config(downsample=True)
                continue                       # retry with bounded curves
            try:
                return json.loads(resp.text)
            except (json.JSONDecodeError, TypeError):
                if not downsampled:            # cut-off JSON without MAX_TOKENS flag
                    downsampled = True
                    config = _config(downsample=True)
                    continue
                return _recover_or_raise(resp.text)   # keep the samples we got
        except Exception as exc:  # noqa: BLE001 — surface message, retry on transient errors
            last_exc = exc
            msg = str(exc).lower()
            # Daily/plan quota exhausted, or the depleted AI-Studio prepaid wallet
            # (India) — will NOT recover by retrying. Stop fast (retrying just
            # burns ~5 min/paper on backoff). 'prepayment' => GCP_PROJECT unset;
            # set it in .env to use Vertex (GCP credit) instead of AI Studio.
            if "exceeded your current quota" in msg or "plan and billing" in msg \
                    or "quota exceeded for metric" in msg or "per day" in msg \
                    or "prepayment" in msg or "prepaid" in msg:
                raise QuotaExhausted(str(exc)[:200])
            # Network / DNS class — never reached Google ($0). Patient retry.
            network = (
                "disconnect" in msg or "connection" in msg                      # dropped connection
                or "timeout" in msg or "timed out" in msg or "reset" in msg     # network
                or "remoteprotocolerror" in msg or "incomplete" in msg          # httpx drops
                or "getaddrinfo" in msg or "11001" in msg                       # DNS lookup failed
                or "name resolution" in msg or "name or service not known" in msg
                or "temporary failure" in msg or "network is unreachable" in msg
                or "ssl" in msg or "handshake" in msg or "eof occurred" in msg
            )
            # Service transient — per-minute rate limit / capacity / server errors.
            service = (
                "429" in msg or "resource_exhausted" in msg
                or "503" in msg or "unavailable" in msg or "overloaded" in msg
                or "500" in msg or "internal" in msg
            )
            # Server-side deadline. Fell through BOTH classifiers above ('deadline'
            # matches no keyword, and '504' was absent), so a single slow paper was
            # burned with zero retries. Retried on its own SMALL budget because each
            # attempt costs a full timeout wall-clock wait and may still be billed.
            deadline = ("504" in msg or "deadline" in msg)
            # Checked before EVERY retry class: a transient error that keeps
            # recurring is indistinguishable from a hang, and each further attempt
            # costs another full timeout. Give the paper up rather than the night.
            if _over_budget():
                raise RuntimeError(
                    f"paper budget {PAPER_MAX_S:.0f}s exhausted after "
                    f"net={net_attempt} svc={svc_attempt} dl={dl_attempt}; "
                    f"last error: {str(exc)[:160]}") from exc
            if network and net_attempt < NET_MAX:
                net_attempt += 1
                time.sleep(min(90, 12 * net_attempt))   # 12,24,36...90 (~7 min total)
                continue
            if service and svc_attempt < SVC_MAX:
                svc_attempt += 1
                time.sleep(min(90, 5 * (2 ** svc_attempt)))
                continue
            if deadline and dl_attempt < DL_MAX:
                dl_attempt += 1
                # Second pass goes straight to downsampled output: the usual cause
                # is an over-long generation, and a shorter one is what fits.
                if not downsampled:
                    downsampled = True
                    config = _config(downsample=True)
                time.sleep(10)
                continue
            raise
    raise RuntimeError(f"Gemini extraction failed after retries: {last_exc}")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if path.lower().endswith(".xml"):
        import re
        for pattern in [r"<tail\b", r"<ce:bibliography\b", r"<ref-list\b"]:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                content = content[:match.start()]
                break
    return content


# --- cheap classifier for s03 --llm --------------------------------------
def classify(title: str, abstract: str) -> str:
    from google.genai import types
    client = _get_client()
    prompt = (
        "Classify this paper for a Heusler experimental-data mining project. "
        "Answer with exactly one word: 'keep' if it reports EXPERIMENTAL "
        "measurements on Heusler/half-Heusler samples, 'skip' if it is a review "
        "or purely computational/DFT, or 'maybe' if unclear.\n\n"
        f"Title: {title}\nAbstract: {abstract}"
    )
    _throttle()
    try:
        resp = client.models.generate_content(
            model=cfg.gemini_model, contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        word = (resp.text or "").strip().lower()
        for label in ("keep", "skip", "maybe"):
            if label in word:
                return label
    except Exception:  # noqa: BLE001 — classification is best-effort
        pass
    return "maybe"
