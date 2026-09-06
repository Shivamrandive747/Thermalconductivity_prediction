"""Stage 4 — LLM extraction (pluggable backend).

For each screened-in paper that has full text and hasn't been extracted yet,
call the active backend (Gemini by default, Claude if configured) and store the
raw JSON response in the `extractions` table. Validation + loading into the
`measurements` table happens in s05, so you can re-validate without re-calling
the API.

Spend guard: stops before crossing MAX_PAPERS_PER_RUN or MAX_ESTIMATED_SPEND_USD
(Google budgets only email you — this actually halts the run).

Usage:
    python -m pipeline.s04_extract                  # experimental ('keep') papers
    python -m pipeline.s04_extract --include-theory # also theoretical papers
    python -m pipeline.s04_extract --include-maybe  # also 'maybe' papers
    python -m pipeline.s04_extract --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys

from tqdm import tqdm

from .config import cfg, PROMPT_DIR
from . import db
from .backends import get_backend, BackendUnavailable, QuotaExhausted

SYSTEM_PROMPT_PATH = PROMPT_DIR / "extraction_system.md"

# Minimum fulltext size that can plausibly contain a paper BODY.
#
# The Elsevier full-text API returns HTTP 200 with a `<coredata>`-only stub — DOI,
# title, a link — when the body is entitlement-blocked. s02 stored those as if they
# were full text (167 of them, 580-2915 bytes), and s04 happily sent them to the LLM,
# which produced confident, richly-detailed output for a paper it had never seen:
# figure numbers ("Fig. 3(a)"), instrument models ("laser flash LFA 457, Netzsch"),
# densities, all fabricated from the title alone. 45 such papers yielded 1,429
# measurements, 524 of which reached the model export (17.4% of all kappa rows).
#
# Detected because an independent source measured Ru2NbGa at 4.6-5.5 W/m/K against
# our extracted 18.0 — and Ru2NbGa turned out to exist ONLY via a stub paper.
#
# A size floor alone is NOT sufficient and was a mistake to rely on: Elsevier returns
# coredata-only responses up to 130,455 B when the article has a long reference list, so
# 226 stubs cleared a 3 KB gate and were extracted as if they were papers. Size is kept
# only as a cheap first filter; the authoritative test is structural (see _has_body).
_MIN_BODY_BYTES = 3000


def _has_body(path: str, kind: str) -> bool:
    """Does this file actually contain article text?

    Authoritative test for the Elsevier stub problem. A <full-text-retrieval-response>
    carrying no <ce:para> and no <ce:section> has no body at all, however large it is --
    observed stubs run from 580 B to 130,455 B, so no size threshold can separate them.
    Anything non-XML (a real PDF) is trusted to the size gate."""
    if kind != "xml":
        return True
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            t = f.read()
    except OSError:
        return False
    if "<full-text-retrieval-response" not in t:
        return True                      # not an Elsevier API response; judge by size
    return ("<ce:para" in t) or ("<ce:section" in t)


def _load_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _pending(conn, include_theory: bool, include_maybe: bool,
             redo: bool = False, extracted_only: bool = False,
             min_richness: float | None = None, campaign: str | None = None):
    labels = ["keep"]                       # experimental first (priority)
    if include_theory:
        labels.append("theory")
    if include_maybe:
        labels.append("maybe")
    labels = tuple(labels)
    placeholders = ",".join("?" for _ in labels)
    # Phase C: extract only gate-certified-rich papers (richness_llm >= min).
    # Not-yet-gated papers (NULL) are EXCLUDED here — gate them first.
    richness_clause = (f"AND COALESCE(p.richness_llm, -1) >= {float(min_richness)}"
                       if min_richness is not None else "")
    # --campaign: restrict to one mining campaign's papers (roadmap C5)
    campaign_clause = "AND p.mining_campaign = ?" if campaign else ""
    # status filter: default = un-extracted or retryable; --redo also re-does
    # already-'ok' papers (Phase C upgrade); --extracted-only re-does ONLY 'ok'
    # papers (pilot / upgrade the existing corpus with the hardened prompt).
    if extracted_only:
        status_clause = "e.status = 'ok'"
    elif redo:
        status_clause = "(e.doi IS NULL OR e.status IN ('failed', 'oversized', 'ok'))"
    else:
        status_clause = "(e.doi IS NULL OR e.status IN ('failed', 'oversized'))"
    params = [*labels] + ([campaign] if campaign else [])
    # Order = extraction-budget allocation (roadmap C5): campaign papers first,
    # then gate class (gold = kappa AND own structure -> the high-confidence
    # structure-property records), then LLM/rule richness as before.
    return conn.execute(
        f"""
        SELECT p.doi, f.path, f.kind
        FROM papers p
        JOIN fulltext f ON f.doi = p.doi
        LEFT JOIN extractions e ON e.doi = p.doi
        WHERE p.screen_label IN ({placeholders})
          AND f.bytes >= {_MIN_BODY_BYTES}
          AND {status_clause}
          {richness_clause}
          {campaign_clause}
        ORDER BY (p.mining_campaign IS NOT NULL) DESC,
                 CASE p.gate_class WHEN 'gold' THEN 2 WHEN 'silver' THEN 1 ELSE 0 END DESC,
                 COALESCE(p.richness_llm, 0) DESC,      -- LLM-gate winners first
                 COALESCE(p.richness_score, 0) DESC
        """,
        params,
    ).fetchall()


def _rows_for_dois(conn, dois: list[str]):
    ph = ",".join("?" for _ in dois)
    return conn.execute(
        f"SELECT p.doi, f.path, f.kind FROM papers p JOIN fulltext f ON f.doi = p.doi "
        f"WHERE p.doi IN ({ph}) AND f.bytes >= {_MIN_BODY_BYTES}", dois).fetchall()


def run(limit: int | None, include_theory: bool, include_maybe: bool,
        redo: bool = False, extracted_only: bool = False,
        dois: list[str] | None = None, min_richness: float | None = None,
        campaign: str | None = None) -> None:
    db.init_db()
    try:
        backend = get_backend()
    except BackendUnavailable as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("Corpus/download/screening stages work without a key; only "
              "extraction needs one. Add GEMINI_API_KEY to .env and re-run.",
              file=sys.stderr)
        sys.exit(2)

    backend_name = cfg.resolve_backend()
    model = cfg.gemini_model if backend_name == "gemini" else cfg.anthropic_model
    system_prompt = _load_prompt()
    from .schema import json_schema
    schema = json_schema()

    with db.connect() as conn:
        if dois:                          # surgical: re-extract exactly these DOIs
            rows = _rows_for_dois(conn, dois)
        else:
            rows = _pending(conn, include_theory, include_maybe, redo,
                            extracted_only, min_richness, campaign)
        # Structural stub check before spending anything. The SQL size gate is a cheap
        # pre-filter; this is what actually stops a title-only response reaching the LLM.
        n_before = len(rows)
        rows = [r for r in rows if _has_body(r["path"], r["kind"])]
        if n_before != len(rows):
            print(f"  skipped {n_before - len(rows)} entitlement stubs (no body text)")
        if limit:
            rows = rows[:limit]
        rows = rows[:cfg.max_papers_per_run]

        spent = 0.0      # ledger driving the guard (actuals when available)
        billed = 0.0     # sum of confirmed billed cost from usage_metadata
        done = 0
        for r in tqdm(rows, desc=f"extract[{backend_name}]", unit="paper"):
            doi, path, kind = r["doi"], r["path"], r["kind"]

            est = backend.estimate_cost_usd(path, kind)
            if spent + est > cfg.max_estimated_spend_usd:
                print(f"\nSpend guard: estimated ${spent:.2f} reached the "
                      f"${cfg.max_estimated_spend_usd:.2f} cap. Stopping. "
                      f"Raise MAX_ESTIMATED_SPEND_USD in .env to continue.")
                break

            try:
                result = backend.extract_paper(doi, path, kind, system_prompt, schema)
                # Real billed usage when the backend reports it (Gemini/Vertex);
                # fall back to the pre-flight estimate otherwise. Charging the
                # ledger with actuals is what keeps the guard honest — the
                # estimate alone drifts badly on small and very large PDFs.
                usage = getattr(backend, "last_usage", None) or {}
                actual = usage.get("cost_usd")
                db.record_extraction(conn, {
                    "doi": doi, "status": "ok", "backend": backend_name,
                    "model": model, "raw_json": json.dumps(result), "error": None,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cost_usd": actual,
                    "api_calls": usage.get("calls"),
                })
                spent += actual if actual is not None else est
                billed += actual if actual is not None else 0.0
                done += 1
            except QuotaExhausted as exc:
                # Daily quota gone — stop cleanly; do NOT mark this paper failed,
                # so a resume run retries it. No point trying the rest today.
                conn.commit()
                print(f"\nDaily quota exhausted ({exc}).")
                print(f"Extracted {done} papers this run. {len(rows) - done} still pending.")
                print("Resume options: wait for the quota reset (midnight Pacific) "
                      "and re-run s04, or switch to Vertex AI ($300 GCP credits) by "
                      "setting GCP_PROJECT/GCP_REGION in .env.")
                return
            except Exception as exc:  # noqa: BLE001 — record and continue
                status = "refused" if "refusal" in str(exc).lower() else "failed"
                db.record_extraction(conn, {
                    "doi": doi, "status": status, "backend": backend_name,
                    "model": model, "raw_json": None, "error": str(exc)[:500],
                })
            conn.commit()  # checkpoint after every paper (resumable)

    per = f" (${billed / done:.4f}/paper)" if done and billed else ""
    print(f"\nExtracted {done} papers — billed ${billed:.4f}{per}, "
          f"ledger ${spent:.2f} / cap ${cfg.max_estimated_spend_usd:.2f}. "
          f"Backend={backend_name} model={model}")
    print("Next: python -m pipeline.s05_extract_collect")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LLM extraction on screened papers.")
    ap.add_argument("--limit", type=int, default=None, help="Cap papers this run.")
    ap.add_argument("--include-theory", action="store_true",
                    help="Also extract theoretical/DFT Heusler papers.")
    ap.add_argument("--include-maybe", action="store_true",
                    help="Also extract papers labelled 'maybe' by screening.")
    ap.add_argument("--redo", action="store_true",
                    help="Also re-extract already-'ok' papers (Phase C upgrade).")
    ap.add_argument("--extracted-only", action="store_true",
                    help="Re-extract ONLY already-'ok' papers (pilot / upgrade existing).")
    ap.add_argument("--dois", default=None,
                    help="Comma-separated DOIs to re-extract exactly (surgical retry).")
    ap.add_argument("--min-richness", type=float, default=None,
                    help="Only extract papers with LLM-gate richness >= this "
                         "(Phase C: use 3 for gate-rich only; excludes ungated).")
    ap.add_argument("--campaign", default=None,
                    help="Only extract papers tagged with this mining campaign "
                         "(e.g. 'tierS'); campaign papers always sort first anyway.")
    args = ap.parse_args()
    dois = [d.strip() for d in args.dois.split(",")] if args.dois else None
    run(args.limit, args.include_theory, args.include_maybe,
        args.redo, args.extracted_only, dois, args.min_richness, args.campaign)


if __name__ == "__main__":
    main()
