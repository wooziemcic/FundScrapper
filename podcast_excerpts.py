# podcast_excerpts.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from tickers import tickers as TICKERS

from tickers import tickers  # Assumed: dict like { "JPM": {...}, "BAC": {...}, ... }


@dataclass
class PodcastSnippet:
    ticker: str
    podcast_id: str
    podcast_name: str
    episode_id: str
    title: str
    published: str
    snippet: str
    transcript_source: str
    sentence_index: int  # index of the sentence where the ticker was detected


def _load_episode_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _split_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter; good enough for windowed snippets.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _build_ticker_regex_map(tickers: List[str]) -> Dict[str, re.Pattern]:
    """
    Build \bTICKER\b regex for each ticker to avoid partial matches.
    """
    patterns: Dict[str, re.Pattern] = {}
    for t in tickers:
        t = (t or "").strip()
        if not t:
            continue
        pat = re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        patterns[t] = pat
    return patterns


def _get_default_tickers() -> List[str]:
    """
    Pull ticker symbols from your tickers.py universe.

    Assumes TICKERS is a dict keyed by symbol, e.g. "JPM", "BAC", etc.
    """
    # If TICKERS is a dict keyed by ticker symbols:
    return sorted(tickers.keys())


def extract_podcast_snippets_for_tickers(
    root_dir: Path,
    tickers: List[str],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    window: int = 2,
) -> List[PodcastSnippet]:
    """
    Walk data/podcasts/<podcast_id>/*.json, load transcripts,
    and extract snippets mentioning the given tickers.

    Args:
        root_dir: root folder where podcast_ingest.py wrote JSONs (e.g. data/podcasts)
        tickers: list like ["JPM", "BAC", ...]
        since/until: UTC datetime bounds on published date (optional)
        window: number of sentences before/after the hit to include in snippet

    Returns:
        List of PodcastSnippet records.
    """
    if until is None:
        until = datetime.now(timezone.utc)
    if since is None:
        since = until - timedelta(days=7)

    if not tickers:
        print("[INFO] No tickers provided; using full Cutler universe from tickers.py")
        tickers = _get_default_tickers()

    ticker_patterns = _build_ticker_regex_map(tickers)
    snippets: List[PodcastSnippet] = []

    print(
        f"[INFO] Scanning podcasts under {root_dir} "
        f"for {len(ticker_patterns)} tickers between {since.isoformat()} and {until.isoformat()}"
    )

    if not root_dir.exists():
        print(f"[WARN] Root directory does not exist: {root_dir}")
        return snippets

    for pod_dir in root_dir.iterdir():
        if not pod_dir.is_dir():
            continue

        for json_path in pod_dir.glob("*.json"):
            ep = _load_episode_json(json_path)
            if not ep:
                continue

            published_str = ep.get("published") or ""
            pub_dt = _parse_iso_utc(published_str)
            if not pub_dt:
                continue

            if pub_dt < since or pub_dt > until:
                continue

            transcript = ep.get("transcript_text") or ""
            if not transcript:
                continue

            sentences = _split_sentences(transcript)
            if not sentences:
                continue

            meta = {
                "podcast_id": ep.get("podcast_id") or pod_dir.name,
                "podcast_name": ep.get("podcast_name") or pod_dir.name,
                "episode_id": ep.get("episode_id") or json_path.stem,
                "title": ep.get("title") or "Untitled Episode",
                "published": published_str,
                "transcript_source": ep.get("transcript_source") or "unknown",
            }

            # For each ticker, scan all sentences
            for ticker, pattern in ticker_patterns.items():
                for idx, sent in enumerate(sentences):
                    if not pattern.search(sent):
                        continue

                    # Found a match – build a window of sentences around it
                    start = max(0, idx - window)
                    end = min(len(sentences), idx + window + 1)
                    snippet_text = " ".join(sentences[start:end]).strip()

                    snippet = PodcastSnippet(
                        ticker=ticker,
                        snippet=snippet_text,
                        sentence_index=idx,
                        **meta,
                    )
                    snippets.append(snippet)

    print(f"[INFO] Extracted {len(snippets)} raw snippets")
    return snippets


def save_snippets_by_ticker(snippets: List[PodcastSnippet], out_path: Path) -> None:
    """
    Save snippets grouped by ticker to a JSON file:

    {
      "JPM": [ {...snippet1...}, {...snippet2...} ],
      "BAC": [ ... ],
      ...
    }
    """
    grouped: Dict[str, List[dict]] = {}
    for sn in snippets:
        grouped.setdefault(sn.ticker, []).append(asdict(sn))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Saved {len(snippets)} snippets for {len(grouped)} tickers to {out_path}"
    )


# ------------------------------
# CLI usage
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract ticker-aware snippets from podcast transcripts."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/podcasts",
        help="Root directory where podcast_ingest.py wrote episode JSON files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/podcast_excerpts.json",
        help="Output JSON file for grouped snippets.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional list of tickers to search. If omitted, use all companies from tickers.py",
    )

    args = parser.parse_args()

    if args.tickers:
        tickers_to_use = args.tickers
    else:
        # default: all tickers from tickers.py
        tickers_to_use = sorted(TICKERS.keys())

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look back this many days from now.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Number of sentences before/after the hit to include in snippet.",
    )

    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)

    root_dir = Path(args.root)
    out_path = Path(args.out)

    tickers_arg = args.tickers or []

    snippets = extract_podcast_snippets_for_tickers(
        root_dir=root_dir,
        tickers=tickers_arg,
        since=since,
        until=now,
        window=args.window,
    )
    save_snippets_by_ticker(snippets, out_path)



def build_ticker_patterns(ticker_list):
    """
    Build regex patterns that match company names / aliases, not single-letter tickers.
    """
    patterns = {}

    for symbol in ticker_list:
        name_variants = TICKERS.get(symbol, [])
        if not name_variants:
            continue

        regex_parts = []
        for name in name_variants:
            # Skip ultra-short tokens like 'C', 'F', etc.
            if len(name.strip()) <= 2:
                continue

            escaped = re.escape(name.strip())
            # Word-boundary match, case-insensitive
            regex_parts.append(rf"\b{escaped}\b")

        if not regex_parts:
            continue

        patterns[symbol] = re.compile("|".join(regex_parts), re.IGNORECASE)

    return patterns
