# podcast_insights.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import openai

# ------------------------------
# OpenAI setup
# ------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please set it before running podcast_insights.py"
    )

openai.api_key = OPENAI_API_KEY

# ------------------------------
# Optional ticker -> company names
# ------------------------------

try:
    # Same module used in podcast_excerpts.py
    from tickers import tickers as CUTLER_TICKERS  # dict like {"ABBV": ["AbbVie"], ...}
except Exception:
    CUTLER_TICKERS: Dict[str, List[str]] = {}


def _get_company_names_for_ticker(ticker: str) -> List[str]:
    """
    Prefer human-readable company names from tickers.py.
    Fallback to [ticker] if nothing is defined.
    """
    names = CUTLER_TICKERS.get(ticker, [])
    if isinstance(names, str):
        names = [names]
    names = [n for n in (names or []) if n]
    return names or [ticker]


# ------------------------------
# Prompt design
# ------------------------------

PODCAST_SYSTEM_PROMPT = """
You are an experienced buy-side equity analyst.

You will receive:
- A stock ticker and its company names.
- Several short excerpts from recent finance / investing podcasts where this ticker was mentioned.

Your job is to infer, using ONLY these podcast excerpts (no outside knowledge):

1) What is the implied stance on the stock?
   - Map this to one of: "buy", "hold", "sell", or "unclear".
   - "buy" = broadly positive / bullish, upside-focused, would add or initiate.
   - "hold" = more neutral / balanced, maintain position, unclear skew.
   - "sell" = clearly negative / bearish, would reduce or exit.
   - "unclear" = there is not enough signal in the snippets.

2) Give a confidence score (0.0–1.0) based purely on the strength and clarity of what is said
   in the excerpts (0.0 = no signal, 1.0 = extremely clear and strong signal).

3) Write a concise overall_summary in plain English (3–6 sentences, max ~180 words)
   that explains how the podcasts are framing the stock: what they like/dislike,
   main drivers, macro/sector tone, and risk/reward.

4) Provide:
   - time_horizon: e.g. "3–6 months", "6–12 months", "multi-year", or "unclear".
   - key_themes: 2–6 bullet-style phrases capturing the big ideas.
   - risks: 0–5 short bullet-style phrases.
   - opportunities: 0–5 short bullet-style phrases.
   - podcast_evidence: 0–6 short strings that reference specific episodes or quotes
     (e.g. "On On the Balance Sheet – 'If You Stick Around Long Enough...' they highlight X").

CRITICAL RULES:
- Base everything ONLY on the snippets provided.
- If the snippets do not say anything meaningful about the stock, set stance="unclear",
  confidence near 0.0, and explain that there was no usable signal.
- Do not fabricate detailed fundamentals, numbers, or events that are not clearly implied
  by the excerpts.
- Output must be a single JSON object with the exact fields described – no markdown,
  no extra commentary.
""".strip()


def _build_user_message(
    ticker: str,
    company_names: List[str],
    snippets: List[Dict[str, Any]],
) -> str:
    """
    Build the user content string that lists all podcast snippets and
    instructs the model to output JSON.
    """
    lines: List[str] = []

    lines.append(f"Ticker: {ticker}")
    if company_names:
        lines.append(f"Company names: {', '.join(company_names)}")
    else:
        lines.append("Company names: (not available)")

    lines.append("")
    lines.append("Below are recent podcast excerpts mentioning this ticker.")
    lines.append("Use ONLY these excerpts to form your view.")
    lines.append("")

    for idx, sn in enumerate(snippets, start=1):
        podcast_name = sn.get("podcast_name") or sn.get("podcast_id") or "Unknown podcast"
        title = sn.get("title") or sn.get("episode_title") or "Untitled episode"
        published = sn.get("published") or sn.get("episode_date") or ""
        transcript_source = sn.get("transcript_source") or "unknown"
        url = sn.get("url") or ""

        lines.append(f"--- Excerpt {idx} ---")
        lines.append(f"Podcast: {podcast_name}")
        lines.append(f"Episode title: {title}")
        if published:
            lines.append(f"Published: {published}")
        if url:
            lines.append(f"URL: {url}")
        lines.append(f"Transcript source: {transcript_source}")

        sent_idx = sn.get("sentence_index")
        if sent_idx is not None:
            lines.append(f"Sentence index in transcript: {sent_idx}")

        snippet_text = (sn.get("snippet") or "").strip()
        if snippet_text:
            lines.append("")
            lines.append("Excerpt text:")
            lines.append(snippet_text)
        else:
            lines.append("")
            lines.append("Excerpt text: [missing or empty]")

        lines.append("")  # blank line between excerpts

    lines.append("")
    lines.append("Now respond with a STRICT JSON object with the following keys:")
    lines.append("")
    lines.append("ticker: string")
    lines.append("company_names: list of strings")
    lines.append('stance: one of ["buy", "hold", "sell", "unclear"]')
    lines.append("confidence: float between 0.0 and 1.0")
    lines.append("overall_summary: short paragraph (max ~180 words)")
    lines.append('time_horizon: short string like "3–6 months", "6–12 months", "multi-year", or "unclear"')
    lines.append("key_themes: list of short strings")
    lines.append("risks: list of short strings")
    lines.append("opportunities: list of short strings")
    lines.append("podcast_evidence: list of short strings pointing to specific episodes/quotes")
    lines.append("")
    lines.append("Output ONLY valid JSON. Do not include backticks or any extra commentary.")

    return "\n".join(lines)


def _fallback_result(
    ticker: str,
    company_names: List[str],
    error_message: str,
) -> Dict[str, Any]:
    """
    Use this when OpenAI fails or the JSON cannot be parsed.
    """
    return {
        "ticker": ticker,
        "company_names": company_names,
        "stance": "unclear",
        "confidence": 0.0,
        "overall_summary": f"OpenAI call failed for this ticker: {error_message}",
        "time_horizon": "unclear",
        "key_themes": [],
        "risks": [],
        "opportunities": [],
        "podcast_evidence": [],
        "raw_model_output": "",
    }


def _normalize_result(
    ticker: str,
    company_names: List[str],
    parsed: Dict[str, Any],
    raw_model_output: str,
) -> Dict[str, Any]:
    """
    Ensure we always return a dict with all expected keys,
    even if the model omitted some.
    """
    out: Dict[str, Any] = dict(parsed or {})

    out.setdefault("ticker", ticker)
    out.setdefault("company_names", company_names)
    out.setdefault("stance", "unclear")
    out.setdefault("confidence", 0.0)
    out.setdefault("overall_summary", "")
    out.setdefault("time_horizon", "unclear")
    out.setdefault("key_themes", [])
    out.setdefault("risks", [])
    out.setdefault("opportunities", [])
    out.setdefault("podcast_evidence", [])
    out["raw_model_output"] = raw_model_output or json.dumps(parsed or {}, ensure_ascii=False)

    return out


def _call_openai_for_ticker(
    ticker: str,
    company_names: List[str],
    snippets: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    """
    Call OpenAI ChatCompletion for a single ticker's podcast snippets.
    Returns a dict ready to be written to podcast_insights_all.json.
    """
    if not snippets:
        return _fallback_result(
            ticker,
            company_names,
            "No podcast snippets available for this ticker.",
        )

    user_content = _build_user_message(ticker, company_names, snippets)

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": PODCAST_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=900,
        )

        content = response["choices"][0]["message"]["content"]
    except Exception as e:
        return _fallback_result(ticker, company_names, str(e))

    # Try to parse JSON
    try:
        parsed = json.loads(content)
    except Exception:
        # If parsing fails, still return something useful
        fallback = _fallback_result(
            ticker,
            company_names,
            "Model returned non-JSON content.",
        )
        fallback["overall_summary"] = content.strip()
        fallback["raw_model_output"] = content
        return fallback

    return _normalize_result(ticker, company_names, parsed, content)


# ------------------------------
# Main CLI
# ------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build AI insights from podcast excerpts for each ticker."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=str,
        required=True,
        help="Input JSON produced by podcast_excerpts.py (grouped by ticker).",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=str,
        required=True,
        help="Output JSON path for podcast insights.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model to use (e.g. gpt-4o-mini, gpt-4o).",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of tickers to process (space-separated). "
             "If omitted, process all tickers in the input JSON.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected input JSON to be a dict of {ticker: [snippets...]}. "
            "Did you pass the output from podcast_excerpts.py?"
        )

    all_tickers = sorted(list(data.keys()))
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        tickers_to_process = [t for t in all_tickers if t.upper() in wanted]
    else:
        tickers_to_process = all_tickers

    print(
        f"[INFO] Building podcast insights for {len(tickers_to_process)} "
        f"tickers using model {args.model}"
    )

    results: List[Dict[str, Any]] = []

    for idx, ticker in enumerate(tickers_to_process, start=1):
        snippets = data.get(ticker) or []
        company_names = _get_company_names_for_ticker(ticker)

        print(f"[INFO] [{idx}/{len(tickers_to_process)}] Processing {ticker} "
              f"with {len(snippets)} snippets...")

        insight = _call_openai_for_ticker(
            ticker=ticker,
            company_names=company_names,
            snippets=snippets,
            model=args.model,
        )
        results.append(insight)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Wrote podcast insights for {len(results)} tickers to {output_path}"
    )


if __name__ == "__main__":
    main()
