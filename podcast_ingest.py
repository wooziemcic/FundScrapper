# podcast_ingest.py

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import openai
import math
import time

import feedparser
import requests
from bs4 import BeautifulSoup

from podcasts_config import PODCASTS, Podcast, get_podcast_by_id, load_podcasts

# ----- OpenAI / env setup -----
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Each chunk must be safely below Whisper's 25 MB limit
WHISPER_CHUNK_BYTES = 20 * 1024 * 1024  # 20 MB per chunk


# ------------------------------
# Dataclasses
# ------------------------------

@dataclass
class EpisodeRecord:
    podcast_id: str
    podcast_name: str
    episode_id: str          # unique per feed (entry.id or link)
    title: str
    published: str           # ISO 8601 string
    audio_url: Optional[str]
    page_url: Optional[str]
    transcript_text: Optional[str]
    transcript_source: str   # "rss_content" | "rss_summary" | "html_page" | "none"


# ------------------------------
# Utility functions
# ------------------------------

def _sanitize_for_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "episode"


def _parse_entry_date(entry) -> Optional[datetime]:
    # feedparser exposes published / updated fields in different ways
    dt = None
    if getattr(entry, "published", None):
        try:
            dt = parsedate_to_datetime(entry.published)
        except Exception:
            dt = None

    if dt is None and getattr(entry, "updated", None):
        try:
            dt = parsedate_to_datetime(entry.updated)
        except Exception:
            dt = None

    if dt is None:
        return None

    # Normalize to timezone-aware UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt



def _entry_audio_url(entry) -> Optional[str]:
    # Many feeds use <enclosure> tags for audio files
    enclosures = getattr(entry, "enclosures", []) or []
    for enc in enclosures:
        url = enc.get("href") or enc.get("url")
        if url:
            return url

    # Fallback: some feeds put audio in <link type="audio/...">
    links = getattr(entry, "links", []) or []
    for ln in links:
        if ln.get("type", "").startswith("audio/") and ln.get("href"):
            return ln["href"]

    return None


def _looks_like_transcript(text: str) -> bool:
    """
    Very rough heuristic: long, sentence-heavy text is *maybe* a transcript
    rather than a 2–3 line summary.
    """
    if not text:
        return False

    # strip HTML tags
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()

    if len(plain) < 800:  # under ~800 chars, it's probably just show notes
        return False

    # ensure multiple reasonably long sentences
    sentences = re.split(r"[.!?]", plain)
    long_sentences = [s for s in sentences if len(s.split()) >= 6]
    return len(long_sentences) >= 10


def _extract_text_from_rss_entry(entry) -> tuple[Optional[str], str]:
    """
    Try to pull a transcript-level text from the RSS entry itself.

    Returns (text, source_label)
    """
    # 1) content:encoded (feedparser => entry.content list)
    contents = getattr(entry, "content", None)
    if contents:
        best = max((c.get("value", "") for c in contents), key=len, default="")
        if _looks_like_transcript(best):
            return best.strip(), "rss_content"

    # 2) summary/detail
    summary = getattr(entry, "summary", "") or ""
    if _looks_like_transcript(summary):
        return summary.strip(), "rss_summary"

    # 3) description (sometimes distinct)
    desc = getattr(entry, "description", "") or ""
    if _looks_like_transcript(desc) and len(desc) > len(summary):
        return desc.strip(), "rss_summary"

    return None, "none"


def _fetch_html_transcript(page_url: str, timeout: int = 10) -> Optional[str]:
    """
    Try to download the episode webpage and extract the main <p> text.

    This is deliberately simple; we can specialize later for certain hosts.
    """
    if not page_url:
        return None

    try:
        resp = requests.get(page_url, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Prefer <article> region if present (common on many publisher sites)
    article = soup.find("article")
    container = article or soup.body
    if not container:
        return None

    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)

    if _looks_like_transcript(text):
        return text

    return None

MAX_WHISPER_BYTES = 24 * 1024 * 1024  # 24 MB safety margin under 25 MB API limit


def _download_audio_range(
    audio_url: str,
    dest_path: Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
    timeout: int = 60,
) -> Optional[Path]:
    """
    Download audio from audio_url into dest_path.

    If start/end are provided, attempt an HTTP Range request (bytes=start-end).
    If the server ignores Range and returns the full file, we simply
    stop reading after the requested length.

    Returns dest_path on success, None on failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (CutlerResearchBot; +https://cutler.example)",
        "Accept": "*/*",
    }
    if start is not None and end is not None:
        headers["Range"] = f"bytes={start}-{end}"

    try:
        resp = requests.get(audio_url, stream=True, timeout=timeout, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [WARN] Failed to download audio: {e}")
        return None

    # If we requested a range, the ideal is a 206 response, but some servers
    # will still return 200 + full content. We'll just stop after the range length.
    max_bytes = None
    if start is not None and end is not None:
        max_bytes = (end - start) + 1

    written = 0
    try:
        with dest_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if max_bytes is not None and written + len(chunk) > max_bytes:
                    to_write = max_bytes - written
                    if to_write > 0:
                        f.write(chunk[:to_write])
                        written += to_write
                    break
                f.write(chunk)
                written += len(chunk)
    except Exception as e:
        print(f"    [WARN] Failed to save audio file: {e}")
        return None

    if written == 0:
        print("    [WARN] Downloaded 0 bytes of audio, skipping.")
        return None

    return dest_path

def _whisper_transcribe_audio(audio_url: str, tmp_dir: Path) -> Optional[str]:
    """
    Use OpenAI Whisper (whisper-1) to transcribe an episode audio file.

    Strategy:
    - Use HEAD to get total Content-Length (if available).
    - If size known:
        - Split into 20 MB chunks.
        - For each chunk, download that byte range and transcribe.
        - Concatenate transcripts in order.
    - If size unknown:
        - Download once (up to ~20 MB) and transcribe that partial audio.

    Returns full concatenated transcript text where possible, or None on failure.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Decide on a clean base name + extension
    last_part = audio_url.split("/")[-1] or "audio"
    sanitized = _sanitize_for_filename(last_part)

    # Try to infer original extension from URL
    valid_exts = [".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"]
    ext = None
    lower_name = sanitized.lower()
    for e in valid_exts:
        if lower_name.endswith(e):
            ext = e
            base = sanitized[: -len(e)]
            break
    if ext is None:
        # Fallback: treat as mp3
        ext = ".mp3"
        base = sanitized

    # First, try HEAD to get total size
    headers = {
        "User-Agent": "Mozilla/5.0 (CutlerResearchBot; +https://cutler.example)",
        "Accept": "*/*",
    }
    try:
        head_resp = requests.head(audio_url, timeout=10, headers=headers, allow_redirects=True)
        head_resp.raise_for_status()
        length_str = head_resp.headers.get("Content-Length")
        total_size = int(length_str) if length_str is not None else None
    except Exception as e:
        print(f"    [WARN] HEAD request failed or no Content-Length: {e}")
        total_size = None

    transcripts: List[str] = []

    def _transcribe_file(path: Path) -> Optional[str]:
        max_attempts = 3
        backoff_seconds = 2

        for attempt in range(1, max_attempts + 1):
            try:
                with path.open("rb") as f:
                    result = openai.Audio.transcribe(
                        model="whisper-1",
                        file=f,
                        language="en",
                    )
            except Exception as e:
                msg = str(e)
                print(
                    f"    [WARN] Whisper transcription failed for {path.name} "
                    f"(attempt {attempt}/{max_attempts}): {msg}"
                )

                # If it's a transient server error (e.g., 5xx / 502), try again
                if attempt < max_attempts and ("502" in msg or "5xx" in msg or "server error" in msg):
                    time.sleep(backoff_seconds)
                    continue

                # Otherwise, give up for this chunk
                return None

            # Success path
            if isinstance(result, dict):
                text = result.get("text")
            else:
                text = getattr(result, "text", None)

            if not text:
                return None

            return text.strip()

        # Shouldn't really get here, but just in case
        return None

    if total_size is None:
        # Fallback: we don't know total size / range not supported -> single partial chunk
        print("    [INFO] Unknown total size; downloading a single ~20MB chunk")
        part_path = tmp_dir / f"{base}_part000{ext}"
        downloaded = _download_audio_range(audio_url, part_path, start=None, end=None)
        if not downloaded:
            return None
        text = _transcribe_file(downloaded)
        if text:
            transcripts.append(text)
    else:
        # Split known size into 20MB chunks
        num_chunks = math.ceil(total_size / WHISPER_CHUNK_BYTES)
        print(
            f"    [INFO] Total size ~{total_size / (1024 * 1024):.1f} MB, "
            f"splitting into {num_chunks} chunk(s) for Whisper"
        )

        for chunk_index in range(num_chunks):
            start = chunk_index * WHISPER_CHUNK_BYTES
            end = min(total_size - 1, start + WHISPER_CHUNK_BYTES - 1)
            part_path = tmp_dir / f"{base}_part{chunk_index:03d}{ext}"

            downloaded = _download_audio_range(audio_url, part_path, start=start, end=end)
            if not downloaded:
                print(f"    [WARN] Failed to download chunk {chunk_index}, stopping.")
                break

            size_bytes = downloaded.stat().st_size
            print(
                f"    [INFO] Chunk {chunk_index+1}/{num_chunks}: "
                f"{size_bytes / (1024 * 1024):.1f} MB"
            )

            text = _transcribe_file(downloaded)
            if text:
                transcripts.append(text)
            else:
                print(f"    [WARN] No transcript for chunk {chunk_index}, continuing.")

    if not transcripts:
        return None

    full_transcript = "\n\n".join(transcripts)
    return full_transcript.strip()

# ------------------------------
# Core ingestion for a single podcast
# ------------------------------

def fetch_episodes_for_podcast(
    podcast: Podcast,
    since: datetime,
    until: Optional[datetime] = None,
    max_episodes: int = 15,
    enable_whisper: bool = False,
    whisper_tmp_dir: Optional[Path] = None
) -> List[EpisodeRecord]:
    """
    Fetch recent episodes for a single podcast within [since, until],
    and try to attach transcripts where possible.
    """
    print(f"Fetching feed for {podcast.name} ...")

    feed = feedparser.parse(podcast.rss)
    if whisper_tmp_dir is None:
        whisper_tmp_dir = Path("data") / "podcast_audio_tmp"

    records: List[EpisodeRecord] = []
    until = until or datetime.utcnow()

    for entry in feed.entries:
        pub_dt = _parse_entry_date(entry)
        if pub_dt is None:
            continue

        if pub_dt < since or pub_dt > until:
            continue

        episode_id = getattr(entry, "id", None) or getattr(entry, "link", None)
        if not episode_id:
            episode_id = f"{entry.title}-{pub_dt.isoformat()}"

        title = getattr(entry, "title", "Untitled Episode")
        page_url = getattr(entry, "link", None)
        audio_url = _entry_audio_url(entry)

        # First, attempt RSS-level transcript
        transcript_text, source = _extract_text_from_rss_entry(entry)

        # If nothing, try the episode HTML page
        if transcript_text is None and page_url:
            html_text = _fetch_html_transcript(page_url)
            if html_text:
                transcript_text = html_text
                source = "html_page"

        # If still nothing and Whisper is enabled, try audio transcription
        if (
            transcript_text is None
            and enable_whisper
            and audio_url is not None
        ):
            print(f"    [INFO] No transcript found; calling Whisper for '{title}'")
            whisper_text = _whisper_transcribe_audio(audio_url, whisper_tmp_dir)
            if whisper_text:
                transcript_text = whisper_text
                source = "whisper"

        rec = EpisodeRecord(
            podcast_id=podcast.id,
            podcast_name=podcast.name,
            episode_id=episode_id,
            title=title,
            published=pub_dt.isoformat(),
            audio_url=audio_url,
            page_url=page_url,
            transcript_text=transcript_text,
            transcript_source=source,
        )
        records.append(rec)

        if len(records) >= max_episodes:
            break

    print(
        f"  -> {len(records)} episodes "
        f"({sum(1 for r in records if r.transcript_text) } with transcripts)"
    )
    return records


# ------------------------------
# High-level ingestion entry point
# ------------------------------

def ingest_podcasts(
    output_root: Path,
    since: datetime,
    until: Optional[datetime] = None,
    podcast_ids: Optional[list[str]] = None,
    max_episodes_per_podcast: int = 15,
    enable_whisper: bool = False,
) -> List[EpisodeRecord]:
    """
    High-level ingestion used by CLI and (later) Streamlit.

    - Iterates over selected podcasts from podcasts_config.PODCASTS.
    - For each, fetches episodes and tries to extract transcripts.
    - Writes one JSON file per episode under:
          <output_root>/<podcast_id>/<episode_slug>.json
    - Returns all EpisodeRecord objects in memory as well.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: List[EpisodeRecord] = []
    whisper_tmp_dir = output_root / "_audio_tmp"

    all_podcasts: List[Podcast] = PODCASTS

    # Filter by ids if provided
    if podcast_ids:
        filtered = []
        requested = set(podcast_ids)
        for p in all_podcasts:
            if p.id in requested:
                filtered.append(p)
        all_podcasts = filtered

    for podcast in all_podcasts:
        records = fetch_episodes_for_podcast(
            podcast,
            since=since,
            until=until,
            max_episodes=max_episodes_per_podcast,
            enable_whisper=enable_whisper,
            whisper_tmp_dir=whisper_tmp_dir,
        )
        if not records:
            continue

        pod_dir = output_root / podcast.id
        pod_dir.mkdir(parents=True, exist_ok=True)

        for rec in records:
            fname = _sanitize_for_filename(rec.episode_id) + ".json"
            path = pod_dir / fname
            with path.open("w", encoding="utf-8") as f:
                json.dump(asdict(rec), f, ensure_ascii=False, indent=2)

        all_records.extend(records)

    return all_records


# ------------------------------
# CLI usage
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest podcast episodes + transcripts from RSS/HTML."
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/podcasts",
        help="Output root directory for episode JSON files.",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look back this many days from now.",
    )

    parser.add_argument(
        "--podcasts",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of podcast IDs (space-separated).",
    )

    parser.add_argument(
        "--max-per-podcast",
        type=int,
        default=15,
        help="Max episodes per podcast within the window.",
    )

    parser.add_argument(
        "--whisper",
        action="store_true",
        help="Enable Whisper transcription for episodes without transcripts.",
    )

    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)

    records = ingest_podcasts(
        output_root=Path(args.out),
        since=since,
        until=now,
        podcast_ids=args.podcasts,
        max_episodes_per_podcast=args.max_per_podcast,
        enable_whisper=args.whisper,
    )

    print(
        f"\nDone. Total episodes: {len(records)} "
        f"(with transcripts: {sum(1 for r in records if r.transcript_text)})"
    )
