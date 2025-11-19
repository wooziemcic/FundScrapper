# podcasts_config.py
"""
Podcast universe configuration for the Cutler Research Platform.

This module reads podcast_sources.csv (created by build_podcast_sources.py)
and exposes a list of Podcast objects the rest of the system can use.

Each Podcast has:
- id:   short slug identifier (used in filenames / UI keys)
- name: display name (as in your CSV)
- rss:  resolved RSS feed URL (rss_url or rss_from_website)
- website_url: publisher's main site (optional)
- apple_url: Apple Podcasts page (optional)
- priority: "core" by default (you can hand-edit to mark some as "secondary")
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SOURCES_CSV_DEFAULT = Path("podcast_sources.csv")


@dataclass
class Podcast:
    id: str
    name: str
    rss: str
    website_url: Optional[str] = None
    apple_url: Optional[str] = None
    priority: str = "core"  # "core" | "secondary" | "experimental"


def _slugify(text: str) -> str:
    text = text.strip().lower()
    # replace non-alphanum with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "podcast"


def load_podcasts(
    csv_path: Path = SOURCES_CSV_DEFAULT,
    *,
    include_without_rss: bool = False,
) -> List[Podcast]:
    """
    Load Podcast definitions from podcast_sources.csv.

    Priority rules for RSS:
    - prefer rss_url if present
    - else rss_from_website
    - else skip (unless include_without_rss=True)
    """
    podcasts: List[Podcast] = []

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("podcast_name") or "").strip()
            if not name:
                continue

            rss_url = (row.get("rss_url") or "").strip()
            rss_from_web = (row.get("rss_from_website") or "").strip()
            rss = rss_url or rss_from_web

            if not rss and not include_without_rss:
                # quietly skip ones we don't have feeds for yet
                continue

            website_url = (row.get("website_url") or "").strip() or None
            apple_url = (row.get("apple_url") or "").strip() or None

            podcast_id = _slugify(name)

            podcasts.append(
                Podcast(
                    id=podcast_id,
                    name=name,
                    rss=rss,
                    website_url=website_url,
                    apple_url=apple_url,
                    priority="core",
                )
            )

    return podcasts


def get_podcast_by_id(pid: str, podcasts: List[Podcast]) -> Optional[Podcast]:
    for p in podcasts:
        if p.id == pid:
            return p
    return None


# Convenience: load once by default if you want a simple import in small scripts
PODCASTS: List[Podcast] = load_podcasts()
