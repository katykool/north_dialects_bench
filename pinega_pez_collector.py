#!/usr/bin/env python3
"""
Build a PEZ-style folder for one LingConLab recording.

Compatible with the following corpora:
pyoza, veegora, pinega, keba, vaduga, dvina, tserkovnoe.

Each audio-linked transcript segment is saved as:
- .wav
- .txt
- _orig.txt

No complete recording transcripts are created.
"""

from __future__ import annotations

import argparse
import csv
import logging
import hashlib
import hmac
import re
import secrets
import sys
import time
import unicodedata

from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm


DEFAULT_URL = "https://lingconlab.ru/pinega/files_html/A-0476.html"
DEFAULT_RECORDING_ID = "A-0476"

USER_AGENT = (
    "LingConLabCorpusBuilder/1.0 "
    "(research corpus; polite downloader)"
)


# ============================================================
# HTTP SESSION
# ============================================================

def make_session() -> requests.Session:

    session = requests.Session()

    session.headers.update({
        "User-Agent": USER_AGENT
    })

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# ============================================================
# SPEAKER PSEUDONYMIZATION
# ============================================================

def load_or_create_secret(out_root: Path) -> bytes:

    key_path = out_root / ".speaker_salt"

    if key_path.exists():
        return key_path.read_bytes()

    key = secrets.token_bytes(32)

    key_path.write_bytes(key)

    try:
        key_path.chmod(0o600)
    except OSError:
        pass

    print(f"[privacy] Created private key: {key_path}")
    print("[privacy] DO NOT upload .speaker_salt to Hugging Face.")

    return key


def pseudonym(label: str, key: bytes) -> str:

    digest = hmac.new(
        key,
        label.strip().encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return "SPK-" + digest[:8].upper()


# ============================================================
# TIMESTAMPS AND TEXT
# ============================================================

def timestamp_ms_to_filename(ms: int) -> str:

    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)

    return (
        f"{hours:02d}-"
        f"{minutes:02d}-"
        f"{seconds:02d}-"
        f"{millis:03d}"
    )


def normalize_text(text: str) -> str:

    text = unicodedata.normalize("NFC", text)

    return re.sub(r"\s+", " ", text).strip()


def raw_source_text(text: str) -> str:

    return unicodedata.normalize("NFC", text).strip()


# ============================================================
# EXTRACT AUDIO-LINKED TRANSCRIPT SEGMENTS
# ============================================================

def extract_segments(
    page_html: str,
    page_url: str,
    recording_id: str
):

    soup = BeautifulSoup(page_html, "html.parser")

    body = soup.body or soup

    segments = []

    current_speaker = "UNK"
    pending_text: list[str] = []

    for node in body.descendants:

        if isinstance(node, NavigableString):

            parent = node.parent

            if not isinstance(parent, Tag):
                continue

            if parent.name in {
                "b", "a", "script", "style", "audio", "source"
            }:
                continue

            pending_text.append(str(node))

            continue

        if not isinstance(node, Tag):
            continue

        if node.name == "br":

            pending_text.append("\n")

        elif node.name == "b":

            label = node.get_text(" ", strip=True).strip()

            if label:

                current_speaker = (
                    re.sub(r":\s*$", "", label).strip() or "UNK"
                )

            pending_text = []

        elif node.name == "audio":

            audio_id = node.get("id", "")

            match = re.fullmatch(
                rf"{re.escape(recording_id)}-(\d+)-(\d+)",
                audio_id
            )

            if not match:

                pending_text = []
                continue

            start_ms, end_ms = map(int, match.groups())

            src = node.get("src")

            if not src:

                source = node.find("source")

                src = source.get("src") if source else None

            if not src:

                pending_text = []
                continue

            raw_text = raw_source_text(
                "".join(pending_text)
            )

            text = normalize_text(raw_text)

            segments.append({

                "start_ms": start_ms,
                "end_ms": end_ms,

                "speaker_label_private": current_speaker,

                "audio_url": urljoin(page_url, src),

                "text": text,
                "text_orig": raw_text,

                "audio_id": audio_id,

            })

            pending_text = []

    return segments


# ============================================================
# DOWNLOAD WAV
# ============================================================

def download_audio(session, url, target, logger, segment_id):

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    started_at = time.monotonic()
    bytes_written = 0
    http_status = ""

    logger.info("wav_start segment=%s url=%s target=%s", segment_id, url, target)

    try:
        with session.get(url, timeout=120, stream=True) as response:
            http_status = response.status_code
            response.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

        with open(tmp, "rb") as f:
            header = f.read(12)

        if header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
            raise RuntimeError(f"???????????? WAV: {url}")

        tmp.replace(target)
        elapsed = time.monotonic() - started_at
        logger.info(
            "wav_ok segment=%s status=%s bytes=%d elapsed_seconds=%.3f path=%s",
            segment_id, http_status, bytes_written, elapsed, target
        )
        return {
            "http_status": http_status,
            "bytes": bytes_written,
            "elapsed_seconds": elapsed,
        }

    except Exception:
        if tmp.exists():
            tmp.unlink()
        elapsed = time.monotonic() - started_at
        logger.exception(
            "wav_error segment=%s status=%s bytes=%d elapsed_seconds=%.3f url=%s",
            segment_id, http_status, bytes_written, elapsed, url
        )
        raise

# ============================================================
# CSV
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str]
):

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main(argv=None):

    parser = argparse.ArgumentParser(
        description=(
            "Build a PEZ-style folder for one LingConLab recording."
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="HTML recording page URL"
    )

    parser.add_argument(
        "--recording-id",
        default=DEFAULT_RECORDING_ID,
        help="Recording ID, including Cyrillic IDs if applicable"
    )

    parser.add_argument(
        "--corpus",
        default="pinega",
        help="Corpus name"
    )

    parser.add_argument(
        "--out",
        default="lingconlab_pez_output",
        help="Output directory for the corpus"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Delay between WAV downloads"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTML request timeout in seconds"
    )

    args = parser.parse_args(argv)

    # --------------------------------------------------------
    # Validate recording ID
    # --------------------------------------------------------

    if not re.fullmatch(
        r"[\w-]+",
        args.recording_id,
        flags=re.UNICODE
    ):

        raise ValueError(
            f"Invalid recording ID: {args.recording_id}"
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pinega_pez")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_format = "%(asctime)s %(levelname)s %(message)s"
    file_handler = logging.FileHandler(out_root / "download.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)


    recording_root = (
        out_root / args.recording_id
    )

    segment_root = (
        recording_root /
        f"{args.recording_id}No0"
    )

    segment_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Fetch HTML
    # --------------------------------------------------------

    session = make_session()

    print(f"[corpus] {args.corpus}")
    print(f"[recording] {args.recording_id}")
    print(f"[fetch] {args.url}")

    with session.get(args.url, timeout=args.timeout) as page_response:
        page_response.raise_for_status()
        page_response.encoding = page_response.apparent_encoding or "utf-8"
        segments = extract_segments(
            page_response.text,
            args.url,
            args.recording_id
        )

    if not segments:

        raise RuntimeError(
            "No audio segments found. "
            "Check the recording ID and HTML structure."
        )

    # --------------------------------------------------------
    # Speaker pseudonyms
    # --------------------------------------------------------

    key = load_or_create_secret(out_root)

    speaker_order = {}

    metadata_rows = []

    errors = []

    print(
        f"[parse] Found {len(segments)} audio-linked segments."
    )

    # --------------------------------------------------------
    # Process segments
    # --------------------------------------------------------

    for index, segment in enumerate(

        tqdm(
            segments,
            desc=f"{args.corpus}/{args.recording_id}",
            unit="сегм.",
            dynamic_ncols=True
        ),

        start=1

    ):

        source_label = segment.pop(
            "speaker_label_private"
        )

        if source_label not in speaker_order:

            speaker_order[source_label] = (
                len(speaker_order) + 1
            )

        speaker_id = pseudonym(
            source_label,
            key
        )

        start = timestamp_ms_to_filename(
            segment["start_ms"]
        )

        stem = (
            f"{index:03d}_{speaker_id}_{start}"
        )

        txt_path = segment_root / f"{stem}.txt"

        orig_path = segment_root / f"{stem}_orig.txt"

        wav_path = segment_root / f"{stem}.wav"

        # Save individual text segments
        txt_path.write_text(
            segment["text"] + "\n",
            encoding="utf-8"
        )

        orig_path.write_text(
            segment["text_orig"] + "\n",
            encoding="utf-8"
        )

        # Download audio segment
        status = "ok"
        download_info = {"http_status": "", "bytes": 0, "elapsed_seconds": 0.0}

        try:
            download_info = download_audio(
                session,
                segment["audio_url"],
                wav_path,
                logger,
                f"{index:03d}/{stem}"
            )
        except Exception as exc:
            status = f"ERROR: {exc}"
            errors.append((stem, str(exc)))

        # Metadata
        metadata_rows.append({

            "corpus": args.corpus,

            "segment_id": f"{index:03d}",

            "recording_id": args.recording_id,

            "speaker_id": speaker_id,

            "start_ms": segment["start_ms"],

            "end_ms": segment["end_ms"],

            "start_seconds": (
                f'{segment["start_ms"] / 1000:.3f}'
            ),

            "end_seconds": (
                f'{segment["end_ms"] / 1000:.3f}'
            ),

            "duration_seconds": (
                f'{(segment["end_ms"] - segment["start_ms"]) / 1000:.3f}'
            ),

            "audio_file": (
                f"{args.recording_id}/"
                f"{args.recording_id}No0/{stem}.wav"
            ),

            "text_file": (
                f"{args.recording_id}/"
                f"{args.recording_id}No0/{stem}.txt"
            ),

            "text_orig_file": (
                f"{args.recording_id}/"
                f"{args.recording_id}No0/{stem}_orig.txt"
            ),

            "download_status": status,

            "download_http_status": download_info["http_status"],

            "download_bytes": download_info["bytes"],

            "download_elapsed_seconds": f'{download_info["elapsed_seconds"]:.3f}',

            "source_page": args.url,

        })

        time.sleep(
            max(0.0, args.delay)
        )

    # --------------------------------------------------------
    # Speaker map
    # --------------------------------------------------------

    speaker_rows = [

        {
            "speaker_id": pseudonym(label, key),
            "speaker_order": order
        }

        for label, order in sorted(
            speaker_order.items(),
            key=lambda item: item[1]
        )

    ]

    write_csv(
        recording_root / "speaker_map.csv",
        speaker_rows,
        ["speaker_id", "speaker_order"]
    )

    # --------------------------------------------------------
    # Recording metadata
    # --------------------------------------------------------

    if metadata_rows:

        write_csv(
            recording_root / "metadata.csv",
            metadata_rows,
            list(metadata_rows[0].keys())
        )

        # Corpus-level metadata.csv.
        # This file is replaced on each invocation;
        # the outer Colab loop must aggregate recording metadata.
        write_csv(
            out_root / "metadata.csv",
            metadata_rows,
            list(metadata_rows[0].keys())
        )

    # --------------------------------------------------------
    # README
    # --------------------------------------------------------

    readme = f"""# {args.corpus} — PEZ-style corpus

Recording: `{args.recording_id}`

Source page: {args.url}

## Structure

- `{args.recording_id}/{args.recording_id}No0/`
  - `.wav` — individual audio segment
  - `.txt` — normalized segment transcription
  - `_orig.txt` — source segment transcription
- `{args.recording_id}/speaker_map.csv`
- `{args.recording_id}/metadata.csv`

## Privacy

Speaker labels are replaced with HMAC-based pseudonyms.

The private `.speaker_salt` file is required to reproduce
the same pseudonyms on later runs.

**Do not upload `.speaker_salt` to a public repository.**

## Notes

Only individual audio-linked transcript segments are saved.
No complete recording transcripts are generated.

Text normalization changes whitespace only.
Words, punctuation and casing are preserved.

Overlapping segments are preserved.

Check the source website's terms and permissions
before redistribution.
"""

    (recording_root / "README.md").write_text(
        readme,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()

    print(f"[done] Corpus: {args.corpus}")

    print(f"[done] Recording: {args.recording_id}")

    print(f"[done] Output: {recording_root}")

    print(f"[done] Segments: {len(metadata_rows)}")

    print(f"[done] Speakers: {len(speaker_rows)}")

    if errors:

        print(
            f"[warning] Audio download failures: {len(errors)}"
        )

        for name, error in errors:

            print(f"  - {name}: {error}")

        return 2

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
