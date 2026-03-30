"""Streamlit GUI for Remixatron.

This app ports the Remixatron web workflow into a single Streamlit interface.
It supports:
- YouTube URL import (via yt-dlp)
- Local audio upload
- Cluster controls and beat-cache reuse
- Remix generation with inline audio playback
- Beat/cluster/segment visualization and jump inspection
- Shared bookmarks file compatible with the Flask web UI
"""

from __future__ import annotations

import bz2
import io
import json
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from Remixatron import InfiniteJukebox

APP_TITLE = "Remixatron • Streamlit Studio"
REMIXATRON_DIR = Path.home() / ".remixatron"
BOOKMARKS_FILE = REMIXATRON_DIR / "remixatron.global.bookmarks"
CACHE_SUFFIX = ".beatmap.bz2"


@dataclass
class SourceInfo:
    title: str
    thumbnail: str
    url: str


def _ensure_app_dirs() -> None:
    REMIXATRON_DIR.mkdir(exist_ok=True)
    if not BOOKMARKS_FILE.exists():
        BOOKMARKS_FILE.write_text("[]", encoding="utf-8")


def load_bookmarks() -> list[dict[str, Any]]:
    _ensure_app_dirs()
    try:
        return json.loads(BOOKMARKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_bookmarks(bookmarks: list[dict[str, Any]]) -> None:
    BOOKMARKS_FILE.write_text(json.dumps(bookmarks, indent=2), encoding="utf-8")


def add_or_update_bookmark(item: dict[str, Any]) -> None:
    bookmarks = load_bookmarks()
    deduped = [b for b in bookmarks if b.get("title") != item.get("title")]
    deduped.append(item)
    deduped.sort(key=lambda x: x.get("title", "").lower())
    save_bookmarks(deduped)


def remove_bookmark(title: str) -> None:
    bookmarks = [b for b in load_bookmarks() if b.get("title") != title]
    save_bookmarks(bookmarks)


def progress_callback_factory(progress_bar, status_box):
    def _cb(percentage: float, message: str) -> None:
        progress_bar.progress(max(0.0, min(1.0, float(percentage))))
        status_box.info(f"{int(percentage * 100):>3}% • {message}")

    return _cb


def fetch_from_youtube(url: str, temp_dir: Path) -> tuple[Path, SourceInfo]:
    out_tpl = str(temp_dir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "--write-info-json",
        "-x",
        "--audio-format",
        "wav",
        "-f",
        "bestaudio",
        "--no-playlist",
        "--print",
        "after_move:filepath",
        "-o",
        out_tpl,
        url,
    ]

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    audio_path = Path(lines[-1])

    info_candidates = sorted(temp_dir.glob("*.info.json"))
    info = {}
    if info_candidates:
        info = json.loads(info_candidates[-1].read_text(encoding="utf-8"))

    return audio_path, SourceInfo(
        title=info.get("title", url),
        thumbnail=info.get("thumbnail", ""),
        url=url,
    )


def fetch_from_upload(uploaded_file, temp_dir: Path) -> tuple[Path, SourceInfo]:
    src_path = temp_dir / uploaded_file.name
    src_path.write_bytes(uploaded_file.getbuffer())

    normalized = temp_dir / "upload.ogg"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src_path), str(normalized)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return normalized, SourceInfo(
        title=uploaded_file.name,
        thumbnail="",
        url=uploaded_file.name,
    )


def make_cache_path(cache_key: str) -> Path:
    return REMIXATRON_DIR / f"{urllib.parse.quote(cache_key, safe='')}{CACHE_SUFFIX}"


def build_beatmap(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": b["id"],
            "start": b["start"] * 1000.0,
            "duration": b["duration"] * 1000.0,
            "segment": b["segment"],
            "cluster": b["cluster"],
            "jump_candidate": b["jump_candidates"],
        }
        for b in beats
    ]


def generate_remix_wav(jukebox: InfiniteJukebox, remix_seconds: int) -> bytes:
    chunks: list[np.ndarray] = []
    elapsed = 0.0

    for entry in jukebox.play_vector:
        beat = jukebox.beats[entry["beat"]]
        chunks.append(np.asarray(beat["buffer"]))
        elapsed += float(beat["duration"])
        if elapsed >= remix_seconds:
            break

    audio = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    sf.write(buf, audio, jukebox.sample_rate, format="WAV")
    return buf.getvalue()


def process_audio(
    source_path: Path,
    cache_key: str,
    clusters: int,
    use_cache: bool,
    callback,
) -> InfiniteJukebox:
    cache_path = make_cache_path(cache_key)

    starting_beat_cache = None
    if use_cache and cache_path.exists():
        with bz2.open(cache_path, "rb") as f:
            starting_beat_cache = json.load(f)

    jukebox = InfiniteJukebox(
        str(source_path),
        clusters=clusters,
        progress_callback=callback,
        start_beat=0,
        do_async=False,
        starting_beat_cache=starting_beat_cache,
    )

    if not cache_path.exists() or not use_cache:
        with bz2.open(cache_path, "wb") as f:
            f.write(json.dumps(jukebox.beats, default=lambda _: "").encode("utf-8"))

    return jukebox


def init_session_state() -> None:
    defaults = {
        "ready": False,
        "source_info": None,
        "jukebox": None,
        "beatmap": [],
        "play_vector": [],
        "audio_bytes": b"",
        "cluster_scores": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_analysis() -> None:
    beatmap = st.session_state["beatmap"]
    if not beatmap:
        return

    clusters = max(b["cluster"] for b in beatmap) + 1
    segments = max(b["segment"] for b in beatmap) + 1

    st.subheader("Track analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beats", len(beatmap))
    c2.metric("Clusters", clusters)
    c3.metric("Segments", segments)
    c4.metric("Seg/Cluster ratio", f"{segments / max(clusters, 1):.3f}")

    df = pd.DataFrame(
        {
            "beat": [b["id"] for b in beatmap],
            "cluster": [b["cluster"] for b in beatmap],
            "segment": [b["segment"] for b in beatmap],
            "duration_ms": [round(b["duration"], 2) for b in beatmap],
            "jump_count": [len(b["jump_candidate"]) for b in beatmap],
        }
    )

    st.caption("Beat-to-cluster map (first 1500 beats)")
    st.scatter_chart(df.head(1500), x="beat", y="cluster", color="segment", size="jump_count")

    selected = st.slider("Inspect beat", 0, len(beatmap) - 1, 0)
    beat = beatmap[selected]
    st.json(
        {
            "beat": beat["id"],
            "cluster": beat["cluster"],
            "segment": beat["segment"],
            "duration_ms": round(beat["duration"], 2),
            "jump_candidate_count": len(beat["jump_candidate"]),
            "jump_candidates_preview": beat["jump_candidate"][:20],
        }
    )

    with st.expander("Beat table"):
        st.dataframe(df, use_container_width=True)


def main() -> None:
    _ensure_app_dirs()
    init_session_state()

    st.set_page_config(page_title=APP_TITLE, page_icon="🎛️", layout="wide")
    st.title(APP_TITLE)
    st.caption("A full Streamlit GUI port of the Remixatron Web UI workflow.")

    with st.sidebar:
        st.header("Source")
        source_mode = st.radio("Input type", ["YouTube URL", "Upload file", "Bookmark"])

        bookmarks = load_bookmarks()
        chosen_bookmark = None
        if source_mode == "Bookmark":
            if not bookmarks:
                st.warning("No bookmarks saved yet.")
            else:
                titles = [b.get("title", "Untitled") for b in bookmarks]
                selected_title = st.selectbox("Saved track", titles)
                chosen_bookmark = next(b for b in bookmarks if b.get("title") == selected_title)

        default_url = chosen_bookmark.get("url", "") if chosen_bookmark else ""
        url = st.text_input("YouTube URL", value=default_url, disabled=source_mode == "Upload file")

        uploaded = st.file_uploader(
            "Upload audio",
            type=["mp3", "wav", "ogg", "m4a", "flac"],
            disabled=source_mode != "Upload file",
        )

        default_clusters = int(chosen_bookmark.get("clusters", 0)) if chosen_bookmark else 0
        clusters = st.number_input("Clusters (0 = auto)", 0, 128, value=default_clusters, step=1)
        use_cache = st.toggle("Reuse beat cache", value=True)
        remix_seconds = st.slider("Remix preview length (seconds)", 20, 600, 180, step=10)

        process_clicked = st.button("Process / Remix", type="primary", use_container_width=True)

    if process_clicked:
        progress_bar = st.progress(0.0)
        status_box = st.empty()

        with tempfile.TemporaryDirectory(prefix="remixatron-st-") as td:
            temp_dir = Path(td)
            callback = progress_callback_factory(progress_bar, status_box)

            try:
                if source_mode in ("YouTube URL", "Bookmark"):
                    if not url.strip():
                        st.error("Please enter a YouTube URL.")
                        st.stop()
                    source_path, source_info = fetch_from_youtube(url.strip(), temp_dir)
                    cache_key = source_info.url
                else:
                    if uploaded is None:
                        st.error("Please upload an audio file.")
                        st.stop()
                    source_path, source_info = fetch_from_upload(uploaded, temp_dir)
                    cache_key = f"upload:{source_info.title}"

                callback(0.05, "Source prepared. Starting analysis…")
                jukebox = process_audio(source_path, cache_key, int(clusters), use_cache, callback)
                callback(0.96, "Generating remix preview…")

                audio_bytes = generate_remix_wav(jukebox, remix_seconds)
                beatmap = build_beatmap(jukebox.beats)

                st.session_state["ready"] = True
                st.session_state["source_info"] = source_info
                st.session_state["jukebox"] = jukebox
                st.session_state["beatmap"] = beatmap
                st.session_state["play_vector"] = jukebox.play_vector
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["cluster_scores"] = getattr(jukebox, "cluster_ratio_log", [])

                callback(1.0, "Done.")
                st.success("Remix is ready.")
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                st.error(f"External tool failed: {stderr or exc}")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    if st.session_state["ready"]:
        source_info = st.session_state["source_info"]

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Now playing")
            st.write(f"**{source_info.title}**")
            st.audio(st.session_state["audio_bytes"], format="audio/wav")

            dl_name = f"{source_info.title[:80].replace('/', '_') or 'remix'}.wav"
            st.download_button(
                "Download remix preview",
                data=st.session_state["audio_bytes"],
                file_name=dl_name,
                mime="audio/wav",
            )

        with right:
            st.subheader("Bookmarks")
            if source_info.url.startswith("http"):
                c1, c2 = st.columns(2)
                if c1.button("⭐ Save bookmark", use_container_width=True):
                    add_or_update_bookmark(
                        {
                            "title": source_info.title,
                            "thumbnail": source_info.thumbnail,
                            "url": source_info.url,
                            "clusters": st.session_state["jukebox"].clusters,
                        }
                    )
                    st.success("Bookmark saved.")
                if c2.button("🗑️ Remove bookmark", use_container_width=True):
                    remove_bookmark(source_info.title)
                    st.info("Bookmark removed.")
            else:
                st.info("Uploaded tracks are local-only and are not bookmarkable.")

            with st.expander("Source metadata", expanded=True):
                st.json(
                    {
                        "title": source_info.title,
                        "url": source_info.url,
                        "thumbnail": source_info.thumbnail,
                        "sample_rate": st.session_state["jukebox"].sample_rate,
                        "tempo_bpm": round(float(st.session_state["jukebox"].tempo), 2),
                        "clusters": int(st.session_state["jukebox"].clusters),
                    }
                )

        render_analysis()

        if st.session_state["cluster_scores"]:
            st.subheader("Auto-cluster diagnostics")
            cluster_df = pd.DataFrame(st.session_state["cluster_scores"])
            st.dataframe(cluster_df, use_container_width=True)


if __name__ == "__main__":
    main()
