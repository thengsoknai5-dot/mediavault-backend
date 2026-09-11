r"""
MediaVault Backend - FastAPI Server
====================================
Copy this file to: C:\Users\nguon\mediavault-backend\main.py
Then run: python -m uvicorn main:app --reload --port 8000

Requirements (already installed):
  fastapi, uvicorn, yt-dlp, python-multipart, deep-translator
"""

import os
import uuid
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="MediaVault API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://mediavault4594.builtwithrocket.new",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: Dict[str, Dict[str, Any]] = {}

# ── Download directory ─────────────────────────────────────────────────────────
DOWNLOAD_DIR = Path(os.path.expanduser("~")) / "mediavault-downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── YouTube cookies (fixes "Sign in to confirm you're not a bot") ──────────────
# Cookies can come from either the YOUTUBE_COOKIES env var (set once in Railway)
# or be updated live via POST /settings/cookies from the app's Settings panel —
# useful since YouTube session cookies expire every few days/weeks and editing
# Railway variables + redeploying every time is slow. Either way they end up in
# the same file on disk, and with_cookies() checks the file fresh on every
# request so an update takes effect immediately with no restart needed.
COOKIES_FILE = str(Path(os.path.expanduser("~")) / "mediavault-cookies.txt")

_env_cookies = os.environ.get("YOUTUBE_COOKIES", "").strip()
if _env_cookies:
    Path(COOKIES_FILE).write_text(_env_cookies, encoding="utf-8")

# ── PO Token provider (fixes "Requested format is not available" from
# YouTube's SABR streaming rollout — yt-dlp/yt-dlp#12482) ──────────────────
# Set POT_PROVIDER_URL to the internal URL of a running
# brainicism/bgutil-ytdlp-pot-provider service (e.g. a second Railway
# service reachable at http://<service-name>.railway.internal:4416). When
# set, every yt-dlp call is told to fetch a Proof-of-Origin token from it,
# which restores the download URLs YouTube otherwise strips from formats.
POT_PROVIDER_URL = os.environ.get("POT_PROVIDER_URL", "").strip()


def merge_pot_provider(extractor_args: Optional[dict]) -> Optional[dict]:
    if not POT_PROVIDER_URL:
        return extractor_args
    merged = dict(extractor_args) if extractor_args else {}
    merged["youtubepot-bgutilhttp"] = {"base_url": [POT_PROVIDER_URL]}
    return merged


def with_cookies(opts: dict) -> dict:
    """Attach cookiefile for YouTube bot-check bypass."""
    if Path(COOKIES_FILE).exists() and Path(COOKIES_FILE).stat().st_size > 0:
        opts["cookiefile"] = COOKIES_FILE
    # Note: we intentionally do NOT force a specific player_client (e.g.
    # android/ios) here. Forcing those clients avoids the bot-check on some
    # videos but strips out format data on others — "Requested format is not
    # available" for videos that work fine elsewhere is a symptom of that.
    # With valid, fresh cookies the default web client is bypassed correctly
    # and keeps the full format list intact.
    return opts


class CookiesUpdateRequest(BaseModel):
    cookies: str


@app.post("/settings/cookies")
async def update_cookies(req: CookiesUpdateRequest):
    """Update the YouTube cookies file at runtime (no redeploy needed)."""
    content = req.cookies.strip()
    if not content:
        raise HTTPException(status_code=400, detail="cookies must not be empty")
    Path(COOKIES_FILE).write_text(content, encoding="utf-8")
    return {"status": "ok", "message": "Cookies updated", "bytes": len(content)}


@app.get("/settings/cookies-status")
async def cookies_status():
    """Whether cookies are currently configured, and roughly how fresh."""
    p = Path(COOKIES_FILE)
    if not p.exists() or p.stat().st_size == 0:
        return {"configured": False}
    return {
        "configured": True,
        "updated_at": p.stat().st_mtime,
        "size_bytes": p.stat().st_size,
    }


# ── Request / Response models ──────────────────────────────────────────────────
class MediaInfoRequest(BaseModel):
    url: str


class SaveRequest(BaseModel):
    url: str
    format_id: Optional[str] = "bestvideo+bestaudio/best"
    output_name: Optional[str] = None
    audio_only: Optional[bool] = False


class TrimRequest(BaseModel):
    file_path: str
    start_time: str   # e.g. "00:00:10" end_time: str     # e.g."00:01:30"
    output_name: Optional[str] = None


class TranslateSubtitleRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "auto"
    target_lang: str = "en"


class DubAudioRequest(BaseModel):
    video_path: str
    audio_path: str
    output_name: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────
def format_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    if not seconds:
        return "0:00"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def run_download(job_id: str, url: str, ydl_opts: dict):
    """Background thread: run yt-dlp download and update job store."""
    import yt_dlp

    jobs[job_id]["status"] = "downloading"
    jobs[job_id]["progress"] = 0

    def progress_hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = int(downloaded / total * 100)
                jobs[job_id]["progress"] = pct
                jobs[job_id]["speed"] = d.get("speed", 0)
                jobs[job_id]["eta"] = d.get("eta", 0)
        elif d["status"] == "finished":
            jobs[job_id]["progress"] = 100
            jobs[job_id]["filename"] = d.get("filename", "")

    ydl_opts["progress_hooks"] = [progress_hook]
    ydl_opts.setdefault("retries", 5)
    ydl_opts.setdefault("fragment_retries", 5)
    ydl_opts.setdefault("socket_timeout", 30)

    # Try yt-dlp's own default client selection FIRST — this is what the
    # proven-working desktop version of this tool uses (no forced client at
    # all), and forcing a specific player_client has caused more breakage
    # than it fixed (e.g. "tv" itself being broken upstream). Only fall back
    # to explicitly forcing "web" or "android" if the default path fails.
    original_format = ydl_opts.get("format")
    formats_to_try = [f for f in [original_format, "bv*+ba/b", "best"] if f]
    seen_f: set = set()
    formats_to_try = [f for f in formats_to_try if not (f in seen_f or seen_f.add(f))]

    attempts = [
        {"extractor_args": None},
        {"extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        # YouTube's "SABR streaming" rollout (yt-dlp/yt-dlp#12482, an
        # ongoing, still-unresolved fight between YouTube and yt-dlp as of
        # 2026) strips download URLs from many formats unless a PO Token is
        # presented. formats=missing_pot tells yt-dlp to include those
        # formats anyway — usually lower quality, but still downloadable —
        # instead of erroring out with nothing at all.
        {"extractor_args": {"youtube": {"player_client": ["web"], "formats": ["missing_pot"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android"], "formats": ["missing_pot"]}}},
    ]

    last_error = None
    for attempt in attempts:
        for fmt in formats_to_try:
            try:
                attempt_opts = dict(ydl_opts)
                attempt_opts["format"] = fmt
                if attempt["extractor_args"] is not None:
                    attempt_opts["extractor_args"] = attempt["extractor_args"]
                attempt_opts["extractor_args"] = merge_pot_provider(attempt_opts.get("extractor_args"))
                with yt_dlp.YoutubeDL(attempt_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["progress"] = 100
                    jobs[job_id]["title"] = info.get("title", "")
                    jobs[job_id]["filename"] = ydl.prepare_filename(info)
                    return
            except Exception as e:
                last_error = e
                continue

    jobs[job_id]["status"] = "error"
    jobs[job_id]["error"] = str(last_error) if last_error else "Download failed"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — frontend polls this to show green/red dot."""
    return {
        "status": "ok",
        "message": "MediaVault backend is running",
        "youtube_cookies_configured": Path(COOKIES_FILE).exists() and Path(COOKIES_FILE).stat().st_size > 0,
        "pot_provider_configured": bool(POT_PROVIDER_URL),
    }


@app.post("/media-info")
def media_info(req: MediaInfoRequest):
    """
    Fetch video metadata using yt-dlp (no download).
    Returns title, duration, thumbnail, formats, etc.
    """
    import yt_dlp

    ydl_opts_base = with_cookies({
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    })

    # Same reasoning as /save: try yt-dlp's default client selection first,
    # only forcing a specific one if that fails.
    attempts = [
        None,
        {"youtube": {"player_client": ["web"]}},
        {"youtube": {"player_client": ["android"]}},
        {"youtube": {"player_client": ["web"], "formats": ["missing_pot"]}},
        {"youtube": {"player_client": ["android"], "formats": ["missing_pot"]}},
    ]
    info = None
    last_error: Exception | None = None
    for extractor_args in attempts:
        try:
            opts = dict(ydl_opts_base)
            if extractor_args is not None:
                opts["extractor_args"] = extractor_args
            opts["extractor_args"] = merge_pot_provider(opts.get("extractor_args"))
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(req.url, download=False)
            break
        except Exception as e:
            last_error = e
            continue
    if info is None:
        raise HTTPException(status_code=400, detail=str(last_error) if last_error else "Failed to fetch media info")

    # Build format list
    raw_formats = info.get("formats", [])
    formats = []
    seen = set()
    for f in raw_formats:
        label = f.get("format_note") or f.get("resolution") or f.get("format_id", "")
        ext = f.get("ext", "")
        fid = f.get("format_id", "")
        key = f"{label}-{ext}"
        if key in seen:
            continue
        seen.add(key)
        filesize = f.get("filesize") or f.get("filesize_approx") or 0
        formats.append({
            "format_id": fid,
            "label": label,
            "ext": ext,
            "resolution": f.get("resolution", ""),
            "fps": f.get("fps"),
            "vcodec": f.get("vcodec", ""),
            "acodec": f.get("acodec", ""),
            "filesize": filesize,
            "filesize_mb": round(filesize / 1024 / 1024, 2) if filesize else None,
        })

    return {
        "title": info.get("title", ""),
        "duration": format_duration(info.get("duration", 0)),
        "duration_seconds": info.get("duration", 0),
        "thumbnail": info.get("thumbnail", ""),
        "channel": info.get("uploader", info.get("channel", "")),
        "view_count": info.get("view_count", 0),
        "upload_date": info.get("upload_date", ""),
        "description": info.get("description", "")[:500] if info.get("description") else "",
        "webpage_url": info.get("webpage_url", req.url),
        "formats": formats,
    }


@app.post("/save")
def save(req: SaveRequest, background_tasks: BackgroundTasks):
    """
    Start a yt-dlp download job.
    Returns a job_id for polling via /save-progress.
    """
    job_id = str(uuid.uuid4())

    output_template = str(DOWNLOAD_DIR / (req.output_name or "%(title)s.%(ext)s"))

    ydl_opts: dict = with_cookies({
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    })

    if req.audio_only:
        ydl_opts["format"] = "bestaudio/best"
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
    else:
        # Deliberately ignore req.format_id: format IDs returned by
        # /media-info come from whichever player client (web/android/ios)
        # yt-dlp used for that call, and the same ID may not exist when
        # /save runs its own extraction — that mismatch caused "Requested
        # format is not available". "bv*+ba/b" is a flexible selector: best
        # video (any codec/container) + best audio, falling back to the
        # single best pre-merged stream if that combination fails.
        ydl_opts["format"] = "bv*+ba/b"
        ydl_opts["merge_output_format"] = "mp4"

    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "url": req.url,
        "speed": 0,
        "eta": 0,
        "filename": "",
        "error": "",
    }

    t = threading.Thread(
        target=run_download,
        args=(job_id, req.url, ydl_opts),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "status": "queued", "download_dir": str(DOWNLOAD_DIR)}


@app.get("/save-progress/{job_id}")
def save_progress(job_id: str):
    """Poll download progress for a given job_id."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}")
def download_file(job_id: str):
    """
    Stream a completed job's file back to the browser so it actually saves
    to the user's own device — previously jobs finished on the Railway
    server with no way to retrieve them, since Railway's disk is ephemeral
    and not visible to the user at all.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed yet (status: {job.get('status')})")
    file_path = job.get("filename")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Output file no longer exists on the server")
    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream",
    )


@app.post("/trim")
def trim(req: TrimRequest):
    """
    Trim a video/audio file using ffmpeg.
    Requires ffmpeg to be installed and on PATH.
    """
    input_path = Path(req.file_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    suffix = input_path.suffix
    out_name = req.output_name or f"{input_path.stem}_trimmed{suffix}"
    output_path = DOWNLOAD_DIR / out_name

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", req.start_time,
        "-to", req.end_time,
        "-c", "copy",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg error: {result.stderr[-500:]}"
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Install ffmpeg and add it to PATH."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ffmpeg timed out")

    return {
        "status": "completed",
        "output_path": str(output_path),
        "output_name": out_name,
    }


@app.post("/translate-subtitle")
def translate_subtitle(req: TranslateSubtitleRequest):
    """
    Translate text using deep-translator (Google Translate backend).
    """
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(
            source=req.source_lang,
            target=req.target_lang,
        )
        # deep-translator has a 5000-char limit per call — chunk if needed
        MAX_CHUNK = 4500
        text = req.text
        if len(text) <= MAX_CHUNK:
            translated = translator.translate(text)
        else:
            chunks = [text[i:i + MAX_CHUNK] for i in range(0, len(text), MAX_CHUNK)]
            translated = " ".join(translator.translate(c) for c in chunks)

        return {
            "status": "ok",
            "source_lang": req.source_lang,
            "target_lang": req.target_lang,
            "original": req.text,
            "translated": translated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dub-audio")
def dub_audio(req: DubAudioRequest):
    """
    Replace the audio track of a video with a new audio file using ffmpeg.
    """
    video_path = Path(req.video_path)
    audio_path = Path(req.audio_path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video_path}")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio not found: {req.audio_path}")

    out_name = req.output_name or f"{video_path.stem}_dubbed.mp4"
    output_path = DOWNLOAD_DIR / out_name

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg error: {result.stderr[-500:]}"
            )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Install ffmpeg and add it to PATH."
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ffmpeg timed out")

    return {
        "status": "completed",
        "output_path": str(output_path),
        "output_name": out_name,
    }


# ── Dev entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
