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
# Set the YOUTUBE_COOKIES env var (Railway → Variables) to the full contents of
# a cookies.txt file exported from a logged-in YouTube session (Netscape format,
# e.g. via the "Get cookies.txt LOCALLY" browser extension). If set, we write it
# to a temp file once at startup and pass it to yt-dlp as cookiefile.
_COOKIES_ENV = os.environ.get("YOUTUBE_COOKIES", "").strip()
COOKIES_FILE: Optional[str] = None
if _COOKIES_ENV:
    _cookies_path = Path(os.path.expanduser("~")) / "mediavault-cookies.txt"
    _cookies_path.write_text(_COOKIES_ENV, encoding="utf-8")
    COOKIES_FILE = str(_cookies_path)


def with_cookies(opts: dict) -> dict:
    """Attach cookiefile + YouTube bot-check mitigations to yt-dlp opts."""
    if COOKIES_FILE:
        opts["cookiefile"] = COOKIES_FILE
    # Datacenter IPs (Railway, AWS, etc.) get flagged by YouTube's bot check
    # even with valid cookies. Using the Android/iOS player client skips the
    # web player's bot-check path entirely — this is the standard yt-dlp
    # workaround as of 2025-2026.
    opts["extractor_args"] = {
        "youtube": {
            "player_client": ["android", "ios", "web"],
        }
    }
    opts["http_headers"] = {
        "User-Agent": "com.google.android.youtube/19.29.37 (Linux; U; Android 11) gzip",
    }
    return opts


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

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["title"] = info.get("title", "")
            jobs[job_id]["filename"] = ydl.prepare_filename(info)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — frontend polls this to show green/red dot."""
    return {
        "status": "ok",
        "message": "MediaVault backend is running",
        "youtube_cookies_configured": COOKIES_FILE is not None,
    }


@app.post("/media-info")
def media_info(req: MediaInfoRequest):
    """
    Fetch video metadata using yt-dlp (no download).
    Returns title, duration, thumbnail, formats, etc.
    """
    import yt_dlp

    ydl_opts = with_cookies({
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        ydl_opts["format"] = req.format_id or "bestvideo+bestaudio/best"
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
