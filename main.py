r"""
MediaVault Backend - FastAPI Server
====================================
Copy this file to: C:\Users\nguon\mediavault-backend\main.py
Then run: python -m uvicorn main:app --reload --port 8000

Requirements (already installed):
  fastapi, uvicorn, yt-dlp, python-multipart, deep-translator, faster-whisper, edge-tts
"""

import os
import uuid
import asyncio
import subprocess
import threading
from datetime import timedelta
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
COOKIES_FILE = str(Path(os.path.expanduser("~")) / "mediavault-cookies.txt")

_env_cookies = os.environ.get("YOUTUBE_COOKIES", "").strip()
if _env_cookies:
    Path(COOKIES_FILE).write_text(_env_cookies, encoding="utf-8")

# ── PO Token provider ───────────────────────────────────────────────────────
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
    return opts


class CookiesUpdateRequest(BaseModel):
    cookies: str


@app.post("/settings/cookies")
async def update_cookies(req: CookiesUpdateRequest):
    content = req.cookies.strip()
    if not content:
        raise HTTPException(status_code=400, detail="cookies must not be empty")
    Path(COOKIES_FILE).write_text(content, encoding="utf-8")
    return {"status": "ok", "message": "Cookies updated", "bytes": len(content)}


@app.get("/settings/cookies-status")
async def cookies_status():
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
    start_time: str
    end_time: str
    output_name: Optional[str] = None


class TranslateSubtitleRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "auto"
    target_lang: str = "en"


class DubAudioRequest(BaseModel):
    video_path: str
    audio_path: str
    output_name: Optional[str] = None


class TranslateVideoRequest(BaseModel):
    url: str
    mode: str = "subtitle"          # "subtitle" or "dub"
    target_lang: Optional[str] = "km"
    voice: Optional[str] = "km-KH-PisethNeural"   # only used for mode="dub"


# ── Helpers ────────────────────────────────────────────────────────────────────
def format_duration(seconds: float) -> str:
    if not seconds:
        return "0:00"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _download_with_fallback(url: str, ydl_opts: dict):
    """Shared multi-attempt yt-dlp download used by /save and /translate-video."""
    import yt_dlp

    original_format = ydl_opts.get("format")
    formats_to_try = [f for f in [original_format, "bv*+ba/b", "best"] if f]
    seen_f: set = set()
    formats_to_try = [f for f in formats_to_try if not (f in seen_f or seen_f.add(f))]

    attempts = [
        {"extractor_args": None},
        {"extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"extractor_args": {"youtube": {"player_client": ["android"]}}},
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
                    return info, ydl.prepare_filename(info)
            except Exception as e:
                last_error = e
                continue

    raise RuntimeError(str(last_error) if last_error else "Download failed")


def run_download(job_id: str, url: str, ydl_opts: dict):
    """Background thread: run yt-dlp download and update job store."""
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

    try:
        info, filename = _download_with_fallback(url, ydl_opts)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["title"] = info.get("title", "")
        jobs[job_id]["filename"] = filename
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Translate / Dub pipeline helpers ────────────────────────────────────────
_whisper_model = None


def get_whisper_model():
    """Lazy-load faster-whisper model once per process."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        size = os.environ.get("WHISPER_MODEL_SIZE", "base")
        _whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
    return _whisper_model


def extract_audio(video_path: Path) -> Path:
    audio_path = video_path.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract error: {result.stderr[-500:]}")
    return audio_path


def transcribe_audio(audio_path: Path):
    """Returns (segments[{start,end,text}], detected_language)."""
    model = get_whisper_model()
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)
    segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments_iter]
    return segments, info.language


def translate_texts(texts, target_lang="km", source_lang="auto"):
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    out = []
    for t in texts:
        if not t.strip():
            out.append("")
            continue
        try:
            out.append(translator.translate(t))
        except Exception:
            out.append(t)  # fall back to original text rather than failing the whole job
    return out


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments, translated_texts, out_path: Path):
    lines = []
    for i, (seg, text) in enumerate(zip(segments, translated_texts), start=1):
        if not text.strip():
            continue
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}")
        lines.append(text)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def burn_subtitles(video_path: Path, srt_path: Path, out_path: Path):
    escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"subtitles='{escaped}'",
           "-c:a", "copy", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn error: {result.stderr[-500:]}")


async def _tts_save(text: str, out_path: Path, voice: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def text_to_speech(text: str, out_path: Path, voice: str):
    asyncio.run(_tts_save(text, out_path, voice))


def run_translate_job(job_id: str, url: str, mode: str, target_lang: str, voice: str):
    """Background thread: download -> transcribe -> translate -> subtitle-burn or dub."""
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["stage"] = "downloading"
        jobs[job_id]["progress"] = 5

        output_template = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")
        ydl_opts = with_cookies({
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "bv*+ba/b",
            "merge_output_format": "mp4",
        })
        info, video_file = _download_with_fallback(url, ydl_opts)
        video_path = Path(video_file)
        jobs[job_id]["title"] = info.get("title", "")
        jobs[job_id]["progress"] = 30

        jobs[job_id]["stage"] = "extracting_audio"
        audio_path = extract_audio(video_path)
        jobs[job_id]["progress"] = 40

        jobs[job_id]["stage"] = "transcribing"
        segments, detected_lang = transcribe_audio(audio_path)
        jobs[job_id]["detected_language"] = detected_lang
        jobs[job_id]["progress"] = 60

        jobs[job_id]["stage"] = "translating"
        texts = [s["text"] for s in segments]
        translated = translate_texts(texts, target_lang=target_lang, source_lang="auto")
        jobs[job_id]["progress"] = 75

        if mode == "dub":
            jobs[job_id]["stage"] = "generating_speech"
            full_text = " ".join(t for t in translated if t)
            dub_audio_path = video_path.with_name(video_path.stem + "_dub_audio.mp3")
            text_to_speech(full_text, dub_audio_path, voice)
            jobs[job_id]["progress"] = 88

            jobs[job_id]["stage"] = "merging"
            out_path = DOWNLOAD_DIR / f"{video_path.stem}_dubbed.mp4"
            cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(dub_audio_path),
                   "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(out_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg dub merge error: {result.stderr[-500:]}")
            jobs[job_id]["filename"] = str(out_path)
        else:
            jobs[job_id]["stage"] = "burning_subtitles"
            srt_path = video_path.with_suffix(".srt")
            build_srt(segments, translated, srt_path)
            out_path = DOWNLOAD_DIR / f"{video_path.stem}_sub.mp4"
            burn_subtitles(video_path, srt_path, out_path)
            jobs[job_id]["filename"] = str(out_path)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "done"
        jobs[job_id]["progress"] = 100
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "MediaVault backend is running",
        "youtube_cookies_configured": Path(COOKIES_FILE).exists() and Path(COOKIES_FILE).stat().st_size > 0,
        "pot_provider_configured": bool(POT_PROVIDER_URL),
    }


@app.post("/media-info")
def media_info(req: MediaInfoRequest):
    import yt_dlp

    ydl_opts_base = with_cookies({
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    })

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


@app.post("/translate-video")
def translate_video(req: TranslateVideoRequest):
    """
    Download a video and produce a Khmer-translated version:
      mode="subtitle" -> Khmer .srt burned into the video (fast, cheap)
      mode="dub"      -> original audio replaced with Khmer TTS voice (slower)
    Poll status with the existing /save-progress/{job_id} and fetch the
    finished file with the existing /download/{job_id}.
    """
    if req.mode not in ("subtitle", "dub"):
        raise HTTPException(status_code=400, detail="mode must be 'subtitle' or 'dub'")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "url": req.url,
        "mode": req.mode,
        "target_lang": req.target_lang,
        "filename": "",
        "error": "",
    }

    t = threading.Thread(
        target=run_translate_job,
        args=(job_id, req.url, req.mode, req.target_lang, req.voice),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "status": "queued", "mode": req.mode}


@app.get("/save-progress/{job_id}")
def save_progress(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}")
def download_file(job_id: str):
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg error: {result.stderr[-500:]}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install ffmpeg and add it to PATH.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ffmpeg timed out")

    return {"status": "completed", "output_path": str(output_path), "output_name": out_name}


@app.post("/translate-subtitle")
def translate_subtitle(req: TranslateSubtitleRequest):
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=req.source_lang, target=req.target_lang)
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg error: {result.stderr[-500:]}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="ffmpeg not found. Install ffmpeg and add it to PATH.")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="ffmpeg timed out")

    return {"status": "completed", "output_path": str(output_path), "output_name": out_name}


# ── Dev entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
