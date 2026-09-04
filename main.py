"""
MediaVault Backend - Personal media management utility
FastAPI service wrapping yt-dlp, ffmpeg, deep-translator, and Coqui TTS.

For personal use on content you own or have permission to use.
"""

import os
import uuid
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import yt_dlp

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="MediaVault Backend")

app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
TRIMMED_DIR = BASE_DIR / "trimmed"
SUBS_DIR = BASE_DIR / "subtitles"
DUBBED_DIR = BASE_DIR / "dubbed"

for d in (DOWNLOADS_DIR, TRIMMED_DIR, SUBS_DIR, DUBBED_DIR):
    d.mkdir(exist_ok=True)

# Serve processed files so the frontend can preview/download them
app.mount("/files/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")
app.mount("/files/trimmed", StaticFiles(directory=TRIMMED_DIR), name="trimmed")
app.mount("/files/dubbed", StaticFiles(directory=DUBBED_DIR), name="dubbed")

# In-memory job tracking (fine for a personal single-user tool;
# swap for a DB/file store if you need persistence across restarts)
JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class MediaInfoRequest(BaseModel):
    url: str


class SaveRequest(BaseModel):
    url: str
    mode: str = "video_audio"   # "video_audio" | "video_only" | "audio_only"
    quality: str = "best"       # e.g. "1080", "720", "480", "best"


class TrimRequest(BaseModel):
    file_path: str  # path returned by /save, relative to downloads dir
    start_time: float  # seconds
    end_time: float    # seconds


class TranslateRequest(BaseModel):
    text: str
    target_lang: str  # e.g. "km" for Khmer, "en", "zh-CN"


class DubRequest(BaseModel):
    file_path: str
    translated_text: str
    target_lang: str
    replace_audio: bool = True


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 1. Media info
# ---------------------------------------------------------------------------

@app.post("/media-info")
def media_info(req: MediaInfoRequest):
    ydl_opts = {"quiet": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch media info: {e}")

    formats = []
    for f in info.get("formats", []):
        if f.get("vcodec") != "none" and f.get("height"):
            formats.append({
                "format_id": f["format_id"],
                "resolution": f"{f.get('height')}p",
                "ext": f.get("ext"),
            })

    subtitles = list(info.get("subtitles", {}).keys()) + list(info.get("automatic_captions", {}).keys())

    return {
        "title": info.get("title"),
        "thumbnail": info.get("thumbnail"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "formats": formats,
        "subtitle_languages": sorted(set(subtitles)),
        "platform": info.get("extractor_key"),
    }


# ---------------------------------------------------------------------------
# 2. Save / download
# ---------------------------------------------------------------------------

def _run_download(job_id: str, url: str, mode: str, quality: str):
    JOBS[job_id] = {"status": "running", "progress": 0}

    def progress_hook(d):
        if d["status"] == "downloading":
            pct = d.get("_percent_str", "0%").strip().replace("%", "")
            try:
                JOBS[job_id]["progress"] = float(pct)
            except ValueError:
                pass
        elif d["status"] == "finished":
            JOBS[job_id]["progress"] = 100

    out_tmpl = str(DOWNLOADS_DIR / f"{job_id}.%(ext)s")

    if mode == "audio_only":
        fmt = "bestaudio/best"
        postprocessors = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}]
    elif mode == "video_only":
        fmt = f"bestvideo[height<={quality}]" if quality != "best" else "bestvideo"
        postprocessors = []
    else:  # video_audio
        fmt = f"bestvideo[height<={quality}]+bestaudio/best" if quality != "best" else "best"
        postprocessors = []

    ydl_opts = {
        "format": fmt,
        "outtmpl": out_tmpl,
        "postprocessors": postprocessors,
        "progress_hooks": [progress_hook],
        "noplaylist": True,
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if mode == "audio_only":
                filename = str(Path(filename).with_suffix(".mp3"))
        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "file_path": Path(filename).name,
            "title": info.get("title"),
        }
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/save")
def save(req: SaveRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_download, job_id, req.url, req.mode, req.quality)
    return {"job_id": job_id}


@app.get("/save/status/{job_id}")
def save_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# 3. Trim
# ---------------------------------------------------------------------------

@app.post("/trim")
def trim(req: TrimRequest):
    src = DOWNLOADS_DIR / req.file_path
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    duration = req.end_time - req.start_time
    if duration <= 0:
        raise HTTPException(status_code=400, detail="end_time must be after start_time")

    out_name = f"trim_{uuid.uuid4().hex[:8]}_{src.name}"
    out_path = TRIMMED_DIR / out_name

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(req.start_time),
        "-i", str(src),
        "-t", str(duration),
        "-c", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not out_path.exists():
        # Fallback: re-encode (stream copy can fail on some keyframe boundaries)
        cmd_reencode = [
            "ffmpeg", "-y",
            "-ss", str(req.start_time),
            "-i", str(src),
            "-t", str(duration),
            str(out_path),
        ]
        result = subprocess.run(cmd_reencode, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg trim failed: {result.stderr[-500:]}")

    return {"file_path": out_name, "url": f"/files/trimmed/{out_name}"}


# ---------------------------------------------------------------------------
# 4. Translate subtitles
# ---------------------------------------------------------------------------

@app.post("/translate-subtitle")
def translate_subtitle(req: TranslateRequest):
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target=req.target_lang).translate(req.text)
        return {"translated_text": translated, "engine": "deep-translator"}
    except Exception as e:
        # Offline fallback
        try:
            import argostranslate.translate
            translated = argostranslate.translate.translate(req.text, "en", req.target_lang)
            return {"translated_text": translated, "engine": "argos-translate (offline)"}
        except Exception as e2:
            raise HTTPException(
                status_code=500,
                detail=f"Translation failed. Online error: {e}. Offline fallback error: {e2}"
            )


# ---------------------------------------------------------------------------
# 5. Voice dubbing (Coqui TTS)
# ---------------------------------------------------------------------------

def _run_dub(job_id: str, file_path: str, translated_text: str, target_lang: str, replace_audio: bool):
    JOBS[job_id] = {"status": "running"}
    try:
        src = DOWNLOADS_DIR / file_path
        if not src.exists():
            src = TRIMMED_DIR / file_path
        if not src.exists():
            raise FileNotFoundError("Source video not found")

        work_dir = DUBBED_DIR / job_id
        work_dir.mkdir(exist_ok=True)

        # 1. Extract a short voice sample from the original audio for cloning
        sample_path = work_dir / "speaker_sample.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-t", "10", "-ac", "1", "-ar", "22050",
            str(sample_path)
        ], capture_output=True, check=True)

        # 2. Generate dubbed speech with Coqui XTTS-v2, cloning the sample voice
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        dubbed_audio_path = work_dir / "dubbed_audio.wav"
        tts.tts_to_file(
            text=translated_text,
            speaker_wav=str(sample_path),
            language=target_lang,
            file_path=str(dubbed_audio_path),
        )

        # 3. Merge dubbed audio into the video
        out_name = f"dubbed_{src.stem}.mp4"
        out_path = DUBBED_DIR / out_name

        if replace_audio:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-i", str(dubbed_audio_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-shortest",
                str(out_path),
            ]
        else:
            # Keep both audio tracks
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-i", str(dubbed_audio_path),
                "-map", "0:v:0", "-map", "0:a:0", "-map", "1:a:0",
                "-c:v", "copy",
                str(out_path),
            ]
        subprocess.run(cmd, capture_output=True, check=True)

        JOBS[job_id] = {
            "status": "done",
            "file_path": out_name,
            "url": f"/files/dubbed/{out_name}",
        }
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/dub-audio")
def dub_audio(req: DubRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(
        _run_dub, job_id, req.file_path, req.translated_text, req.target_lang, req.replace_audio
    )
    return {"job_id": job_id}


@app.get("/dub-audio/status/{job_id}")
def dub_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
