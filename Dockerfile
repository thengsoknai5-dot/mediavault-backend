FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# yt-dlp is NOT version-pinned in requirements.txt on purpose: YouTube changes
# how it serves video pages very frequently (sometimes daily), and each yt-dlp
# release patches whatever broke. Without this explicit upgrade step, Docker's
# layer cache would keep reusing whatever yt-dlp version was installed on the
# very first build forever, since requirements.txt itself never changes. This
# step runs after COPY . . so it re-executes on every deploy regardless of
# cache, always fetching the newest yt-dlp from PyPI.
RUN pip install --no-cache-dir --upgrade yt-dlp

CMD ["python", "main.py"]
