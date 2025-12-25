#!/usr/bin/env python3
"""
Single-threaded YouTube → frames pipeline.

For each example in the annotation JSON:
  - Extract video_id from example["video"]
  - If frames for this video_id already exist, skip
  - Download the video via yt-dlp into a temp directory
  - Use ffmpeg to extract resized JPG frames at specified fps
  - Delete the downloaded video file

Result:
  <output_dir>/video_{max_frames}frames_fps{fps}/<video_id>/<video_id>_%06d.jpg
"""

import argparse
import json
import time
import subprocess
from pathlib import Path
from typing import Set
from tqdm import tqdm
import random 
import os
import re

YT_BASE_URL = "https://www.youtube.com/watch?v="
VIDEO_ID_FIELD = "video"
DURATION_FIELD = "duration"
DURATION_THRESHOLD = 1200  # filter out videos longer than 20 minutes
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
COOKIES_PATH = "cookies.txt"
FMT_STRING = "bv[height<=480]/b[height<=480]/bv/b"
MAX_SUSPICIOUS_ERRORS = 20
CONSEC_SUSPICIOUS_ERRORS = 0

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    """Run a shell command, print its output on failure, and return (code, stdout)."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print("---- ERROR OUTPUT START ----")
        print(result.stdout)
        print("---- ERROR OUTPUT END ----")
    return result.returncode, result.stdout


def download_video(video_id: str, tmp_video_dir: Path) -> Path | None:
    """
    Download a YouTube video (video-only) to tmp_video_dir as <video_id>.mp4.

    Behavior:
      - Single attempt (no retries)
      - If failure looks "ban-like" (UNPLAYABLE / challenge / bot check), wait 10–60s
        with random jitter, increasing as errors stack.
      - Track consecutive suspicious errors; abort after MAX_SUSPICIOUS_ERRORS.
    """
    global CONSEC_SUSPICIOUS_ERRORS

    tmp_video_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_video_dir / f"{video_id}.mp4"

    if out_path.exists():
        # Success => reset ban-like error streak
        CONSEC_SUSPICIOUS_ERRORS = 0
        return out_path

    fmt = FMT_STRING

    cmd = [
        "yt-dlp",
        f"{YT_BASE_URL}{video_id}",
        "-f",
        fmt,  # video-only
        "--merge-output-format",
        "mp4",  # ensure mp4 container
        "-o",
        str(out_path),
        "--no-overwrites",
        "--cookies",
        COOKIES_PATH,
        "--user-agent",
        USER_AGENT,
        "--sleep-interval",
        "15",
        "--max-sleep-interval",
        "30",
        "--concurrent-fragments",
        "4",  # download 4 fragments concurrently for faster download
    ]

    exit_code, output = run_cmd(cmd)
    # Normalize error text
    error_text = (output or "").lower()

    if exit_code == 0 and out_path.exists():
        # success: reset suspicious error streak
        CONSEC_SUSPICIOUS_ERRORS = 0
        return out_path

    # -----------------------------
    # Failure: decide how to back off
    # -----------------------------
    print(f"[download][fail] {video_id} (exit_code={exit_code})")

    # Heuristics for "ban-like" / challenge errors
    suspicious = any(
        key in error_text
        for key in [
            "unplayable",                        # playability status: UNPLAYABLE
            "this content isn’t available",      # common for soft blocks / bans
            "this content isn't available",
            "confirm you’re not a bot",
            "confirm you're not a bot",
            "sign in to confirm",
            "n challenge solving failed",        # EJS / challenge issues
            "challenge required",
            "player response playability status",
        ]
    )

    if suspicious:
        CONSEC_SUSPICIOUS_ERRORS += 1

        # choose a backoff window depending on how many in a row
        if CONSEC_SUSPICIOUS_ERRORS == 1:
            low, high = 10.0, 15.0   # first suspicious error
        elif CONSEC_SUSPICIOUS_ERRORS <= 3:
            low, high = 20.0, 30.0   # 2nd–3rd suspicious
        else:
            low, high = 40.0, 60.0   # 4th+ suspicious in a row

        sleep_for = random.uniform(low, high)
        print(
            f"[backoff] suspicious error #{CONSEC_SUSPICIOUS_ERRORS} "
            f"for {video_id} → sleeping {sleep_for:.1f}s"
        )
        time.sleep(sleep_for)

        if CONSEC_SUSPICIOUS_ERRORS >= MAX_SUSPICIOUS_ERRORS:
            print(
                f"[fatal] Hit {CONSEC_SUSPICIOUS_ERRORS} consecutive suspicious errors "
                f"(>= {MAX_SUSPICIOUS_ERRORS}). Aborting to avoid ban."
            )
            raise SystemExit(1)
    else:
        # normal failure: break suspicious streak and sleep briefly
        if CONSEC_SUSPICIOUS_ERRORS > 0:
            print("[info] Non-suspicious failure; resetting suspicious error streak.")
        CONSEC_SUSPICIOUS_ERRORS = 0
        sleep_for = random.uniform(0.5, 2.0)
        print(f"[backoff] normal failure for {video_id} → sleeping {sleep_for:.1f}s")
        time.sleep(sleep_for)

    return None


def extract_and_resize_frames(
    video_path: Path,
    output_dir: Path,
    video_id: str,
    fps: int = 2,
    max_frames: int = 14400,
    size: int = 224,
) -> Path | None:
    """
    Use ffmpeg to extract frames from video_path directly as
    resized JPGs at the given fps.
    
    Output directory:
        output_dir / "video_{max_frames}frames_fps{fps}" / video_id
    Files:
        <video_id>_000001.jpg, <video_id>_000002.jpg, ...
    """
    # Root directory for all videos' frames (dynamic naming)
    dataset_frames_root = output_dir / f"video_{max_frames}frames_fps{fps}"
    dataset_frames_root.mkdir(parents=True, exist_ok=True)
    
    # Per-video directory
    video_frames_dir = dataset_frames_root / video_id
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # If frames already exist, skip extraction
    existing = list(video_frames_dir.glob(f"{video_id}_*.jpg"))
    if existing:
        print(f"[frames][skip] {video_id}: {len(existing)} frames already exist")
        return video_frames_dir
    
    frame_pattern = video_frames_dir / f"{video_id}_%06d.jpg"
    
    # fps + resize in one go for efficiency
    vf_filter = f"fps={fps},scale={size}:{size}:flags=fast_bilinear"
    
    cmd = [
        "ffmpeg",
        "-threads",
        "0",  # use all available CPU cores
        "-i",
        str(video_path),
        "-vf",
        vf_filter,
        "-vframes",
        str(max_frames),
        "-q:v",
        "5",  # JPEG quality (2=best, 31=worst; 5 is good balance)
        str(frame_pattern),
    ]
    
    exit_code, _ = run_cmd(cmd)
    if exit_code != 0:
        print(f"[frames][fail] {video_id} (exit_code={exit_code})")
        # Clean up partially generated frames
        for f in video_frames_dir.glob(f"{video_id}_*.jpg"):
            f.unlink(missing_ok=True)
        return None
    
    return video_frames_dir


def delete_file(path: Path) -> None:
    """Delete a file if it exists."""
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[cleanup][warn] Failed to delete {path}: {e}")


def extract_video_id(raw_video_field: str) -> str:
    """
    Extract YouTube ID from the 'video' field.
    
    Examples:
        "abcd1234.mp4"      -> "abcd1234"
        "abcd1234"          -> "abcd1234"
        "abcd1234.webm"     -> "abcd1234"
    """
    return raw_video_field.split("/")[-1].split(".")[0]


def extract_youtube_id(vid: str) -> str:
    """
    Extract 11-character YouTube ID from a text line.
    Handles formats like:
        "v_r-iXUXMP4DY" -> "r-iXUXMP4DY"
        "wUgPzvcKK5c_210.0_360.0" -> "wUgPzvcKK5c"
        "-gNwItPwMhM_210.0_360.0" -> "-gNwItPwMhM"
    
    YouTube IDs are exactly 11 characters.
    """
    # Get bare filename without path or extension
    base = os.path.splitext(os.path.basename(vid))[0]
    
    if len(base) == 11:
        return base

    # 1) Strip trailing time segments like:
    #    _210.0_360.0, _60.0_210.0, _360_510, or even just _210.0
    #    This pattern matches one or two "_<number or float>" at the end.
    base = re.sub(r'(_\d+(?:\.\d+)?){1,2}$', '', base)

    # 2) Strip leading "v_" prefix (common in your filenames)
    base = re.sub(r'^v_', '', base)

    # 3) Find the first 11-char YouTube-like token (letters, digits, _ or -)
    m = re.search(r'[A-Za-z0-9_-]{11}', base)
    if m:
        return m.group(0)

    # 4) Fallback: try searching the original string, just in case
    m = re.search(r'[A-Za-z0-9_-]{11}', vid)
    if m:
        return m.group(0)

    # 5) Last-resort fallback + debug
    print(f"Warning: could not find 11-char YouTube id in {vid}")
    return base


def process_dataset_single_thread(
    annotation_json_path: Path,
    output_dir: Path,
    tmp_video_dir: Path,
    fps: int = 2,
    max_frames: int = 14400,
    size: int = 224,
) -> None:
    """
    Single-threaded pipeline:
    For each example:
      - Resolve video_id
      - If frames already exist, skip
      - Else download -> extract frames -> delete video file
    
    Args:
        annotation_json_path: Path to annotation JSON or text file with video IDs
        output_dir: Root directory for frame output
        tmp_video_dir: Temporary directory for video downloads
        fps: Frames per second for extraction
        max_frames: Maximum frames per video
        size: Output frame size
    """
    print(f"[load] Reading from {annotation_json_path}")
    
    # Detect file type and load accordingly
    if annotation_json_path.suffix == '.txt':
        # Text file: one video ID per line (extract 11-char YouTube ID)
        with annotation_json_path.open("r") as f:
            examples = [{"video": line.strip()} for line in f if line.strip()]
    else:
        # JSON file: existing format
        with annotation_json_path.open("r") as f:
            examples = json.load(f)
    
    print(f"[info] Total examples in annotation: {len(examples)}")
    print(f"[info] Output directory: {output_dir / f'video_{max_frames}frames_fps{fps}'}")
    
    if Path(COOKIES_PATH).exists():
        print(f"[info] Using cookies from: {COOKIES_PATH}")
    else:
        raise FileNotFoundError(f"[error] Cookies file not found: {COOKIES_PATH}")

    seen_ids: Set[str] = set()
    stats = {
        'checked': 0,
        'skipped_exists': 0,
        'skipped_duration': 0,
        'downloaded': 0,
        'errored': 0,
    }
    
    # Process each example one by one
    for ex in tqdm(examples, desc="Processing videos"):
        if isinstance(ex, str):
            raw_video_field = ex
        else:
            raw_video_field = ex[VIDEO_ID_FIELD]    
        video_id = extract_video_id(raw_video_field)
        video_id = extract_youtube_id(video_id)

        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        stats['checked'] += 1

        if DURATION_FIELD in ex and ex[DURATION_FIELD] > DURATION_THRESHOLD:
            stats['skipped_duration'] += 1
            continue

        frames_dir = output_dir / f"video_{max_frames}frames_fps{fps}" / video_id
        if frames_dir.exists() and any(frames_dir.glob(f"{video_id}_*.jpg")):
            stats['skipped_exists'] += 1
            continue

        # 1) Download video
        t0 = time.perf_counter()
        video_path = download_video(video_id, tmp_video_dir)
        t1 = time.perf_counter()
        if video_path is None:
            print(f"[timing] {video_id}: download failed after {t1 - t0:.2f}s")
            stats['errored'] += 1
            continue

        # 2) Extract + resize frames
        out_dir = extract_and_resize_frames(
            video_path=video_path,
            output_dir=output_dir,
            video_id=video_id,
            fps=fps,
            max_frames=max_frames,
            size=size,
        )
        t2 = time.perf_counter()

        # 3) Remove intermediate video file
        delete_file(video_path)

        print(
            f"[timing] {video_id}: "
            f"download={t1 - t0:.2f}s, ffmpeg={t2 - t1:.2f}s, total={t2 - t0:.2f}s"
        )

        if out_dir is not None:
            print(f"[ok] {video_id} -> {out_dir}")
            stats['downloaded'] += 1
        else:
            print(f"[fail] {video_id} (frame extraction failed)")
            stats['errored'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Videos checked:          {stats['checked']}")
    print(f"  ├─ Already exist:      {stats['skipped_exists']}")
    print(f"  ├─ Skipped (duration): {stats['skipped_duration']}")
    print(f"  ├─ Newly downloaded:   {stats['downloaded']}")
    print(f"  └─ Errored:            {stats['errored']}")
    print(f"{'='*60}")



def main():
    parser = argparse.ArgumentParser(
        description="Single-thread pipeline: yt-dlp download → ffmpeg frames → delete video."
    )
    parser.add_argument(
        "--annotation-json-path",
        type=Path,
        required=True,
        help="Path to the training annotation JSON (with a 'video' field per example).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for frame output (video_{max_frames}frames_fps{fps} will be created inside).",
    )
    parser.add_argument(
        "--tmp-video-dir",
        type=Path,
        required=True,
        help="Temporary directory to store downloaded mp4 videos before frame extraction.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for ffmpeg extraction.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=14400,
        help="Maximum frames per video.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output frame size (size x size).",
    )
    
    args = parser.parse_args()
    
    process_dataset_single_thread(
        annotation_json_path=args.annotation_json_path,
        output_dir=args.output_dir,
        tmp_video_dir=args.tmp_video_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        size=args.size,
    )


if __name__ == "__main__":
    main()

