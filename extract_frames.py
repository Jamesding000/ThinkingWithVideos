#!/usr/bin/env python3
import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def run_ffmpeg(input_path: Path, output_pattern: Path,
               fps: float = 2.0, max_frames: int = 14400):
    """
    Call ffmpeg to extract frames at given fps.

    Example command:
      ffmpeg -i input.mp4 -vf fps=2 -vframes 14400 /out_dir/videoid_%06d.jpg
    """
    cmd = [
        "ffmpeg",
        "-y",                    # overwrite without asking
        "-i", str(input_path),
        "-vf", f"fps={fps}",
        "-vframes", str(max_frames),
        str(output_pattern),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _process_one_video(args):
    """
    Worker function for a single video.
    """
    vid_path, raw_root, out_base_dir, fps, max_frames, dataset_name = args

    # Preserve directory structure relative to raw_root
    # e.g. raw_root = .../nextqa/raw_videos
    #      vid_path = .../nextqa/raw_videos/0000/2440175990.mp4
    #      rel = 0000/2440175990.mp4
    rel = vid_path.relative_to(raw_root)
    # 0000/2440175990
    rel_no_ext = rel.with_suffix("")
    # Output frame directory:
    #   .../nextqa/video_14400frames_fps2/0000/2440175990
    frame_dir = out_base_dir / rel_no_ext
    frame_dir.mkdir(parents=True, exist_ok=True)

    stem = rel_no_ext.name  # "2440175990"
    frame_pattern = frame_dir / f"{stem}_%06d.jpg"

    # Skip if frames already exist
    existing_frames = list(frame_dir.glob(f"{stem}_*.jpg"))
    if existing_frames:
        return f"[{dataset_name}] [skip] {rel_no_ext}: {len(existing_frames)} frames already exist"

    try:
        run_ffmpeg(vid_path, frame_pattern, fps=fps, max_frames=max_frames)
        return f"[{dataset_name}] [ok] {rel_no_ext}"
    except subprocess.CalledProcessError as e:
        return f"[{dataset_name}] [error] {rel_no_ext}: {e}"


def extract_for_dataset(input_dir: str,
                        output_dir: str,
                        dataset_name: str,
                        fps: float = 2.0,
                        max_frames: int = 14400,
                        num_workers: int | None = None):
    """
    input_dir: directory that contains raw videos for one dataset
               e.g. /data/.../nextqa/raw_videos
    output_dir: directory to write frame folders
               e.g. /data/.../nextqa/video_14400frames_fps2
    dataset_name: 'charades' or 'nextqa'
    """
    raw_video_dir = Path(input_dir).resolve()
    out_base_dir = Path(output_dir).resolve()

    if not raw_video_dir.exists():
        raise FileNotFoundError(f"Raw video directory not found: {raw_video_dir}")

    out_base_dir.mkdir(parents=True, exist_ok=True)

    # search recursively for video files
    video_files = [
        p for p in raw_video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not video_files:
        print(f"[{dataset_name}] No video files found in {raw_video_dir}")
        return

    print(f"[{dataset_name}] Found {len(video_files)} videos in {raw_video_dir}")

    # Decide how many workers to use
    if num_workers is None or num_workers <= 0:
        num_workers = max(1, cpu_count() // 2)  # a safe default

    print(f"[{dataset_name}] Using {num_workers} parallel workers")

    # pass raw_root into the worker so it can compute relative paths
    jobs = [
        (vid_path, raw_video_dir, out_base_dir, fps, max_frames, dataset_name)
        for vid_path in video_files
    ]

    # Parallel processing with tqdm progress bar
    with Pool(processes=num_workers) as pool:
        for msg in tqdm(
            pool.imap_unordered(_process_one_video, jobs),
            total=len(jobs),
            desc=f"Extracting frames ({dataset_name})"
        ):
            if "error" in msg:
                print(msg)


def main():
    parser = argparse.ArgumentParser(description="Extract 2 FPS frames for a given dataset.")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Input directory containing raw videos for a dataset",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory to save extracted frames",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Name of the dataset to process (for logging only)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to extract (default: 2.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=14400,
        help="Maximum frames per video (default: 14400 ≈ 2 fps * 2 hours)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count()//2)",
    )
    args = parser.parse_args()

    extract_for_dataset(
        args.input_dir,
        args.output_dir,
        args.dataset,
        fps=args.fps,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
