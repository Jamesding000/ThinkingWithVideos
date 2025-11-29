#!/usr/bin/env python
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

YT_BASE_URL = "https://www.youtube.com/watch?v="


def download_video(vid: str, out_dir: Path) -> None:
    """
    Download video-only stream (prefer itag 243) for a YouTube ID,
    saved as <vid>.f243.mp4. No audio.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"{vid}.f243.mp4"

    if target_path.exists():
        print(f"[skip] {target_path.name} already exists.")
        return

    # Prefer itag 243 (360p video-only); fallback to <=480p video-only, then bestvideo
    fmt = "243/bv[height<=480]/bv/bestvideo"

    cmd = [
        "yt-dlp",
        f"{YT_BASE_URL}{vid}",
        "-f", fmt,                          # video-only
        "--merge-output-format", "mp4",     # remux to mp4 container
        "-o", str(out_dir / f"{vid}.f243.%(ext)s"),
        "--ignore-errors",
        "--no-overwrites",
    ]

    print(f"[download] {vid} -> {target_path.name}")
    subprocess.run(cmd, check=False)


def main(data_base_dir: Path, annotation_json_path: Path) -> None:
    TRAIN_JSON = annotation_json_path
    VAL_IDS_JSON = data_base_dir / "val_video_ids.json"
    TEST_IDS_JSON = data_base_dir / "test_video_ids.json"
    OUTPUT_DIR = data_base_dir / "raw_videos"

    all_ids = set()

    # --- training: ids from annotation ---
    with TRAIN_JSON.open() as f:
        train_data = json.load(f)
    for example in train_data:
        vid = example["video"].split(".")[0]
        all_ids.add(vid)

    # --- val / test: ids from separate files ---
    # with VAL_IDS_JSON.open() as f:
    #     val_ids = json.load(f)
    # with TEST_IDS_JSON.open() as f:
    #     test_ids = json.load(f)

    # all_ids.update(val_ids)
    # all_ids.update(test_ids)

    # --- download all unique ids ---
    for vid in tqdm(sorted(all_ids)):
        download_video(vid, OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_base_dir",
        type=Path,
        required=True,
        help="Base directory containing val/test IDs and raw_videos/ folder.",
    )
    parser.add_argument(
        "--annotation_json_path",
        type=Path,
        required=True,
        help="Path to the VidChapters/MTVR annotation JSON (training annotations).",
    )
    args = parser.parse_args()

    main(args.data_base_dir, args.annotation_json_path)

