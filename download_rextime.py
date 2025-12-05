import os
from datasets import load_dataset
import argparse, subprocess
from pathlib import Path
from tqdm import tqdm

def get_video_ids():
    split = "validation"   # or "test"

    ds = load_dataset("ReXTime/ReXTime", split = split)
    ids = sorted({row["vid"] for row in ds})[:5]

    out_dir = "/home/yiqunh/ThinkingWithVideos/rextime"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    out_file = os.path.join(out_dir, "rextime_video_ids.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(ids))

def download_one(vid: str, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    mp4 = dst_dir / f"{vid}.mp4"
    if mp4.exists():
        print(f"[skip] {mp4.name} already exists")
        return
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={vid}",
        "-f", "243/bv[height<=480]/bv/bestvideo",
        "-S", "res,ext",
        "-o", str(dst_dir / f"{vid}.%(ext)s"),
        "--merge-output-format", "mp4",
        "--no-overwrites",
        "--ignore-errors",
    ]
    subprocess.run(cmd, check=False)

def main(ids_file, video_root):
    with ids_file.open() as fh:
        ids = [line.strip() for line in fh if line.strip()]
    dst = Path(video_root)
    for vid in tqdm(ids):
        download_one(vid, dst)

if __name__ == "__main__":
    get_video_ids()
    main(
        Path("/home/yiqunh/ThinkingWithVideos/rextime/rextime_video_ids.txt"),
        Path("/home/yiqunh/ThinkingWithVideos/rextime/rextime_videos")
    )
