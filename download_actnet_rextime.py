import os
from datasets import load_dataset
import argparse, subprocess
from pathlib import Path
from tqdm import tqdm
import json

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def get_video_ids_rextime(limit):
    split = "validation"   # or "test"
    ds = load_dataset("ReXTime/ReXTime", split = split)
    ids = sorted({row["vid"] for row in ds})[:limit]

    out_dir = "/home/yiqunh/ThinkingWithVideos/rextime"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    out_file = os.path.join(out_dir, "rextime_video_ids.txt")
    with open(out_file, "w", encoding = "utf-8") as f:
        f.write("\n".join(ids))



def get_video_ids_actnet(limit):
    val_data = Path("processed_data/actnet/actnet_val_1.json")
    with val_data.open() as f:
        entries = json.load(f)
    
    seen = []
    for entry in entries:
        vid = entry["video"]
        yt_id = vid.replace(".mp4", "")
        if yt_id not in seen:
            seen.append(yt_id)
        if limit and len(seen) >= limit:
            break
    
    out_dir = "/home/yiqunh/ThinkingWithVideos/actnet"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    out_file = os.path.join(out_dir, "actnet_video_ids.txt")
    with open(out_file, "w", encoding = "utf-8") as f:
        f.write("\n".join(seen))



def download_one_video(vid, dst_dir, args):
    dst_dir.mkdir(parents = True, exist_ok = True)
    mp4 = dst_dir / f"{vid}.mp4"
    if mp4.exists():
        print(f"{mp4.name} already exists")
        return

    if args.type == "actnet":
        url = f"https://www.youtube.com/watch?v={vid.split('v_')[-1]}"
    else:
        url = f"https://www.youtube.com/watch?v={vid}"

    cmd = [
        "yt-dlp",
        url,
        "-f", "243/bv[height<=480]/bv/bestvideo",
        "-S", "res,ext",
        "-o", str(dst_dir / f"{vid}.%(ext)s"),
        "--merge-output-format", "mp4",
        "--no-overwrites",
        "--ignore-errors",
    ]
    subprocess.run(cmd, check = False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type = str, choices = ["rextime", "actnet"], required = True)
    parser.add_argument("--limit", type = int, default = 5)
    return parser.parse_args()



def main(ids_file, video_root, args):
    with ids_file.open() as f:
        ids = [line.strip() for line in f if line.strip()]
    dst = Path(video_root)
    for vid in tqdm(ids):
        download_one_video(vid, dst, args)


if __name__ == "__main__":
    args = parse_args()
    limit = args.limit
    if args.type == "rextime":
        get_video_ids_rextime(limit)
        ids_file = Path("/home/yiqunh/ThinkingWithVideos/rextime/rextime_video_ids.txt")
        video_root = Path("/home/yiqunh/ThinkingWithVideos/rextime/rextime_videos")
    else:
        get_video_ids_actnet(limit)
        ids_file = Path("/home/yiqunh/ThinkingWithVideos/actnet/actnet_video_ids.txt")
        video_root = Path("/home/yiqunh/ThinkingWithVideos/actnet/actnet_videos")
    
    main(ids_file, video_root, args)
