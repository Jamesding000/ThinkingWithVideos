import json
import os
from pathlib import Path
from download_and_extract_frames import extract_youtube_id

def get_entries(raw_data):
    entries = []
    index = 0
    for video_id in raw_data.keys():
        data = raw_data[video_id]
        duration = data["duration"]
        timestamps = data["timestamps"]
        sentences = data["sentences"]

        for i in range(len(timestamps)):
            start, end = timestamps[i]
            
            text = "Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.30 - 30.42 seconds. Now I will give you the textual sentence: "
            text += sentences[i] 
            text += "Please return its start time and end time.\n"

            solution = f"<think></think><answer>The event happens in the {start} - {end}</answer>"
            
            entries.append({
                "id": f"{index:06d}",
                "video": extract_youtube_id(video_id) + ".mp4",
                "text": text,
                "solution": solution,
                "duration": duration 
            })
            index += 1

    return entries


def get_video_ids_actnet(input_file, out_dir, limit=None):
    val_data = Path(input_file)
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
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    out_file = os.path.join(out_dir, "actnet_video_ids.txt")
    with open(out_file, "w", encoding = "utf-8") as f:
        f.write("\n".join(seen))


def main():
    input_files = ["/data/user_data/jamesdin/data/actnet/val_1.json", "/data/user_data/jamesdin/data/actnet/val_2.json"]
    output_dir = "/data/user_data/jamesdin/data/actnet/"

    for file in input_files:
        input_path = Path(file)
        output_file = output_dir + "actnet_" + file.split("/")[-1]
        output_path = Path(output_file)

        with input_path.open("r", encoding = "utf-8") as src:
            raw_data = json.load(src)

        entries = get_entries(raw_data)

        with output_path.open("w", encoding = "utf-8") as destination:
            json.dump(entries, destination, indent = 4, ensure_ascii = False)

        print(f"Wrote {len(entries)} entries to {output_path}")
    
        # The videos in both files are the same
        get_video_ids_actnet(output_path, output_dir, limit=None)

if __name__ == "__main__":
    main()