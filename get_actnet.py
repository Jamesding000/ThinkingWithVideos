import json
from pathlib import Path

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
                "video": video_id + ".mp4",
                "text": text,
                "solution": solution,
                "duration": duration 
            })
            index += 1

    return entries

def main():
    input_files = ["actnet_data/val_1.json", "actnet_data/val_2.json"]
    output_dir = "processed_data/actnet/"

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

if __name__ == "__main__":
    main()