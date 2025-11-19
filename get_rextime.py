import argparse
import json
from pathlib import Path

from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", 
        type = str, 
        choices = ["validation", "test"], 
        default = "validation"
    )
    return parser.parse_args()

def build_solution(data, is_validation):
    s0, e0 = data["span"]
    text = data["answer"]
    ans = data["ans"]

    text = text.replace(f"<s0>", f"{s0:.2f}")
    text = text.replace(f"<e0>", f"{e0:.2f}")
    if is_validation:
        index = ord(ans) - ord('A')
        options = data["options"]
        option = options[index]
        text = text.replace(f"<option>", option)
    return text


def convert_data(data, is_validation):
    return {
        "id": data["qid"],
        "video": "actnet_videos/" + data["vid"] + ".mp4",
        "text": "\nBased on the content of the video, answer the following question: " 
            + data["question"]
            + "\nIn the <answer> </answer> tag, first specify the exact time period in seconds of the video segment that support your answer, then, provide your final answer with a short sentence.\nFormat your response as follows:\n<think>...</think>\n<answer>From [start_time] to [end_time], [your answer]</answer>\n",
        "solution": build_solution(data, is_validation),
        "duration": float(data["duration"]),
        "data_source": "rextime"
    }

def main():
    args = parse_args()
    split = args.split
    if split == "validation":
        is_validation = True
    else:
        is_validation = False

    dataset = load_dataset("ReXTime/ReXTime")

    records = []
    for data in dataset[split]:
        formatted_data = convert_data(data, is_validation)
        records.append(formatted_data)

    output_dir = "processed_data/rextime/"

    records.sort(key = lambda item: item["id"])
    output = Path(output_dir + "rextime_" + args.split + ".json")
    output.parent.mkdir(parents = True, exist_ok = True)
    output.write_text(
        json.dumps(records, indent = 4, ensure_ascii = False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()