# python eval_rextime.py \
#   --model-path models/Qwen2.5-VL-7B-Instruct \
#   --video-root /home/yiqunh/ThinkingWithVideos/rextime/rextime_videos \
#   --validation-file processed_data/rextime/rextime_validation.json \
#   --output-dir outputs/rextime_eval/qwen2_5_vl_7b

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

sys.path.append("/home/yiqunh/ThinkingWithVideos/verl")

from verl.utils.dataset.video_vl_utils import (
    cached_process_vision_info,
    process_vision_info,
)

from eval_prompts import THINK_GENERAL

STOP_SEQUENCES = [
    "</answer>",
    " </answer>",
    "</answer>\n",
    " </answer>\n"
]

TIME_SPAN_REGEX = re.compile(
    r"from\s+([0-9]+(?:\.[0-9]+)?)\s*(?:seconds|second|sec|s)?\s+to\s+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
NUMBER_REGEX = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


@dataclass
class SampleResult:
    sample_id: str
    video: str
    question: str
    prediction_text: str
    gt_span: Tuple[float, float]
    pred_span: Tuple[float, float]
    metrics: Dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type = str, default = "models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--validation-file", type = str, default = "processed_data/rextime/rextime_validation.json")
    parser.add_argument("--video-root", type = str, default = "data/rextime")
    parser.add_argument("--output-dir", type = str, default = "outputs/rextime_eval/qwen2_5_vl_7b",)
    parser.add_argument("--no-cache", action = "store_true")
    return parser.parse_args()


def resolve_video_path(video_field, video_root):
    rel = Path(video_field)
    candidates = [
        rel,
        (video_root / rel),
        (REPO_ROOT / rel),
        (video_root / rel.name),
    ]
    for cand in candidates:
        expanded = cand.expanduser()
        if not expanded.is_absolute():
            expanded = (REPO_ROOT / expanded).resolve()
        if expanded.exists():
            return expanded
    raise FileNotFoundError(f"Unable to locate video for '{video_field}'. Tried: {candidates}")


def format_prompt(raw_text, duration, template):
    cleaned = raw_text.strip()
    return template.format(duration = duration, input_text = cleaned)


def build_messages(video_path, prompt, args):
    return [
        {
            "role": "user",
            "content": [
                 {
                    "type": "video",
                    "video": str(video_path),
                    "fps": 2.0,
                    "parallel": True,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def move_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def extract_time_range(text):
    if not text:
        return (-1.0, -1.0)
    answer_block = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    candidate_text = answer_block.group(1) if answer_block else text
    match = TIME_SPAN_REGEX.search(candidate_text)
    if match:
        start, end = float(match.group(1)), float(match.group(2))
        if end >= start:
            return (start, end)
    numbers = [float(x) for x in NUMBER_REGEX.findall(candidate_text)]
    for i in range(len(numbers) - 1):
        start, end = numbers[i], numbers[i + 1]
        if end >= start:
            return (start, end)
    return (-1.0, -1.0)


def parse_ground_truth(solution):
    match = TIME_SPAN_REGEX.search(solution)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    numbers = [float(x) for x in NUMBER_REGEX.findall(solution)]
    if len(numbers) >= 2:
        start, end = numbers[0], numbers[1]
        return (min(start, end), max(start, end))
    raise ValueError(f"Ground-truth solution has no parseable span: {solution}")


def compute_temporal_metrics(gt, pred):
    gt_start, gt_end = gt
    pred_start, pred_end = pred
    metrics = {
        "valid": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "recall@0.3": 0.0,
        "recall@0.5": 0.0,
        "recall@0.7": 0.0,
    }
    if pred_start < 0 or pred_end < 0 or pred_end <= pred_start:
        return metrics
    if gt_end <= gt_start:
        return metrics

    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)

    eps = 1e-4
    inter_length = max(0, inter_end - inter_start)
    pred_length = pred_end - pred_start + eps
    gt_length = gt_end - gt_start + eps
    union_length = pred_length + gt_length - inter_length + eps
    iou = inter_length / max(union_length, eps)
    precision = inter_length / max(pred_length, eps)
    recall = inter_length / max(gt_length, eps)

    metrics.update(
        {
            "valid": 1.0,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "recall@0.3": 1.0 if iou >= 0.3 else 0.0,
            "recall@0.5": 1.0 if iou >= 0.5 else 0.0,
            "recall@0.7": 1.0 if iou >= 0.7 else 0.0,
        }
    )
    return metrics


def summarize(records):
    records = list(records)
    summary = {
        "total_samples": len(records),
        "valid_predictions": sum(1 for r in records if r.metrics["valid"] > 0),
    }
    denom = max(summary["valid_predictions"], 1)
    for key in ["iou", "precision", "recall", "recall@0.3", "recall@0.5", "recall@0.7"]:
        summary[f"mean_{key}"] = float(
            sum(r.metrics[key] for r in records if r.metrics["valid"] > 0) / denom
        )
    summary["parse_failures"] = summary["total_samples"] - summary["valid_predictions"]
    return summary


def main():
    args = parse_args()

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = "bfloat16"
    prompt_template = THINK_GENERAL

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map = "cuda" if device.type == "cuda" else None,
        torch_dtype = torch_dtype if device.type == "cuda" else torch.float32,
        trust_remote_code = True,
        use_sliding_window = True,
        attn_implementation = "flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code = True, use_fast = True)

    samples = json.loads(Path(args.validation_file).read_text())

    video_root = Path(args.video_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents = True, exist_ok = True)
    pred_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    pred_path.parent.mkdir(parents = True, exist_ok = True)

    results = []

    with pred_path.open("w", encoding="utf-8") as writer:
        for idx, sample in enumerate(tqdm(samples, desc = "Evaluating", unit = "sample")):
            try:
                video_path = resolve_video_path(sample["video"], video_root)
            except FileNotFoundError as exc:
                gt_span = parse_ground_truth(sample["solution"])
                pred_span = (-1.0, -1.0)

                result = SampleResult(
                    sample_id = sample["id"],
                    video = sample["video"],
                    question = sample["text"],
                    prediction_text = str(exc),
                    gt_span = gt_span,
                    pred_span = pred_span,
                    metrics = compute_temporal_metrics(gt_span, pred_span),
                )
                results.append(result)
                writer.write(json.dumps(result.__dict__, default = list) + "\n")
                continue

            prompt = format_prompt(sample["text"], sample["duration"], prompt_template)
            messages = build_messages(video_path, prompt, args)
            image_inputs, video_inputs, video_kwargs = cached_process_vision_info(messages, return_video_kwargs = True)
            fps_inputs = video_kwargs.get("fps") if video_kwargs else None

            llm_input = processor(
                text = [processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)],
                images = image_inputs,
                videos = video_inputs,
                fps = fps_inputs,
                padding = True,
                padding_side = "left",
                return_tensors = "pt",
            )
            llm_input = move_to_device(llm_input, device)

            with torch.inference_mode():
                generated = model.generate(
                    **llm_input,
                    max_new_tokens = 512,
                    temperature = 0.0,
                    top_p = 0.9,
                    do_sample = False,
                    stop_strings = STOP_SEQUENCES,
                    tokenizer = processor.tokenizer,
                )
            input_len = llm_input["input_ids"].shape[-1]
            decoded = processor.batch_decode(
                generated[:, input_len:],
                skip_special_tokens = True,
                clean_up_tokenization_spaces = False,
            )[0].strip()

            gt_span = parse_ground_truth(sample["solution"])
            pred_span = extract_time_range(decoded)
            metrics = compute_temporal_metrics(gt_span, pred_span)

            result = SampleResult(
                sample_id = sample["id"],
                video = sample["video"],
                question = sample["text"],
                prediction_text = decoded,
                gt_span = gt_span,
                pred_span = pred_span,
                metrics = metrics,
            )
            results.append(result)
            writer.write(json.dumps(result.__dict__, default=list) + "\n")

    summary = summarize(results)
    metrics_path.write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
