# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

from download_and_extract_frames import extract_youtube_id

from transformers.video_utils import VideoMetadata

# from qwen_vl_utils import process_vision_info
from verl.utils.dataset.video_vl_utils import process_vision_info, cached_process_vision_info
from verl.utils.dataset.vision_utils import process_video, cached_process_video

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, GenerationConfig, AutoConfig, AutoModelForVision2Seq
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from peft import AutoPeftModelForCausalLM
from collections import defaultdict

import sys
import re
import vllm

# TEMPLATE = os.getenv("MY_PROMPT_TEMPLATE", default="Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.30 - 30.42 seconds. Now I will give you the textual sentence: {input_text}. Please return its start time and end time.")
VQA_TEMPLATE = """You are a careful visual question answering assistant.

You will see a video and a question about it. Your job is to:
1. Think step by step about the visual content and the question.
2. Then give a single **integer** answer in digits (e.g., 0, 1, 2), not words (no "two", "three", etc.).
3. Wrap **only** that integer inside <answer> and </answer> tags.
4. Your final output must end with exactly one line in this format, and nothing else:

<answer>INTEGER</answer>

Example:
If the correct answer is 3, you must end with:
<answer>3</answer>

Now use the video and question below.

Video duration: {duration} seconds.
Question: {input_text}

First, think carefully about the video. Then, at the end, output ONLY the final answer in the required format:

<answer>"""

TEMPLATE = os.getenv("MY_PROMPT_TEMPLATE", default="Please find the visual event described by a sentence in the video, determining its starting and ending times. The format should be: 'The event happens in the start time - end time'. For example, The event 'person turn a light on' happens in the 24.30 - 30.42 seconds. Now I will give you the textual sentence: {input_text}. Please return its start time and end time.")

VIDEO_INFO_CACHE = {}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    # chunk_size = math.ceil(len(lst) / n)  # integer division
    # return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    res = [[] for i in range(n)]
    for i, x in enumerate(lst):
        res[i % n].append(x)
    return res

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def ranki_print(s):
    print(f'[cuda:{args.chunk_idx}] {s}', flush=True)

class VideoQADataset(Dataset):
    def __init__(self, args, processor):
        with open(args.gt_file) as file:
            gt_questions = json.load(file)
        gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
        if os.path.exists(args.answers_file):
            with open(args.answers_file, "r") as f:
                id_set = [json.loads(row)['id'] for row in f.readlines()]
                id_set = set(id_set)
                gt_questions = [sample for sample in gt_questions if sample['id'] not in id_set]
        # TODO: remove this after debugging
        self.data = gt_questions
        # Use prompt template from args if provided, otherwise use default
        self.template = getattr(args, 'prompt_template', TEMPLATE) if hasattr(args, 'prompt_template') and args.prompt_template else TEMPLATE
        content_video = {
            "type": "video",
            "video": None,
        }
        content_image = {
            "type": "image",
            "image": None,
        }
        if args.max_frames is not None:
            content_video['max_frames'] = args.max_frames
        if args.max_pixels is not None:
            content_video['max_pixels'] = args.max_pixels
            content_image["max_pixels"] = 448*448
        if args.resized_height is not None:
            content_video['resized_height'] = args.resized_height
            content_image['resized_height'] = args.resized_height
        if args.resized_width is not None:
            content_video['resized_width'] = args.resized_width
            content_image['resized_width'] = args.resized_width
        self.content_video = content_video
        self.content_image = content_image
        self.video_dir = args.video_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while idx < len(self.data):
            try:
                return self.getitem(idx)
            except Exception as e:
                video_id = self.data[idx]['video_id'] if 'video_id' in self.data[idx] else self.data[idx]['video']
                print(f"[ERROR] idx={idx}, video={video_id}, error: {e}")
                idx += 1
        raise RuntimeError("All samples from current idx onward failed.")

    def getitem(self, idx):
        sample = self.data[idx]

        # 1. Normalize question / answer fields
        if "question" not in sample:
            sample["question"] = sample["text"]
        if "answer" not in sample:
            sample["answer"] = str(sample["solution"])

        # 2. Build multimodal spec from sample
        if "video" in sample or "video_id" in sample:
            video_name = sample["video_id"] if "video_id" in sample else sample["video"]
            video_path, fps = None, 2.0

            # Try pre-extracted frames first
            video_frame_path = os.path.join(self.video_dir, video_name.split(".")[0])
            
            if not os.path.exists(video_frame_path):
                video_frame_path = os.path.join(self.video_dir, extract_youtube_id(video_name.split(".")[0]))
                
            if os.path.exists(video_frame_path):
                frame_paths = os.listdir(video_frame_path)
                frame_paths = sorted(
                    frame_paths,
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                frame_paths = [os.path.join(video_frame_path, fp) for fp in frame_paths]
                total_frames = len(frame_paths)

                if args.max_frames is not None and total_frames > args.max_frames:
                    idxes = (
                        torch.linspace(0, total_frames - 1, args.max_frames)
                        .round()
                        .long()
                        .tolist()
                    )
                    if "videommmu" in self.video_dir:
                        idxes.append(total_frames - 1)
                    frame_paths = [frame_paths[i] for i in idxes]
                video_path = frame_paths

                # approximate fps from #frames and annotated duration
                fps = len(frame_paths) / float(sample["duration"])
            else:
                # Fallback to raw video file if no frame dir exists
                for ext in [".mp4", ".webm", ".mkv", ".avi"]:
                    path = os.path.join(self.video_dir, video_name.split(".")[0] + ext)
                    if os.path.exists(path):
                        video_path = path
                        break

            if video_path is None:
                raise FileNotFoundError(f"Video file for {video_name} not found.")

            question = self.template.format(
                input_text=sample["question"], duration=sample["duration"]
            )

            # Spec that verl.process_video() understands
            content_mm = self.content_video.copy()
            content_mm["video"] = video_path
            content_mm["fps"] = fps
            content_mm["draw_number"] = True
            content_mm["parallel"] = True

        elif "image" in sample:
            img_template = self.template.replace(
                "Video duration: {duration} seconds.", "This is an image."
            ).replace(
                "video and a question", "image and a question"
            )
            question = img_template.format(input_text=sample["question"], duration=0)
            image_path = os.path.join(self.video_dir, sample["image"])

            content_mm = self.content_image.copy()
            content_mm["image"] = image_path

        else:
            raise ValueError(
                f"Not a multimodal sample! Neither 'video' nor 'image' key found in sample: {sample}"
            )

        # 3. Messages -> raw prompt
        messages = [
            {
                "role": "user",
                "content": [
                    content_mm,
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print('input_text', text)

        # 4. Build multi_modal_data & mm_processor_kwargs
        multi_modal_data = {}
        mm_processor_kwargs = {}

        # Images (rare in this script, but keep for completeness)
        if content_mm.get("type") == "image":
            # vLLM will decode paths itself; no extra processing needed
            multi_modal_data["image"] = [content_mm["image"]]
            mm_processor_kwargs["fps"] = []

        # Videos: pre-extracted frames + metadata (MATCH RL FORMAT)
        if content_mm.get("type") == "video":
            # Use verl's process_video so we load frames (no re-decoding .mp4)
            video_spec = content_mm.copy()
            frames, fps_eff = process_video(video_spec)  # list[PIL.Image], float

            n_frames = len(frames)
            if n_frames == 0:
                raise RuntimeError(f"No frames loaded for {video_name}")

            # This metadata structure matches the working RL pipeline:
            #   ([PIL frames...], {'total_num_frames': ..., 'fps': ..., 'duration': ..., 'frames_indices': ..., 'do_sample_frames': False})
            metadata = {
                "total_num_frames": n_frames,
                "fps": float(fps_eff),
                "duration": float(n_frames) / float(fps_eff) if fps_eff > 0 else float(
                    sample.get("duration", n_frames / 2.0)
                ),
                "frames_indices": np.arange(n_frames),
                "do_sample_frames": False,  # crucial: Qwen3 wrapper uses this
            }

            # NOTE: each video item is a TUPLE (frames, metadata_dict)
            multi_modal_data["video"] = [(frames, metadata)]
            mm_processor_kwargs["fps"] = [float(fps_eff)]

        # 5. Pack llm_inputs for vLLM
        llm_inputs = {
            "prompt": text,
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": mm_processor_kwargs,
        }
        sample["llm_inputs"] = llm_inputs

        return sample


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auto_detect_world_size(model_path):
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")
    world_sizes = set()
    ranks = set()
    for filename in os.listdir(model_path):
        m = pattern.match(filename)
        if m:
            ws = int(m.group(1))
            rank = int(m.group(2))
            world_sizes.add(ws)
            ranks.add(rank)
    if not world_sizes:
        raise RuntimeError("未找到模型分片文件")
    # 如果所有文件的world_size一致，直接用
    if len(world_sizes) == 1:
        return list(world_sizes)[0]
    # 否则用rank数量猜测
    return max(ranks) + 1


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    use_flash_attn = True
    qwen_path = "/data/user_data/jamesdin/models/Qwen3-VL-2B-Thinking"
    try:
        if args.backend == 'vllm':
            llm = vllm.LLM(
                model=args.model_path,
                dtype='bfloat16',
                seed=args.seed,
                max_num_batched_tokens=8192,
                max_model_len=65536,  # reduce the max model context length to 65536 to fit in memory
                gpu_memory_utilization=0.7,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model_path, 
                use_sliding_window=True,
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if use_flash_attn else "eager",
            ).eval()
    except Exception as e:
        ranki_print(f'Load model Except: {e}')
        model_path = args.model_path
        state_dict = defaultdict(list)
        world_size = auto_detect_world_size(model_path)
        for rank in range(world_size):
            filepath = f"{model_path}/model_world_size_{world_size}_rank_{rank}.pt"
            print('loading', filepath)
            this_state_dict = torch.load(filepath)
            for key, value in this_state_dict.items():
                state_dict[key].append(value.to_local())
        for key in state_dict:
            state_dict[key] = torch.cat(state_dict[key], dim=0)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path, 
            use_sliding_window=True,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        ).eval()
        model.load_state_dict(state_dict)
        model.eval()

    processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True, use_fast=True)
    ranki_print("Load model and processor success!")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            ranki_print(f'mkdir Except: {e}')
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    args.answers_file = answers_file

    dataset = VideoQADataset(args, processor)
    batch_size = 8 // args.repeat_times
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True, 
        drop_last=False,
        collate_fn=lambda batch: batch, 
    )
    print('#' * 50)
    print(f"dataloader length: {len(dataloader)}")
    ans_file = open(answers_file, "a")

    target_template = ["</{tag}>", " </{tag}>", "</{tag}>\n", " </{tag}>\n", "</{tag}>\n\n", " </{tag}>\n\n"] 
    target_tags = ["tool_call", "answer"]
    target_sequences = [template.format(tag=tag) for tag in target_tags for template in target_template]
    if args.repeat_times > 1:
        sampling_params = vllm.SamplingParams(
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop=target_sequences,
            include_stop_str_in_output=True,  # Important
            n=args.repeat_times,
        )
    else:
        sampling_params = vllm.SamplingParams(
            temperature=0.01,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop=target_sequences,
            include_stop_str_in_output=True,  # Important
        )

    for samples in tqdm(dataloader, desc=f"cuda:{args.chunk_idx}"):
        if args.backend == 'vllm':
            batch_llm_inputs = [sample['llm_inputs'] for sample in samples]
            
            # Normalize video structure: [frames, metadata] -> (frames, metadata)
            for req in batch_llm_inputs:
                if "video" in req['multi_modal_data']:
                    mm_data = req['multi_modal_data']
                    mm_data['video'] = [(v[0], v[1]) for v in mm_data['video']]
                    req['multi_modal_data'] = mm_data
                    
            # print("batch_llm_inputs[0].keys()", batch_llm_inputs[0].keys())
            # print("batch_llm_inputs[0]", batch_llm_inputs[0])

            outputs = llm.generate(batch_llm_inputs, sampling_params=sampling_params)

        else:
            text = [sample["llm_inputs"]["prompt"] for sample in samples]

            image_inputs = [
                image
                for sample in samples
                if "image" in sample["llm_inputs"]["multi_modal_data"]
                for image in sample["llm_inputs"]["multi_modal_data"]["image"]
            ]

            # unwrap (video, metadata) -> video
            video_inputs = [
                video_meta[0]
                for sample in samples
                if "video" in sample["llm_inputs"]["multi_modal_data"]
                for video_meta in sample["llm_inputs"]["multi_modal_data"]["video"]
            ]

            fps_inputs = [
                fps
                for sample in samples
                if "video" in sample["llm_inputs"]["multi_modal_data"]
                for fps in sample["llm_inputs"]["mm_processor_kwargs"]["fps"]
            ]
            ranki_print(f'Inference: fps_inputs={fps_inputs}')
            # Collect videos_kwargs from each sample (we stored video_metadata, do_sample_frames, etc.)
            if video_inputs:
                # each sample has its own mm_processor_kwargs; but for batch we can just
                # pass a list of metadata objects (one per video)
                video_metadata_batch = []
                for sample in samples:
                    if "video_metadata" in sample["llm_inputs"]["mm_processor_kwargs"]:
                        video_metadata_batch.extend(
                            sample["llm_inputs"]["mm_processor_kwargs"]["video_metadata"]
                        )

                inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    videos_kwargs={
                        "video_metadata": video_metadata_batch,
                        "do_sample_frames": False,
                        "return_metadata": True,
                    },
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                inputs = processor(
                    text=text,
                    images=image_inputs,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            input_ids = inputs.input_ids.cuda()
            attention_mask = inputs.attention_mask.cuda()
            pixel_values_videos = inputs.pixel_values_videos.cuda()
            video_grid_thw = inputs.video_grid_thw.cuda()
            ranki_print(f'video_grid_thw={video_grid_thw}')
            with torch.inference_mode():
                if args.repeat_times > 1:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        max_new_tokens=1024,
                        temperature=1,
                        top_p=1,
                        do_sample=True,
                        num_return_sequences=args.repeat_times,
                        stop_strings=target_sequences,
                        tokenizer=processor.tokenizer,
                    )
                else:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        max_new_tokens=1024,
                        top_k=1,
                        do_sample=False,
                        stop_strings=target_sequences,
                        tokenizer=processor.tokenizer,
                    )
            generated_ids_trimmed = []
            for i in range(batch_size):
                in_ids = inputs.input_ids[i]
                for j in range(args.repeat_times):
                    out_ids = generated_ids[i * args.repeat_times + j]
                    out_ids_trimmed = out_ids[len(in_ids):]
                    generated_ids_trimmed.append(out_ids_trimmed)
            outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        for sample_idx, sample in enumerate(samples):
            for repeat_id in range(args.repeat_times):
                if args.backend == 'vllm':
                    pred_text = outputs[sample_idx].outputs[repeat_id].text.strip()
                else:
                    idx = sample_idx * args.repeat_times + repeat_id
                    pred_text = outputs[idx].strip()
                sample_set = {
                    'id': sample['id'],
                    'repeat_id': repeat_id,
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'pred': pred_text,
                    'text': sample['llm_inputs']['prompt'],
                }
                if 'solution' in sample:
                    sample_set['solution'] = sample['solution']
                if 'question_type' in sample:
                    sample_set['question_type'] = sample['question_type']
                ans_file.write(json.dumps(sample_set) + "\n")
                # ranki_print(f"input={sample['llm_inputs']['prompt']}, output={pred_text}")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--prompt-template", type=str, default=None, help="Prompt template string. If not provided, uses default TEMPLATE.")
    
    global args
    args = parser.parse_args()
    args.seed = 42

    set_seed(args.seed)
    run_inference(args)
