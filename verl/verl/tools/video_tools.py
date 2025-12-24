import logging
import os
import numpy as np
from uuid import uuid4
from PIL import Image
import base64
from io import BytesIO

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.dataset.video_vl_utils import process_vision_info

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def video_clip_to_image_messages(video_name, video_base, start_time, end_time, duration, max_frames=120):
    """
    构造 image_messages，所有帧都传递，clip 区间用 video_start 和 video_end 指定。
    """
    video_path = os.path.join(video_base, video_name.split('.')[0])
    frame_files = sorted(
        [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    frame_paths = [os.path.join(video_path, frame_file) for frame_file in frame_files]
    fps = len(frame_paths) / duration if duration > 0 else 1
    video_message = [{
        'content': [
            {
                "type": "video",
                "video": frame_paths,
                "min_pixels": 4*28*28,
                "max_pixels": 224*224,
                "max_frames": max_frames,
                "fps": fps,
                "draw_number": True,
                "parallel": True,
                "video_start": start_time,
                "video_end": end_time,
            }
        ]
    }]
    # 让 process_vision_info 或下游根据 video_start/video_end 再做 clip
    _, video_inputs = process_vision_info(video_message)
    video_input = np.array(video_inputs[0])
    image_messages = []
    for frame in video_input:
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        base64_str = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        part_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_str}",
                "detail": "low",
            },
        }
        image_messages.append(part_message)
    return image_messages

def call_azure_openai_chat_completion(client, image_messages, prompt, model="gpt-4-vision-preview", temperature=0.0):
    messages = [{
        "role": "user",
        "content": image_messages + [{"type": "text", "text": prompt}]
    }]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content

class GetVideoClipFrameTool(BaseTool):
    """Tool to get raw frame pixel arrays from a video segment."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": 0.0}
        return instance_id

    def _load_frames_from_directory(self, video_path, start_time, end_time, duration, **kwargs):
        """Load frames from pre-extracted frame directory for a given time range."""
        from PIL import Image
        from download_and_extract_frames import extract_youtube_id
        
        video_base = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        
        # Look for frame directory
        video_frame_root = os.path.join(video_base, video_name.split(".")[0])
        if not os.path.exists(video_frame_root):
            video_frame_root = os.path.join(video_base, extract_youtube_id(video_name.split(".")[0]))
        
        if not os.path.exists(video_frame_root):
            return None  # Frame directory not found, caller should try raw video
        
        # Load all frames
        frame_files = sorted(
            [f for f in os.listdir(video_frame_root) if f.endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        frame_paths = [os.path.join(video_frame_root, f) for f in frame_files]
        
        # Calculate which frames correspond to the time range
        total_frames = len(frame_paths)
        clip_fps = total_frames / duration if duration > 0 else 2.0
        
        start_frame_idx = int(start_time * clip_fps)
        end_frame_idx = int(end_time * clip_fps)
        
        # Clip to valid range
        start_frame_idx = max(0, start_frame_idx)
        end_frame_idx = min(total_frames, end_frame_idx)
        
        # Extract frames for this segment
        selected_frame_paths = frame_paths[start_frame_idx:end_frame_idx]
        
        if len(selected_frame_paths) == 0:
            raise ValueError(f"No frames found for time range {start_time}-{end_time}")
        
        # Load frames as PIL Images
        frames = [Image.open(fp).convert("RGB") for fp in selected_frame_paths]
        
        return {
            "frames": frames,
            "frame_paths": selected_frame_paths,
            "fps": clip_fps,
        }

    def _execute(self, instance_id: str, parameters: dict, **kwargs):
        try:
            # Parse parameters
            video_path = kwargs.get("video_path")
            duration = kwargs.get("duration")
            if video_path is None or duration is None:
                raise ValueError("video_path and duration are required in kwargs")
            
            start_time = float(parameters.get("start_time"))
            end_time = float(parameters.get("end_time"))
            max_frames = kwargs.get("max_frames", 64)
            fps = kwargs.get("fps", 2.0)
            draw_number = kwargs.get("draw_number", True)
            parallel = kwargs.get("parallel", True)
            
            # Try loading from pre-extracted frame directory first
            frame_data = self._load_frames_from_directory(
                video_path, start_time, end_time, duration, **kwargs
            )
            
            if frame_data is not None:
                # Successfully loaded from frame directory
                frames = frame_data["frames"]
                selected_frame_paths = frame_data["frame_paths"]
                clip_fps = frame_data["fps"]
                
                # Build the video element for vLLM
                ele = {
                    "type": "video",
                    "video": selected_frame_paths,
                    "max_pixels": kwargs.get("max_pixels", 224*224),
                    "max_frames": max_frames,
                    "fps": fps,
                    "draw_number": draw_number,
                    "parallel": parallel,
                }
                
                return_content = {
                    "ele": ele,
                    "video": frames,
                    "fps": fps,
                    "start_time": start_time,
                    "end_time": end_time,
                }
                return {"type": "result", "content": return_content}, 0.0, {}
            
            # Fallback: Load from raw video file (original functionality)
            video_base = os.path.dirname(video_path)
            video_name = os.path.basename(video_path)
            image_messages = video_clip_to_image_messages(
                video_name, video_base, start_time, end_time, duration, max_frames=max_frames
            )
            return_content = {
                "image_messages": image_messages,
                "start_time": start_time,
                "end_time": end_time,
            }
            return {"type": "result", "content": return_content}, 0.0, {}
            
        except Exception as e:
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        return self._execute(instance_id, parameters, **kwargs)

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class GetVideoClipCaptionTool(BaseTool):
    """Tool to get the caption string for a video segment using Azure OpenAI."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": 0.0}
        return instance_id

    def _execute(self, instance_id: str, parameters: dict, video_name: str, video_base: str, duration: float, client, **kwargs):
        try:
            start_time = float(parameters.get("start_time"))
            end_time = float(parameters.get("end_time"))
            image_messages = video_clip_to_image_messages(
                video_name, video_base, start_time, end_time, duration, max_frames=kwargs.get("max_frames", 120)
            )
            prompt = "Please provide a detailed caption for this video segment, describing all important visual details including people, actions, objects, scenes, emotions, and significant events in sequence. Be as comprehensive and specific as possible."
            prompt = "Please provide an objective and detailed caption for this video segment, focusing on accurately describing the visible subjects and their actions, the scene, and any changes, strictly based on the visual content. Do not make assumptions or add information not clearly present in the video."

            caption = call_azure_openai_chat_completion(client, image_messages, prompt, model=kwargs.get("model", "gpt-4-vision-preview"), temperature=0.0)
            return_content = {
                "caption": caption,
                "start_time": start_time,
                "end_time": end_time
            }
            return {"type": "result", "content": return_content}, 0.0, {}
        except Exception as e:
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}

    async def execute(self, instance_id: str, parameters: dict, video_name: str, video_base: str, duration: float, client, **kwargs):
        return self._execute(instance_id, parameters, video_name, video_base, duration, client, **kwargs)

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

class AskVideoClipQuestionTool(BaseTool):
    """Tool to answer a question about a video segment using Azure OpenAI."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": 0.0}
        return instance_id

    def _execute(self, instance_id: str, parameters: dict, video_name: str, video_base: str, duration: float, client, **kwargs):
        try:
            start_time = float(parameters.get("start_time"))
            end_time = float(parameters.get("end_time"))
            question = parameters.get("question")
            image_messages = video_clip_to_image_messages(
                video_name, video_base, start_time, end_time, duration, max_frames=kwargs.get("max_frames", 120)
            )
            prompt = f"Please answer the following question based on this video segment: {question}"
            prompt = f"Please answer the following question based only on the visual content of this video segment: {question} Describe only what is clearly visible in the frames. Do not make assumptions, guesses, or add any information that is not directly shown in the video."

            answer = call_azure_openai_chat_completion(client, image_messages, prompt, model=kwargs.get("model", "gpt-4-vision-preview"), temperature=0.0)
            return_content = {
                "question": question,
                "answer": answer,
                "start_time": start_time,
                "end_time": end_time,
            }
            return {"type": "result", "content": return_content}, 0.0, {}
        except Exception as e:
            return {"type": "error", "content": f"Error: {self.name} tool execution failed: {e}"}, 0.0, {"error": str(e)}

    async def execute(self, instance_id: str, parameters: dict, video_name: str, video_base: str, duration: float, client, **kwargs):
        return self._execute(instance_id, parameters, video_name, video_base, duration, client, **kwargs)

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


from omegaconf import OmegaConf
from verl.tools.schemas import OpenAIFunctionToolSchema

def load_tool_schema(tools_config_file, class_name):
    """
    根据 tools_config_file 和 class_name，返回对应 tool 的 OpenAIFunctionToolSchema 实例。
    """
    tools_config = OmegaConf.load(tools_config_file)
    for tool_config in tools_config.tools:
        if tool_config.class_name == class_name:
            if tool_config.get("tool_schema", None) is None:
                return None
            tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
            return OpenAIFunctionToolSchema.parse_obj(tool_schema_dict)
    raise ValueError(f"Tool class {class_name} not found in config file.")



def main():
    from openai import AzureOpenAI

    # 路径和参数
    yaml_path = "verl/verl/tools/config/video_tool_config.yaml"
    video_name = "3MSZA.mp4"
    video_base = "data/charades/video_14400frames_fps2"
    duration = 31.0
    start_time = 24.3
    end_time = 30.4
    max_frames = 120

    # 加载 tool schema
    frame_schema = load_tool_schema(yaml_path, "verl.tools.video_tools.GetVideoClipFrameTool")
    caption_schema = load_tool_schema(yaml_path, "verl.tools.video_tools.GetVideoClipCaptionTool")
    qa_schema = load_tool_schema(yaml_path, "verl.tools.video_tools.AskVideoClipQuestionTool")

    # Azure OpenAI client
    api_version="gpt-4.1-2025-04-14"
    client = AzureOpenAI(
        api_key="KC0QHv33A8jozezM4Fw2BrZkClIuQv3q_GPT_AK",
        api_version=api_version,
        azure_endpoint="https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl"
    )

    # 实例化工具
    frame_tool = GetVideoClipFrameTool({}, frame_schema)
    caption_tool = GetVideoClipCaptionTool({}, caption_schema)
    qa_tool = AskVideoClipQuestionTool({}, qa_schema)

    parameters = {
        "start_time": start_time,
        "end_time": end_time
    }
    qa_parameters = {
        "start_time": start_time,
        "end_time": end_time,
        "question": "What is the main action happening in this video segment?"
    }

    import asyncio
    import json

    async def run_tools():
        frame_result, _, _ = await frame_tool.execute(
            instance_id="test1", parameters=parameters,
            video_name=video_name, video_base=video_base, duration=duration, max_frames=max_frames
        )
        # print("Frame Tool Result:", frame_result)

        caption_result, _, _ = await caption_tool.execute(
            instance_id="test2", parameters=parameters,
            video_name=video_name, video_base=video_base, duration=duration, client=client, max_frames=max_frames, model=api_version
        )
        print("Caption Tool Result:", json.dumps(caption_result))

        qa_result, _, _ = await qa_tool.execute(
            instance_id="test3", parameters=qa_parameters,
            video_name=video_name, video_base=video_base, duration=duration, client=client, max_frames=max_frames, model=api_version
        )
        print("QA Tool Result:", json.dumps(qa_result))

    asyncio.run(run_tools())

if __name__ == "__main__":
    main()
