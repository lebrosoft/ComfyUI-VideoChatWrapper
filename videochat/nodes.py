from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor
)
# import torch
from pathlib import Path
import folder_paths
import os

# from .logger import logger
from .utils import calculate_file_hash, strip_path

#  .\python_embeded\python.exe -m pip install qwen_vl_utils
from qwen_vl_utils import process_vision_info

# https://huggingface.co/OpenGVLab/VideoChat-R1_7B/tree/main
# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
class ModelLoader:
    def __init__(self):
        self.model_checkpoint = None
        self.model = None
        self.processor = None
        # self.device = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )
        # self.bf16_support = (
        #     torch.cuda.is_available()
        #     and torch.cuda.get_device_capability(self.device)[0] >= 8
        # )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["VideoChat-R1_7B", ],{"default": "VideoChat-R1_7B"},),
            }
        }

    RETURN_TYPES = ("MODEL", "PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "VideoChatWrapper"
  
    def load_model(self, model):
        model_id = f"OpenGVLab/{model}"
        # put downloaded model to model/videochat dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "videochat", os.path.basename(model_id)
        )
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )
        if self.model is None:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint, 
                torch_dtype="auto", 
                device_map="auto",
                # attn_implementation="flash_attention_2"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        return self.model, self.processor
    

class VideoSummary:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "processor": ("PROCESSOR",),
                "video_path": ("STRING", ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "video_summary"
    CATEGORY = "VideoChatWrapper"

    def video_summary(self, model, processor, video_path):
        question = "Describe this video in detail."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": f"""{question}
                    """},
                ],
            }
        ]

        #In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return (output_text,)

video_extensions = ['webm', 'mp4', 'mkv', 'mov']

def load_video(**kwargs):
    kwargs['video'] = strip_path(kwargs['video'])
    # print("video_path: ", kwargs['video'])
    return (kwargs['video'],)
    
class LoadVideoUpload:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1].lower() in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),
                    },
                }

    CATEGORY = "VideoChatWrapper"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "load_video"

    def load_video(self, **kwargs):
        kwargs['video'] = folder_paths.get_annotated_filepath(strip_path(kwargs['video']))
        return load_video(**kwargs)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        image_path = folder_paths.get_annotated_filepath(video)
        return calculate_file_hash(image_path)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True
    

