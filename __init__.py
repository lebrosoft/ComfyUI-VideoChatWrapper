# from .videochat.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .videochat.nodes import ModelLoader, VideoSummary, LoadVideoUpload
from .nodes.audio import ResampleAudio, ConcatAudio, SplitAudio, JoinAudio, AudioCombine, ConvertAudioChannels

NODE_CLASS_MAPPINGS = {
    "VCW_ModelLoader": ModelLoader,
    "VCW_VideoSummary": VideoSummary,
    "VCW_LoadVideo": LoadVideoUpload,
    "ResampleAudio": ResampleAudio,
    "ConcatAudio": ConcatAudio,
    "SplitAudio": SplitAudio,
    "JoinAudio": JoinAudio,
    "AudioCombine": AudioCombine,
    "ConvertAudioChannels": ConvertAudioChannels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VCW_ModelLoader": "Load Model (VideoChat)",
    "VCW_VideoSummary": "Video Summary (VideoChat)",
    "VCW_LoadVideo": "Load Video (Upload) (VideoChat) ",
    "ResampleAudio": "Resample Audio",
    "ConcatAudio": "Concat Audio",
    "SplitAudio": "Split Audio",
    "JoinAudio": "Join Audio",
    "AudioCombine": "Audio Combine",
    "ConvertAudioChannels": "Convert Audio Channels",
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]