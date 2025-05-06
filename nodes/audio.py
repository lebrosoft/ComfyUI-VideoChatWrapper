
import sys
import warnings
from typing import Any, Dict

import torch
import torchaudio
from torchaudio.transforms import Resample

from typing import TypedDict
from typing import Tuple
from torch import Tensor

import comfy.model_management

#  pip install git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft
# from audiocraft.data.audio_utils import convert_audio

# from .node_def import BASE_NODE_CATEGORY, AudioData

BASE_NODE_CATEGORY = "jida"
AudioData = Dict[str, Any]

NODE_CATEGORY = BASE_NODE_CATEGORY + "/audio"

class AUDIO(TypedDict):
    """
    Required Fields:
        waveform (torch.Tensor): A tensor containing the audio data. Shape: [Batch, Channels, frames].
        sample_rate (int): The sample rate of the audio data.
    """

    waveform: Tensor
    sample_rate: int

def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav

# def convert_audio(wav: torch.Tensor, from_rate: float,
#                   to_rate: float, to_channels: int) -> torch.Tensor:
#     """Convert audio to new sample rate and number of audio channels."""
#     wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
#     wav = convert_audio_channels(wav, to_channels)
#     return wav

# https://github.com/christian-byrne/audio-separation-nodes-comfyui/blob/master/src/combine_video_with_audio.py
# https://github.com/kale4eat/ComfyUI-speech-dataset-toolkit/blob/main/edit_nodes.py
# https://github.com/eigenpunk/ComfyUI-audio/blob/main/util_nodes.py
# https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/data/audio_utils.py
class ConvertAudioChannels:
    """
    convert an AUDIO's number of channels
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "to_channels": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = "Convert an AUDIO's number of channels."

    def convert(self, audio, to_channels):
        # converted = []
        clip = audio["waveform"]
        expand_dim = len(clip.shape) == 2
        if expand_dim:
            clip = clip.unsqueeze(0)
        # conv_clip = convert_audio(clip, from_rate, to_rate, to_channels)
        conv_clip = convert_audio_channels(clip, to_channels)
        conv_clip = conv_clip.squeeze(0) if expand_dim else conv_clip
        audio_data = {"waveform": conv_clip, "sample_rate": audio["sample_rate"]}
        return (audio_data,)



# combine two audio tracks(mono or stereo) by overlaying their waveforms
class AudioCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
            "optional": {
                "method": (
                    ["add", "mean", "subtract", "multiply", "divide"],
                    {
                        "default": "add",
                        "tooltip": "The method used to combine the audio waveforms.",
                    },
                )
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = "Combine two audio tracks by overlaying their waveforms."

    def main(
        self,
        audio_1: AUDIO,
        audio_2: AUDIO,
        method: str = "add",
    ) -> Tuple[AUDIO]:

        waveform_1: torch.Tensor = audio_1["waveform"]
        input_sample_rate_1: int = audio_1["sample_rate"]

        waveform_2: torch.Tensor = audio_2["waveform"]
        input_sample_rate_2: int = audio_2["sample_rate"]

        # Resample the audio if the sample rates are different
        if input_sample_rate_1 != input_sample_rate_2:
            device: torch.device = comfy.model_management.get_torch_device()
            if input_sample_rate_1 < input_sample_rate_2:
                resample = Resample(input_sample_rate_1, input_sample_rate_2).to(device)
                waveform_1: torch.Tensor = resample(waveform_1.to(device))
                waveform_1.to("cpu")
                output_sample_rate = input_sample_rate_2
            else:
                resample = Resample(input_sample_rate_2, input_sample_rate_1).to(device)
                waveform_2: torch.Tensor = resample(waveform_2.to(device))
                waveform_2.to("cpu")
                output_sample_rate = input_sample_rate_1
        else:
            output_sample_rate = input_sample_rate_1

        # Ensure the audio is the same length
        min_length = min(waveform_1.shape[-1], waveform_2.shape[-1])
        if waveform_1.shape[-1] != min_length:
            waveform_1 = waveform_1[..., :min_length]
        if waveform_2.shape[-1] != min_length:
            waveform_2 = waveform_2[..., :min_length]

        match method:
            case "add":
                waveform = waveform_1 + waveform_2
            case "subtract":
                waveform = waveform_1 - waveform_2
            case "multiply":
                waveform = waveform_1 * waveform_2
            case "divide":
                waveform = waveform_1 / waveform_2
            case "mean":
                waveform = (waveform_1 + waveform_2) / 2

        return (
            {
                "waveform": waveform,
                "sample_rate": output_sample_rate,
            },
        )

# join/combine multiple audio tracks into a single audio track
class JoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audios": ("AUDIO",),
                "silent_interval": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "join_audio"
    DESCRIPTION = "Join/combine multiple audio tracks into a single audio track."

    def join_audio(self, audios: list[AudioData], silent_interval: float):
        if len(audios) == 0:
            raise ValueError("Audios size is zero.")
        if not all(
            [audio["sample_rate"] == audios[0]["sample_rate"] for audio in audios]
        ):
            raise ValueError("sample_rate must be the same.")
        if not all(
            [
                audio["waveform"].size(1) == audios[0]["waveform"].size(1)
                for audio in audios
            ]
        ):
            raise ValueError("All audio must be either stereo or monaural.")

        samples = int(silent_interval * audios[0]["sample_rate"])
        interval = torch.zeros((audios[0]["waveform"].size(1), samples))
        new_tensors = []
        for audio in audios:
            new_tensors.append(audio)
            new_tensors.append(interval)

        new_tensors.pop()

        new_waveforms = torch.concat(new_tensors, dim=-1)
        new_audio = {"waveform": new_waveforms, "sample_rate": audios[0]["sample_rate"]}
        return (new_audio,)

# split an audio track into two audio tracks(left and right)
class SplitAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "second": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio1", "audio2")
    FUNCTION = "split_audio"
    DESCRIPTION = "Split an audio track into two audio tracks(left and right)."

    def split_audio(self, audio: AudioData, second: float):
        sample = max(0, int(second * audio["sample_rate"]) - 1)
        view1 = audio["waveform"][..., :sample]
        view2 = audio["waveform"][..., sample:]
        audio1 = {
            "waveform": view1.detach().clone(),
            "sample_rate": audio["sample_rate"],
        }
        audio2 = {
            "waveform": view2.detach().clone(),
            "sample_rate": audio["sample_rate"],
        }
        return (audio1, audio2)

# concatenate two audio tracks
class ConcatAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "silent_interval": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    CATEGORY = NODE_CATEGORY
    DESCRIPTION = "Concatenate two audio tracks."
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concat_audio"

    def concat_audio(
        self, audio1: AudioData, audio2: AudioData, silent_interval: float
    ):
        if audio1["sample_rate"] != audio2["sample_rate"]:
            raise ValueError(
                f"sample_rate is different.\naudio1: {0}\naudio2: {1}".format(
                    audio1["sample_rate"], audio2["sample_rate"]
                )
            )

        audio1_ch = "stereo" if audio1["waveform"].size(1) > 1 else "monaural"
        audio2_ch = "stereo" if audio2["waveform"].size(1) > 1 else "monaural"
        if not audio1_ch == audio2_ch:
            raise ValueError(f"naudio1 is {audio1_ch} but audio2 is {audio2_ch}")

        samples = int(silent_interval * audio1["sample_rate"])
        interval = torch.zeros(
            (audio1["waveform"].size(0), audio1["waveform"].size(1), samples)
        )
        new_waveform = torch.concat(
            [audio1["waveform"], interval, audio2["waveform"]], dim=-1
        )
        new_audio = {"waveform": new_waveform, "sample_rate": audio1["sample_rate"]}
        return (new_audio,)

# resample an audio track
class ResampleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "new_freq": ("INT", {"default": 32000, "min": 1, "max": 2 ** 32}),
                "resampling_method": (["sinc_interp_hann", "sinc_interp_kaiser"],),
                "lowpass_filter_width": (
                    "INT",
                    {"default": 6, "min": 0, "max": 2**32},
                ),
                "rolloff": (
                    "FLOAT",
                    {
                        "default": 0.99,
                        "min": 0,
                        "max": sys.float_info.max,
                    },
                ),
            },
            "optional": {
                "beta": (
                    "FLOAT",
                    {
                        "default": 14.769656459379492,
                        "min": 0.0,
                        "max": sys.float_info.max,
                    },
                ),
            },
        }

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "resample"
    DESCRIPTION = "Resample an audio track."

    def resample(
        self,
        audio: AudioData,
        new_freq: int,
        resampling_method: str,
        lowpass_filter_width: int,
        rolloff: float,
        beta=None,
    ):
        transform = torchaudio.transforms.Resample(
            orig_freq=audio["sample_rate"],
            new_freq=new_freq,
            resampling_method=resampling_method,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            beta=beta,
        )
        new_audio = {"waveform": transform(audio["waveform"]), "sample_rate": new_freq}
        return (new_audio,)