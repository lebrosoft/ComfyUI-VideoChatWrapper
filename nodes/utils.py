import wave

def is_stereo(wav_file_path):
    """
    Checks if a WAV file is stereo or mono.

    Args:
        wav_file_path: Path to the WAV file.

    Returns:
        True if the WAV file is stereo, False if it's mono.
    """
    try:
        with wave.open(wav_file_path, 'rb') as wf:
            num_channels = wf.getnchannels()
            return num_channels == 2
    except wave.Error as e:
         print(f"Error processing {wav_file_path}: {e}")
         return False

# Example usage
file_path = 'path/to/your/audio.wav'
if is_stereo(file_path):
    print(f"{file_path} is a stereo WAV file.")
else:
    print(f"{file_path} is a mono WAV file.")