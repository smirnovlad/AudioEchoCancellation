import wave
import numpy as np

def create_silent_wav(filename, duration=5, sample_rate=44100, channels=1):
    """
    Create a silent WAV file.
    
    Args:
        filename: Output WAV file path
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels (1=mono, 2=stereo)
    """
    # Calculate total number of frames
    n_frames = int(duration * sample_rate)
    
    # Create silent audio data (all zeros)
    silent_data = np.zeros(n_frames, dtype=np.int16)
    
    # Convert numpy array to bytes
    silent_bytes = silent_data.tobytes()
    
    # Create WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)      # Number of channels
        wav_file.setsampwidth(2)             # Sample width in bytes (2 bytes for 16-bit)
        wav_file.setframerate(sample_rate)   # Sample rate in Hz
        wav_file.writeframes(silent_bytes)   # Write audio data
    
    print(f"Created silent WAV file: {filename}")
    print(f"Duration: {duration} seconds, Sample rate: {sample_rate} Hz, Channels: {channels}")

# Create a 5-second silent WAV file
create_silent_wav("silent_5sec.wav")