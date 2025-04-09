import wave
import numpy as np

# Параметры аудио
sample_rate = 16000  # Частота дискретизации
channels = 1  # Моно
sample_width = 2  # 16 бит

# Длительность в секундах
duration_seconds = 40

# Создаем массив нулей для тишины
num_samples = sample_rate * duration_seconds
silent_data = np.zeros(num_samples, dtype=np.int16)

# Создаем WAV файл
with wave.open('silent_audio.wav', 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    wf.writeframes(silent_data.tobytes())