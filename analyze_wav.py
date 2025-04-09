#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
import numpy as np
import soundfile as sf
import argparse

def analyze_wav_with_wave(file_path):
    """Анализирует WAV файл с использованием библиотеки wave"""
    with wave.open(file_path, 'rb') as wf:
        # Получаем основные параметры
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        frames_count = wf.getnframes()
        duration = frames_count / frame_rate
        
        # Читаем данные
        frames = wf.readframes(frames_count)
        
        print("\n=== Информация о WAV файле (библиотека wave) ===")
        print(f"Путь к файлу: {file_path}")
        print(f"Количество каналов: {channels}")
        print(f"Битность (bytes): {sample_width} ({sample_width * 8} бит)")
        print(f"Частота дискретизации: {frame_rate} Гц")
        print(f"Количество фреймов: {frames_count}")
        print(f"Длительность: {duration:.3f} секунд ({duration * 1000:.1f} мс)")
        print(f"Размер файла: {os.path.getsize(file_path) / 1024:.2f} КБ")
        
        return frames, frame_rate, channels

def analyze_wav_with_soundfile(file_path):
    """Анализирует WAV файл с использованием библиотеки soundfile"""
    try:
        data, samplerate = sf.read(file_path)
        duration = len(data) / samplerate
        
        channels = 1 if len(data.shape) == 1 else data.shape[1]
        
        print("\n=== Информация о WAV файле (библиотека soundfile) ===")
        print(f"Путь к файлу: {file_path}")
        print(f"Форма данных: {data.shape}")
        print(f"Тип данных: {data.dtype}")
        print(f"Количество каналов: {channels}")
        print(f"Частота дискретизации: {samplerate} Гц")
        print(f"Длительность: {duration:.3f} секунд ({duration * 1000:.1f} мс)")
        
        # Базовая статистика по амплитуде
        print("\n=== Статистика амплитуд ===")
        
        if channels == 1:
            channel_data = data
            print(f"Минимальная амплитуда: {np.min(channel_data):.6f}")
            print(f"Максимальная амплитуда: {np.max(channel_data):.6f}")
            print(f"Средняя амплитуда: {np.mean(channel_data):.6f}")
            print(f"Стандартное отклонение: {np.std(channel_data):.6f}")
            print(f"RMS (среднеквадратичное значение): {np.sqrt(np.mean(np.square(channel_data))):.6f}")
        else:
            for i in range(channels):
                channel_data = data[:, i]
                print(f"\nКанал {i+1}:")
                print(f"Минимальная амплитуда: {np.min(channel_data):.6f}")
                print(f"Максимальная амплитуда: {np.max(channel_data):.6f}")
                print(f"Средняя амплитуда: {np.mean(channel_data):.6f}")
                print(f"Стандартное отклонение: {np.std(channel_data):.6f}")
                print(f"RMS (среднеквадратичное значение): {np.sqrt(np.mean(np.square(channel_data))):.6f}")
        
        return data, samplerate
    except Exception as e:
        print(f"Ошибка при анализе файла с soundfile: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Анализатор WAV файлов')
    parser.add_argument('file_path', help='Путь к WAV файлу')
    args = parser.parse_args()
    
    file_path = args.file_path
    
    if not os.path.exists(file_path):
        print(f"Ошибка: файл {file_path} не найден")
        return
    
    try:
        frames, frame_rate, channels = analyze_wav_with_wave(file_path)
        analyze_wav_with_soundfile(file_path)
    except Exception as e:
        print(f"Ошибка при анализе файла: {e}")

if __name__ == "__main__":
    main() 