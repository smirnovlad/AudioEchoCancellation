#!/usr/bin/env python3
"""
Утилиты для работы с аудиофайлами и логированием.
"""

import os
import wave
import logging
import argparse
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List

# Проверяем наличие soundfile для дополнительных возможностей анализа
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logging.warning("Библиотека soundfile не установлена. Расширенный анализ амплитуд будет недоступен.")

def get_wav_info(file_path: str) -> Dict[str, Any]:
    """
    Получает информацию о WAV файле.
    
    Args:
        file_path: Путь к WAV файлу
        
    Returns:
        dict: Словарь с информацией о файле
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Вычисляем длительность в миллисекундах
            duration_ms = n_frames * 1000 / frame_rate
            
            # Добавляем размер файла
            file_size_kb = os.path.getsize(file_path) / 1024
            
            # Читаем фреймы для возможного анализа амплитуд
            frames = wf.readframes(wf.getnframes())
            
            result = {
                'file_path': file_path,
                'n_channels': n_channels,
                'sample_width': sample_width,
                'sample_width_bits': sample_width * 8,
                'frame_rate': frame_rate,
                'n_frames': n_frames,
                'duration_ms': duration_ms,
                'duration_sec': duration_ms / 1000,
                'file_size_kb': file_size_kb,
                'raw_frames': frames
            }
            
            # Добавляем анализ амплитуд, если доступен soundfile
            if SOUNDFILE_AVAILABLE:
                try:
                    data, _ = sf.read(file_path)
                    channels = 1 if len(data.shape) == 1 else data.shape[1]
                    amplitude_stats = []
                    
                    if channels == 1:
                        channel_data = data
                        stats = {
                            'channel': 1,
                            'min': float(np.min(channel_data)),
                            'max': float(np.max(channel_data)),
                            'mean': float(np.mean(channel_data)),
                            'std': float(np.std(channel_data)),
                            'rms': float(np.sqrt(np.mean(np.square(channel_data))))
                        }
                        amplitude_stats.append(stats)
                    else:
                        for i in range(channels):
                            channel_data = data[:, i]
                            stats = {
                                'channel': i+1,
                                'min': float(np.min(channel_data)),
                                'max': float(np.max(channel_data)),
                                'mean': float(np.mean(channel_data)),
                                'std': float(np.std(channel_data)),
                                'rms': float(np.sqrt(np.mean(np.square(channel_data))))
                            }
                            amplitude_stats.append(stats)
                    
                    result['amplitude_stats'] = amplitude_stats
                except Exception as e:
                    logging.debug(f"Не удалось выполнить анализ амплитуд: {e}")
            
            return result
    except Exception as e:
        error_msg = f"Ошибка при получении информации о WAV файле {file_path}: {e}"
        logging.error(error_msg)
        return {
            'error': error_msg,
            'n_channels': 1,
            'sample_width': 2,
            'sample_width_bits': 16,
            'frame_rate': 16000,
            'n_frames': 0,
            'duration_ms': 0,
            'duration_sec': 0,
            'file_size_kb': 0
        }


def log_file_info(file_path: str, description: str = "") -> Optional[Dict[str, Any]]:
    """
    Логирует информацию о WAV файле и возвращает информацию о нём.
    
    Args:
        file_path: Путь к WAV файлу
        description: Описание файла для логов
        
    Returns:
        tuple: (file_info, channels) - информация о файле и количество каналов
    """
    if not file_path or not os.path.exists(file_path):
        return None, None
        
    logging.info(f"File info: {description}: {file_path}")
    file_info = get_wav_info(file_path)
    
    logging.info(f"  Частота дискретизации: {file_info['frame_rate']} Гц")
    logging.info(f"  Длительность: {file_info['duration_sec']:.3f} сек")
    logging.info(f"  Каналов: {file_info['n_channels']}")
    if 'amplitude_stats' in file_info and file_info['amplitude_stats']:
        for stats in file_info['amplitude_stats']:
            logging.info(f"  Канал {stats['channel']}: мин={stats['min']:.6f}, макс={stats['max']:.6f}, RMS={stats['rms']:.6f}")
    
    return file_info

def log_audio_files(files: List[Tuple[str, str]]):
    for file_path, description in files:
        file_info = log_file_info(file_path, description)

def get_wav_amplitude_stats(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Извлекает статистику амплитуд из WAV файла.
    
    Args:
        file_path: Путь к WAV файлу
        
    Returns:
        Optional[List[Dict[str, Any]]]: Список статистики по каналам или None в случае ошибки
    """
    file_info = get_wav_info(file_path)
    return file_info.get('amplitude_stats')


def format_wav_info(file_info: Dict[str, Any]) -> str:
    """
    Форматирует информацию о WAV файле для вывода.
    
    Args:
        file_info: Словарь с информацией о файле
        
    Returns:
        str: Отформатированная строка с информацией
    """
    result = []
    
    # Проверяем наличие ошибки
    if 'error' in file_info:
        result.append(f"Ошибка: {file_info['error']}")
        return "\n".join(result)
    
    # Основная информация
    result.append("=== Информация о WAV файле ===")
    result.append(f"Путь к файлу: {file_info.get('file_path', 'Не указан')}")
    result.append(f"Количество каналов: {file_info.get('n_channels', 'Неизвестно')}")
    result.append(f"Битность: {file_info.get('sample_width', 'Неизвестно')} байт ({file_info.get('sample_width_bits', 'Неизвестно')} бит)")
    result.append(f"Частота дискретизации: {file_info.get('frame_rate', 'Неизвестно')} Гц")
    result.append(f"Количество фреймов: {file_info.get('n_frames', 'Неизвестно')}")
    result.append(f"Длительность: {file_info.get('duration_sec', 0):.3f} секунд ({file_info.get('duration_ms', 0):.1f} мс)")
    result.append(f"Размер файла: {file_info.get('file_size_kb', 0):.2f} КБ")
    
    # Статистика амплитуд, если доступна
    if 'amplitude_stats' in file_info and file_info['amplitude_stats']:
        result.append("\n=== Статистика амплитуд ===")
        for stats in file_info['amplitude_stats']:
            if file_info.get('n_channels', 1) > 1:
                result.append(f"\nКанал {stats['channel']}:")
            result.append(f"Минимальная амплитуда: {stats['min']:.6f}")
            result.append(f"Максимальная амплитуда: {stats['max']:.6f}")
            result.append(f"Средняя амплитуда: {stats['mean']:.6f}")
            result.append(f"Стандартное отклонение: {stats['std']:.6f}")
            result.append(f"RMS (среднеквадратичное значение): {stats['rms']:.6f}")
    
    return "\n".join(result)


def setup_logger(log_level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Настраивает логирование
    
    Args:
        log_level: Уровень логирования
        log_file: Путь к файлу логов (необязательно)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """
    Основная функция для запуска из командной строки
    """
    parser = argparse.ArgumentParser(description='Анализатор WAV файлов')
    parser.add_argument('file_path', help='Путь к WAV файлу для анализа')
    parser.add_argument('--log-file', '-l', type=str, default=None,
                      help='Путь к файлу для сохранения логов')
    parser.add_argument('--json', '-j', type=str, default=None,
                      help='Путь к файлу для сохранения результатов в формате JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Подробный вывод логов')
    parser.add_argument('--no-output', '-n', action='store_true',
                      help='Не выводить результаты в лог')
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level, args.log_file)
    
    # Анализируем файл
    logging.info(f"Анализ WAV файла: {args.file_path}")
    file_info = get_wav_info(args.file_path)
    
    # Выводим информацию
    if not args.no_output:
        formatted_info = format_wav_info(file_info)
        for line in formatted_info.split("\n"):
            logging.info(line)
    
    # Сохраняем результаты в JSON, если указан путь
    if args.json:
        try:
            import json
            
            # Создаем копию результатов, исключая бинарные данные
            json_safe_results = file_info.copy()
            if 'raw_frames' in json_safe_results:
                json_safe_results['raw_frames'] = f"<binary data, {len(json_safe_results['raw_frames'])} bytes>"
            
            with open(args.json, 'w', encoding='utf-8') as f:
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Результаты сохранены в JSON файл: {args.json}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении результатов в JSON: {e}")
    
    return file_info


if __name__ == "__main__":
    main()
