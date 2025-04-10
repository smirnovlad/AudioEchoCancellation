#!/usr/bin/env python3
"""
Скрипт для пакетной обработки предварительно созданных тестовых данных
с использованием WebRTC AEC (подавление акустического эха).

Этот скрипт:
1. Обрабатывает все поддиректории в тестовых каталогах (music, agent_speech)
2. Поддерживает структуру директорий:
   - В каждом тестовом каталоге (music, agent_speech) находятся две поддиректории:
     * clear_reference/ - для обработки чистого референсного сигнала
     * reference_by_micro/ - для обработки сигнала через микрофон
   - В каждой поддиректории находятся директории с разными уровнями громкости:
     * volume_01/ - 10% громкости
     * volume_04/ - 40% громкости
     * volume_07/ - 70% громкости
     * volume_10/ - 100% громкости (исходная)
     * volume_13/ - 130% громкости
3. Применяет модуль AEC к парам файлов (reference_new.wav и original_input.wav)
4. Сохраняет обработанные файлы и основные метрики (задержка, количество обработанных фреймов) в каждой поддиректории
5. Для расчета дополнительных метрик (ERLE и др.) используйте скрипт metrics.py
"""

import os
import sys
import wave
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, Union, List

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_aec_test.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Класс для сериализации NumPy типов в JSON
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Примечание: функция calculate_aec_metrics перенесена в отдельный скрипт metrics.py
# Для расчета дополнительных метрик (ERLE и др.) используйте скрипт metrics.py

# Импорт из модуля AEC
try:
    # Добавляем родительскую директорию в путь импорта
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from aec import WebRTCAECSession
    logging.info("WebRTCAECSession успешно импортирован")
except ImportError:
    logging.error("Не удалось импортировать WebRTCAECSession из пакета aec")
    sys.exit(1)

# Импортируем функции из aec_test_tool.py и utils.py
try:
    from aec_test_tool import calculate_metrics
    from visualization import visualize_audio_processing
    # Импортируем функции из utils.py для анализа WAV файлов
    from utils import get_wav_info, log_file_info, format_wav_info, get_wav_amplitude_stats
    logging.info("Функции из необходимых модулей успешно импортированы")
except ImportError as e:
    logging.error(f"Не удалось импортировать функции из необходимых модулей: {e}")
    sys.exit(1)

def find_test_directories(base_dir="tests"):
    """
    Находит все директории для тестирования
    
    Args:
        base_dir: Путь к корневой директории с тестами
        
    Returns:
        dictionary: Словарь {тест_папка: список_поддиректорий}
    """
    test_dirs = {}
    
    # Главные директории тестов
    main_test_dirs = ["music", "agent_speech", "agent_user_speech", "agent_speech_30_sec"]
    
    for main_dir in main_test_dirs:
        main_dir_path = os.path.join(base_dir, main_dir)
        if not os.path.exists(main_dir_path):
            logging.warning(f"Директория {main_dir_path} не существует, пропускаем")
            continue
        
        # Проверяем наличие mono/stereo поддиректорий для определенных типов тестов
        if main_dir in ["agent_speech", "agent_user_speech", "agent_speech_30_sec"]:
            # Проверяем наличие mono/stereo поддиректорий
            channel_dirs = ["mono", "stereo"]
            has_channel_dirs = False
            
            for channel_dir in channel_dirs:
                channel_dir_path = os.path.join(main_dir_path, channel_dir)
                if os.path.exists(channel_dir_path) and os.path.isdir(channel_dir_path):
                    has_channel_dirs = True
                    logging.info(f"Найдена поддиректория {channel_dir} в {main_dir_path}")
                    
                    # Обрабатываем поддиректорию с указанным channel_dir
                    found_sub_dirs = find_test_subdirectories(channel_dir_path)
                    
                    if found_sub_dirs:
                        # Используем main_dir/channel_dir как ключ для словаря
                        test_dirs[f"{main_dir}/{channel_dir}"] = found_sub_dirs
                        logging.info(f"Найдено {len(found_sub_dirs)} поддиректорий в {main_dir}/{channel_dir}")
                    else:
                        logging.warning(f"В директории {channel_dir_path} не найдено поддиректорий с тестовыми данными")
            
            # Если не найдены mono/stereo поддиректории, обрабатываем как обычную директорию
            if not has_channel_dirs:
                found_sub_dirs = find_test_subdirectories(main_dir_path)
                
                if found_sub_dirs:
                    test_dirs[main_dir] = found_sub_dirs
                    logging.info(f"Найдено {len(found_sub_dirs)} поддиректорий в {main_dir}")
                else:
                    logging.warning(f"В директории {main_dir_path} не найдено поддиректорий с тестовыми данными")
        else:
            # Для директорий без mono/stereo (например, music) - обычная обработка
            found_sub_dirs = find_test_subdirectories(main_dir_path)
            
            if found_sub_dirs:
                test_dirs[main_dir] = found_sub_dirs
                logging.info(f"Найдено {len(found_sub_dirs)} поддиректорий в {main_dir}")
            else:
                logging.warning(f"В директории {main_dir_path} не найдено поддиректорий с тестовыми данными")
    
    return test_dirs

def find_test_subdirectories(base_dir):
    """
    Находит все поддиректории в тестовой директории
    
    Args:
        base_dir: Путь к тестовой директории
        
    Returns:
        list: Список поддиректорий
    """
    found_sub_dirs = []
    
    # Ищем поддиректории типа clear_reference и reference_by_micro
    ref_types = ["clear_reference", "reference_by_micro"]
    
    for ref_type in ref_types:
        ref_type_path = os.path.join(base_dir, ref_type)
        if not os.path.exists(ref_type_path):
            logging.warning(f"Директория {ref_type_path} не существует, пропускаем")
            continue
        
        if ref_type == "clear_reference":
            # Для clear_reference ищем директории с уровнями громкости (volume_XX)
            for d in os.listdir(ref_type_path):
                sub_dir_path = os.path.join(ref_type_path, d)
                if os.path.isdir(sub_dir_path) and d.startswith("volume_"):
                    found_sub_dirs.append(sub_dir_path)
                    logging.info(f"Найдена директория: {sub_dir_path}")
        else:
            # Для reference_by_micro сначала ищем директории с задержками (delay_XX)
            for delay_dir in os.listdir(ref_type_path):
                delay_dir_path = os.path.join(ref_type_path, delay_dir)
                if os.path.isdir(delay_dir_path) and delay_dir.startswith("delay_"):
                    # Внутри директории с задержкой ищем директории с уровнями громкости
                    for vol_dir in os.listdir(delay_dir_path):
                        vol_dir_path = os.path.join(delay_dir_path, vol_dir)
                        if os.path.isdir(vol_dir_path) and vol_dir.startswith("volume_"):
                            found_sub_dirs.append(vol_dir_path)
                            logging.info(f"Найдена директория: {vol_dir_path}")
    
    return found_sub_dirs

def process_audio_with_aec(
        processed_file_path,
        reference_file_info,
        original_input_file_info,
        reference_by_micro_volumed_file_info,
        reference_by_micro_volumed_delayed_file_info,
        frame_rate=16000,
        n_channels=1, 
        visualize=True,
        output_dir="results",
        frame_size_ms=10.0
    ):
    """
    Обрабатывает аудиофайлы с использованием AEC и сохраняет результаты.
    
    Args:
        original_input_file: Путь к входному файлу
        processed_file_path: Путь к файлу для сохранения результата
        reference_file_path: Путь к файлу с референсным сигналом
        frame_rate: Частота дискретизации
        n_channels: Количество каналов (1 или 2)
        visualize: Флаг для визуализации результатов
        output_dir: Директория для сохранения результатов
        frame_size_ms: Размер фрейма в миллисекундах
        
    Returns:
        dict: Словарь с метриками
    """
    import wave
    import numpy as np
    import logging
    from visualization import visualize_audio_processing
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "aec_processing.log")),
            logging.StreamHandler()
        ]
    )
    
    try:
        logging.info(f"Обработка файла {original_input_file_info['file_path']} с помощью AEC...")
        logging.info(f"Используем количество каналов: {n_channels}")
        
        # Проверяем длительность
        if abs(reference_file_info['duration_ms'] - original_input_file_info['duration_ms']) > 500: # Если разница более 500 мс
            logging.warning(f"Длительности файлов существенно различаются: {reference_file_info['duration_sec']:.3f}s vs {original_input_file_info['duration_sec']:.3f}s")
            logging.warning("Для обработки будет использована минимальная длина")
        else:
            logging.info(f"Длительности файлов примерно совпадают (разница {abs(reference_file_info['duration_ms'] - original_input_file_info['duration_ms']):.2f} мс)")

        # Настраиваем AEC и масштабируем сигналы
        frame_size = int(frame_rate * frame_size_ms / 1000)
        
        # Инициализация сессии AEC
        aec_session = WebRTCAECSession(
            session_id=f"batch_{os.path.basename(original_input_file_info['file_path'])}",
            frame_rate=frame_rate,
            n_channels=n_channels,
            batch_mode=True,
            frame_size_ms=frame_size_ms
        )
        
        # Оценка и установка задержки
        delay_samples, delay_ms, confidence = aec_session.auto_set_delay(
            reference_file_info['raw_frames'],
            original_input_file_info['raw_frames'],
            actual_delay_ms=654
        )
        logging.info(f"Обнаружена задержка: {delay_samples} семплов ({delay_ms:.2f} мс), уверенность: {confidence:.4f}")
        
        # Получаем данные корреляции из последнего вычисления, если такой метод существует
        correlation_data = {}
        if hasattr(aec_session, 'get_last_correlation_data'):
            correlation_data = aec_session.get_last_correlation_data()
            logging.info(f"Получены данные корреляции из AEC-сессии для передачи в визуализацию")
        
        # Оптимизация параметров AEC для лучшего качества
        aec_session.optimize_for_best_quality()
        
        # Подготовка к пофреймовой обработке
        frame_size_bytes = frame_size * 2 * n_channels  # 2 байта на сэмпл (16 бит)
        
        # Разделение на фреймы
        ref_frames = [
            reference_file_info['raw_frames'][i:i+frame_size_bytes]
            for i in range(0, len(reference_file_info['raw_frames']), frame_size_bytes)
        ]
        in_frames = [
            original_input_file_info['raw_frames'][i:i+frame_size_bytes]
            for i in range(0, len(original_input_file_info['raw_frames']), frame_size_bytes)
        ]

        min_frames = min(len(ref_frames), len(in_frames))
        logging.info(f"Количество фреймов: референс={len(ref_frames)}, вход={len(in_frames)}, обработка={min_frames}")
        
        # Анализ и логирование информации о фреймах
        if len(ref_frames) > 0 and len(in_frames) > 0:
            ref_frame_size = len(ref_frames[0])
            in_frame_size = len(in_frames[0])
            ref_frame_samples = ref_frame_size // (2 * n_channels)  # 2 байта на сэмпл (16 бит)
            in_frame_samples = in_frame_size // (2 * n_channels)
            
            # Информация о размере фреймов
            logging.info(f"Размер референсного фрейма: {ref_frame_size} байт, {ref_frame_samples} сэмплов, длительность {ref_frame_samples/frame_rate*1000:.2f} мс")
            logging.info(f"Размер входного фрейма: {in_frame_size} байт, {in_frame_samples} сэмплов, длительность {in_frame_samples/frame_rate*1000:.2f} мс")
            
            # Дополнительная информация о первом фрейме каждого типа
            try:
                ref_frame_np = np.frombuffer(ref_frames[0], dtype=np.int16)
                in_frame_np = np.frombuffer(in_frames[0], dtype=np.int16)
                
                ref_channels_data = ref_frame_np.reshape(-1, n_channels) if n_channels > 1 else ref_frame_np
                in_channels_data = in_frame_np.reshape(-1, n_channels) if n_channels > 1 else in_frame_np
                
                # Информация о каналах
                if n_channels > 1:
                    ref_ch1_min, ref_ch1_max = np.min(ref_channels_data[:,0]), np.max(ref_channels_data[:,0])
                    ref_ch2_min, ref_ch2_max = np.min(ref_channels_data[:,1]), np.max(ref_channels_data[:,1])
                    in_ch1_min, in_ch1_max = np.min(in_channels_data[:,0]), np.max(in_channels_data[:,0])
                    in_ch2_min, in_ch2_max = np.min(in_channels_data[:,1]), np.max(in_channels_data[:,1])
                    
                    logging.info(f"Референсный фрейм: Канал 1 [мин={ref_ch1_min}, макс={ref_ch1_max}], Канал 2 [мин={ref_ch2_min}, макс={ref_ch2_max}]")
                    logging.info(f"Входной фрейм: Канал 1 [мин={in_ch1_min}, макс={in_ch1_max}], Канал 2 [мин={in_ch2_min}, макс={in_ch2_max}]")
                else:
                    ref_min, ref_max = np.min(ref_channels_data), np.max(ref_channels_data)
                    in_min, in_max = np.min(in_channels_data), np.max(in_channels_data)
                    
                    logging.info(f"Референсный фрейм: [мин={ref_min}, макс={ref_max}]")
                    logging.info(f"Входной фрейм: [мин={in_min}, макс={in_max}]")
                
                # Информация об энергии сигнала
                ref_energy = np.sum(ref_frame_np.astype(np.float32)**2) / len(ref_frame_np)
                in_energy = np.sum(in_frame_np.astype(np.float32)**2) / len(in_frame_np)
                
                logging.info(f"Энергия референсного фрейма: {ref_energy:.2f}")
                logging.info(f"Энергия входного фрейма: {in_energy:.2f}")
                logging.info(f"Соотношение энергий (вход/референс): {in_energy/ref_energy if ref_energy > 0 else 'N/A'}")
            except Exception as e:
                logging.warning(f"Не удалось проанализировать содержимое фреймов: {e}")
        
        # Буферизация для компенсации задержки
        delay_frames = int(delay_samples / frame_size)
        # pre_buffer_size = max(5, delay_frames) # Минимум 5 фреймов для предварительной буферизации
        pre_buffer_size = delay_frames

        # Предварительная буферизация референсных фреймов
        for i in range(min(pre_buffer_size, len(ref_frames))):
            logging.info(f"Буферизация референсного фрейма #{i}, размер: {len(ref_frames[i])} байт")
            aec_session.add_reference_frame(ref_frames[i])
        
        # Обработка фреймов
        processed_frames = []
        
        # Интервал логирования подробной информации о фреймах
        frame_log_interval = min(100, min_frames // 10 if min_frames > 10 else 1)
        
        for i in range(min_frames):
            # Добавляем референсный фрейм с учетом задержки
            ref_idx = i + pre_buffer_size
            if ref_idx < len(ref_frames):
                # Подробное логирование для первого фрейма и каждые frame_log_interval фреймов
                if i == 0 or i % frame_log_interval == 0:
                    ref_data_np = np.frombuffer(ref_frames[ref_idx], dtype=np.int16)
                    ref_energy = np.sum(ref_data_np.astype(np.float32)**2) / len(ref_data_np)
                    logging.info(f"Референсный фрейм #{ref_idx}: размер={len(ref_frames[ref_idx])} байт, энергия={ref_energy:.2f}")
                
                aec_session.add_reference_frame(ref_frames[ref_idx])
            
            # Обрабатываем входной фрейм с дополнительным логированием
            if i == 0 or i % frame_log_interval == 0:
                in_data_np = np.frombuffer(in_frames[i], dtype=np.int16)
                in_energy = np.sum(in_data_np.astype(np.float32)**2) / len(in_data_np)
                logging.info(f"Входной фрейм #{i}: размер={len(in_frames[i])} байт, энергия={in_energy:.2f}")
            
            processed_frame = aec_session.process_frame(in_frames[i])
            
            # Подробное логирование обработанного фрейма
            if i == 0 or i % frame_log_interval == 0:
                proc_data_np = np.frombuffer(processed_frame, dtype=np.int16)
                proc_energy = np.sum(proc_data_np.astype(np.float32)**2) / len(proc_data_np)
                logging.info(f"Обработанный фрейм #{i}: размер={len(processed_frame)} байт, энергия={proc_energy:.2f}")
                
                # Если это первый фрейм, добавим более подробную информацию
                if i == 0:
                    if n_channels > 1:
                        proc_channels_data = proc_data_np.reshape(-1, n_channels)
                        proc_ch1_min, proc_ch1_max = np.min(proc_channels_data[:,0]), np.max(proc_channels_data[:,0])
                        proc_ch2_min, proc_ch2_max = np.min(proc_channels_data[:,1]), np.max(proc_channels_data[:,1])
                        logging.info(f"Обработанный фрейм: Канал 1 [мин={proc_ch1_min}, макс={proc_ch1_max}], Канал 2 [мин={proc_ch2_min}, макс={proc_ch2_max}]")
                    else:
                        proc_min, proc_max = np.min(proc_data_np), np.max(proc_data_np)
                        logging.info(f"Обработанный фрейм: [мин={proc_min}, макс={proc_max}]")
            
            processed_frames.append(processed_frame)
            
            # Логируем прогресс каждые 100 фреймов
            if i % 100 == 0 or i == min_frames - 1:
                logging.info(f"Обработано {i+1}/{min_frames} фреймов ({(i+1)/min_frames*100:.1f}%)")
        
        # Объединяем обработанные фреймы в итоговый сигнал
        processed_data = b''.join(processed_frames)
        
        # Получаем финальную статистику
        final_stats = aec_session.get_statistics()
        logging.info(f"Статистика обработки AEC: обработано {final_stats['processed_frames']} фреймов")
        
        if 'echo_frames' in final_stats:
            echo_frames = final_stats["echo_frames"]
            echo_percentage = (echo_frames / final_stats["processed_frames"] * 100) if final_stats["processed_frames"] > 0 else 0
            logging.info(f"Фреймов с обнаруженным эхо: {echo_frames} ({echo_percentage:.2f}%)")
        
        # Сохраняем обработанный сигнал
        with wave.open(processed_file_path, 'wb') as out_wf:
            out_wf.setnchannels(original_input_file_info['n_channels'])
            out_wf.setsampwidth(2)  # 16 бит = 2 байта
            out_wf.setframerate(original_input_file_info['frame_rate'])
            out_wf.writeframes(processed_data)
            
        logging.info(f"Обработанный файл сохранен как {processed_file_path}")

        logging.info("\n>>> Анализ файла с обработанным аудио <<<")
        processed_file_info = log_file_info(processed_file_path, "Файл с обработанным аудио")
        if not processed_file_info:
            logging.error(f"Не удалось получить информацию о файле с обработанным аудио {processed_file_path}")
            return {}

        # Расчёт основных метрик
        metrics = {
            "echo_frames": final_stats.get("echo_frames", 0),
            "echo_percentage": echo_percentage,
            "processed_frames": final_stats.get("processed_frames", 0),
            "delay_samples": delay_samples,
            "delay_ms": delay_ms,
            "delay_confidence": confidence,
        }
        
        # # Добавляем данные корреляции в метрики для визуализации
        # if correlation_data:
        #     metrics.update({
        #         "delay_correlation": correlation_data.get('correlation'),
        #         "delay_lags": correlation_data.get('lags'),
        #     })
        #     logging.info(f"Данные корреляции добавлены в метрики для визуализации")
        
        # Примечание: Дополнительный расчёт метрик качества (ERLE и другие)
        # теперь выполняется отдельным скриптом metrics.py
        logging.info("Для расчета дополнительных метрик (ERLE и др.) используйте скрипт metrics.py")

        # Визуализация результатов обработки
        if visualize:
            logging.info("Создание визуализаций...")
            try:
                from visualization import visualize_audio_processing
                
                # Создаем директорию для результатов, если ее нет
                os.makedirs(output_dir, exist_ok=True)

                # Затем вызываем visualize_audio_processing
                visualize_audio_processing(
                    output_dir=output_dir,
                    reference_file_info=reference_file_info,
                    reference_by_micro_volumed_file_info=reference_by_micro_volumed_file_info,
                    reference_by_micro_volumed_delayed_file_info=reference_by_micro_volumed_delayed_file_info,
                    original_input_file_info=original_input_file_info,
                    processed_file_info=processed_file_info,
                    frame_rate=frame_rate,
                    n_channels=n_channels,
                    metrics=metrics  # передаем метрики с данными корреляции
                )
                
                logging.info(f"Визуализации созданы в директории {output_dir}")

            except ImportError:
                logging.warning("Не удалось импортировать модуль визуализации.")
            except Exception as e:
                logging.error(f"Ошибка при создании визуализаций: {e}")
                logging.exception("Подробная информация об ошибке:")

        return metrics, processed_file_info
        
    except Exception as e:
        logging.error(f"Ошибка при обработке с AEC: {e}")
        logging.exception("Подробная информация об ошибке:")
        return {}

def extract_audio_file_info(base_path, audio_file_name: str, description: str = "") -> Dict[str, Any]:
        logging.info(f"\n>>> Анализ файла: {description} <<<")
        audio_file_path = os.path.join(base_path, audio_file_name)        
        audio_file_info = {}
        if os.path.exists(audio_file_path):
            audio_file_info = log_file_info(audio_file_path, description)
        else:
            logging.error(f"Файл {audio_file_path} не найден")

        return audio_file_info

def validate_audio_files(
    reference_file_info,
    original_input_file_info,
    reference_by_micro_volumed_file_info,
    reference_by_micro_volumed_delayed_file_info
):
    """
    Проверяет совместимость критически важных параметров аудиофайлов для алгоритма AEC.
    
    Args:
        reference_file_info: Информация о референсном файле
        original_input_file_info: Информация о входном файле
        reference_by_micro_volumed_file_info: Информация о референсном файле с измененной громкостью
        reference_by_micro_volumed_delayed_file_info: Информация о задержанном референсном файле
    
    Returns:
        bool: True если параметры совместимы, False в противном случае
    """
    logging.info("=== Валидация параметров аудиофайлов ===")
    
    # Словарь для хранения общих проблем
    validation_issues = {
        "critical": [],    # Критические проблемы, делающие обработку невозможной
        "warnings": [],    # Предупреждения, обработка возможна, но с искажениями
        "info": []         # Информационные сообщения
    }
    
    # 1. Проверка частоты дискретизации (sample rate / frame rate)
    frame_rates = [
        reference_file_info['frame_rate'],
        original_input_file_info['frame_rate'],
        reference_by_micro_volumed_file_info['frame_rate'],
        reference_by_micro_volumed_delayed_file_info['frame_rate']
    ]
    
    if len(set(frame_rates)) > 1:
        validation_issues["critical"].append(f"Частоты дискретизации не совпадают: {frame_rates}")
    else:
        validation_issues["info"].append(f"Частоты дискретизации совпадают: {frame_rates[0]} Гц")
    
    # 2. Проверка разрядности (sample width)
    sample_widths = [
        reference_file_info['sample_width'],
        original_input_file_info['sample_width'],
        reference_by_micro_volumed_file_info['sample_width'],
        reference_by_micro_volumed_delayed_file_info['sample_width']
    ]
    
    if len(set(sample_widths)) > 1:
        validation_issues["critical"].append(f"Разрядности не совпадают: {sample_widths}. Возможны искажения.")
    else:
        validation_issues["info"].append(f"Разрядности совпадают: {sample_widths[0]} байт")
    
    # 3. Проверка количества каналов
    channels = [
        reference_file_info['n_channels'],
        original_input_file_info['n_channels'],
        reference_by_micro_volumed_file_info['n_channels'],
        reference_by_micro_volumed_delayed_file_info['n_channels']
    ]
    
    if len(set(channels)) > 1:
        validation_issues["critical"].append(f"Количество каналов отличается: {channels}")
    else:
        validation_issues["info"].append(f"Количество каналов совпадает: {channels[0]}")
    
    # 4. Проверка длительности
    # Допустимая разница в длительности (1500 мс)
    duration_threshold_ms = 1500
    
    # Проверка между входным файлом и референсным
    ref_in_duration_diff = abs(reference_file_info['duration_ms'] - original_input_file_info['duration_ms'])
    if ref_in_duration_diff > duration_threshold_ms:
        validation_issues["warnings"].append(
            f"Существенная разница в длительности между референсным и входным файлами: "
            f"{ref_in_duration_diff:.2f} мс > {duration_threshold_ms} мс"
        )
    
    # Проверка разницы длительности между обычным и задержанным референсным файлами
    # Это нормально если задержанный файл длиннее
    ref_delayed_diff = reference_by_micro_volumed_delayed_file_info['duration_ms'] - reference_by_micro_volumed_file_info['duration_ms']
    expected_diff_threshold = 1000  # Ожидаем задержку не более 1000 мс
    
    if ref_delayed_diff < 0:
        validation_issues["warnings"].append(
            f"Задержанный референсный файл короче обычного: {ref_delayed_diff:.2f} мс"
        )
    elif ref_delayed_diff > expected_diff_threshold:
        validation_issues["info"].append(
            f"Задержка между референсными файлами: {ref_delayed_diff:.2f} мс (> {expected_diff_threshold} мс)"
        )
    else:
        validation_issues["info"].append(
            f"Задержка между референсными файлами: {ref_delayed_diff:.2f} мс"
        )
    
    # 5. Проверка амплитудных характеристик, если доступны
    has_amplitude_stats = all('amplitude_stats' in file_info for file_info in [
        reference_file_info, original_input_file_info, 
        reference_by_micro_volumed_file_info, reference_by_micro_volumed_delayed_file_info
    ])
    
    if has_amplitude_stats:
        # Проверяем отношение RMS между оригинальным и измененным референсным файлами
        ref_rms = reference_file_info['amplitude_stats'][0]['rms']
        ref_vol_rms = reference_by_micro_volumed_file_info['amplitude_stats'][0]['rms']
        vol_ratio = ref_vol_rms / ref_rms if ref_rms > 0 else 0
        
        # Проверяем, находится ли соотношение громкости в ожидаемом диапазоне (0.1-10)
        if vol_ratio < 0.1 or vol_ratio > 10:
            validation_issues["warnings"].append(
                f"Необычное соотношение громкости между референсным и измененным референсным: {vol_ratio:.2f}x"
            )
        else:
            validation_issues["info"].append(
                f"Соотношение громкости между референсным и измененным: {vol_ratio:.2f}x"
            )
        
        # Проверяем, имеют ли файлы разумный динамический диапазон
        for idx, file_info in enumerate([reference_file_info, original_input_file_info, 
                                     reference_by_micro_volumed_file_info, 
                                     reference_by_micro_volumed_delayed_file_info]):
            file_name = ["референсный", "входной", "референсный с громкостью", "задержанный референсный"][idx]
            rms = file_info['amplitude_stats'][0]['rms']
            if rms < 100:  # Очень тихий сигнал
                validation_issues["warnings"].append(
                    f"Очень тихий сигнал в {file_name} файле: RMS={rms:.2f}"
                )
    
    # Вывод результатов проверки
    if validation_issues["critical"]:
        logging.error("Критические проблемы совместимости аудиофайлов:")
        for issue in validation_issues["critical"]:
            logging.error(f"  ● {issue}")
        logging.error("Обработка может не дать корректных результатов!")
    
    if validation_issues["warnings"]:
        logging.warning("Предупреждения о совместимости аудиофайлов:")
        for issue in validation_issues["warnings"]:
            logging.warning(f"  ● {issue}")
    
    if validation_issues["info"]:
        logging.info("Информация о совместимости аудиофайлов:")
        for issue in validation_issues["info"]:
            logging.info(f"  ● {issue}")
    
    # Если есть критические проблемы, возвращаем False
    success = len(validation_issues["critical"]) == 0
    logging.info(f"Валидация {'успешно завершена' if success else 'не пройдена'}")
    return success

def process_test_directory(dir_path, output_dir=None, frame_size_ms=10.0, visualize=True, verbose=False):
    """
    Обрабатывает одну тестовую директорию
    
    Args:
        dir_path: Путь к тестовой директории
        output_dir: Директория для сохранения результатов (если None, используется dir_path)
        frame_size_ms: Размер фрейма в мс для AEC
        visualize: Создавать ли визуализации
        verbose: Подробный вывод
        
    Returns:
        dict: Метрики и результаты обработки
    """
    # Задаем уровень логирования в зависимости от verbose
    log_level = logging.info if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # Всегда сохраняем метрики в той же директории, где находятся аудиофайлы
    output_dir = dir_path
    
    # Настраиваем файловый логгер для конкретной директории
    dir_log_file = os.path.join(output_dir, "aec_processing.log")
    file_handler = logging.FileHandler(dir_log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(log_level)
    
    # Добавляем обработчик к корневому логгеру
    root_logger = logging.getLogger()
    # Удаляем предыдущие файловые обработчики, если они были
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename != "batch_aec_test.log":
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Начало обработки директории: {dir_path}")
    logging.info(f"Результаты будут сохранены в: {output_dir}")
    
    # Используем utils.py для подробного анализа всех WAV файлов
    logging.info("\n=== Детальный анализ входных файлов ===")

    # Проверяем наличие необходимых файлов    
    reference_file_info = extract_audio_file_info(dir_path, "reference.wav", description="Референсный файл")
    original_input_file_info = extract_audio_file_info(dir_path, "original_input.wav", description="Входной файл")
    reference_by_micro_volumed_file_info = extract_audio_file_info(dir_path, "reference_by_micro_volumed.wav", description="Файл с референсным аудио на входе в микрофон (с измененной громкостью)")
    reference_by_micro_volumed_delayed_file_info = extract_audio_file_info(dir_path, "reference_by_micro_volumed_delayed.wav", description="Файл с референсным аудио на входе в микрофон (с измененной громкостью и задержкой)")

    if not validate_audio_files(
        reference_file_info,
        original_input_file_info,
        reference_by_micro_volumed_file_info,
        reference_by_micro_volumed_delayed_file_info
    ):
        logging.error("Ошибка валидации аудиофайлов. Проверьте файлы и повторите попытку.")
        return None

    # Определяем имя выходного файла
    processed_file_path = os.path.join(output_dir, "processed_input.wav")
    
    # Запускаем обработку с помощью AEC
    try:
        # Обрабатываем аудио с нашей оптимизированной функцией
        metrics, processed_file_info = process_audio_with_aec(
            processed_file_path,
            reference_file_info,
            original_input_file_info,
            reference_by_micro_volumed_file_info,
            reference_by_micro_volumed_delayed_file_info,
            frame_rate=original_input_file_info['frame_rate'],
            n_channels=original_input_file_info['n_channels'],
            visualize=visualize,
            output_dir=output_dir,
            frame_size_ms=frame_size_ms
        )

        # Проверяем длительность выходного файла
        if os.path.exists(processed_file_path):                    
            # Проверяем, не сильно ли отличается длительность
            if abs(processed_file_info['duration_sec'] - original_input_file_info['duration_sec']) > 0.5:  # допуск 0.5 сек
                logging.warning(f"Длительность выходного файла ({processed_file_info['duration_ms']:.2f} мс) существенно отличается от входного ({original_input_file_info['duration_ms']:.2f} мс)")
        else:
            logging.error(f"Выходной файл {processed_file_path} не был создан!")

        # Сохраняем метрики в JSON файл
        metrics_file = os.path.join(output_dir, "aec_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyJSONEncoder)
        
        logging.info(f"Метрики сохранены в {metrics_file}")
        logging.info(f"Обработка директории {dir_path} завершена успешно")
        
        # Удаляем файловый обработчик для этой директории
        root_logger.removeHandler(file_handler)
        file_handler.close()
        
        return metrics
    
    except Exception as e:
        logging.error(f"Ошибка при обработке директории {dir_path}: {e}")
        logging.exception("Подробная информация об ошибке:")
        
        # Удаляем файловый обработчик для этой директории
        root_logger.removeHandler(file_handler)
        file_handler.close()
        
        return None

def process_all_tests(test_dirs, args):
    """
    Обрабатывает все найденные директории с тестами
    
    Args:
        test_dirs: Список директорий с тестами
        args: Аргументы командной строки
        
    Returns:
        dict: Результаты тестов
    """
    results = {}
    
    # Проходим по всем тестовым директориям
    for main_dir, sub_dirs in test_dirs.items():
        # Обрабатываем поддиректории
        for sub_dir in sub_dirs:
            # Всегда сохраняем результаты в исходной директории тестов
            output_dir = sub_dir
            
            # Обрабатываем директорию
            metrics = process_test_directory(
                sub_dir, 
                output_dir, 
                args.frame_size_ms, 
                args.visualize, 
                args.verbose
            )
            
            if metrics:
                # Сохраняем результаты
                results[sub_dir] = metrics
    
    # # Сохраняем сводный JSON файл с результатами в корневой директории тестов
    # results_file = os.path.join(args.tests_dir, "aec_test_results.json")
    
    # try:
    #     with open(results_file, 'w') as f:
    #         json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
    #     logging.info(f"Сводные результаты тестов сохранены в {results_file}")
    # except Exception as e:
    #     logging.error(f"Ошибка при сохранении сводных результатов: {e}")
    
    logging.info("Обработка всех тестовых директорий завершена")
    return results

def process_test_directories(args):
    """
    Обрабатывает тестовые директории в соответствии с аргументами командной строки.
    
    Args:
        args: Аргументы командной строки
    """
    # Если указана конкретная директория для тестирования
    if args.test_dir:
        if os.path.exists(args.test_dir):
            logging.info(f"Обработка указанной директории: {args.test_dir}")
            
            # Обрабатываем директории и сохраняем результаты прямо в них
            results = process_directory_by_level(args.test_dir, args)
            
            # Сохраняем сводные результаты в JSON файл в указанной директории
            results_file = os.path.join(args.test_dir, "aec_test_results.json")
            try:
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
                logging.info(f"Сводные результаты тестов сохранены в {results_file}")
            except Exception as e:
                logging.error(f"Ошибка при сохранении сводных результатов: {e}")
            
        else:
            logging.error(f"Указанная директория не существует: {args.test_dir}")
            sys.exit(1)
    else:
        # Иначе обрабатываем все директории
        logging.info(f"Поиск тестовых директорий в {args.tests_dir}")
        
        # Обрабатываем директории и сохраняем результаты прямо в них
        results = process_all_tests(find_test_directories(args.tests_dir), args)
        
        # Сохраняем результаты в корневой директории тестов
        results_file = os.path.join(args.tests_dir, "aec_test_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
            logging.info(f"Сводные результаты тестов сохранены в {results_file}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении сводных результатов: {e}")

def process_directory_by_level(dir_path, args):
    """
    Определяет уровень вложенности тестовой директории и обрабатывает её соответствующим образом.
    
    Args:
        dir_path: Путь к тестовой директории
        args: Аргументы командной строки
        
    Returns:
        dict: Словарь с результатами обработки
    """
    dir_name = os.path.basename(dir_path)
    parent_dir = os.path.basename(os.path.dirname(dir_path))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
    
    logging.info(f"Анализ директории: {dir_path}")
    logging.info(f"Имя директории: {dir_name}, родительская: {parent_dir}, прародительская: {grandparent_dir}")
    
    # Словарь для хранения результатов обработки
    results = {}
    
    # Проверяем, является ли текущая директория директорией типа mono или stereo
    if dir_name in ["mono", "stereo"]:
        logging.info(f"Обнаружена директория для канального формата: {dir_path}")
        
        # Устанавливаем количество каналов
        n_channels = 1 if dir_name == "mono" else 2
        logging.info(f"Установлено количество каналов: {n_channels}")
        
        # Ищем поддиректории reference_by_micro и clear_reference
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path) and item in ["reference_by_micro", "clear_reference"]:
                logging.info(f"Обрабатываем поддиректорию {item}: {item_path}")
                # Рекурсивно вызываем для обработки reference_by_micro и clear_reference директорий
                subdir_results = process_directory_by_level(item_path, args)
                # Объединяем результаты
                results.update(subdir_results)
    
    # Проверяем, является ли текущая директория директорией с уровнем громкости (volume_XX)
    elif dir_name.startswith("volume_"):
        logging.info(f"Обнаружена директория с уровнем громкости: {dir_path}")
        # Всегда сохраняем результаты в той же директории, где находятся файлы
        output_dir = dir_path
        metrics = process_test_directory(dir_path, output_dir, 
                                        args.frame_size_ms, args.visualize, args.verbose)
        if metrics:
            results[dir_path] = metrics
        
    # Проверяем, является ли текущая директория директорией с задержкой (delay_XX)
    elif dir_name.startswith("delay_"):
        logging.info(f"Обнаружена директория с задержкой: {dir_path}")
        # Ищем поддиректории с уровнями громкости
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path) and item.startswith("volume_"):
                logging.info(f"Обрабатываем поддиректорию с уровнем громкости: {item_path}")
                # Всегда сохраняем результаты в исходной директории
                output_dir = item_path
                
                metrics = process_test_directory(item_path, output_dir, 
                                               args.frame_size_ms, args.visualize, args.verbose)
                if metrics:
                    results[item_path] = metrics
        
    # Проверяем, является ли текущая директория директорией reference_by_micro или clear_reference
    elif dir_name in ["reference_by_micro", "clear_reference"]:
        logging.info(f"Обнаружена директория {dir_name}")
        # Для clear_reference ищем директории с уровнями громкости
        if dir_name == "clear_reference":
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path) and item.startswith("volume_"):
                    logging.info(f"Обрабатываем поддиректорию с уровнем громкости: {item_path}")
                    # Всегда сохраняем результаты в исходной директории
                    output_dir = item_path
                    
                    metrics = process_test_directory(item_path, output_dir, 
                                                   args.frame_size_ms, args.visualize, args.verbose)
                    if metrics:
                        results[item_path] = metrics
        # Для reference_by_micro ищем директории с задержками
        else:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path) and item.startswith("delay_"):
                    logging.info(f"Обрабатываем поддиректорию с задержкой: {item_path}")
                    # Рекурсивно вызываем для обработки delay_XX директорий
                    delay_results = process_directory_by_level(item_path, args)
                    # Объединяем результаты
                    results.update(delay_results)
        
    # Проверяем, является ли текущая директория корневой директорией тестового сценария
    elif dir_name in ["music", "agent_speech", "agent_user_speech", "agent_speech_30_sec"]:
        logging.info(f"Обнаружена корневая директория тестового сценария: {dir_path}")
        
        # Проверяем наличие mono/stereo директорий для определенных типов тестов
        if dir_name in ["agent_speech", "agent_user_speech", "agent_speech_30_sec"]:
            # Сначала проверяем наличие mono/stereo поддиректорий
            channel_dirs = ["mono", "stereo"]
            has_channel_dirs = False
            
            for channel_dir in channel_dirs:
                channel_dir_path = os.path.join(dir_path, channel_dir)
                if os.path.exists(channel_dir_path) and os.path.isdir(channel_dir_path):
                    has_channel_dirs = True
                    logging.info(f"Обрабатываем поддиректорию {channel_dir}: {channel_dir_path}")
                    # Рекурсивно вызываем для обработки mono/stereo директорий
                    channel_results = process_directory_by_level(channel_dir_path, args)
                    # Объединяем результаты
                    results.update(channel_results)
                    
            # Если нет mono/stereo поддиректорий, ищем стандартные reference_by_micro и clear_reference
            if not has_channel_dirs:
                # Ищем поддиректории reference_by_micro и clear_reference
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path) and item in ["reference_by_micro", "clear_reference"]:
                        logging.info(f"Обрабатываем поддиректорию {item}: {item_path}")
                        # Рекурсивно вызываем для обработки reference_by_micro и clear_reference директорий
                        subdir_results = process_directory_by_level(item_path, args)
                        # Объединяем результаты
                        results.update(subdir_results)
        else:
            # Для директорий без mono/stereo структуры (например, music)
            # Ищем поддиректории reference_by_micro и clear_reference
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path) and item in ["reference_by_micro", "clear_reference"]:
                    logging.info(f"Обрабатываем поддиректорию {item}: {item_path}")
                    # Рекурсивно вызываем для обработки reference_by_micro и clear_reference директорий
                    subdir_results = process_directory_by_level(item_path, args)
                    # Объединяем результаты
                    results.update(subdir_results)
    else:
        # Если директория не соответствует известной структуре, пробуем обработать как обычную директорию
        logging.warning(f"Директория {dir_path} не соответствует известной структуре тестов")
        logging.warning("Попытка обработать как обычную тестовую директорию")
        
        # Проверяем наличие необходимых файлов
        reference_file_path_path = os.path.join(dir_path, "reference.wav")
        original_input_file_path = os.path.join(dir_path, "original_input.wav")
        
        if os.path.exists(reference_file_path_path) and os.path.exists(original_input_file_path):
            logging.info(f"Найдены необходимые файлы в директории {dir_path}, обрабатываем")
            # Всегда сохраняем результаты в исходной директории
            output_dir = dir_path
            
            metrics = process_test_directory(dir_path, output_dir, 
                                           args.frame_size_ms, args.visualize, args.verbose)
            if metrics:
                results[dir_path] = metrics
        else:
            logging.warning(f"В директории {dir_path} не найдены необходимые файлы для обработки")
            # Проверяем поддиректории
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    # Проверяем наличие файлов в поддиректории
                    ref_file = os.path.join(item_path, "reference.wav")
                    in_file = os.path.join(item_path, "original_input.wav")
                    if os.path.exists(ref_file) and os.path.exists(in_file):
                        logging.info(f"Найдены необходимые файлы в поддиректории {item_path}, обрабатываем")
                        # Всегда сохраняем результаты в исходной директории
                        output_dir = item_path
                        
                        metrics = process_test_directory(item_path, output_dir, 
                                                       args.frame_size_ms, args.visualize, args.verbose)
                        if metrics:
                            results[item_path] = metrics
    
    return results
    
def main():
    parser = argparse.ArgumentParser(description="Пакетная обработка тестовых данных с WebRTC AEC")
    parser.add_argument("--tests-dir", "-t", default="tests", 
                      help="Директория с тестовыми данными (по умолчанию: tests)")
    parser.add_argument("--test-dir", "-d", default=None,
                      help="Конкретная директория для тестирования (если указана, обрабатывается только она). "
                           "Может быть любого уровня вложенности: agent_user_speech, reference_by_micro, "
                           "delay_XX или volume_XX. Скрипт автоматически определит уровень и обработает соответствующие данные.")
    parser.add_argument("--results-dir", "-r", default=None, 
                      help="Директория для сохранения результатов (не используется, результаты сохраняются в исходных директориях)")
    parser.add_argument("--visualize", "-v", action="store_true", default=True,
                      help="Создавать визуализации (по умолчанию: включено)")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                      help="Не создавать визуализации")
    parser.add_argument("--frame-size-ms", "-f", type=float, default=10.0,
                      help="Размер фрейма в миллисекундах (по умолчанию: 10.0)")
    parser.add_argument("--verbose", action="store_true", default=False,
                      help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    log_level = logging.info if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Запуск пакетной обработки тестовых данных с WebRTC AEC")
    logging.info("Примечание: Результаты будут сохранены в исходных директориях с тестами")
    logging.info("Для расчета дополнительных метрик, используйте скрипт metrics.py")
    logging.info("Для создания отчетов и графиков, используйте скрипт report.py")
    
    # Обрабатываем директории в соответствии с аргументами
    process_test_directories(args)
    
    logging.info("Обработка завершена")
    logging.info("Базовые метрики сохранены в файлах aec_metrics.json в соответствующих директориях")
    logging.info("Далее можно запустить: python metrics.py --test-dir <директория_с_тестами>")

if __name__ == "__main__":
    main() 