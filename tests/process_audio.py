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

# Импортируем функции из aec_test_tool.py
try:
    from aec_test_tool import calculate_metrics
    from visualization import visualize_audio_processing
    logging.info("Функции из aec_test_tool.py успешно импортированы")
except ImportError:
    logging.error("Не удалось импортировать функции из aec_test_tool.py")
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
    main_test_dirs = ["music", "agent_speech", "agent_user_speech"]
    
    for main_dir in main_test_dirs:
        main_dir_path = os.path.join(base_dir, main_dir)
        if not os.path.exists(main_dir_path):
            logging.warning(f"Директория {main_dir_path} не существует, пропускаем")
            continue
        
        # Ищем поддиректории типа clear_reference и reference_by_micro
        ref_types = ["clear_reference", "reference_by_micro"]
        found_sub_dirs = []
        
        for ref_type in ref_types:
            ref_type_path = os.path.join(main_dir_path, ref_type)
            if not os.path.exists(ref_type_path):
                logging.warning(f"Директория {ref_type_path} не существует, пропускаем")
                continue
            
            if ref_type == "clear_reference":
                # Для clear_reference ищем директории с уровнями громкости (volume_XX)
                for d in os.listdir(ref_type_path):
                    sub_dir_path = os.path.join(ref_type_path, d)
                    if os.path.isdir(sub_dir_path) and d.startswith("volume_"):
                        found_sub_dirs.append(sub_dir_path)
                        logging.debug(f"Найдена директория: {sub_dir_path}")
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
                                logging.debug(f"Найдена директория: {vol_dir_path}")
        
        if found_sub_dirs:
            test_dirs[main_dir] = found_sub_dirs
            logging.info(f"Найдено {len(found_sub_dirs)} поддиректорий в {main_dir}")
        else:
            logging.warning(f"В директории {main_dir_path} не найдено поддиректорий с тестовыми данными")
    
    return test_dirs

def process_audio_with_aec(input_file, output_file, reference_file, sample_rate=16000, channels=1, 
                           visualize=True, output_dir="results", frame_size_ms=10.0):
    """
    Обрабатывает аудиофайлы с использованием AEC и сохраняет результаты.
    
    Args:
        input_file: Путь к входному файлу
        output_file: Путь к файлу для сохранения результата
        reference_file: Путь к файлу с референсным сигналом
        sample_rate: Частота дискретизации
        channels: Количество каналов (1 или 2)
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
        logging.info(f"Обработка файла {input_file} с помощью AEC...")
        
        # Чтение референсного файла и определение его параметров
        with wave.open(reference_file, 'rb') as ref_wf:
            ref_data = ref_wf.readframes(ref_wf.getnframes())
            ref_frames_count = ref_wf.getnframes()
            ref_rate = ref_wf.getframerate()
            ref_channels = ref_wf.getnchannels()
            ref_duration_ms = ref_frames_count * 1000 / ref_rate
            logging.info(f"Референсный файл: длительность {ref_duration_ms:.2f} мс ({ref_duration_ms/1000:.2f} с), {ref_rate} Гц, {ref_channels} канала(ов)")
            
        # Чтение входного файла и определение его параметров
        with wave.open(input_file, 'rb') as in_wf:
            in_data = in_wf.readframes(in_wf.getnframes())
            in_frames_count = in_wf.getnframes()
            in_rate = in_wf.getframerate()
            in_channels = in_wf.getnchannels()
            in_duration_ms = in_frames_count * 1000 / in_rate
            logging.info(f"Входной файл: длительность {in_duration_ms:.2f} мс ({in_duration_ms/1000:.2f} с), {in_rate} Гц, {in_channels} канала(ов)")
        
        # Определяем, нужно ли конвертировать моно в стерео или наоборот
        # Для WebRTC AEC используем количество каналов из входного файла
        channels = in_channels
        logging.info(f"Используем количество каналов: {channels}")
        
        # Попытка чтения файла reference_by_micro_volumed.wav
        ref_vol_file = os.path.join(os.path.dirname(reference_file), "reference_by_micro_volumed.wav")
        ref_vol_data = None
        
        if os.path.exists(ref_vol_file):
            try:
                with wave.open(ref_vol_file, 'rb') as ref_vol_wf:
                    ref_vol_data = ref_vol_wf.readframes(ref_vol_wf.getnframes())
                    ref_vol_frames_count = ref_vol_wf.getnframes()
                    ref_vol_rate = ref_vol_wf.getframerate()
                    ref_vol_duration_ms = ref_vol_frames_count * 1000 / ref_vol_rate
                    logging.info(f"Референсный файл с измененной громкостью: длительность {ref_vol_duration_ms:.2f} мс ({ref_vol_duration_ms/1000:.2f} с), {ref_vol_rate} Гц")
            except Exception as e:
                logging.warning(f"Не удалось прочитать файл reference_by_micro_volumed.wav: {e}")
                ref_vol_data = None
        
        # Попытка чтения задержанного референсного файла (reference_by_micro_volumed_delayed.wav)
        reference_delayed_file = os.path.join(os.path.dirname(reference_file), "reference_by_micro_volumed_delayed.wav")
        ref_delayed_data = None
        
        if os.path.exists(reference_delayed_file):
            try:
                with wave.open(reference_delayed_file, 'rb') as ref_delayed_wf:
                    ref_delayed_data = ref_delayed_wf.readframes(ref_delayed_wf.getnframes())
                    ref_delayed_frames_count = ref_delayed_wf.getnframes()
                    ref_delayed_rate = ref_delayed_wf.getframerate()
                    ref_delayed_duration_ms = ref_delayed_frames_count * 1000 / ref_delayed_rate
                    logging.info(f"Задержанный референсный файл: длительность {ref_delayed_duration_ms:.2f} мс ({ref_delayed_duration_ms/1000:.2f} с), {ref_delayed_rate} Гц")
                    
                    # Вычисляем задержку как разницу длительностей
                    delay_diff_ms = ref_delayed_duration_ms - ref_duration_ms
                    logging.info(f"Разница длительностей между референсным и задержанным референсным файлами: {delay_diff_ms:.2f} мс")
            except Exception as e:
                logging.warning(f"Не удалось прочитать задержанный референсный файл: {e}")
                ref_delayed_data = None
        
        # Если количество каналов не совпадает, конвертируем данные
        if ref_channels != in_channels:
            logging.warning(f"Количество каналов в файлах не совпадает: {ref_channels} vs {in_channels}")
            logging.warning("Выполняется конвертация каналов...")
            
            # Для корректной работы WebRTC AEC, приводим оба файла к одинаковому количеству каналов
            # Это надо делать здесь, а не внутри WebRTC, иначе будут ошибки
        
        # Проверяем частоты дискретизации
        if ref_rate != in_rate:
            logging.warning(f"Частоты дискретизации файлов не совпадают: {ref_rate} vs {in_rate}")
            logging.warning("Для корректной работы AEC необходимы одинаковые частоты дискретизации")
            
            # В этом случае нужно выполнить ресемплинг одного из файлов
            # Это делается вне этой функции, в процессе подготовки данных
            
        # Проверяем количество фреймов
        if ref_frames_count != in_frames_count:
            logging.warning(f"Количество фреймов в файлах не совпадает: {ref_frames_count} vs {in_frames_count}")
            logging.warning("Для корректной работы AEC будет использована минимальная длина")
        
        # Преобразуем байты в NumPy массивы
        ref_np = np.frombuffer(ref_data, dtype=np.int16)
        in_np = np.frombuffer(in_data, dtype=np.int16)
        
        # Настраиваем AEC и масштабируем сигналы
        frame_size = int(sample_rate * frame_size_ms / 1000)
        
        # Инициализация сессии AEC
        aec_session = WebRTCAECSession(
            session_id=f"batch_{os.path.basename(input_file)}",
            sample_rate=sample_rate,
            channels=channels,
            batch_mode=True,
            frame_size_ms=frame_size_ms
        )
        
        # Оценка и установка задержки
        delay_samples, delay_ms, confidence = aec_session.auto_set_delay(ref_data, in_data)
        logging.info(f"Обнаружена задержка: {delay_samples} семплов ({delay_ms:.2f} мс), уверенность: {confidence:.4f}")
        
        # Получаем данные корреляции из последнего вычисления, если такой метод существует
        correlation_data = {}
        if hasattr(aec_session, 'get_last_correlation_data'):
            correlation_data = aec_session.get_last_correlation_data()
            logging.info(f"Получены данные корреляции из AEC-сессии для передачи в визуализацию")
        
        # Оптимизация параметров AEC для лучшего качества
        aec_session.optimize_for_best_quality()
        
        # Подготовка к пофреймовой обработке
        frame_size_bytes = frame_size * 2 * channels  # 2 байта на сэмпл (16 бит)
        
        # Разделение на фреймы
        ref_frames = [ref_data[i:i+frame_size_bytes] for i in range(0, len(ref_data), frame_size_bytes)]
        in_frames = [in_data[i:i+frame_size_bytes] for i in range(0, len(in_data), frame_size_bytes)]
        
        min_frames = min(len(ref_frames), len(in_frames))
        logging.info(f"Количество фреймов: референс={len(ref_frames)}, вход={len(in_frames)}, обработка={min_frames}")
        
        # Буферизация для компенсации задержки
        delay_frames = int(delay_samples / frame_size)
        pre_buffer_size = max(5, delay_frames) # Минимум 5 фреймов для предварительной буферизации
        
        # Предварительная буферизация референсных фреймов
        for i in range(min(pre_buffer_size, len(ref_frames))):
            aec_session.add_reference_frame(ref_frames[i])
        
        # Обработка фреймов
        processed_frames = []
        
        for i in range(min_frames):
            # Добавляем референсный фрейм с учетом задержки
            ref_idx = i + pre_buffer_size
            if ref_idx < len(ref_frames):
                aec_session.add_reference_frame(ref_frames[ref_idx])
            
            # Обрабатываем входной фрейм
            processed_frame = aec_session.process_frame(in_frames[i])
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
        with wave.open(output_file, 'wb') as out_wf:
            out_wf.setnchannels(in_channels)
            out_wf.setsampwidth(2)  # 16 бит = 2 байта
            out_wf.setframerate(in_rate)
            out_wf.writeframes(processed_data)
            
        logging.info(f"Обработанный файл сохранен как {output_file}")
        
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
                
                # Логируем частоту дискретизации всех файлов
                logging.info("=== Проверка частоты дискретизации файлов ===")
                
                # Проверяем основные файлы
                logging.info(f"Используемая частота дискретизации для визуализации: {sample_rate} Гц")
                
                # Проверяем reference_by_micro_volumed.wav
                ref_vol_file = os.path.join(os.path.dirname(reference_file), "reference_by_micro_volumed.wav")
                if os.path.exists(ref_vol_file):
                    with wave.open(ref_vol_file, 'rb') as wf:
                        logging.info(f"reference_by_micro_volumed.wav: {wf.getframerate()} Гц, {wf.getnchannels()} канала(ов)")
                
                # Проверяем reference_by_micro_volumed_delayed.wav
                ref_delayed_file = os.path.join(os.path.dirname(reference_file), "reference_by_micro_volumed_delayed.wav")
                if os.path.exists(ref_delayed_file):
                    with wave.open(ref_delayed_file, 'rb') as wf:
                        logging.info(f"reference_by_micro_volumed_delayed.wav: {wf.getframerate()} Гц, {wf.getnchannels()} канала(ов)")
                
                # Проверяем original_input.wav
                orig_input_file = os.path.join(os.path.dirname(input_file), "original_input.wav")
                if os.path.exists(orig_input_file):
                    with wave.open(orig_input_file, 'rb') as wf:
                        logging.info(f"original_input.wav: {wf.getframerate()} Гц, {wf.getnchannels()} канала(ов)")
                
                # Проверяем my_voice.wav
                my_voice_file = os.path.join(os.path.dirname(input_file), "my_voice.wav")
                if os.path.exists(my_voice_file):
                    with wave.open(my_voice_file, 'rb') as wf:
                        logging.info(f"my_voice.wav: {wf.getframerate()} Гц, {wf.getnchannels()} канала(ов)")

                # Затем вызываем visualize_audio_processing
                visualize_audio_processing(
                    output_dir=output_dir,
                    reference_data=ref_data,
                    reference_file_path=reference_file,
                    reference_by_micro_volumed_data=ref_vol_data,
                    reference_by_micro_volumed_file_path=ref_vol_file,
                    reference_by_micro_volumed_delayed_data=ref_delayed_data,
                    reference_by_micro_volumed_delayed_file_path=reference_delayed_file,
                    input_data=in_data,
                    input_file_path=input_file,
                    processed_data=processed_data,
                    processed_file_path=output_file,
                    reference_delayed_data=ref_delayed_data,
                    reference_delayed_file_path=reference_delayed_file,
                    sample_rate=sample_rate,
                    channels=channels,
                    metrics=metrics  # передаем метрики с данными корреляции
                )
                
                logging.info(f"Визуализации созданы в директории {output_dir}")

            except ImportError:
                logging.warning("Не удалось импортировать модуль визуализации.")
            except Exception as e:
                logging.error(f"Ошибка при создании визуализаций: {e}")
                logging.exception("Подробная информация об ошибке:")

        return metrics
        
    except Exception as e:
        logging.error(f"Ошибка при обработке с AEC: {e}")
        logging.exception("Подробная информация об ошибке:")
        return {}

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
    log_level = logging.DEBUG if verbose else logging.INFO
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
    
    # Проверяем наличие необходимых файлов
    reference_file = os.path.join(dir_path, "reference.wav")
    input_file = os.path.join(dir_path, "original_input.wav")
    
    if not os.path.exists(reference_file):
        logging.error(f"Файл {reference_file} не найден")
        return None
    
    if not os.path.exists(input_file):
        logging.error(f"Файл {input_file} не найден")
        return None
    
    # Определяем имя выходного файла
    output_file = os.path.join(output_dir, "processed_input.wav")
    
    # Обрабатываем файлы с помощью AEC
    logging.info(f"Обработка директории: {dir_path}")
    logging.info(f"Референсный файл: {reference_file}")
    logging.info(f"Входной файл: {input_file}")
    logging.info(f"Выходной файл: {output_file}")
    
    # Получаем длительность входного файла для проверки после обработки
    input_duration = 0
    try:
        with wave.open(input_file, 'rb') as wf:
            input_frames = wf.getnframes()
            input_rate = wf.getframerate()
            input_duration = input_frames / input_rate
            input_duration_ms = input_frames * 1000 / input_rate
            logging.info(f"Длительность входного файла: {input_duration_ms:.2f} мс ({input_duration:.2f} с)")
    except Exception as e:
        logging.error(f"Ошибка при чтении входного файла: {e}")
    
    # Запускаем обработку с помощью AEC
    try:
        # Определяем sample_rate и channels из входного файла
        with wave.open(input_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            logging.info(f"Параметры аудио: {sample_rate}Hz, {channels} каналов")
        
        # Обрабатываем аудио с нашей оптимизированной функцией
        metrics = process_audio_with_aec(
            input_file,
            output_file,
            reference_file,
            sample_rate=sample_rate,
            channels=channels,
            visualize=visualize,
            output_dir=output_dir,
            frame_size_ms=frame_size_ms
        )
        
        # Проверяем длительность выходного файла
        if os.path.exists(output_file):
            try:
                with wave.open(output_file, 'rb') as wf:
                    output_frames = wf.getnframes()
                    output_rate = wf.getframerate()
                    output_duration = output_frames / output_rate
                    output_duration_ms = output_frames * 1000 / output_rate
                    logging.info(f"Длительность выходного файла: {output_duration_ms:.2f} мс ({output_duration:.2f} с)")
                    
                    # Проверяем, не сильно ли отличается длительность
                    if input_duration > 0 and abs(output_duration - input_duration) > 0.5:  # допуск 0.5 сек
                        logging.warning(f"Длительность выходного файла ({output_duration_ms:.2f} мс) существенно отличается от входного ({input_duration_ms:.2f} мс)")
                        if output_duration < input_duration * 0.8:  # если выходной файл меньше 80% входного
                            logging.error("Выходной файл значительно короче входного. Возможно, обработка была прервана!")
            except Exception as e:
                logging.error(f"Ошибка при проверке выходного файла: {e}")
        else:
            logging.error(f"Выходной файл {output_file} не был создан!")
        
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

def process_single_directory(test_dir, args):
    """
    Обрабатывает одну конкретную тестовую директорию.
    
    Args:
        test_dir: Путь к тестовой директории
        args: Аргументы командной строки
    """
    try:
        # Проверяем наличие необходимых файлов
        reference_file = os.path.join(test_dir, "reference.wav")
        input_file = os.path.join(test_dir, "original_input.wav")
        reference_delayed_file = os.path.join(test_dir, "reference_by_micro_volumed_delayed.wav")
        
        if not os.path.exists(reference_file):
            logging.warning(f"Файл {reference_file} не найден, пропускаем директорию")
            return
            
        if not os.path.exists(input_file):
            logging.warning(f"Файл {input_file} не найден, пропускаем директорию")
            return
        
        # Определяем директорию для результатов
        results_dir = args.results_dir if args.results_dir else test_dir
        
        # Запускаем обработку для этой директории
        process_test_directory(test_dir, results_dir, args.frame_size_ms, args.visualize, args.verbose)
        
        logging.info(f"Обработка директории {test_dir} завершена")
        
    except Exception as e:
        logging.error(f"Ошибка при обработке директории {test_dir}: {e}")
        logging.exception("Подробная информация об ошибке:")

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
    
    # Проверяем, является ли текущая директория директорией с уровнем громкости (volume_XX)
    if dir_name.startswith("volume_"):
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
        
    # Проверяем, является ли текущая директория корневой директорией тестового сценария (music, agent_speech, agent_user_speech)
    elif dir_name in ["music", "agent_speech", "agent_user_speech"]:
        logging.info(f"Обнаружена корневая директория тестового сценария: {dir_path}")
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
        reference_file = os.path.join(dir_path, "reference.wav")
        input_file = os.path.join(dir_path, "original_input.wav")
        
        if os.path.exists(reference_file) and os.path.exists(input_file):
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
    log_level = logging.DEBUG if args.verbose else logging.INFO
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