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
     * reference_01/ - 10% громкости
     * reference_04/ - 40% громкости
     * reference_07/ - 70% громкости
     * reference_10/ - 100% громкости (исходная)
     * reference_13/ - 130% громкости
3. Применяет модуль AEC к парам файлов (reference_new.wav и original_input.wav)
4. Сохраняет обработанные файлы и статистику в каждой поддиректории
5. Генерирует сводный отчет для всех тестов, группируя результаты по типам референсных сигналов
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

# Импорт из модуля AEC
try:
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
    main_test_dirs = ["music", "agent_speech"]
    
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
                
            # Ищем директории с разными уровнями громкости (reference_XX)
            for d in os.listdir(ref_type_path):
                sub_dir_path = os.path.join(ref_type_path, d)
                if os.path.isdir(sub_dir_path) and d.startswith("reference_"):
                    found_sub_dirs.append(sub_dir_path)
                    logging.debug(f"Найдена директория: {sub_dir_path}")
        
        if found_sub_dirs:
            test_dirs[main_dir] = found_sub_dirs
            logging.info(f"Найдено {len(found_sub_dirs)} поддиректорий в {main_dir}")
        else:
            logging.warning(f"В директории {main_dir_path} не найдено поддиректорий с уровнями громкости")
    
    return test_dirs

def optimized_process_audio_with_aec(input_file, output_file, reference_file, sample_rate=16000, channels=1, 
                           visualize=True, output_dir="results", frame_size_ms=10.0):
    """
    Оптимизированная версия функции обработки аудиофайла с помощью WebRTC AEC.
    Обеспечивает полную обработку аудиофайла и корректно обрабатывает NumPy типы.
    
    Args:
        input_file: Путь к входному файлу
        output_file: Путь к выходному файлу
        reference_file: Путь к референсному файлу
        sample_rate: Частота дискретизации
        channels: Количество каналов
        visualize: Создавать ли визуализации
        output_dir: Директория для сохранения результатов
        frame_size_ms: Размер фрейма в миллисекундах
        
    Returns:
        dict: Метрики качества AEC
    """
    try:
        logging.info(f"Обработка файла {input_file} с помощью AEC...")
        
        # Чтение референсного файла и определение его параметров
        with wave.open(reference_file, 'rb') as ref_wf:
            ref_data = ref_wf.readframes(ref_wf.getnframes())
            ref_frames_count = ref_wf.getnframes()
            ref_rate = ref_wf.getframerate()
            ref_channels = ref_wf.getnchannels()
            logging.info(f"Референсный файл: {ref_frames_count} фреймов, {ref_rate} Гц, {ref_channels} канала(ов), длительность {ref_frames_count/ref_rate:.2f} с")
            
        # Чтение входного файла и определение его параметров
        with wave.open(input_file, 'rb') as in_wf:
            in_data = in_wf.readframes(in_wf.getnframes())
            in_frames_count = in_wf.getnframes()
            in_rate = in_wf.getframerate()
            in_channels = in_wf.getnchannels()
            logging.info(f"Входной файл: {in_frames_count} фреймов, {in_rate} Гц, {in_channels} канала(ов), длительность {in_frames_count/in_rate:.2f} с")
        
        # Определяем, нужно ли конвертировать моно в стерео или наоборот
        # Для WebRTC AEC используем количество каналов из входного файла
        channels = in_channels
        logging.info(f"Используем количество каналов: {channels}")
        
        # Если количество каналов не совпадает, конвертируем данные
        if ref_channels != in_channels:
            logging.warning(f"Несоответствие количества каналов: референсный файл ({ref_channels}) и входной файл ({in_channels})")
            
            if ref_channels == 1 and in_channels == 2:
                # Конвертируем моно в стерео для референсного файла
                logging.info("Конвертация референсного файла из моно в стерео")
                # Преобразуем байты в массив int16
                ref_samples = np.frombuffer(ref_data, dtype=np.int16)
                # Дублируем каналы
                ref_stereo = np.column_stack((ref_samples, ref_samples))
                # Преобразуем обратно в байты
                ref_data = ref_stereo.tobytes()
                logging.info(f"Референсный файл преобразован в стерео: {len(ref_data)} байт")
            elif ref_channels == 2 and in_channels == 1:
                # Конвертируем стерео в моно для входного файла
                logging.info("Конвертация входного файла из стерео в моно")
                # Преобразуем байты в массив int16 и формируем двумерный массив (кадры x каналы)
                in_samples = np.frombuffer(in_data, dtype=np.int16).reshape(-1, 2)
                # Усредняем каналы
                in_mono = np.mean(in_samples, axis=1, dtype=np.int16)
                # Преобразуем обратно в байты
                in_data = in_mono.tobytes()
                logging.info(f"Входной файл преобразован в моно: {len(in_data)} байт")
                channels = 1  # Обновляем количество каналов
        
        # Инициализация сессии AEC с правильным количеством каналов
        aec_session = WebRTCAECSession(
            session_id="batch_test_session",
            sample_rate=sample_rate,
            channels=channels,
            batch_mode=False,
            frame_size_ms=frame_size_ms
        )
        
        # Логируем размер фрейма из AEC сессии
        frame_size = aec_session.frame_size
        frame_size_ms_actual = frame_size / sample_rate * 1000
        frame_size_bytes = frame_size * 2 * channels  # 2 байта на сэмпл (16 бит)
        
        logging.info(f"Размер фрейма в AEC сессии: {frame_size} сэмплов, {frame_size_bytes} байт, {frame_size_ms_actual:.2f} мс")
        
        # Оценка и установка задержки
        delay_samples, delay_ms, confidence = aec_session.auto_set_delay(ref_data, in_data)
        
        # Вычисляем задержку в фреймах
        delay_frames = int(delay_samples / frame_size)
        logging.info(f"Задержка в фреймах: {delay_frames} фреймов")
        
        # Разделение на фреймы с учетом правильного размера фрейма
        ref_frames = []
        for i in range(0, len(ref_data), frame_size_bytes):
            frame = ref_data[i:i+frame_size_bytes]
            if len(frame) == frame_size_bytes:  # Только полные фреймы
                ref_frames.append(frame)
            elif len(frame) > 0:  # Остаток (неполный фрейм)
                # Дополняем нулями до полного размера фрейма
                frame = frame + b'\x00' * (frame_size_bytes - len(frame))
                ref_frames.append(frame)
                logging.debug(f"Добавлен дополненный фрейм референсного сигнала размером {len(frame)} байт")
        
        in_frames = []
        for i in range(0, len(in_data), frame_size_bytes):
            frame = in_data[i:i+frame_size_bytes]
            if len(frame) == frame_size_bytes:  # Только полные фреймы
                in_frames.append(frame)
            elif len(frame) > 0:  # Остаток (неполный фрейм)
                # Дополняем нулями до полного размера фрейма
                frame = frame + b'\x00' * (frame_size_bytes - len(frame))
                in_frames.append(frame)
                logging.debug(f"Добавлен дополненный фрейм входного сигнала размером {len(frame)} байт")
        
        logging.info(f"Референсный файл разделен на {len(ref_frames)} фреймов")
        logging.info(f"Входной файл разделен на {len(in_frames)} фреймов")
        
        # Определяем количество фреймов для обработки
        min_frames = min(len(ref_frames), len(in_frames))
        logging.info(f"Будет обработано {min_frames} фреймов (примерно {min_frames * frame_size_ms_actual / 1000:.2f} секунд)")
        
        # Проверяем, не обрабатываем ли мы слишком мало фреймов
        expected_frames = max(len(ref_frames), len(in_frames))
        if min_frames < expected_frames * 0.9:  # Если обрабатываем менее 90% ожидаемого
            logging.warning(f"Обрабатывается только {min_frames} из {expected_frames} ожидаемых фреймов ({min_frames/expected_frames*100:.1f}%)")
            logging.warning("Это может привести к значительному сокращению длительности выходного файла")
        
        # Предварительная буферизация с учетом задержки
        pre_buffer_size = min(delay_frames, min_frames)
        pre_buffer_ms = pre_buffer_size * frame_size_ms_actual
        logging.info(f"Предварительная буферизация: {pre_buffer_size} фреймов ({pre_buffer_ms:.2f} мс)")
        
        for i in range(pre_buffer_size):
            ref_frame = ref_frames[i]
            aec_session.add_reference_frame(ref_frame)
            
        # Обработка фреймов в правильной последовательности
        processed_frames = []
        
        # Отслеживаем прогресс
        progress_step = max(1, min_frames // 10)  # Логируем каждые 10% или каждый фрейм, если фреймов мало
        
        for i in range(min_frames):
            # Добавляем референсный фрейм с учетом задержки
            ref_idx = i + pre_buffer_size
            if ref_idx < len(ref_frames):
                aec_session.add_reference_frame(ref_frames[ref_idx])
            
            # Обрабатываем входной фрейм
            processed_frame = aec_session.process_frame(in_frames[i])
            processed_frames.append(processed_frame)
            
            # Логируем прогресс каждые 10% или в конце
            if i % progress_step == 0 or i == min_frames - 1:
                progress_percent = (i+1) / min_frames * 100
                logging.info(f"Обработано {i+1}/{min_frames} фреймов ({progress_percent:.1f}%)")
                
                # Проверяем на ошибки в NumPy типах
                if processed_frame is None or len(processed_frame) == 0:
                    logging.error(f"Фрейм {i+1} обработан с ошибкой: получен пустой фрейм")
        
        # Получаем финальную статистику
        final_stats = aec_session.get_statistics()
        
        # Выводим информацию о фреймах с эхо
        total_frames = final_stats.get("processed_frames", 0)
        echo_frames = final_stats.get("echo_frames", 0)
        
        # Проверяем, что total_frames не равно 0
        if total_frames > 0:
            # Ограничиваем echo_frames значением total_frames
            echo_frames = min(echo_frames, total_frames)
            echo_percentage = (echo_frames / total_frames * 100)
        else:
            echo_frames = 0
            echo_percentage = 0
        
        logging.info(f"Статистика обнаружения эха:")
        logging.info(f"  Всего обработано фреймов: {total_frames}")
        logging.info(f"  Фреймов с обнаруженным эхо: {echo_frames} ({echo_percentage:.2f}%)")
        
        # Проверяем, что все фреймы обработаны
        if len(processed_frames) < min_frames:
            logging.error(f"Не все фреймы были обработаны: {len(processed_frames)}/{min_frames}")
        else:
            logging.info(f"Все {len(processed_frames)} фреймов успешно обработаны")
            
        # Сохранение обработанного аудио
        logging.info(f"Сохранение обработанного аудио в {output_file}")
        with wave.open(output_file, 'wb') as out_wf:
            out_wf.setnchannels(channels)
            out_wf.setsampwidth(2)  # 16-bit audio = 2 bytes
            out_wf.setframerate(sample_rate)
            out_wf.writeframes(b''.join(processed_frames))
            
        logging.info(f"Обработка завершена, результат сохранен в {output_file}")
        
        # Объединяем все обработанные фреймы для вычисления метрик
        processed_data = b''.join(processed_frames)
        
        # Расчет метрик качества
        metrics = calculate_metrics(ref_data, processed_data, in_data)
        metrics["echo_frames"] = echo_frames
        metrics["echo_frames_percentage"] = echo_percentage
        metrics["delay_samples"] = delay_samples
        metrics["delay_ms"] = delay_ms
        metrics["delay_confidence"] = confidence
        
        # Выводим основные метрики
        logging.info("\nМетрики качества AEC:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                logging.info(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Получаем масштабированные сигналы
        try:
            scaled_mic_data, scaled_ref_data = aec_session.get_scaled_signals(in_data, ref_data)
        except Exception as e:
            logging.error(f"Ошибка при получении масштабированных сигналов: {e}")
            scaled_mic_data, scaled_ref_data = None, None
        
        # Визуализация
        if visualize:
            # Создаем директорию для результатов, если она не существует
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # Вызываем функцию визуализации с масштабированными сигналами
                visualization_results = visualize_audio_processing(
                    output_dir=output_dir,
                    reference_data=ref_data,
                    input_data=in_data,
                    processed_data=processed_data,
                    scaled_mic_data=scaled_mic_data,
                    scaled_ref_data=scaled_ref_data,
                    metrics=metrics,
                    sample_rate=sample_rate,
                    max_delay_ms=1000
                )
                
                # Добавляем результаты визуализации в метрики
                if visualization_results:
                    metrics.update(visualization_results)
            except Exception as e:
                logging.error(f"Ошибка при визуализации: {e}")
                logging.exception("Подробная информация об ошибке визуализации:")
        
        # Проверяем длительность созданного файла
        try:
            with wave.open(output_file, 'rb') as wf:
                output_frames = wf.getnframes()
                output_rate = wf.getframerate()
                output_duration = output_frames / output_rate
                expected_duration = min_frames * frame_size / sample_rate
                logging.info(f"Выходной файл: {output_frames} фреймов, {output_duration:.2f} с (ожидалось примерно {expected_duration:.2f} с)")
                
                # Проверяем соответствие длительности исходному входному файлу
                with wave.open(input_file, 'rb') as in_wf:
                    input_duration = in_wf.getnframes() / in_wf.getframerate()
                    if abs(output_duration - input_duration) > input_duration * 0.1:  # более 10% разница
                        logging.warning(f"Длительность выходного файла ({output_duration:.2f} с) отличается от входного ({input_duration:.2f} с) на {abs(output_duration - input_duration):.2f} с")
                        if output_duration < input_duration * 0.8:  # если выходной файл меньше 80% входного
                            logging.error("Выходной файл значительно короче входного. Это может указывать на проблемы с обработкой!")
        except Exception as e:
            logging.error(f"Ошибка при проверке выходного файла: {e}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Ошибка при обработке аудио: {e}")
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
    
    if output_dir is None:
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
    reference_file = os.path.join(dir_path, "reference_new.wav")
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
            logging.info(f"Длительность входного файла: {input_duration:.2f} секунд ({input_frames} фреймов)")
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
        metrics = optimized_process_audio_with_aec(
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
                    logging.info(f"Длительность выходного файла: {output_duration:.2f} секунд ({output_frames} фреймов)")
                    
                    # Проверяем, не сильно ли отличается длительность
                    if input_duration > 0 and abs(output_duration - input_duration) > 0.5:  # допуск 0.5 сек
                        logging.warning(f"Длительность выходного файла ({output_duration:.2f} с) существенно отличается от входного ({input_duration:.2f} с)")
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
    Обрабатывает все тестовые директории и собирает результаты
    
    Args:
        test_dirs: Словарь с тестовыми директориями
        args: Аргументы командной строки
        
    Returns:
        dict: Сводные результаты по всем тестам
    """
    results = {}
    
    # Создаем директорию для общих результатов, если указана
    summary_dir = None
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        summary_dir = args.results_dir
    
    # Обработка всех тестов
    for main_dir, sub_dirs in test_dirs.items():
        results[main_dir] = {}
        
        for sub_dir in sub_dirs:
            if "music" in sub_dir:
                continue
            dir_name = os.path.basename(sub_dir)
            
            # Определяем директорию для результатов
            if args.results_dir:
                output_dir = os.path.join(args.results_dir, main_dir, dir_name)
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = sub_dir
            
            # Обрабатываем директорию
            metrics = process_test_directory(
                sub_dir, 
                output_dir=output_dir,
                frame_size_ms=args.frame_size_ms,
                visualize=args.visualize,
                verbose=args.verbose
            )
            
            if metrics:
                results[main_dir][dir_name] = metrics
                logging.info(f"Обработка директории {sub_dir} завершена")
            else:
                logging.error(f"Не удалось обработать директорию {sub_dir}")
    
    # Генерируем сводный отчет
    if summary_dir:
        logging.info(f"Генерируем сводный отчет в {summary_dir}")
        generate_summary_report(results, summary_dir)
    
    return results

def generate_summary_report(results, output_dir=None):
    """
    Генерирует сводный отчет по всем тестам
    
    Args:
        results: Результаты тестов
        output_dir: Директория для сохранения отчета
    """
    if not results:
        logging.error("Нет результатов для генерации отчета")
        return
    
    if output_dir is None:
        output_dir = "results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Формируем отчет
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {},
        "detailed_results": results
    }
    
    # Вычисляем средние значения метрик по всем тестам
    all_metrics = defaultdict(list)
    
    for main_dir, sub_dirs in results.items():
        report["summary"][main_dir] = {}
        
        # Собираем метрики для каждой поддиректории
        dir_metrics = defaultdict(list)
        
        for dir_name, metrics in sub_dirs.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    dir_metrics[metric_name].append(value)
                    all_metrics[metric_name].append(value)
        
        # Вычисляем средние значения для директории
        for metric_name, values in dir_metrics.items():
            if values:
                report["summary"][main_dir][metric_name] = {
                    "mean": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values)
                }
    
    # Вычисляем общие средние значения
    report["summary"]["overall"] = {}
    for metric_name, values in all_metrics.items():
        if values:
            report["summary"]["overall"][metric_name] = {
                "mean": np.mean(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values)
            }
    
    # Сохраняем отчет в JSON файл
    report_file = os.path.join(output_dir, "aec_summary_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
    
    logging.info(f"Сводный отчет сохранен в {report_file}")
    
    # Генерируем графики
    generate_comparison_charts(results, output_dir)
    
    return report

def generate_comparison_charts(results, output_dir):
    """
    Генерирует сравнительные графики для различных уровней громкости,
    группируя их по типам референсных сигналов (clear_reference и reference_by_micro)
    
    Args:
        results: Результаты тестов
        output_dir: Директория для сохранения графиков
    """
    # Создаем папку для графиков
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Метрики для визуализации
    metrics_to_plot = [
        "erle_db", 
        "snr_db", 
        "correlation_improvement", 
        "echo_frames_percentage"
    ]
    
    for main_dir, sub_dirs in results.items():
        # Создаем словари для группировки результатов по типам референсов
        ref_types = {
            "clear_reference": {},
            "reference_by_micro": {}
        }
        
        # Группируем результаты по типам референсов
        for dir_name, metrics in sub_dirs.items():
            # Определяем тип референса из пути
            path_parts = dir_name.split(os.sep)
            
            # Проверяем, содержит ли путь clear_reference или reference_by_micro
            if "clear_reference" in path_parts:
                ref_type = "clear_reference"
            elif "reference_by_micro" in path_parts:
                ref_type = "reference_by_micro"
            else:
                logging.warning(f"Неизвестный тип референса для {dir_name}, пропускаем")
                continue
            
            # Получаем имя директории с уровнем громкости (reference_XX)
            volume_dir = path_parts[-1]  # Последняя часть пути - имя директории с громкостью
            
            # Извлекаем уровень громкости из имени директории (например, reference_04 -> 0.4)
            if volume_dir.startswith("reference_"):
                volume_level = float(volume_dir.split("_")[1]) / 10.0
                
                # Добавляем метрики в соответствующую группу
                if volume_level not in ref_types[ref_type]:
                    ref_types[ref_type][volume_level] = {}
                
                for metric in metrics_to_plot:
                    if metric in metrics and isinstance(metrics[metric], (int, float)):
                        ref_types[ref_type][volume_level][metric] = metrics[metric]
        
        # Генерируем графики для каждого типа референса
        for ref_type, volume_data in ref_types.items():
            if not volume_data:  # Пропускаем, если нет данных
                continue
                
            # Получаем упорядоченные уровни громкости и соответствующие значения метрик
            volume_levels = sorted(volume_data.keys())
            metrics_values = {metric: [] for metric in metrics_to_plot}
            
            for level in volume_levels:
                for metric in metrics_to_plot:
                    if level in volume_data and metric in volume_data[level]:
                        metrics_values[metric].append(volume_data[level][metric])
                    else:
                        metrics_values[metric].append(None)
            
            # Создаем график для каждой метрики
            for metric in metrics_to_plot:
                if not any(v is not None for v in metrics_values[metric]):
                    continue
                
                plt.figure(figsize=(10, 6))
                plt.plot(volume_levels, metrics_values[metric], 'o-', linewidth=2)
                plt.grid(True)
                plt.xlabel('Уровень громкости (коэффициент)')
                plt.ylabel(metric)
                
                # Используем понятные названия для типов референсов
                ref_type_display = "Чистый референс" if ref_type == "clear_reference" else "Референс через микрофон"
                plt.title(f'{main_dir}: {ref_type_display} - Зависимость {metric} от уровня громкости')
                
                # Создаем имя файла из типа референса и метрики
                filename = f"{main_dir}_{ref_type}_{metric}.png"
                plt.savefig(os.path.join(charts_dir, filename))
                plt.close()
    
    logging.info(f"Графики сохранены в {charts_dir}")

def main():
    parser = argparse.ArgumentParser(description="Пакетная обработка тестовых данных с WebRTC AEC")
    parser.add_argument("--tests-dir", "-t", default="tests", 
                      help="Директория с тестовыми данными (по умолчанию: tests)")
    parser.add_argument("--results-dir", "-r", default=None, 
                      help="Директория для сохранения результатов (по умолчанию: исходные директории)")
    parser.add_argument("--visualize", "-v", action="store_true", default=True,
                      help="Создавать визуализации (по умолчанию: включено)")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                      help="Не создавать визуализации")
    parser.add_argument("--frame-size-ms", "-f", type=float, default=10.0,
                      help="Размер фрейма в миллисекундах (по умолчанию: 10.0)")
    parser.add_argument("--verbose", action="store_true", default=False,
                      help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Вывод информации о режиме сохранения результатов
    if args.results_dir:
        logging.info(f"Результаты будут сохранены в отдельную директорию: {args.results_dir}")
        logging.info("Структура исходных директорий будет сохранена.")
        logging.info(f"Сводный отчет будет сохранен в: {args.results_dir}")
    else:
        logging.info("Результаты будут сохранены в исходные директории с тестовыми файлами.")
        logging.info("Каждая директория будет содержать свой лог-файл 'aec_processing.log'.")
        logging.info("Сводный отчет будет сохранен в директорию 'results_summary'.")
    
    # Находим все тестовые директории
    test_dirs = find_test_directories(args.tests_dir)
    
    if not test_dirs:
        logging.error("Не найдено тестовых директорий")
        return
    
    # Обрабатываем все тесты
    results = process_all_tests(test_dirs, args)
    
    # Генерируем сводный отчет только если указана директория results_dir
    if args.results_dir:
        logging.info(f"Сводный отчет сохранен в {args.results_dir}")
    else:
        # Если директория results_dir не указана, создаем отдельную директорию для сводного отчета
        summary_dir = "results_summary"
        os.makedirs(summary_dir, exist_ok=True)
        generate_summary_report(results, summary_dir)
        logging.info(f"Сводный отчет сохранен в {summary_dir}")
    
    logging.info("Пакетная обработка тестов завершена!")

if __name__ == "__main__":
    main() 