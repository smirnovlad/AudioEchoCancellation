#!/usr/bin/env python3
"""
Скрипт для расчёта метрик на основе созданных файлов.

Этот скрипт:
1. Принимает директорию с тестовыми данными, аналогично batch_aec_test.py
2. Находит все поддиректории, содержащие обработанные файлы
3. Для каждой поддиректории:
   - Считывает существующие метрики из aec_metrics.json
   - Рассчитывает дополнительные метрики на основе аудиофайлов
   - Сохраняет обновленные метрики обратно в aec_metrics.json
4. Не выполняет AEC обработку - работает только с уже обработанными файлами

Использование:
python metrics.py --test-dir tests/music
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
        logging.FileHandler("metrics.log", mode='w'),
        logging.StreamHandler()
    ]
)

# Класс для сериализации NumPy типов в JSON (из batch_aec_test.py)
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def calculate_aec_metrics(ref_data, in_data, processed_data, ref_delayed_data=None, 
                         sample_rate=16000, channels=1, predicted_delay_samples=None, 
                         predicted_delay_ms=None):
    """
    Расчет метрик качества для AEC обработки.
    
    Args:
        ref_data (bytes): Референсные данные
        in_data (bytes): Входные данные
        processed_data (bytes): Обработанные данные
        ref_delayed_data (bytes, optional): Задержанные референсные данные
        sample_rate (int): Частота дискретизации
        channels (int): Количество каналов
        predicted_delay_samples (int, optional): Предсказанная задержка в отсчетах
        predicted_delay_ms (float, optional): Предсказанная задержка в миллисекундах
    
    Returns:
        dict: Словарь с рассчитанными метриками
    """
    metrics = {}
    
    try:
        # Преобразуем байтовые данные в numpy массивы
        ref_np = np.frombuffer(ref_data, dtype=np.int16)
        in_np = np.frombuffer(in_data, dtype=np.int16)
        processed_np = np.frombuffer(processed_data, dtype=np.int16)
        
        # Обрезаем массивы до одинаковой длины
        min_len = min(len(ref_np), len(in_np), len(processed_np))
        ref_np = ref_np[:min_len]
        in_np = in_np[:min_len]
        processed_np = processed_np[:min_len]
        
        # Нормализуем данные
        ref_norm = ref_np.astype(np.float32) / 32768.0
        in_norm = in_np.astype(np.float32) / 32768.0
        proc_norm = processed_np.astype(np.float32) / 32768.0
        
        # Расчет ERLE (Echo Return Loss Enhancement) в дБ
        # ERLE = 10 * log10(power_of_input / power_of_processed)
        input_power = np.mean(in_norm**2)
        processed_power = np.mean(proc_norm**2)
        
        if processed_power > 0 and input_power > 0:
            erle_db = 10 * np.log10(input_power / processed_power)
            metrics["erle_db"] = float(erle_db)  # Явно преобразуем в float для совместимости
            logging.info(f"Рассчитан ERLE: {erle_db:.2f} дБ")
        else:
            logging.warning("Не удалось рассчитать ERLE: нулевая мощность сигнала")
        
        # Расчет разницы между предсказанной и реальной задержкой
        if ref_delayed_data is not None:
            # Вычисляем "реальную" задержку между исходным референсным и задержанным референсным сигналами
            ref_delayed_np = np.frombuffer(ref_delayed_data, dtype=np.int16)
            
            # Определяем минимальную длину для корреляции (до 5 секунд)
            correlation_window = int(5 * sample_rate)  # 5 секунд для корреляции
            actual_correlation_window = min(correlation_window, len(ref_np), len(ref_delayed_np))
            
            # Обрезаем оба сигнала до минимальной длины для корреляции
            ref_for_corr = ref_np[:actual_correlation_window]
            ref_delayed_for_corr = ref_delayed_np[:actual_correlation_window]
            
            # Если данные стерео, берем только левый канал
            if channels == 2:
                if len(ref_for_corr) % 2 == 0 and len(ref_delayed_for_corr) % 2 == 0:
                    ref_for_corr = ref_for_corr.reshape(-1, 2)[:, 0]
                    ref_delayed_for_corr = ref_delayed_for_corr.reshape(-1, 2)[:, 0]
                    logging.info("Используется левый канал для анализа стерео данных")
                else:
                    logging.warning("Нечетное количество элементов для стерео данных, используются исходные массивы")
            
            # Нормализуем данные для корреляции
            ref_for_corr = ref_for_corr.astype(np.float32) / 32768.0
            ref_delayed_for_corr = ref_delayed_for_corr.astype(np.float32) / 32768.0
            
            # Вычисляем корреляцию
            corr = np.correlate(ref_delayed_for_corr, ref_for_corr, 'full')
            lags = np.arange(len(corr)) - (len(ref_for_corr) - 1)
            
            # Находим максимальную корреляцию
            max_corr_idx = np.argmax(np.abs(corr))
            real_delay_samples = lags[max_corr_idx]
            real_delay_ms = real_delay_samples * 1000.0 / sample_rate
            
            # Добавляем метрики задержки
            metrics.update({
                "real_delay_samples": real_delay_samples,
                "real_delay_ms": real_delay_ms,
            })
            
            # Если есть предсказанная задержка, вычисляем разницу
            if predicted_delay_samples is not None and predicted_delay_ms is not None:
                delay_diff_samples = predicted_delay_samples - real_delay_samples
                delay_diff_ms = predicted_delay_ms - real_delay_ms
                delay_abs_diff_ms = abs(delay_diff_ms)
                
                metrics.update({
                    "delay_diff_samples": delay_diff_samples,
                    "delay_diff_ms": delay_diff_ms,
                    "delay_abs_diff_ms": delay_abs_diff_ms,
                })
                
                logging.info(f"Реальная задержка: {real_delay_samples} отсчетов ({real_delay_ms:.2f} мс)")
                logging.info(f"Разница между предсказанной и реальной задержкой: {delay_diff_samples} отсчетов ({delay_diff_ms:.2f} мс)")
    
    except Exception as e:
        logging.error(f"Ошибка при расчете метрик: {e}")

    return metrics

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

def calculate_metrics_for_directory(dir_path):
    """
    Рассчитывает метрики для одной директории.
    
    Args:
        dir_path: Путь к директории с обработанными файлами
        
    Returns:
        dict: Словарь с рассчитанными метриками или None в случае ошибки
    """
    logging.info(f"Расчет метрик для директории: {dir_path}")
    
    # Проверяем наличие необходимых файлов
    metrics_file = os.path.join(dir_path, "aec_metrics.json")
    if not os.path.exists(metrics_file):
        logging.warning(f"Файл метрик {metrics_file} не найден, пропускаем директорию")
        return None
    
    reference_file = os.path.join(dir_path, "reference_volumed.wav")
    input_file = os.path.join(dir_path, "original_input.wav")
    processed_file = os.path.join(dir_path, "processed_input.wav")
    reference_delayed_file = os.path.join(dir_path, "reference_volumed_delayed.wav")
    
    if not os.path.exists(reference_file):
        logging.warning(f"Файл {reference_file} не найден, пропускаем директорию")
        return None
    
    if not os.path.exists(input_file):
        logging.warning(f"Файл {input_file} не найден, пропускаем директорию")
        return None
    
    if not os.path.exists(processed_file):
        logging.warning(f"Файл {processed_file} не найден, пропускаем директорию")
        return None
    
    # Загружаем существующие метрики
    try:
        with open(metrics_file, 'r') as f:
            existing_metrics = json.load(f)
        logging.info(f"Загружены существующие метрики из {metrics_file}")
    except Exception as e:
        logging.error(f"Ошибка при загрузке метрик из {metrics_file}: {e}")
        existing_metrics = {}
    
    # Загружаем аудиофайлы
    try:
        # Загружаем референсный файл
        with wave.open(reference_file, 'rb') as wf:
            ref_data = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
        
        # Загружаем входной файл
        with wave.open(input_file, 'rb') as wf:
            in_data = wf.readframes(wf.getnframes())
        
        # Загружаем обработанный файл
        with wave.open(processed_file, 'rb') as wf:
            processed_data = wf.readframes(wf.getnframes())
        
        # Загружаем задержанный референсный файл, если он существует
        ref_delayed_data = None
        if os.path.exists(reference_delayed_file):
            with wave.open(reference_delayed_file, 'rb') as wf:
                ref_delayed_data = wf.readframes(wf.getnframes())
                logging.info(f"Загружен задержанный референсный файл: {reference_delayed_file}")
    
    except Exception as e:
        logging.error(f"Ошибка при загрузке аудиофайлов: {e}")
        return None
    
    # Извлекаем предсказанные задержки из существующих метрик
    predicted_delay_samples = existing_metrics.get('delay_samples')
    predicted_delay_ms = existing_metrics.get('delay_ms')
    
    # Рассчитываем дополнительные метрики
    try:
        # Получаем дополнительные метрики
        new_metrics = calculate_aec_metrics(
            ref_data, 
            in_data, 
            processed_data, 
            ref_delayed_data,
            sample_rate=sample_rate,
            channels=channels,
            predicted_delay_samples=predicted_delay_samples,
            predicted_delay_ms=predicted_delay_ms
        )
        
        # Проверяем наличие ключевой метрики ERLE
        if 'erle_db' in new_metrics:
            logging.info(f"Рассчитан ERLE: {new_metrics['erle_db']:.2f} дБ")
        else:
            logging.warning("ERLE не был рассчитан")
        
        # Обновляем существующие метрики
        existing_metrics.update(new_metrics)
        
        # Сохраняем обновленные метрики
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2, cls=NumpyJSONEncoder)
        logging.info(f"Обновленные метрики сохранены в {metrics_file}")
        
        return existing_metrics
    
    except Exception as e:
        logging.error(f"Ошибка при расчете метрик: {e}")
        return None

def create_summary_file(results, test_dir):
    """
    Создает сводный файл со всеми результатами тестов.
    
    Args:
        results: Словарь с результатами всех обработанных директорий
        test_dir: Путь к корневой директории с тестами
        
    Returns:
        str: Путь к созданному файлу summary_results.json
    """
    # Создаем путь к файлу summary_results.json
    summary_file = os.path.join(test_dir, "summary_results.json")
    logging.info(f"Создание сводного файла результатов: {summary_file}")
    
    try:
        # Сохраняем результаты в JSON формате
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
        logging.info(f"Сводный файл результатов успешно создан: {summary_file}")
        
        # Вычисляем общую статистику
        erle_values = [metrics.get('erle_db') for metrics in results.values() if metrics.get('erle_db') is not None]
        delay_diff_values = [metrics.get('delay_abs_diff_ms') for metrics in results.values() if metrics.get('delay_abs_diff_ms') is not None]
        
        statistics = {
            "total_tests": len(results),
            "erle_statistics": {},
            "delay_statistics": {}
        }
        
        if erle_values:
            statistics["erle_statistics"] = {
                "count": len(erle_values),
                "min": min(erle_values),
                "max": max(erle_values),
                "mean": sum(erle_values) / len(erle_values)
            }
        
        if delay_diff_values:
            statistics["delay_statistics"] = {
                "count": len(delay_diff_values),
                "min": min(delay_diff_values),
                "max": max(delay_diff_values),
                "mean": sum(delay_diff_values) / len(delay_diff_values)
            }
        
        # Сохраняем статистику
        statistics_file = os.path.join(test_dir, "summary_statistics.json")
        with open(statistics_file, 'w') as f:
            json.dump(statistics, f, indent=2, cls=NumpyJSONEncoder)
        logging.info(f"Файл статистики успешно создан: {statistics_file}")
        
        return summary_file
    
    except Exception as e:
        logging.error(f"Ошибка при создании сводного файла результатов: {e}")
        return None

def process_directory_by_level(dir_path):
    """
    Определяет уровень вложенности тестовой директории и обрабатывает её соответствующим образом.
    
    Args:
        dir_path: Путь к тестовой директории
        
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
        metrics = calculate_metrics_for_directory(dir_path)
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
                metrics = calculate_metrics_for_directory(item_path)
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
                    metrics = calculate_metrics_for_directory(item_path)
                    if metrics:
                        results[item_path] = metrics
        # Для reference_by_micro ищем директории с задержками
        else:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path) and item.startswith("delay_"):
                    logging.info(f"Обрабатываем поддиректорию с задержкой: {item_path}")
                    # Рекурсивно вызываем для обработки delay_XX директорий
                    delay_results = process_directory_by_level(item_path)
                    # Объединяем результаты
                    results.update(delay_results)
        
    # Проверяем, является ли текущая директория корневой директорией тестового сценария
    elif dir_name in ["music", "agent_speech", "agent_user_speech"]:
        logging.info(f"Обнаружена корневая директория тестового сценария: {dir_path}")
        # Ищем поддиректории reference_by_micro и clear_reference
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path) and item in ["reference_by_micro", "clear_reference"]:
                logging.info(f"Обрабатываем поддиректорию {item}: {item_path}")
                # Рекурсивно вызываем для обработки reference_by_micro и clear_reference директорий
                subdir_results = process_directory_by_level(item_path)
                # Объединяем результаты
                results.update(subdir_results)
    
    else:
        # Если директория не соответствует известной структуре, пробуем обработать все поддиректории
        logging.warning(f"Директория {dir_path} не соответствует известной структуре тестов")
        logging.warning("Сканируем все поддиректории")
        
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                # Проверяем, есть ли файл метрик в этой директории
                metrics_file = os.path.join(item_path, "aec_metrics.json")
                processed_file = os.path.join(item_path, "processed_input.wav")
                
                if os.path.exists(metrics_file) and os.path.exists(processed_file):
                    logging.info(f"Обнаружены файлы метрик и обработанный файл в {item_path}, обрабатываем")
                    metrics = calculate_metrics_for_directory(item_path)
                    if metrics:
                        results[item_path] = metrics
                else:
                    # Если нет файлов метрик, рекурсивно проверяем поддиректории
                    subdir_results = process_directory_by_level(item_path)
                    results.update(subdir_results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Расчет метрик на основе обработанных файлов")
    parser.add_argument("--test-dir", "-d", required=True,
                      help="Директория с тестовыми данными (обязательный параметр)")
    parser.add_argument("--verbose", action="store_true", default=False,
                      help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Начало расчета метрик для директории: {args.test_dir}")
    
    # Проверяем наличие указанной директории
    if not os.path.exists(args.test_dir):
        logging.error(f"Указанная директория не существует: {args.test_dir}")
        sys.exit(1)
    
    # Обрабатываем директорию
    results = process_directory_by_level(args.test_dir)
    
    # Выводим общую статистику по результатам
    directories_processed = len(results)
    erle_values = [metrics.get('erle_db') for metrics in results.values() if metrics.get('erle_db') is not None]
    
    logging.info(f"Обработка завершена. Обработано директорий: {directories_processed}")
    
    if erle_values:
        avg_erle = sum(erle_values) / len(erle_values)
        min_erle = min(erle_values)
        max_erle = max(erle_values)
        logging.info(f"Статистика ERLE: среднее={avg_erle:.2f} дБ, мин={min_erle:.2f} дБ, макс={max_erle:.2f} дБ")
    else:
        logging.warning("Не удалось рассчитать ни одного значения ERLE")
    
    # Создаем сводный файл результатов
    if results:
        summary_file = create_summary_file(results, args.test_dir)
        if summary_file:
            logging.info(f"Сводный файл результатов успешно создан: {summary_file}")
        else:
            logging.error("Не удалось создать сводный файл результатов")
    else:
        logging.warning("Нет результатов для создания сводного файла")
    
    logging.info("Расчет метрик завершен")

if __name__ == "__main__":
    main()
