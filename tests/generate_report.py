#!/usr/bin/env python3
"""
Скрипт для анализа результатов тестов AEC и генерации отчетов

Этот скрипт:
1. Загружает результаты тестов из summary_results.json, созданного metrics.py
2. Анализирует зависимости показателей от различных параметров:
   - Уровень громкости
   - Задержка
   - Тип референсного сигнала
3. Генерирует различные графики и статистические отчеты
4. Сохраняет результаты анализа в структурированном виде

Использование:
python report.py --test-dir tests/agent_user_speech/reference_by_micro/delay_200
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("report_generation.log", mode='w'),
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


def load_test_results(test_dir: str) -> Dict[str, Any]:
    """
    Загружает результаты тестов из файла summary_results.json в указанной директории
    
    Args:
        test_dir: Директория с файлом summary_results.json
        
    Returns:
        dict: Словарь с результатами тестов
    """
    # Путь к файлу summary_results.json
    summary_file = os.path.join(test_dir, "summary_results.json")
    
    # Проверяем существование файла
    if not os.path.exists(summary_file):
        logging.error(f"Файл {summary_file} не найден")
        return {}
    
    # Загружаем результаты из файла
    try:
        with open(summary_file, 'r') as f:
            results = json.load(f)
            logging.info(f"Загружены результаты из {summary_file}: {len(results)} тестов")
            return results
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла {summary_file}: {e}")
        return {}


def extract_test_parameters(path: str) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    """
    Извлекает параметры теста из пути к директории
    
    Args:
        path: Путь к директории с тестовыми данными
        
    Returns:
        tuple: (тип_референса, задержка, громкость)
    """
    path_parts = path.split(os.sep)
    
    # Инициализируем параметры
    ref_type = None
    delay_value = None
    volume_level = None
    
    # Определяем тип референса
    if "clear_reference" in path_parts:
        ref_type = "clear_reference"
    elif "reference_by_micro" in path_parts:
        ref_type = "reference_by_micro"
    
    # Извлекаем задержку и громкость
    for part in path_parts:
        # Извлекаем задержку (например, delay_250 -> 250 мс)
        if part.startswith("delay_"):
            try:
                delay_value = int(part.split("_")[1])
            except (IndexError, ValueError):
                pass
        
        # Извлекаем уровень громкости (например, volume_04 -> 0.4)
        if part.startswith("volume_"):
            try:
                volume_level = float(part.split("_")[1]) / 10.0
            except (IndexError, ValueError):
                pass
    
    # Для clear_reference задержка всегда 0
    if ref_type == "clear_reference":
        delay_value = 0
    
    return ref_type, delay_value, volume_level


def group_results_by_params(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[int, Dict[float, Dict[str, Any]]]]]:
    """
    Группирует результаты тестов по параметрам: категория, тип референса, задержка, громкость
    
    Args:
        results: Словарь с результатами тестов
        
    Returns:
        dict: Структурированный словарь результатов
    """
    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    for path, metrics in results.items():
        # Определяем категорию из пути (music, agent_speech и т.д.)
        path_parts = path.split(os.sep)
        category = None
        
        for part in path_parts:
            if part in ["music", "agent_speech", "agent_user_speech"]:
                category = part
                break
        
        if not category:
            logging.warning(f"Не удалось определить категорию для {path}, пропускаем")
            continue
        
        # Извлекаем параметры теста
        ref_type, delay_value, volume_level = extract_test_parameters(path)
        
        if not all([ref_type, delay_value is not None, volume_level is not None]):
            logging.warning(f"Не удалось определить все параметры для {path}, пропускаем")
            continue
        
        # Добавляем метрики в структуру
        grouped_results[category][ref_type][delay_value][volume_level] = metrics
    
    return grouped_results


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Вычисляет статистические показатели для списка значений
    
    Args:
        values: Список числовых значений
        
    Returns:
        dict: Словарь со статистиками
    """
    if not values:
        return {}
    
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
        "count": len(values)
    }


def generate_summary_report(grouped_results: Dict[str, Dict[str, Dict[int, Dict[float, Dict[str, Any]]]]]) -> Dict[str, Any]:
    """
    Генерирует сводный отчет по группированным результатам
    
    Args:
        grouped_results: Структурированные результаты тестов
        
    Returns:
        dict: Сводный отчет
    """
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {},
        "detailed_by_category": {},
        "detailed_by_reference_type": {},
        "detailed_by_delay": {},
        "detailed_by_volume": {}
    }
    
    # Метрики для анализа
    key_metrics = ["erle_db", "delay_diff_ms", "echo_percentage", "delay_confidence", "delay_abs_diff_ms"]
    all_metrics = defaultdict(list)
    
    # Собираем и группируем метрики по разным параметрам
    for category, ref_types in grouped_results.items():
        report["detailed_by_category"][category] = {}
        
        for ref_type, delays in ref_types.items():
            if ref_type not in report["detailed_by_reference_type"]:
                report["detailed_by_reference_type"][ref_type] = {}
            
            for delay, volumes in delays.items():
                if delay not in report["detailed_by_delay"]:
                    report["detailed_by_delay"][delay] = {}
                
                for volume, metrics in volumes.items():
                    if volume not in report["detailed_by_volume"]:
                        report["detailed_by_volume"][volume] = {}
                    
                    # Собираем метрики
                    for metric in key_metrics:
                        if metric in metrics and isinstance(metrics[metric], (int, float)):
                            # Добавляем в общий список для всех категорий
                            all_metrics[metric].append(metrics[metric])
                            
                            # По категории
                            if metric not in report["detailed_by_category"][category]:
                                report["detailed_by_category"][category][metric] = []
                            report["detailed_by_category"][category][metric].append(metrics[metric])
                            
                            # По типу референса
                            if metric not in report["detailed_by_reference_type"][ref_type]:
                                report["detailed_by_reference_type"][ref_type][metric] = []
                            report["detailed_by_reference_type"][ref_type][metric].append(metrics[metric])
                            
                            # По задержке
                            if metric not in report["detailed_by_delay"][delay]:
                                report["detailed_by_delay"][delay][metric] = []
                            report["detailed_by_delay"][delay][metric].append(metrics[metric])
                            
                            # По громкости
                            if metric not in report["detailed_by_volume"][volume]:
                                report["detailed_by_volume"][volume][metric] = []
                            report["detailed_by_volume"][volume][metric].append(metrics[metric])
    
    # Вычисляем статистики для всех групп
    # Общие статистики
    report["summary"]["overall"] = {}
    for metric, values in all_metrics.items():
        report["summary"]["overall"][metric] = calculate_statistics(values)
    
    # По категории
    for category, metrics in report["detailed_by_category"].items():
        report["summary"][category] = {}
        for metric, values in metrics.items():
            report["summary"][category][metric] = calculate_statistics(values)
        report["detailed_by_category"][category] = {metric: calculate_statistics(values) for metric, values in metrics.items()}
    
    # По типу референса
    for ref_type, metrics in report["detailed_by_reference_type"].items():
        if ref_type not in report["summary"]:
            report["summary"][ref_type] = {}
        for metric, values in metrics.items():
            report["summary"][ref_type][metric] = calculate_statistics(values)
        report["detailed_by_reference_type"][ref_type] = {metric: calculate_statistics(values) for metric, values in metrics.items()}
    
    # По задержке
    for delay, metrics in report["detailed_by_delay"].items():
        report["detailed_by_delay"][delay] = {metric: calculate_statistics(values) for metric, values in metrics.items()}
    
    # По громкости
    for volume, metrics in report["detailed_by_volume"].items():
        report["detailed_by_volume"][volume] = {metric: calculate_statistics(values) for metric, values in metrics.items()}
    
    return report


def generate_comparison_charts(grouped_results: Dict[str, Dict[str, Dict[int, Dict[float, Dict[str, Any]]]]], output_dir: str):
    """
    Генерирует сравнительные графики для различных параметров
    
    Args:
        grouped_results: Структурированные результаты тестов
        output_dir: Директория для сохранения графиков
    """
    # Создаем папку для графиков
    charts_dir = os.path.join(output_dir, "charts")
    
    # Очищаем директорию с графиками, если она существует
    if os.path.exists(charts_dir):
        logging.info(f"Очистка существующей директории с графиками: {charts_dir}")
        try:
            for item in os.listdir(charts_dir):
                item_path = os.path.join(charts_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
        except Exception as e:
            logging.error(f"Ошибка при очистке директории {charts_dir}: {e}")
            logging.exception("Подробная информация об ошибке:")
    
    os.makedirs(charts_dir, exist_ok=True)
    
    # Метрики для визуализации
    metrics_to_plot = [
        "erle_db", 
        "delay_diff_ms", 
        "echo_percentage", 
        "delay_confidence",
        "delay_abs_diff_ms"
    ]
    
    # Создаем сравнительные графики метрик
    generate_comparative_metric_charts(grouped_results, charts_dir, metrics_to_plot)
    
    # 3. Сравнение типов референса
    for category in grouped_results:
        # Получаем средние значения метрик для каждого типа референса
        ref_type_metrics = {}
        
        for ref_type in grouped_results[category]:
            ref_type_metrics[ref_type] = defaultdict(list)
            
            for delay in grouped_results[category][ref_type]:
                for volume in grouped_results[category][ref_type][delay]:
                    for metric in metrics_to_plot:
                        if metric in grouped_results[category][ref_type][delay][volume] and \
                           isinstance(grouped_results[category][ref_type][delay][volume][metric], (int, float)):
                            ref_type_metrics[ref_type][metric].append(
                                grouped_results[category][ref_type][delay][volume][metric]
                            )
        
        # Создаем гистограммы для сравнения типов референса
        for metric in metrics_to_plot:
            valid_ref_types = []
            mean_values = []
            std_values = []
            
            for ref_type, metrics in ref_type_metrics.items():
                if metric in metrics and metrics[metric]:
                    valid_ref_types.append("Чистый референс" if ref_type == "clear_reference" else "Референс через микрофон")
                    mean_values.append(np.mean(metrics[metric]))
                    std_values.append(np.std(metrics[metric]))
            
            if len(valid_ref_types) > 1:  # Создаем график только если есть что сравнивать
                plt.figure(figsize=(10, 6))
                bars = plt.bar(valid_ref_types, mean_values, yerr=std_values, capsize=10)
                
                # Добавляем значения над столбцами
                for i, bar in enumerate(bars):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_values[i] + 0.1,
                            f'{mean_values[i]:.2f}', ha='center', va='bottom')
                
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.ylabel(metric)
                plt.title(f'{category}: Сравнение типов референса по {metric}')
                
                # Создаем имя файла
                filename = f"{category}_compare_ref_types_{metric}.png"
                plt.savefig(os.path.join(charts_dir, filename))
                plt.close()
    
    logging.info(f"Графики сохранены в {charts_dir}")


def generate_comparative_metric_charts(grouped_results: Dict[str, Dict[str, Dict[int, Dict[float, Dict[str, Any]]]]], 
                                      charts_dir: str, metrics_to_plot: List[str]):
    """
    Генерирует графики сравнения метрик при разных значениях задержки и уровня громкости
    
    Args:
        grouped_results: Структурированные результаты тестов
        charts_dir: Директория для сохранения графиков
        metrics_to_plot: Список метрик для визуализации
    """
    # Цвета для отображения разных уровней громкости/задержек
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    line_styles = ['-', '--', '-.', ':']
    marker_styles = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    # Логирование начала генерации графиков
    logging.info(f"Начало генерации сравнительных графиков для метрик: {metrics_to_plot}")
    
    # Проходим по всем категориям
    for category, ref_types in grouped_results.items():
        for ref_type, delays in ref_types.items():
            # 1. Создаем графики зависимости метрик от громкости с разными задержками
            # Сначала соберем данные по всем задержкам и уровням громкости
            volume_levels_by_delay = {}
            metrics_by_delay = {metric: {} for metric in metrics_to_plot}
            
            # Проверяем наличие метрики erle_db в данных
            has_erle_data = False
            
            for delay, volumes in delays.items():
                if not volumes:  # Пропускаем, если нет данных
                    continue
                
                # Получаем упорядоченные уровни громкости
                volume_levels_by_delay[delay] = sorted(volumes.keys())
                
                for metric in metrics_to_plot:
                    metrics_by_delay[metric][delay] = []
                    for level in volume_levels_by_delay[delay]:
                        # Проверка наличия метрики и её типа
                        if metric in volumes[level]:
                            value = volumes[level][metric]
                            # Преобразование к числовому формату, если возможно
                            if isinstance(value, (int, float)):
                                metrics_by_delay[metric][delay].append(value)
                                if metric == 'erle_db':
                                    has_erle_data = True
                                    logging.info(f"ERLE значение найдено: {value} для задержки {delay} и громкости {level}")
                            else:
                                try:
                                    # Попытка преобразовать строковое значение в число
                                    value = float(value)
                                    metrics_by_delay[metric][delay].append(value)
                                    if metric == 'erle_db':
                                        has_erle_data = True
                                        logging.info(f"ERLE значение преобразовано в число: {value}")
                                except (ValueError, TypeError):
                                    metrics_by_delay[metric][delay].append(None)
                                    logging.info(f"Не удалось преобразовать значение {value} к числовому типу")
                        else:
                            metrics_by_delay[metric][delay].append(None)
            
            # Логируем наличие данных ERLE
            if 'erle_db' in metrics_to_plot:
                if has_erle_data:
                    logging.info(f"Найдены данные ERLE для категории {category}, типа референса {ref_type}")
                else:
                    logging.warning(f"Данные ERLE отсутствуют для категории {category}, типа референса {ref_type}")
            
            # Теперь создаем графики для каждой метрики, где каждая линия - отдельная задержка
            for metric in metrics_to_plot:
                # Проверяем, есть ли данные для этой метрики
                has_data = False
                for delay in delays:
                    if delay in metrics_by_delay[metric] and metrics_by_delay[metric][delay]:
                        if any(v is not None for v in metrics_by_delay[metric][delay]):
                            has_data = True
                            break
                
                if not has_data:
                    logging.warning(f"Нет данных для построения графика {metric} для {category}/{ref_type}")
                    if metric == 'erle_db':
                        # Дополнительная диагностика для erle_db
                        logging.error(f"КРИТИЧНО для erle_db: Проверка наличия данных по каждой задержке:")
                        for delay in delays:
                            if delay in metrics_by_delay[metric]:
                                logging.error(f"  Задержка {delay}: {metrics_by_delay[metric][delay]}")
                            else:
                                logging.error(f"  Задержка {delay}: нет данных")
                    continue
                
                # Для erle_db вывести дополнительную диагностику
                if metric == 'erle_db':
                    logging.info(f"=== ДИАГНОСТИКА ДЛЯ ERLE_DB (график по громкости) ===")
                    for delay in sorted(metrics_by_delay[metric].keys()):
                        delay_data = metrics_by_delay[metric][delay]
                        valid_count = sum(1 for v in delay_data if v is not None)
                        logging.info(f"  Задержка {delay}: имеет {valid_count} валидных точек из {len(delay_data)}")
                
                plt.figure(figsize=(12, 8))
                
                # Сортируем задержки для правильного порядка в легенде
                sorted_delays = sorted(metrics_by_delay[metric].keys())
                
                for i, delay in enumerate(sorted_delays):
                    # Получаем цвет и стиль линии
                    color_idx = i % len(colors)
                    style_idx = i % len(line_styles)
                    marker_idx = i % len(marker_styles)
                    
                    # Проверяем, есть ли данные для этой задержки
                    if not metrics_by_delay[metric][delay] or not any(v is not None for v in metrics_by_delay[metric][delay]):
                        continue
                    
                    # Отфильтровываем None значения перед построением
                    valid_x = []
                    valid_y = []
                    for idx, val in enumerate(metrics_by_delay[metric][delay]):
                        if val is not None and idx < len(volume_levels_by_delay[delay]):
                            valid_x.append(volume_levels_by_delay[delay][idx])
                            valid_y.append(val)
                    
                    if not valid_x or not valid_y:
                        continue
                    
                    # Строим линию для этой задержки
                    plt.plot(
                        valid_x, 
                        valid_y, 
                        marker=marker_styles[marker_idx],
                        linestyle=line_styles[style_idx], 
                        color=colors[color_idx], 
                        linewidth=2,
                        label=f'Задержка: {delay} мс'
                    )
                
                plt.grid(True)
                plt.xlabel('Уровень громкости (коэффициент)')
                plt.ylabel(metric)
                
                ref_type_display = "Чистый референс" if ref_type == "clear_reference" else "Референс через микрофон"
                plt.title(f'{category}: {ref_type_display} - Зависимость {metric} от уровня громкости')
                
                # Добавляем легенду
                plt.legend(loc='best')
                
                # Сохраняем график
                filename = f"{category}_{ref_type}_{metric}_by_volume.png"
                plt.savefig(os.path.join(charts_dir, filename))
                plt.close()
                logging.info(f"Создан график {filename}")
            
            # 2. Создаем графики зависимости метрик от задержки с разными уровнями громкости
            # Только для reference_by_micro имеет смысл строить такие графики
            if ref_type == "reference_by_micro":
                # Перегруппируем данные по громкости
                volume_grouped_data = defaultdict(dict)
                
                for delay, volumes in delays.items():
                    for volume, metrics_data in volumes.items():
                        volume_grouped_data[volume][delay] = metrics_data
                
                # Для каждой метрики создаем график зависимости от задержки
                for metric in metrics_to_plot:
                    has_data = False
                    # Добавляем более детальное логирование для анализа проблемы
                    logging.info(f"Проверка наличия данных {metric} для графика по задержке в категории {category}")
                    
                    # Сначала проверим наличие данных в общем
                    all_values = []
                    for volume in volume_grouped_data:
                        for delay in volume_grouped_data[volume]:
                            if metric in volume_grouped_data[volume][delay]:
                                value = volume_grouped_data[volume][delay][metric]
                                logging.info(f"Найдено значение {metric}: {value} (тип: {type(value)}) для громкости {volume} и задержки {delay}")
                                
                                # Пытаемся преобразовать значение к числу, если оно не числовое
                                if isinstance(value, (int, float)):
                                    all_values.append(value)
                                    has_data = True
                                elif isinstance(value, str):
                                    try:
                                        numeric_value = float(value)
                                        all_values.append(numeric_value)
                                        has_data = True
                                        logging.info(f"Строковое значение {value} успешно преобразовано в число {numeric_value}")
                                    except ValueError:
                                        logging.warning(f"Не удалось преобразовать строковое значение '{value}' в число")
                    
                    if not has_data:
                        logging.warning(f"Нет данных для построения графика {metric} по задержке для {category}")
                        if metric == 'erle_db':
                            logging.error(f"КРИТИЧНО: Метрика ERLE не найдена для графика по задержке, хотя ранее она была обнаружена!")
                        continue
                    
                    logging.info(f"Найдено {len(all_values)} значений для метрики {metric} в категории {category}")
                    
                    plt.figure(figsize=(12, 8))
                    
                    # Сортируем уровни громкости для правильного порядка в легенде
                    sorted_volumes = sorted(volume_grouped_data.keys())
                    
                    for i, volume in enumerate(sorted_volumes):
                        delay_data = volume_grouped_data[volume]
                        
                        if not delay_data:  # Пропускаем, если нет данных
                            continue
                        
                        # Получаем цвет и стиль линии
                        color_idx = i % len(colors)
                        style_idx = i % len(line_styles)
                        marker_idx = i % len(marker_styles)
                        
                        # Получаем упорядоченные задержки и соответствующие значения метрик
                        delay_values = sorted(delay_data.keys())
                        metric_values = []
                        
                        # Логируем данные для этой громкости
                        logging.info(f"Проверка громкости {volume} для метрики {metric}, найдено {len(delay_values)} точек задержки")
                        
                        for delay in delay_values:
                            if metric in delay_data[delay]:
                                value = delay_data[delay][metric]
                                if isinstance(value, (int, float)):
                                    metric_values.append(value)
                                    logging.info(f"  Задержка {delay} мс: {value}")
                                else:
                                    try:
                                        # Попытка преобразовать строковое значение в число
                                        numeric_value = float(value)
                                        metric_values.append(numeric_value)
                                        logging.info(f"  Задержка {delay} мс: {value} (преобразовано в {numeric_value})")
                                    except (ValueError, TypeError):
                                        metric_values.append(None)
                                        logging.warning(f"  Задержка {delay} мс: {value} (невозможно преобразовать)")
                            else:
                                metric_values.append(None)
                                logging.warning(f"  Задержка {delay} мс: метрика не найдена")
                        
                        # Отфильтровываем None значения перед построением
                        valid_x = []
                        valid_y = []
                        for idx, val in enumerate(metric_values):
                            if val is not None and idx < len(delay_values):
                                valid_x.append(delay_values[idx])
                                valid_y.append(val)
                        
                        if not valid_x or not valid_y:
                            logging.warning(f"Для громкости {volume} нет валидных точек для построения графика {metric}")
                            continue
                        
                        logging.info(f"Строим линию для громкости {volume} с {len(valid_x)} точками данных")
                        
                        # Строим линию для этой громкости
                        plt.plot(
                            valid_x, 
                            valid_y, 
                            marker=marker_styles[marker_idx],
                            linestyle=line_styles[style_idx], 
                            color=colors[color_idx], 
                            linewidth=2,
                            label=f'Громкость: {volume:.1f}'
                        )
                    
                    plt.grid(True)
                    plt.xlabel('Задержка (мс)')
                    plt.ylabel(metric)
                    
                    plt.title(f'{category}: Референс через микрофон - Зависимость {metric} от задержки')
                    
                    # Добавляем легенду
                    plt.legend(loc='best')
                    
                    # Сохраняем график
                    filename = f"{category}_{ref_type}_{metric}_by_delay.png"
                    plt.savefig(os.path.join(charts_dir, filename))
                    plt.close()
                    logging.info(f"Создан график {filename}")


def save_report(report: Dict[str, Any], output_dir: str):
    """
    Сохраняет отчет в файл
    
    Args:
        report: Сводный отчет
        output_dir: Директория для сохранения отчета
    """
    # Очищаем директорию отчёта, если она существует
    if os.path.exists(output_dir):
        logging.info(f"Очистка существующей директории отчёта: {output_dir}")
        try:
            # Удаляем файлы, но не трогаем поддиректории
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
        except Exception as e:
            logging.error(f"Ошибка при очистке директории {output_dir}: {e}")
            logging.exception("Подробная информация об ошибке:")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем отчет в JSON файл
    report_file = os.path.join(output_dir, "aec_analysis_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
    
    logging.info(f"Отчет сохранен в {report_file}")
    
    # Сохраняем общую сводку в текстовый файл
    summary_file = os.path.join(output_dir, "aec_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Анализ AEC тестов (сгенерирован {report['generated_at']})\n")
        f.write("=" * 80 + "\n\n")
        
        # Общая сводка
        f.write("ОБЩАЯ СТАТИСТИКА\n")
        f.write("-" * 80 + "\n")
        for metric, stats in report["summary"]["overall"].items():
            f.write(f"{metric}:\n")
            for stat_name, value in stats.items():
                f.write(f"  {stat_name}: {value:.2f}\n" if isinstance(value, float) else f"  {stat_name}: {value}\n")
            f.write("\n")
        
        # Статистика по категориям
        f.write("\nСТАТИСТИКА ПО КАТЕГОРИЯМ\n")
        f.write("-" * 80 + "\n")
        for category, metrics in report["summary"].items():
            if category == "overall":
                continue
            if category in ["clear_reference", "reference_by_micro"]:
                continue
            
            f.write(f"\n{category}:\n")
            for metric, stats in metrics.items():
                f.write(f"  {metric}:\n")
                for stat_name, value in stats.items():
                    f.write(f"    {stat_name}: {value:.2f}\n" if isinstance(value, float) else f"    {stat_name}: {value}\n")
        
        # Статистика по типу референса
        f.write("\nСТАТИСТИКА ПО ТИПУ РЕФЕРЕНСА\n")
        f.write("-" * 80 + "\n")
        ref_types = {"clear_reference": "Чистый референс", "reference_by_micro": "Референс через микрофон"}
        for ref_type, display_name in ref_types.items():
            if ref_type in report["summary"]:
                f.write(f"\n{display_name}:\n")
                for metric, stats in report["summary"][ref_type].items():
                    f.write(f"  {metric}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"    {stat_name}: {value:.2f}\n" if isinstance(value, float) else f"    {stat_name}: {value}\n")
    
    logging.info(f"Сводка сохранена в {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Анализ результатов тестов AEC и генерация отчетов")
    parser.add_argument("--test-dir", "-d", required=True, 
                      help="Директория с файлом summary_results.json")
    parser.add_argument("--output-dir", "-o", default=None, 
                      help="Директория для сохранения отчета (по умолчанию: {test_dir}/report_output)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                      help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Настраиваем уровень логирования
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    
    logging.info("Запуск анализа результатов тестов AEC")
    
    # Проверяем наличие указанной директории
    if not os.path.exists(args.test_dir):
        logging.error(f"Указанная директория не существует: {args.test_dir}")
        sys.exit(1)
    
    # Определяем директорию для сохранения отчета
    if args.output_dir is None:
        # По умолчанию создаем директорию report_output внутри test_dir
        output_dir = os.path.join(args.test_dir, "report_output")
    else:
        output_dir = args.output_dir
    
    # Загружаем результаты тестов из summary_results.json
    results = load_test_results(args.test_dir)
    
    if not results:
        logging.error("Не найдены результаты тестов. Завершение.")
        return
    
    # Группируем результаты по параметрам
    grouped_results = group_results_by_params(results)
    
    # Генерируем сводный отчет
    report = generate_summary_report(grouped_results)
    
    # Сохраняем отчет
    save_report(report, output_dir)
    
    # Генерируем графики
    generate_comparison_charts(grouped_results, output_dir)
    
    logging.info(f"Анализ завершен. Отчет сгенерирован в {output_dir}")


if __name__ == "__main__":
    main() 