import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
from typing import Optional, Tuple, Dict, List, Any
from utils import log_file_info
from plot import *

def calculate_signal_delay(in_channel, ref_channel, frame_rate):
    """
    Вычисляет задержку между входным и референсным сигналами.
    
    Args:
        in_channel: Входной сигнал (numpy массив)
        ref_channel: Референсный сигнал (numpy массив)
        frame_rate: Частота дискретизации
        
    Returns:
        tuple: (lag, delay_ms, correlation, lags, confidence) - 
               lag - задержка в семплах,
               delay_ms - задержка в миллисекундах,
               correlation - массив корреляции,
               lags - массив лагов,
               confidence - уверенность определения задержки
    """
    # Вычисляем кросс-корреляцию
    correlation = signal.correlate(in_channel, ref_channel, mode='full')
    lags = signal.correlation_lags(len(in_channel), len(ref_channel), mode='full')
    
    # Находим индекс максимальной корреляции только с положительной задержкой
    # Положительные лаги означают, что входной сигнал задержан относительно референсного
    positive_lags_indices = np.where(lags > 0)[0]
    
    if len(positive_lags_indices) > 0:
        # Ищем максимум корреляции только среди положительных лагов
        max_corr_idx = positive_lags_indices[np.argmax(np.abs(correlation[positive_lags_indices]))]
        lag = lags[max_corr_idx]
        
        # Логируем информацию об использовании только положительных значений
        logging.info(f"Определение задержки по положительным значениям лагов: выбран лаг {lag}")
    else:
        # Для случая, когда все лаги отрицательные или нулевые (редкая ситуация)
        # Возвращаемся к исходному поведению
        logging.warning("Не найдено положительных значений лагов, используется прежний метод поиска")
        center_index = len(lags) // 2  # Индекс, соответствующий нулевой задержке
        positive_lags_indices = np.where(lags >= 0)[0]
        max_corr_idx = positive_lags_indices[np.argmax(np.abs(correlation[positive_lags_indices]))]
        lag = lags[max_corr_idx]
    
    # Вычисляем задержку в миллисекундах
    delay_ms = lag * 1000 / frame_rate
    
    # Вычисляем уверенность (нормализованная корреляция)
    max_corr = np.abs(correlation[max_corr_idx])
    auto_corr_ref = np.sum(ref_channel**2)
    auto_corr_in = np.sum(in_channel**2)
    confidence = max_corr / np.sqrt(auto_corr_ref * auto_corr_in) if auto_corr_ref > 0 and auto_corr_in > 0 else 0
    
    logging.info(f"Обнаружена задержка: {delay_ms:.2f} мс (лаг: {lag} семплов, уверенность: {confidence:.2f})")
    
    return lag, delay_ms, correlation, lags, confidence


def format_array_info(array_name, array, frame_rate):
    """
    Форматирует информацию о массиве аудиоданных в понятное описание.
    
    Args:
        array_name: Название массива
        array: NumPy массив с аудиоданными
        frame_rate: Частота дискретизации
        
    Returns:
        str: Отформатированное описание
    """
    if array is None:
        return f"{array_name}: массив не предоставлен"
        
    if len(array.shape) > 1:  # Стерео
        format_type = "стерео"
        n_samples = array.shape[0]
        n_channels = array.shape[1]
        shape_str = f"{array.shape} (сэмплов: {n_samples}, каналов: {n_channels})"
    else:  # Моно
        format_type = "моно"
        n_samples = len(array)
        shape_str = f"{array.shape} (сэмплов: {n_samples})"
    
    duration_sec = n_samples / frame_rate
    duration_ms = n_samples * 1000 / frame_rate
    
    return f"{array_name}: {shape_str}, формат: {format_type}, длительность: {duration_sec:.3f} с ({duration_ms:.1f} мс)"

def calculate_and_log_statistics(
    in_array,
    ref_array,
    processed_array,
    processed_data,
    delay_ms,
    lag,
    results
):
    """
    Вычисляет и логирует статистику по аудио сигналам.
    
    Args:
        in_array: Массив входного сигнала
        ref_array: Массив референсного сигнала
        processed_array: Массив обработанного сигнала
        processed_data: Исходные обработанные данные в байтах (для проверки наличия)
        delay_ms: Задержка в миллисекундах
        lag: Задержка в семплах
        results: Словарь для сохранения результатов
        
    Returns:
        dict: Обновленный словарь с результатами
    """
    # Вычисляем RMS для входного и референсного сигналов
    in_rms = np.sqrt(np.mean(in_array**2))
    ref_rms = np.sqrt(np.mean(ref_array**2))
    
    # Вычисляем разницу между входным и референсным сигналами
    # Обеспечиваем одинаковую длину массивов
    min_length = min(len(in_array), len(ref_array))
    diff_array = in_array[:min_length] - ref_array[:min_length]
    diff_rms = np.sqrt(np.mean(diff_array**2))
    
    # Логируем основную статистику
    logging.info("Статистика сигналов:")
    logging.info(f"  RMS входного сигнала: {in_rms:.6f}")
    logging.info(f"  RMS референсного сигнала: {ref_rms:.6f}")
    logging.info(f"  Соотношение вход/референс: {in_rms/ref_rms if ref_rms > 0 else 0:.6f}")
    logging.info(f"  RMS разницы (вход - референс): {diff_rms:.6f}")
    logging.info(f"  Обнаруженная задержка: {delay_ms:.2f} мс")
    
    # Обновляем результаты
    results.update({
        "delay_ms": delay_ms,
        "input": {
            "rms": in_rms
        },
        "reference": {
            "rms": ref_rms
        },
        "original_diff": {
            "rms": diff_rms
        }
    })
    
    # Если есть обработанные данные, вычисляем дополнительную статистику
    if processed_data is not None:
        # Корректируем обработанный сигнал с учетом задержки
        if lag >= 0:
            # Входной сигнал задержан относительно референсного
            corrected_proc_array = np.roll(processed_array, -lag)
            corrected_ref_array = ref_array
        else:
            # Референсный сигнал задержан относительно входного
            corrected_ref_array = np.roll(ref_array, lag)
            corrected_proc_array = processed_array
        
        # Вычисляем RMS обработанного сигнала
        proc_rms = np.sqrt(np.mean(corrected_proc_array**2))
        
        # Вычисляем разницу между обработанным и референсным сигналами
        min_length = min(len(corrected_proc_array), len(corrected_ref_array))
        diff_processed = corrected_proc_array[:min_length] - corrected_ref_array[:min_length]
        diff_proc_rms = np.sqrt(np.mean(diff_processed**2))
        
        # Логируем статистику обработанного сигнала
        logging.info(f"  RMS обработанного сигнала: {proc_rms:.6f}")
        logging.info(f"  RMS разницы (обработанный - референс): {diff_proc_rms:.6f}")
        
        # Вычисляем улучшение после обработки
        improvement = None
        if diff_rms > 0:
            improvement = diff_rms / diff_proc_rms if diff_proc_rms > 0 else float('inf')
            logging.info(f"  Улучшение после обработки: {improvement:.2f}x ({20*np.log10(improvement):.2f} дБ)")
        
        # Добавляем информацию об обработанном сигнале в результаты
        results["processed"] = {
            "rms": proc_rms,
            "diff_rms": diff_proc_rms,
            "improvement": improvement
        }
    
    return results

def visualize_one_channel(
    plt_figure,
    channel_num,
    num_plots,
    num_cols,
    ref_array,
    ref_by_micro_volumed_array,
    ref_by_micro_volumed_delayed_array,
    in_array,
    processed_array,
    time_axis_ms,
    frame_rate,
    metrics=None,
    ref_delay_ms=None
):
    """
    Визуализирует графики для одного канала аудио.
    
    Args:
        plt_figure: Matplotlib figure для построения графиков
        channel_num: Номер канала (1 или 2)
        num_plots: Общее количество графиков в фигуре
        num_cols: Количество колонок в фигуре
        ref_array: Массив оригинального референсного сигнала
        ref_by_micro_volumed_array: Массив референсного сигнала с измененной громкостью
        ref_by_micro_volumed_delayed_array: Массив задержанного референсного сигнала
        in_array: Массив входного сигнала
        processed_array: Массив обработанного сигнала
        time_axis_ms: Временная ось в миллисекундах
        frame_rate: Частота дискретизации
        metrics: Предварительно рассчитанные метрики (необязательно)
        ref_delay_ms: Задержка референсного сигнала (опционально)
    """
    # Извлекаем данные для конкретного канала
    if len(ref_array.shape) > 1 and ref_array.shape[1] > channel_num - 1:
        ref_channel = ref_array[:, channel_num - 1]
    else:
        ref_channel = ref_array
        
    if len(ref_by_micro_volumed_array.shape) > 1 and ref_by_micro_volumed_array.shape[1] > channel_num - 1:
        ref_by_micro_volumed_channel = ref_by_micro_volumed_array[:, channel_num - 1]
    else:
        ref_by_micro_volumed_channel = ref_by_micro_volumed_array
        
    if len(ref_by_micro_volumed_delayed_array.shape) > 1 and ref_by_micro_volumed_delayed_array.shape[1] > channel_num - 1:
        ref_by_micro_volumed_delayed_channel = ref_by_micro_volumed_delayed_array[:, channel_num - 1]
    else:
        ref_by_micro_volumed_delayed_channel = ref_by_micro_volumed_delayed_array
        
    if len(in_array.shape) > 1 and in_array.shape[1] > channel_num - 1:
        in_channel = in_array[:, channel_num - 1]
    else:
        in_channel = in_array
        
    if len(processed_array.shape) > 1 and processed_array.shape[1] > channel_num - 1:
        processed_channel = processed_array[:, channel_num - 1]
    else:
        processed_channel = processed_array
        
    # Находим минимальную длину массивов для корректного отображения
    min_display_length = min(len(ref_channel), len(in_channel), len(processed_channel))
    
    # Обрезаем массивы до минимальной длины
    ref_channel = ref_channel[:min_display_length]
    in_channel = in_channel[:min_display_length]
    processed_channel = processed_channel[:min_display_length]
    ref_by_micro_volumed_channel = ref_by_micro_volumed_channel[:min_display_length]
    ref_by_micro_volumed_delayed_channel = ref_by_micro_volumed_delayed_channel[:min_display_length]
    
    # Вычисляем задержку между сигналами, если не предоставлены метрики
    if metrics and all(k in metrics for k in ['delay_samples', 'delay_ms', 'confidence', 'delay_correlation', 'delay_lags']):
        # Используем предрассчитанные данные
        lag = metrics['delay_samples']
        delay_ms = metrics['delay_ms']
        confidence = metrics['confidence']
        correlation = metrics['delay_correlation']
        lags = metrics['delay_lags']
    else:
        # Рассчитываем корреляцию для текущего канала
        lag, delay_ms, correlation, lags, confidence = calculate_signal_delay(in_channel, ref_channel, frame_rate)
    
    # Вычисляем позицию в сетке для текущего канала
    col_offset = channel_num - 1
    
    # 1. График двух референсных сигналов на входе в микро (до и после задержки)
    plot_reference_signals(
        plt_figure,
        (num_plots, num_cols, 1 + col_offset),
        ref_by_micro_volumed_channel = {
            'channel': ref_by_micro_volumed_channel,
            'audio_file': 'ref_by_micro_volumed.wav'
        },
        ref_by_micro_volumed_delayed_channel = {
            'channel': ref_by_micro_volumed_delayed_channel,
            'audio_file': 'ref_by_micro_volumed_delayed.wav'
        },
        sample_rate=frame_rate,
        ref_delay_ms=ref_delay_ms,
        channel_num=channel_num
    )
    
    # 2. График кросс-корреляции между обычным и сдвинутым референсными сигналами
    plot_reference_correlation(
        plt_figure,
        (num_plots, num_cols, num_cols + 1 + col_offset),
        ref_by_micro_volumed_channel,
        ref_by_micro_volumed_delayed_channel,
        frame_rate,
        channel_num=channel_num
    )
    
    # 3. График оригинального и на входе в микро референсных сигналов
    plot_original_and_by_micro_volumed_reference_signals(
        plt_figure,
        (num_plots, num_cols, num_cols*2 + 1 + col_offset),
        ref_channel = {
            'channel': ref_channel,
            'audio_file': 'reference.wav'
        },
        ref_by_micro_volumed_channel = {
            'channel': ref_by_micro_volumed_channel,
            'audio_file': 'ref_by_micro_volumed.wav'
        },
        sample_rate=frame_rate,
        channel_num=channel_num
    )
    
    # 4. График корреляции оригинального и на входе в микро референсных сигналов
    ref_delay_ms = plot_original_and_by_micro_volumed_reference_correlation(
        plt_figure,
        (num_plots, num_cols, num_cols*3 + 1 + col_offset),
        ref_channel,
        ref_by_micro_volumed_channel,
        frame_rate,
        channel_num=channel_num
    )
    
    # 5. График оригинального референсного и входного сигналов
    plot_corrected_original_and_by_micro_volumed_reference_signals(
        plt_figure,
        (num_plots, num_cols, num_cols*4 + 1 + col_offset),
        reference_channel = {
            'channel': ref_channel,
            'audio_file': 'reference.wav',
        },
        ref_by_micro_volumed_channel = {
            'channel': ref_by_micro_volumed_channel,
            'audio_file': 'ref_by_micro_volumed.wav',
        },
        sample_rate=frame_rate,
        lag=lag,
        delay_ms=ref_delay_ms,
        channel_num=channel_num
    )
    
    # 6. Исходные сигналы с указанием задержки
    plot_original_signals(
        plt_figure,
        (num_plots, num_cols, num_cols*5 + 1 + col_offset),
        ref_channel ={
            'channel': ref_channel,
            'audio_file': 'reference.wav'
        },
        in_channel = {
            'channel': in_channel,
            'audio_file': 'original_input.wav'
        },
        time_axis_ms=time_axis_ms,
        delay_ms=delay_ms,
        channel_num=channel_num
    )
    
    # 7. Кросс-корреляция между входным и референсным сигналами
    plot_input_reference_correlation(
        plt_figure,
        (num_plots, num_cols, num_cols*6 + 1 + col_offset),
        lags,
        correlation,
        delay_ms,
        confidence,
        frame_rate,
        channel_num=channel_num
    )
    
    # 8. Сигналы после корректировки задержки
    plot_corrected_signals(
        plt_figure,
        (num_plots, num_cols, num_cols*7 + 1 + col_offset),
        ref_channel = {
            'channel': ref_channel,
            'audio_file': 'reference.wav'
        },
        in_channel = {
            'channel': in_channel,
            'audio_file': 'original_input.wav'
        },
        time_axis_ms=time_axis_ms,
        lag=lag,
        delay_ms=delay_ms,
        channel_num=channel_num
    )
    
    # 9. Сравнение исходного и обработанного сигналов
    plot_processed_signals(
        plt_figure,
        (num_plots, num_cols, num_cols*8 + 1 + col_offset),
        in_channel = {
            'channel': in_channel,
            'audio_file': 'original_input.wav',
        },
        processed_channel = {
            'channel': processed_channel,
            'audio_file': 'processed_input.wav',
        },
        time_axis_ms=time_axis_ms,
    )
    
    # Возвращаем данные о задержке для использования в статистике
    return lag, delay_ms, correlation, lags, confidence

def visualize_audio_processing(
    output_dir: str,
    reference_file_info: Optional[Dict[str, Any]],
    reference_by_micro_volumed_file_info: Optional[Dict[str, Any]],
    reference_by_micro_volumed_delayed_file_info: Optional[Dict[str, Any]],
    original_input_file_info: Optional[Dict[str, Any]],
    processed_file_info: Optional[Dict[str, Any]],
    input_file_path: Optional[str] = None,
    output_prefix: str = "aec",
    frame_rate: int = 16000,
    n_channels: int = 1,
    metrics: Optional[Dict[str, Any]] = None
):
    """
    Единая функция для визуализации результатов обработки аудио
    
    Args:
        output_dir: Директория для сохранения результатов
        reference_data: Данные оригинального референсного сигнала (байты)
        reference_file_path: Путь к оригинальному референсному файлу
        reference_by_micro_volumed_data: Данные референсного сигнала с измененной громкостью (байты)
        reference_by_micro_volumed_delayed_data: Данные задержанного референсного сигнала (байты)
        input_data: Входные данные (байты)
        input_file_path: Путь к входному файлу
        processed_data: Обработанные данные (байты)
        reference_delayed_data: Устаревший параметр, используйте reference_by_micro_volumed_delayed_data
        output_prefix: Префикс для имен выходных файлов
        frame_rate: Частота дискретизации
        n_channels: Количество каналов (1 для моно, 2 для стерео)
        metrics: Метрики обработки, включая данные корреляции
    """
    results = {}
    
    try:
        # Создаем директорию для результатов, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Логируем полученную частоту дискретизации и количество каналов
        logging.info(f"Получена частота дискретизации {frame_rate} Гц")
        logging.info(f"Начальное количество каналов: {n_channels}")
        
        # Если задан n_channels=1 (значение по умолчанию), пробуем определить из файлов
        if n_channels == 1:
            # Проверяем файлы для определения количества каналов через log_file_info
            _, detected_n_channels = log_file_info(input_file_path, "Проверка каналов входного файла")
            if detected_n_channels and detected_n_channels in [1, 2]:
                n_channels = detected_n_channels
                logging.info(f"visualize_audio_processing: Определено количество каналов из файла: {n_channels}")
        
        # Проверяем корректность значения
        if n_channels not in [1, 2]:
            logging.warning(f"Неверное количество каналов ({n_channels}), используем значение по умолчанию: 1")
            n_channels = 1

        # Логируем размеры данных
        if reference_by_micro_volumed_file_info:
            logging.info(f"  Размер reference_by_micro_volumed_data: {len(reference_by_micro_volumed_file_info['raw_frames'])} байт")
        if original_input_file_info:
            logging.info(f"  Размер input_data: {len(original_input_file_info['raw_frames'])} байт")

        # Передаем количество каналов в функцию bytes_to_numpy
        ref_array = bytes_to_numpy(reference_file_info['raw_frames'], frame_rate, n_channels)
        ref_by_micro_volumed_array = bytes_to_numpy(reference_by_micro_volumed_file_info['raw_frames'], frame_rate, n_channels)
        ref_by_micro_volumed_delayed_array = bytes_to_numpy(reference_by_micro_volumed_delayed_file_info['raw_frames'], frame_rate, n_channels)
        in_array = bytes_to_numpy(original_input_file_info['raw_frames'], frame_rate, n_channels)
        processed_array = bytes_to_numpy(processed_file_info['raw_frames'], frame_rate, n_channels)
        
        # Логируем подробную информацию о массивах
        logging.info(format_array_info("Референсный массив с измененной громкостью (ref_by_micro_volumed_array)", ref_by_micro_volumed_array, frame_rate))
        logging.info(format_array_info("Задержанный референсный массив (ref_by_micro_volumed_delayed_array)", ref_by_micro_volumed_delayed_array, frame_rate))
        logging.info(format_array_info("Входной массив (in_array)", in_array, frame_rate))
        logging.info(format_array_info("Обработанный массив (processed_array)", processed_array, frame_rate))
        
        # Вычисляем длительность в зависимости от формата данных
        if len(ref_by_micro_volumed_array.shape) > 1:  # Стерео
            ref_by_micro_volumed_duration = ref_by_micro_volumed_array.shape[0] / frame_rate
        else:  # Моно
            ref_by_micro_volumed_duration = len(ref_by_micro_volumed_array) / frame_rate

        if len(ref_by_micro_volumed_delayed_array.shape) > 1:  # Стерео
            ref_by_micro_volumed_delayed_duration = ref_by_micro_volumed_delayed_array.shape[0] / frame_rate
        else:  # Моно
            ref_by_micro_volumed_delayed_duration = len(ref_by_micro_volumed_delayed_array) / frame_rate            

        duration_diff_ms = ref_by_micro_volumed_delayed_duration - ref_by_micro_volumed_duration
        logging.info(f"Разница длительностей между референсным и задержанным референсным файлами: {duration_diff_ms:.2f} мс")

        # Проверяем наличие второго канала хотя бы в одном из сигналов
        has_second_channel = any([
            len(ref_array.shape) > 1 and ref_array.shape[1] > 1,
            len(ref_by_micro_volumed_array.shape) > 1 and ref_by_micro_volumed_array.shape[1] > 1,
            len(ref_by_micro_volumed_delayed_array.shape) > 1 and ref_by_micro_volumed_delayed_array.shape[1] > 1,
            len(in_array.shape) > 1 and in_array.shape[1] > 1,
            len(processed_array.shape) > 1 and processed_array.shape[1] > 1
        ])
        
        logging.info(f"Наличие второго канала: {has_second_channel}")

        # Вычисляем длительность сигналов в миллисекундах для лога
        ref_by_micro_volumed_duration_ms = len(ref_by_micro_volumed_array) * 1000 / frame_rate

        in_duration_ms = len(in_array) * 1000 / frame_rate

        ref_by_micro_volumed_delayed_duration_ms = len(ref_by_micro_volumed_delayed_array) * 1000 / frame_rate

        # Логируем информацию о длительности
        logging.info(f"Длительность reference_by_micro_volumed.wav: {ref_by_micro_volumed_duration_ms:.2f} мс")
        logging.info(f"Длительность original_input.wav: {in_duration_ms:.2f} мс")
        logging.info(f"Длительность reference_by_micro_volumed_delayed.wav: {ref_by_micro_volumed_delayed_duration_ms:.2f} мс")
        
        # Проверяем минимальную длину для создания временной оси
        min_length = min(
            len(ref_array.flatten()) if len(ref_array.shape) > 0 else 0,
            len(ref_by_micro_volumed_array.flatten()) if len(ref_by_micro_volumed_array.shape) > 0 else 0,
            len(ref_by_micro_volumed_delayed_array.flatten()) if len(ref_by_micro_volumed_delayed_array.shape) > 0 else 0,
            len(in_array.flatten()) if len(in_array.shape) > 0 else 0,
            len(processed_array.flatten()) if len(processed_array.shape) > 0 else 0
        )
        
        # Создаем временную ось подходящей длины
        time_axis = np.arange(min_length // (1 if n_channels == 1 else 2)) / frame_rate
        # Создаем временную ось в миллисекундах
        time_axis_ms = time_axis * 1000  # Преобразуем секунды в миллисекунды
        
        logging.info(f"Создана временная ось длиной {len(time_axis_ms)} точек")
        
        # Определяем количество графиков
        num_plots = 9  # Базовое количество графиков
        
        # Определяем количество колонок в зависимости от наличия второго канала
        num_cols = 2 if has_second_channel else 1

        # Создаем график с учетом второго канала
        one_plot_height = 6
        one_plot_width = 30
        fig_width = 2 * one_plot_width if has_second_channel else one_plot_width  # Шире если есть второй канал
        fig_height = one_plot_height * num_plots  # Высота остается прежней
        
        plt.figure(figsize=(fig_width, fig_height))

        # Визуализируем первый канал и получаем данные о задержке
        lag, delay_ms, correlation, lags, confidence = visualize_one_channel(
            plt,
            channel_num=1,
            num_plots=num_plots,
            num_cols=num_cols,
            ref_array=ref_array,
            ref_by_micro_volumed_array=ref_by_micro_volumed_array,
            ref_by_micro_volumed_delayed_array=ref_by_micro_volumed_delayed_array,
            in_array=in_array,
            processed_array=processed_array,
            time_axis_ms=time_axis_ms,
            frame_rate=frame_rate,
            metrics=metrics,
            ref_delay_ms=None if 'ref_delay_ms' not in locals() else ref_delay_ms
        )
        
        # Если есть второй канал, визуализируем его
        if has_second_channel:
            visualize_one_channel(
                plt,
                channel_num=2,
                num_plots=num_plots,
                num_cols=num_cols,
                ref_array=ref_array,
                ref_by_micro_volumed_array=ref_by_micro_volumed_array,
                ref_by_micro_volumed_delayed_array=ref_by_micro_volumed_delayed_array,
                in_array=in_array,
                processed_array=processed_array,
                time_axis_ms=time_axis_ms,
                frame_rate=frame_rate,
                metrics=metrics,
                ref_delay_ms=None if 'ref_delay_ms' not in locals() else ref_delay_ms
            )

        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, f'{output_prefix}_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save each subplot as a separate file
        fig = plt.gcf()
        for i, ax in enumerate(fig.get_axes()):
            # Create a new figure for each subplot
            subplot_fig = plt.figure(figsize=(one_plot_width, one_plot_height))
            subplot_ax = subplot_fig.add_subplot(111)
            
            # Copy the content from the original subplot
            for line in ax.get_lines():
                subplot_ax.plot(line.get_xdata(), line.get_ydata(), 
                               color=line.get_color(), 
                               linestyle=line.get_linestyle(),
                               label=line.get_label(),
                               alpha=line.get_alpha())
            
            # Copy other elements like title, labels, grid, etc.
            subplot_ax.set_title(ax.get_title())
            subplot_ax.set_xlabel(ax.get_xlabel())
            subplot_ax.set_ylabel(ax.get_ylabel())
            subplot_ax.set_xlim(ax.get_xlim())
            subplot_ax.set_ylim(ax.get_ylim())
            
            # Copy grid settings - fix the AttributeError
            subplot_ax.grid(True)  # Just enable grid for all individual plots
            
            # Copy legend if present
            if ax.get_legend():
                subplot_ax.legend()
            
            # Copy vertical and horizontal lines
            for line in ax.get_lines():
                if len(line.get_xdata()) == 2:
                    if line.get_xdata()[0] == line.get_xdata()[1]:  # vertical line
                        subplot_ax.axvline(x=line.get_xdata()[0], color=line.get_color(), 
                                          linestyle=line.get_linestyle(), label=line.get_label())
            
            # Save individual plot
            plot_number = i + 1
            subplot_file = os.path.join(plots_dir, f'plot_{plot_number}.png')
            subplot_fig.tight_layout()
            subplot_fig.savefig(subplot_file)
            plt.close(subplot_fig)
            
        # Save the original combined figure
        signals_file = os.path.join(output_dir, f'{output_prefix}_signals.png')
        plt.savefig(signals_file)
        plt.close()
        logging.info(f"Визуализация сигналов сохранена в {signals_file}")
        logging.info(f"Отдельные графики сохранены в {plots_dir}")
        
        # Вычисляем и логируем статистику
        results = calculate_and_log_statistics(
            in_array,
            ref_by_micro_volumed_array,  # Используем ref_by_micro_volumed_array вместо ref_array
            processed_array, 
            processed_file_info['raw_frames'],
            delay_ms,
            lag,
            results
        )
    
    except ImportError:
        logging.warning("Не удалось импортировать необходимые библиотеки для визуализации")
    except Exception as e:
        logging.error(f"Ошибка при визуализации: {e}")
        logging.exception("Подробная информация об ошибке:")


def bytes_to_numpy(audio_bytes, frame_rate=16000, n_channels=2):
    """
    Преобразует аудио-данные из байтов в numpy массив.
    
    Args:
        audio_bytes: Аудио-данные в формате байтов
        frame_rate: Частота дискретизации
        n_channels: Количество каналов (1 для моно, 2 для стерео)
        
    Returns:
        numpy.ndarray: Аудио-данные в формате numpy массива
    """
    try:
        if audio_bytes is None:
            logging.warning("bytes_to_numpy: Получен None вместо байтовых данных")
            # Возвращаем пустой массив правильного формата
            return np.array([], dtype=np.float32)
            
        # Проверяем, что размер данных не нулевой
        if len(audio_bytes) == 0:
            logging.warning("bytes_to_numpy: Получен пустой массив байтов")
            return np.array([], dtype=np.float32)
            
        # Логируем размер входных данных в байтах
        logging.info(f"bytes_to_numpy: Получено {len(audio_bytes)} байт данных")
        
        # Проверяем, что длина байтов четная (для int16)
        if len(audio_bytes) % 2 != 0:
            logging.warning(f"bytes_to_numpy: Нечетное количество байтов: {len(audio_bytes)}. Обрезаем последний байт.")
            audio_bytes = audio_bytes[:-1]
        
        # Преобразуем байты в numpy массив
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Проверяем, что есть данные после преобразования
        if len(audio_array) == 0:
            logging.warning("bytes_to_numpy: После преобразования получен пустой массив")
            return np.array([], dtype=np.float32)
        
        # Преобразуем одномерный массив в двумерный для стерео данных
        if n_channels == 2:
            # Проверяем, что длина массива четная
            if len(audio_array) % 2 == 0:
                # Преобразуем в массив [n_samples, 2]
                audio_array = audio_array.reshape(-1, 2)
                logging.info(f"bytes_to_numpy: Преобразовано в стерео массив размером {audio_array.shape}")
            else:
                logging.warning(f"bytes_to_numpy: Нечетное количество элементов для стерео данных: {len(audio_array)}. Добавляем нулевой семпл.")
                # Добавляем один нулевой семпл, чтобы сделать массив четной длины
                audio_array = np.append(audio_array, [0])
                audio_array = audio_array.reshape(-1, 2)
    
        # Вычисляем длительность в зависимости от формата данных
        if len(audio_array.shape) > 1:  # Стерео
            duration = audio_array.shape[0] / frame_rate
        else:  # Моно
            duration = len(audio_array) / frame_rate
            
        logging.info(f"bytes_to_numpy: Длительность: {duration:.3f} сек")
        
        # Логируем статистику данных
        if len(audio_array) > 0:
            logging.info(f"bytes_to_numpy: Мин/Макс/Среднее значения: {np.min(audio_array):.2f}/{np.max(audio_array):.2f}/{np.mean(audio_array):.2f}")
        
        return audio_array
    except Exception as e:
        logging.error(f"Ошибка при преобразовании байтов в numpy массив: {e}")
        logging.exception("Подробная информация об ошибке:")
        # В случае ошибки возвращаем пустой массив
        return np.array([], dtype=np.float32)
