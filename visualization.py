import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
from typing import Optional, Tuple, Dict, List, Any
from utils import log_file_info

def plot_reference_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    ref_by_micro_volumed_delayed_channel,
    sample_rate,
    ref_delay_ms=None
):
    """
    Отображает график сравнения исходного и задержанного референсных сигналов.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_array: Массив с исходным референсным сигналом
        ref_by_micro_volumed_delayed_array: Массив с задержанным референсным сигналом
        sample_rate: Частота дискретизации
        ref_delay_ms: Задержка в мс между сигналами (если известна)
    """
    plt_figure.subplot(*subplot_position)
    
    # Определяем максимальную длительность для отображения (в мс)
    # Используем длину обработанного файла, но не менее 8000 мс и не более 40000 мс
    max_display_ms = max(8000, min(40000, max(len(ref_channel), len(ref_by_micro_volumed_delayed_channel)) * 1000 / sample_rate))
    display_samples = int(max_display_ms * sample_rate / 1000)
    
    # Определяем шаг деления оси X в зависимости от длительности
    if max_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для длительности до 10 секунд
    elif max_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для длительности от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для длительности более 20 секунд
    
    # Дополняем оба массива нулями до display_samples
    padded_ref = np.zeros(display_samples)
    padded_ref[:len(ref_channel)] = ref_channel
    
    padded_ref_delayed = np.zeros(display_samples)
    padded_ref_delayed[:len(ref_by_micro_volumed_delayed_channel)] = ref_by_micro_volumed_delayed_channel
    
    # Используем дополненные массивы
    ref_channel = padded_ref
    ref_by_micro_volumed_delayed_channel = padded_ref_delayed
    
    logging.info(f"Референсные сигналы дополнены нулями до {max_display_ms} мс ({display_samples} семплов)")
    logging.info(f"Исходная длина ref_channel: {len(ref_channel)} семплов")
    logging.info(f"Исходная длина ref_by_micro_volumed_delayed_channel: {len(ref_by_micro_volumed_delayed_channel)} семплов")
    
    # Создаем расширенную временную ось для отображения
    display_time_ms = np.arange(display_samples) * 1000 / sample_rate
    
    # Строим графики используя подготовленные массивы с одинаковой длиной
    plt_figure.plot(display_time_ms, ref_channel, label=f'Референс на входе в микро', alpha=0.7, color='blue')
    plt_figure.plot(display_time_ms, ref_by_micro_volumed_delayed_channel, label=f'Задержанный референс на входе в микро', alpha=0.7, color='red')
    
    # Если есть задержка, вычисленная по корреляции, отображаем её
    if ref_delay_ms is not None:
        plt_figure.axvline(x=ref_delay_ms, color='g', linestyle='--', 
                  label=f'Вычисленная задержка: {ref_delay_ms:.2f} мс')
    
    plt_figure.title(f'{subplot_position[-1]}. Сравнение референсных сигналов на входе в микро (с задержкой и без)')
    
    # Добавляем более детальные деления на оси X с динамическим шагом
    plt_figure.xticks(np.arange(0, max_display_ms + 1, tick_step_ms))
    plt_figure.xlim([0, max_display_ms])  # Динамический диапазон
    plt_figure.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
    
    plt_figure.xlabel('Время (мс)')
    plt_figure.ylabel('Амплитуда')
    plt_figure.grid(True)
    plt_figure.legend()

def plot_reference_correlation(
    plt_figure,
    subplot_position,
    ref_channel,
    ref_by_micro_volumed_delayed_channel,
    sample_rate
):
    """
    Отображает график кросс-корреляции между обычным и сдвинутым референсными сигналами.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_array: Массив с исходным референсным сигналом
        ref_by_micro_volumed_delayed_array: Массив с задержанным референсным сигналом
        sample_rate: Частота дискретизации
    
    Returns:
        float: Вычисленная задержка в миллисекундах
    """
    plt_figure.subplot(*subplot_position)

    # Определяем минимальную длину для корреляции (до 5 секунд)
    correlation_window = int(5 * sample_rate)  # 5 секунд для корреляции
    actual_correlation_window = min(correlation_window, len(ref_channel), len(ref_by_micro_volumed_delayed_channel))

    # Логируем информацию о корреляционном окне
    logging.info(f"Используем корреляционное окно: {actual_correlation_window} семплов ({actual_correlation_window/sample_rate:.1f} сек)")

    # Обрезаем оба сигнала до минимальной длины для корреляции
    ref_for_corr = ref_channel[:actual_correlation_window]
    ref_delayed_for_corr = ref_by_micro_volumed_delayed_channel[:actual_correlation_window]

    # Вычисляем корреляцию
    corr = np.correlate(ref_delayed_for_corr, ref_for_corr, 'full')
    corr_time = np.arange(len(corr)) - (len(ref_for_corr) - 1)
    corr_time_ms = corr_time * 1000.0 / sample_rate

    # Находим индекс середины корреляции (нулевая задержка)
    center_idx = len(ref_for_corr) - 1
    
    # Анализируем только положительные задержки (> 0 мс)
    positive_indices = np.where(corr_time > 0)[0]
    
    if len(positive_indices) > 0:
        # Находим максимальное значение корреляции только на положительных задержках
        max_corr_idx = positive_indices[np.argmax(np.abs(corr[positive_indices]))]
        ref_delay_samples = corr_time[max_corr_idx]
        ref_delay_ms = ref_delay_samples * 1000.0 / sample_rate
    else:
        # Если по какой-то причине нет положительных значений
        max_corr_idx = np.argmax(np.abs(corr))
        ref_delay_samples = corr_time[max_corr_idx]
        ref_delay_ms = ref_delay_samples * 1000.0 / sample_rate
        logging.warning("Не найдено положительных значений задержки, используется максимальная корреляция")

    # Логируем информацию о найденной задержке
    logging.info(f"Корреляция между референсными сигналами на входе в микро (с задержкой и без): макс.индекс={max_corr_idx}, задержка={ref_delay_samples} сэмплов ({ref_delay_ms:.2f} мс)")

    # Строим график корреляции
    plt_figure.plot(corr_time_ms, corr, color='green')
    plt_figure.axvline(x=ref_delay_ms, color='r', linestyle='--', 
                label=f'Измеренная задержка: {ref_delay_ms:.2f} мс')

    plt_figure.title(f'{subplot_position[-1]}. Кросс-корреляция между референсными сигналами на входе в микро (с задержкой и без)')
    plt_figure.xlabel('Задержка (мс)')
    plt_figure.ylabel('Корреляция')
    plt_figure.legend()

    # Определяем максимальный диапазон для отображения корреляции и шаг
    max_corr_display_ms = max(1500, min(ref_delay_ms * 2, 5000))
    
    # Определяем шаг деления оси X в зависимости от диапазона
    if max_corr_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для диапазона до 10 секунд
    elif max_corr_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для диапазона от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для диапазона более 20 секунд
    
    # Устанавливаем диапазон по X с учетом найденной задержки
    # Отображаем как отрицательную часть (для контекста), так и положительную (где ищем задержку)
    plt_figure.xlim([-100, max_corr_display_ms])
    plt_figure.xticks(np.arange(-100, max_corr_display_ms + 1, tick_step_ms))
    # Добавляем сетку
    plt_figure.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Добавляем вертикальную линию на нулевой задержке для наглядности
    plt_figure.axvline(x=0, color='gray', linestyle=':', 
                label='Нулевая задержка')
    
    return ref_delay_ms

def plot_original_and_by_micro_volumed_reference_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    ref_by_micro_volumed_channel,
    sample_rate
):
    """
    Отображает график сравнения оригинального референсного сигнала и 
    референсного сигнала с измененной громкостью на входе в микрофон.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_channel: Массив с оригинальным референсным сигналом
        ref_by_micro_volumed_channel: Массив с референсным сигналом измененной громкости
        sample_rate: Частота дискретизации
    """
    plt_figure.subplot(*subplot_position)
    
    # Определяем максимальную длительность для отображения (в мс)
    # Используем длину обработанного файла, но не менее 8000 мс и не более 40000 мс
    max_display_ms = max(8000, min(40000, max(len(ref_channel), len(ref_by_micro_volumed_channel)) * 1000 / sample_rate))
    display_samples = int(max_display_ms * sample_rate / 1000)
    
    # Определяем шаг деления оси X в зависимости от длительности
    if max_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для длительности до 10 секунд
    elif max_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для длительности от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для длительности более 20 секунд
    
    # Дополняем оба массива нулями до display_samples
    padded_ref = np.zeros(display_samples)
    padded_ref[:len(ref_channel)] = ref_channel
    
    padded_ref_by_micro_volumed = np.zeros(display_samples)
    padded_ref_by_micro_volumed[:len(ref_by_micro_volumed_channel)] = ref_by_micro_volumed_channel
    
    # Используем дополненные массивы
    ref_channel = padded_ref
    ref_by_micro_volumed_channel = padded_ref_by_micro_volumed
    
    logging.info(f"Оригинальный и с измененной громкостью референсные сигналы дополнены нулями до {max_display_ms} мс ({display_samples} семплов)")
    
    # Создаем расширенную временную ось для отображения
    display_time_ms = np.arange(display_samples) * 1000 / sample_rate
    
    # Строим графики используя подготовленные массивы с одинаковой длиной
    plt_figure.plot(display_time_ms, ref_channel, label=f'Оригинальный референс', alpha=0.7, color='blue')
    plt_figure.plot(display_time_ms, ref_by_micro_volumed_channel, label=f'Референс на входе в микро и другой громкостью', alpha=0.7, color='purple')
    
    plt_figure.title(f'{subplot_position[-1]}. Сравнение оригинального и с измененной громкостью референсных сигналов')
    
    # Добавляем более детальные деления на оси X с динамическим шагом
    plt_figure.xticks(np.arange(0, max_display_ms + 1, tick_step_ms))
    plt_figure.xlim([0, max_display_ms])  # Динамический диапазон
    plt_figure.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
    
    plt_figure.xlabel('Время (мс)')
    plt_figure.ylabel('Амплитуда')
    plt_figure.grid(True)
    plt_figure.legend()

def plot_original_reference_and_input_signals(
    plt_figure,
    subplot_position,
    ref_by_micro_volumed_delayed_channel,
    in_channel,
    sample_rate
):
    """
    Отображает график сравнения задержанного референсного сигнала с измененной 
    громкостью и входного сигнала микрофона.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_by_micro_volumed_delayed_channel: Массив с задержанным референсным сигналом
        in_channel: Массив с входным сигналом микрофона
        sample_rate: Частота дискретизации
    """
    plt_figure.subplot(*subplot_position)
    
    # Определяем максимальную длительность для отображения (в мс)
    # Используем длину обработанного файла, но не менее 8000 мс и не более 40000 мс
    max_display_ms = max(8000, min(40000, max(len(ref_by_micro_volumed_delayed_channel), len(in_channel)) * 1000 / sample_rate))
    display_samples = int(max_display_ms * sample_rate / 1000)
    
    # Определяем шаг деления оси X в зависимости от длительности
    if max_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для длительности до 10 секунд
    elif max_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для длительности от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для длительности более 20 секунд
    
    # Дополняем оба массива нулями до display_samples
    padded_ref_delayed = np.zeros(display_samples)
    padded_ref_delayed[:len(ref_by_micro_volumed_delayed_channel)] = ref_by_micro_volumed_delayed_channel
    
    padded_in = np.zeros(display_samples)
    padded_in[:len(in_channel)] = in_channel
    
    # Используем дополненные массивы
    ref_by_micro_volumed_delayed_channel = padded_ref_delayed
    in_channel = padded_in
    
    logging.info(f"Задержанный референсный и входной сигналы дополнены нулями до {max_display_ms} мс ({display_samples} семплов)")
    
    # Создаем расширенную временную ось для отображения
    display_time_ms = np.arange(display_samples) * 1000 / sample_rate
    
    # Строим графики используя подготовленные массивы с одинаковой длиной
    plt_figure.plot(display_time_ms, ref_by_micro_volumed_delayed_channel, label=f'Задержанный референс на входе в микро', alpha=0.7, color='red')
    plt_figure.plot(display_time_ms, in_channel, label=f'Входной сигнал', alpha=0.7, color='green')
    
    plt_figure.title(f'{subplot_position[-1]}. Задержанный референсный сигнал в микро и входной сигнал')
    
    # Добавляем более детальные деления на оси X с динамическим шагом
    plt_figure.xticks(np.arange(0, max_display_ms + 1, tick_step_ms))
    plt_figure.xlim([0, max_display_ms])  # Динамический диапазон
    plt_figure.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
    
    plt_figure.xlabel('Время (мс)')
    plt_figure.ylabel('Амплитуда')
    plt_figure.grid(True)
    plt_figure.legend()

def plot_input_reference_correlation(
    plt_figure,
    subplot_position,
    lags,
    correlation,
    delay_ms,
    confidence,
    sample_rate
):
    """
    Отображает график кросс-корреляции между входным и референсным сигналами.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        lags: Массив задержек
        correlation: Массив значений корреляции
        delay_ms: Вычисленная задержка в миллисекундах
        confidence: Уверенность определения задержки
        sample_rate: Частота дискретизации
    """
    plt_figure.subplot(*subplot_position)
    # Преобразуем лаги в миллисекунды для оси X
    lags_ms = lags * 1000 / sample_rate
    plt_figure.plot(lags_ms, np.abs(correlation), color='green')
    plt_figure.axvline(x=delay_ms, color='r', linestyle='--', 
                label=f'Измеренная задержка: {delay_ms:.2f} мс, уверенность: {confidence:.2f}')
                
    # Добавляем вертикальную линию на нулевой задержке для наглядности
    plt_figure.axvline(x=0, color='gray', linestyle=':', 
                label='Нулевая задержка')

    plt_figure.title(f'{subplot_position[-1]}. Кросс-корреляция между входным и референсным сигналами')
    plt_figure.xlabel('Задержка (мс)')
    plt_figure.ylabel('Корреляция')
    plt_figure.legend()

    # Определяем максимальный диапазон для отображения корреляции и шаг
    max_corr_display_ms = max(1500, min(delay_ms * 2, 5000))
    
    # Определяем шаг деления оси X в зависимости от диапазона
    if max_corr_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для диапазона до 10 секунд
    elif max_corr_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для диапазона от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для диапазона более 20 секунд
    
    # Устанавливаем диапазон по X от -100 до динамического максимума
    plt_figure.xlim([-100, max_corr_display_ms])
    plt_figure.xticks(np.arange(-100, max_corr_display_ms + 1, tick_step_ms))
    # Добавляем сетку
    plt_figure.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Добавляем аннотацию о том, что анализ выполняется только для положительных задержек
    plt_figure.annotate("Учитываются только положительные значения задержки",
                    xy=(0.02, 0.95), xycoords='axes fraction', 
                    color='black', fontsize=9, alpha=0.8)

def plot_original_and_by_micro_volumed_reference_correlation(
    plt_figure,
    subplot_position,
    ref_channel,
    ref_by_micro_volumed_channel,
    sample_rate
):
    """
    Отображает график кросс-корреляции между оригинальным и изменённым референсными сигналами.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_channel: Массив с оригинальным референсным сигналом
        ref_by_micro_volumed_channel: Массив с референсным сигналом измененной громкости
        sample_rate: Частота дискретизации
    
    Returns:
        float: Вычисленная задержка в миллисекундах
    """
    plt_figure.subplot(*subplot_position)

    # Определяем минимальную длину для корреляции (до 5 секунд)
    correlation_window = int(5 * sample_rate)
    actual_correlation_window = min(correlation_window, len(ref_channel), len(ref_by_micro_volumed_channel))

    # Обрезаем оба сигнала до минимальной длины для корреляции
    ref_for_corr = ref_channel[:actual_correlation_window]
    ref_by_micro_for_corr = ref_by_micro_volumed_channel[:actual_correlation_window]

    # Вычисляем корреляцию
    corr = np.correlate(ref_by_micro_for_corr, ref_for_corr, 'full')
    corr_time = np.arange(len(corr)) - (len(ref_for_corr) - 1)
    corr_time_ms = corr_time * 1000.0 / sample_rate

    # Анализируем только положительные задержки (> 0 мс)
    positive_indices = np.where(corr_time > 0)[0]
    
    if len(positive_indices) > 0:
        # Находим максимальное значение корреляции только на положительных задержках
        max_corr_idx = positive_indices[np.argmax(np.abs(corr[positive_indices]))]
        ref_delay_samples = corr_time[max_corr_idx]
        ref_delay_ms = ref_delay_samples * 1000.0 / sample_rate
    else:
        # Если по какой-то причине нет положительных значений
        max_corr_idx = np.argmax(np.abs(corr))
        ref_delay_samples = corr_time[max_corr_idx]
        ref_delay_ms = ref_delay_samples * 1000.0 / sample_rate
        logging.warning("Не найдено положительных значений задержки в оригинальной корреляции, используется максимальная корреляция")

    # Строим график корреляции
    plt_figure.plot(corr_time_ms, corr, color='blue')
    plt_figure.axvline(x=ref_delay_ms, color='r', linestyle='--', 
                label=f'Измеренная задержка: {ref_delay_ms:.2f} мс')

    plt_figure.title(f'{subplot_position[-1]}. Кросс-корреляция между референсным сигналом и референсным сигналом на входе в микро')
    plt_figure.xlabel('Задержка (мс)')
    plt_figure.ylabel('Корреляция')
    plt_figure.legend()

    # Определяем максимальный диапазон для отображения корреляции и шаг
    max_corr_display_ms = max(1500, min(ref_delay_ms * 2, 5000))
    
    # Определяем шаг деления оси X в зависимости от диапазона
    if max_corr_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для диапазона до 10 секунд
    elif max_corr_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для диапазона от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для диапазона более 20 секунд
    
    # Устанавливаем диапазон по X с учетом найденной задержки
    plt_figure.xlim([-100, max_corr_display_ms])
    plt_figure.xticks(np.arange(-100, max_corr_display_ms + 1, tick_step_ms))
    plt_figure.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Добавляем вертикальную линию на нулевой задержке для наглядности
    plt_figure.axvline(x=0, color='gray', linestyle=':', 
                label='Нулевая задержка')
    
    return ref_delay_ms

def calculate_signal_delay(in_channel, ref_channel, sample_rate):
    """
    Вычисляет задержку между входным и референсным сигналами.
    
    Args:
        in_channel: Входной сигнал (numpy массив)
        ref_channel: Референсный сигнал (numpy массив)
        sample_rate: Частота дискретизации
        
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
    delay_ms = lag * 1000 / sample_rate
    
    # Вычисляем уверенность (нормализованная корреляция)
    max_corr = np.abs(correlation[max_corr_idx])
    auto_corr_ref = np.sum(ref_channel**2)
    auto_corr_in = np.sum(in_channel**2)
    confidence = max_corr / np.sqrt(auto_corr_ref * auto_corr_in) if auto_corr_ref > 0 and auto_corr_in > 0 else 0
    
    logging.info(f"Обнаружена задержка: {delay_ms:.2f} мс (лаг: {lag} семплов, уверенность: {confidence:.2f})")
    
    return lag, delay_ms, correlation, lags, confidence


def format_array_info(array_name, array, sample_rate):
    """
    Форматирует информацию о массиве аудиоданных в понятное описание.
    
    Args:
        array_name: Название массива
        array: NumPy массив с аудиоданными
        sample_rate: Частота дискретизации
        
    Returns:
        str: Отформатированное описание
    """
    if array is None:
        return f"{array_name}: массив не предоставлен"
        
    if len(array.shape) > 1:  # Стерео
        format_type = "стерео"
        n_samples = array.shape[0]
        channels = array.shape[1]
        shape_str = f"{array.shape} (сэмплов: {n_samples}, каналов: {channels})"
    else:  # Моно
        format_type = "моно"
        n_samples = len(array)
        shape_str = f"{array.shape} (сэмплов: {n_samples})"
    
    duration_sec = n_samples / sample_rate
    duration_ms = n_samples * 1000 / sample_rate
    
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

def visualize_audio_processing(
    output_dir: str,
    reference_data: Optional[bytes] = None,
    reference_file_path: Optional[str] = None,
    reference_by_micro_volumed_data: Optional[bytes] = None,
    reference_by_micro_volumed_file_path: Optional[str] = None,
    reference_by_micro_volumed_delayed_data: Optional[bytes] = None,
    reference_by_micro_volumed_delayed_file_path: Optional[str] = None,
    input_data: Optional[bytes] = None,
    input_file_path: Optional[str] = None,
    processed_data: Optional[bytes] = None,
    processed_file_path: Optional[str] = None,
    output_prefix: str = "aec",
    sample_rate: int = 16000,
    channels: int = 1,
    metrics: Optional[Dict[str, Any]] = None
):
    """
    Единая функция для визуализации результатов обработки аудио
    
    Args:
        output_dir: Директория для сохранения результатов
        reference_data: Данные оригинального референсного сигнала (байты)
        reference_file_path: Путь к оригинальному референсному файлу
        reference_by_micro_volumed_data: Данные референсного сигнала с измененной громкостью (байты)
        reference_by_micro_volumed_file_path: Путь к файлу референсного сигнала с измененной громкостью
        reference_by_micro_volumed_delayed_data: Данные задержанного референсного сигнала (байты)
        reference_by_micro_volumed_delayed_file_path: Путь к файлу задержанного референсного сигнала
        input_data: Входные данные (байты)
        input_file_path: Путь к входному файлу
        processed_data: Обработанные данные (байты)
        processed_file_path: Путь к обработанному файлу
        reference_delayed_data: Устаревший параметр, используйте reference_by_micro_volumed_delayed_data
        reference_delayed_file_path: Устаревший параметр, используйте reference_by_micro_volumed_delayed_file_path
        output_prefix: Префикс для имен выходных файлов
        sample_rate: Частота дискретизации
        channels: Количество каналов (1 для моно, 2 для стерео)
        metrics: Метрики обработки, включая данные корреляции
    """
    results = {}
    
    try:
        # Создаем директорию для результатов, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Логируем полученную частоту дискретизации и количество каналов
        logging.info(f"Получена частота дискретизации {sample_rate} Гц")
        logging.info(f"Начальное количество каналов: {channels}")
        
        # Если задан channels=1 (значение по умолчанию), пробуем определить из файлов
        if channels == 1:
            # Проверяем файлы для определения количества каналов через log_file_info
            _, detected_channels = log_file_info(input_file_path, "Проверка каналов входного файла")
            if detected_channels and detected_channels in [1, 2]:
                channels = detected_channels
                logging.info(f"visualize_audio_processing: Определено количество каналов из файла: {channels}")
        
        # Проверяем корректность значения
        if channels not in [1, 2]:
            logging.warning(f"Неверное количество каналов ({channels}), используем значение по умолчанию: 1")
            channels = 1

        # Логируем информацию о файлах
        ref_file_info, _ = log_file_info(reference_file_path, "Оригинальное референсное аудио")
        ref_by_micro_volumed_file_info, _ = log_file_info(reference_by_micro_volumed_file_path, "Референсное аудио на входе в микро с измененной громкостью")
        ref_by_micro_volumed_delayed_file_info, _ = log_file_info(reference_by_micro_volumed_delayed_file_path, "Референсное аудио на входе в микро с измененной громкостью и задержкой")
        input_file_info, _ = log_file_info(input_file_path, "Входной файл")
        processed_file_info, _ = log_file_info(processed_file_path, "Обработанный файл")

        if ref_by_micro_volumed_delayed_file_info:
            ref_by_micro_volumed_delayed_duration_ms = ref_by_micro_volumed_delayed_file_info['duration_ms']
            
            # Вычисляем разницу в длительности между референсным и задержанным референсным файлами
            if 'reference' in ref_by_micro_volumed_delayed_file_info:
                duration_diff_ms = ref_by_micro_volumed_delayed_file_info['duration_ms'] - ref_by_micro_volumed_delayed_file_info['reference']['duration_ms']
                logging.info(f"Разница длительностей между референсным и задержанным референсным файлами: {duration_diff_ms:.2f} мс")

        # Логируем размеры данных
        if reference_by_micro_volumed_data:
            logging.info(f"  Размер reference_by_micro_volumed_data: {len(reference_by_micro_volumed_data)} байт")
        if input_data:
            logging.info(f"  Размер input_data: {len(input_data)} байт")
        
        # Передаем количество каналов в функцию bytes_to_numpy
        ref_array = bytes_to_numpy(reference_data, sample_rate, channels)
        ref_by_micro_volumed_array = bytes_to_numpy(reference_by_micro_volumed_data, sample_rate, channels)
        ref_by_micro_volumed_delayed_array = bytes_to_numpy(reference_by_micro_volumed_delayed_data, sample_rate, channels)
        in_array = bytes_to_numpy(input_data, sample_rate, channels)
        processed_array = bytes_to_numpy(processed_data, sample_rate, channels)
        
        # Логируем подробную информацию о массивах
        logging.info(format_array_info("Референсный массив с измененной громкостью (ref_by_micro_volumed_array)", ref_by_micro_volumed_array, sample_rate))
        logging.info(format_array_info("Задержанный референсный массив (ref_by_micro_volumed_delayed_array)", ref_by_micro_volumed_delayed_array, sample_rate))
        logging.info(format_array_info("Входной массив (in_array)", in_array, sample_rate))
        logging.info(format_array_info("Обработанный массив (processed_array)", processed_array, sample_rate))
        
        # Вычисляем длительность в зависимости от формата данных
        if len(ref_by_micro_volumed_array.shape) > 1:  # Стерео
            ref_by_micro_volumed_duration = ref_by_micro_volumed_array.shape[0] / sample_rate
        else:  # Моно
            ref_by_micro_volumed_duration = len(ref_by_micro_volumed_array) / sample_rate

        if len(ref_by_micro_volumed_delayed_array.shape) > 1:  # Стерео
            ref_by_micro_volumed_delayed_duration = ref_by_micro_volumed_delayed_array.shape[0] / sample_rate
        else:  # Моно
            ref_by_micro_volumed_delayed_duration = len(ref_by_micro_volumed_delayed_array) / sample_rate

        if len(in_array.shape) > 1:  # Стерео
            in_duration = in_array.shape[0] / sample_rate
        else:  # Моно
            in_duration = len(in_array) / sample_rate
            
        logging.info(f"  Длительность ref_vol: {ref_by_micro_volumed_duration:.3f} сек")
        logging.info(f"  Длительность in: {in_duration:.3f} сек")
        logging.info(f"  Длительность ref_vol_delayed: {ref_by_micro_volumed_delayed_duration:.3f} сек")

        # Для каждого массива определяем одноканальную версию для отображения
        if len(ref_array.shape) > 1:
            ref_channel = ref_array[:, 0]  # Берем первый канал для отображения
        else:
            ref_channel = ref_array

        if len(ref_by_micro_volumed_array.shape) > 1:
            ref_by_micro_volumed_channel = ref_by_micro_volumed_array[:, 0]  # Берем первый канал для отображения
        else:
            ref_by_micro_volumed_channel = ref_by_micro_volumed_array

        if len(ref_by_micro_volumed_delayed_array.shape) > 1:
            ref_by_micro_volumed_delayed_channel = ref_by_micro_volumed_delayed_array[:, 0]
        else:
            ref_by_micro_volumed_delayed_channel = ref_by_micro_volumed_delayed_array

        if len(in_array.shape) > 1:
            in_channel = in_array[:, 0]  # Берем первый канал для отображения
        else:
            in_channel = in_array

        if len(processed_array.shape) > 1:
            processed_channel = processed_array[:, 0]
        else:
            processed_channel = processed_array

        # Вычисляем задержку между сигналами
        if metrics and all(k in metrics for k in ['delay_samples', 'delay_ms', 'confidence', 'delay_correlation', 'delay_lags']):
            # Используем предрассчитанные данные из AEC
            lag = metrics['delay_samples']
            delay_ms = metrics['delay_ms']
            confidence = metrics['confidence']
            correlation = metrics['delay_correlation']
            lags = metrics['delay_lags']
            logging.info(f"Используются предрассчитанные данные корреляции из AEC, задержка: {delay_ms:.2f} мс")
        else:
            # Если данные не предоставлены, рассчитываем корреляцию
            lag, delay_ms, correlation, lags, confidence = calculate_signal_delay(in_channel, ref_channel, sample_rate)
            logging.info(f"Корреляция была рассчитана внутри функции визуализации")
        
        # Вычисляем длительность сигналов в миллисекундах
        ref_by_micro_volumed_duration_ms = len(ref_by_micro_volumed_array) * 1000 / sample_rate
        if ref_by_micro_volumed_file_info:
            assert ref_by_micro_volumed_duration_ms == ref_by_micro_volumed_file_info['duration_ms']

        in_duration_ms = len(in_array) * 1000 / sample_rate
        if input_file_info:
            assert in_duration_ms == input_file_info['duration_ms']

        ref_by_micro_volumed_delayed_duration_ms = len(ref_by_micro_volumed_delayed_array) * 1000 / sample_rate
        if ref_by_micro_volumed_delayed_file_info:
            assert ref_by_micro_volumed_delayed_duration_ms == ref_by_micro_volumed_delayed_file_info['duration_ms']

        # Логируем информацию о длительности
        logging.info(f"Длительность reference_by_micro_volumed.wav: {ref_by_micro_volumed_duration_ms:.2f} мс")
        logging.info(f"Длительность original_input.wav: {in_duration_ms:.2f} мс")
        logging.info(f"Длительность reference_by_micro_volumed_delayed.wav: {ref_by_micro_volumed_delayed_duration_ms:.2f} мс")
        logging.info(f"Длительность processed_input.wav: {len(processed_channel) * 1000 / sample_rate:.2f} мс")

        # Проверяем и согласовываем длины массивов
        if len(in_channel) != len(processed_channel):
            logging.warning(f"Размеры входного ({len(in_channel)}) и обработанного ({len(processed_channel)}) сигналов не совпадают!")
            logging.info("Приведение длины массивов для корректного отображения...")
        
        # Находим минимальную длину массивов для корректного отображения
        min_display_length = min(len(ref_channel), len(in_channel), len(processed_channel))
        logging.info(f"Минимальная длина массивов для отображения: {min_display_length} семплов")

        # Обрезаем массивы до минимальной длины
        ref_channel = ref_channel[:min_display_length]
        in_channel = in_channel[:min_display_length]
        processed_channel = processed_channel[:min_display_length]

        # Создаем временную ось подходящей длины
        time_axis = np.arange(min_display_length) / sample_rate
        # Создаем временную ось в миллисекундах
        time_axis_ms = time_axis * 1000  # Преобразуем секунды в миллисекунды
        
        # Определяем количество графиков
        num_plots = 9  # Только 8 графиков
        
        # Создаем график
        plt.figure(figsize=(12, 4 * num_plots))
        
        # 1. График двух референсных сигналов на входе в микро (до и после задержки)
        plot_reference_signals(
            plt,
            (num_plots, 1, 1),
            ref_by_micro_volumed_channel,
            ref_by_micro_volumed_delayed_channel,
            sample_rate,
            ref_delay_ms if 'ref_delay_ms' in locals() else None
        )
        
        # 2. График кросс-корреляции между обычным и сдвинутым референсными сигналами
        ref_delay_ms = plot_reference_correlation(
            plt,
            (num_plots, 1, 2),
            ref_by_micro_volumed_channel,
            ref_by_micro_volumed_delayed_channel,
            sample_rate
        )

        # График оригинального и на входе в микро референсных сигналов
        plot_original_and_by_micro_volumed_reference_signals(
            plt,
            (num_plots, 1, 3),
            ref_channel,
            ref_by_micro_volumed_channel,
            sample_rate,
        )

        # График оригинального и на входе в микро референсных сигналов
        plot_original_and_by_micro_volumed_reference_correlation(
            plt,
            (num_plots, 1, 4),
            ref_channel,
            ref_by_micro_volumed_channel,
            sample_rate,
        )
        
        # График оригинального референсного и входного сигналов
        plot_original_reference_and_input_signals(
            plt,
            (num_plots, 1, 5),
            ref_by_micro_volumed_delayed_channel,
            in_channel,
            sample_rate,
        )

        # 6. Исходные сигналы с указанием задержки
        plot_original_signals(
            plt,
            (num_plots, 1, 6),
            ref_channel,
            in_channel,
            time_axis_ms,
            delay_ms
        )

        # 7. Кросс-корреляция между входным и референсным сигналами
        plot_input_reference_correlation(
            plt,
            (num_plots, 1, 7),
            lags,
            correlation,
            delay_ms,
            confidence,
            sample_rate
        )

        # 8. Сигналы после корректировки задержки
        plot_corrected_signals(
            plt,
            (num_plots, 1, 8),
            ref_channel,
            in_channel,
            time_axis_ms,
            lag,
            delay_ms
        )

        # 9. Сравнение исходного и обработанного сигналов
        plot_processed_signals(
            plt,
            (num_plots, 1, 9),
            in_channel,
            processed_channel,
            time_axis_ms,
            sample_rate,
            channels
        )

        plt.tight_layout()
        signals_file = os.path.join(output_dir, f'{output_prefix}_signals.png')
        plt.savefig(signals_file)
        plt.close()
        logging.info(f"Визуализация сигналов сохранена в {signals_file}")
        
        # Вычисляем и логируем статистику
        results = calculate_and_log_statistics(
            in_array,
            ref_by_micro_volumed_array,  # Используем ref_by_micro_volumed_array вместо ref_array
            processed_array, 
            processed_data,
            delay_ms,
            lag,
            results
        )
    
    except ImportError:
        logging.warning("Не удалось импортировать необходимые библиотеки для визуализации")
    except Exception as e:
        logging.error(f"Ошибка при визуализации: {e}")
        logging.exception("Подробная информация об ошибке:")


def bytes_to_numpy(audio_bytes, sample_rate=16000, channels=2):
    """
    Преобразует аудио-данные из байтов в numpy массив.
    
    Args:
        audio_bytes: Аудио-данные в формате байтов
        sample_rate: Частота дискретизации
        channels: Количество каналов (1 для моно, 2 для стерео)
        
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
        if channels == 2:
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
            duration = audio_array.shape[0] / sample_rate
        else:  # Моно
            duration = len(audio_array) / sample_rate
            
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

def plot_original_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    in_channel,
    time_axis_ms,
    delay_ms
):
    """
    Отображает график исходных сигналов с указанием задержки.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_channel: Массив с референсным сигналом для отображения
        in_channel: Массив с входным сигналом для отображения
        time_axis_ms: Временная ось в миллисекундах
        delay_ms: Вычисленная задержка в миллисекундах
    """
    plt_figure.subplot(*subplot_position)
    plt_figure.plot(time_axis_ms, ref_channel, label='Референсный', alpha=0.7, color='blue')
    plt_figure.plot(time_axis_ms, in_channel, label='Входной', alpha=0.7, color='green')
    plt_figure.title(f'{subplot_position[-1]}. Референсный и входной сигналы (задержка: {delay_ms:.2f} мс)')

    # Определяем максимальное значение для отображения и шаг
    max_display_ms = time_axis_ms[-1]
    
    # Определяем шаг деления оси X в зависимости от длительности
    if max_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для длительности до 10 секунд
    elif max_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для длительности от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для длительности более 20 секунд
        
    # Добавляем более детальные деления на оси X
    plt_figure.xticks(np.arange(0, max_display_ms + 1, tick_step_ms))
    plt_figure.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X

    plt_figure.xlabel('Время (мс)')
    plt_figure.ylabel('Амплитуда')
    plt_figure.legend()
    plt_figure.grid(True)

def plot_corrected_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    in_channel,
    time_axis_ms,
    lag,
    delay_ms
):
    """
    Отображает график сигналов после корректировки задержки.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        ref_channel: Массив с референсным сигналом для отображения
        in_channel: Массив с входным сигналом для отображения
        time_axis_ms: Временная ось в миллисекундах
        lag: Задержка в семплах
        delay_ms: Задержка в миллисекундах
    """
    plt_figure.subplot(*subplot_position)
    
    # Определяем corrected_in_array и corrected_ref_array
    if lag >= 0:
        # Входной сигнал задержан относительно референсного
        corrected_in_array = np.roll(in_channel, -lag)
        corrected_ref_array = ref_channel
    else:
        # Референсный сигнал задержан относительно входного
        corrected_ref_array = np.roll(ref_channel, lag)
        corrected_in_array = in_channel
        
    # Используем те же цвета, что и на графике оригинальных сигналов
    plt_figure.plot(time_axis_ms, corrected_ref_array, label='Референсный', alpha=0.7, color='blue')
    plt_figure.plot(time_axis_ms, corrected_in_array, label='Входной', alpha=0.7, color='green')
    
    plt_figure.title(f'{subplot_position[-1]}. Сигналы с учётом задержки ({delay_ms:.2f} мс)')
    
    # Определяем максимальное значение для отображения и шаг
    max_display_ms = time_axis_ms[-1]
    
    # Определяем шаг деления оси X в зависимости от длительности
    if max_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для длительности до 10 секунд
    elif max_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для длительности от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для длительности более 20 секунд
        
    # Добавляем более детальные деления на оси X
    plt_figure.xticks(np.arange(0, max_display_ms + 1, tick_step_ms))
    plt_figure.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
    
    plt_figure.xlabel('Время (мс)')
    plt_figure.ylabel('Амплитуда')
    plt_figure.grid(True)
    plt_figure.legend()

def plot_processed_signals(
    plt_figure,
    subplot_position,
    in_channel,
    processed_channel,
    time_axis_ms,
    sample_rate,
    channels
):
    """
    Отображает график сравнения исходного и обработанного сигналов.
    
    Args:
        plt_figure: Объект figure из matplotlib
        subplot_position: Позиция для subplot (tuple из трех чисел: rows, cols, index)
        in_channel: Массив с входным сигналом для отображения
        processed_data: Обработанные данные (байты)
        time_axis_ms: Временная ось в миллисекундах
        sample_rate: Частота дискретизации
        channels: Количество каналов
    """
    plt_figure.subplot(*subplot_position)
    
    # Проверяем и приводим к одинаковой длине, если размеры массивов не совпадают
    if len(in_channel) != len(processed_channel):
        logging.info(f"Размеры входного ({len(in_channel)}) и обработанного ({len(processed_channel)}) сигналов не совпадают. Приводим к одинаковой длине.")
        min_length = min(len(in_channel), len(processed_channel))
        in_channel = in_channel[:min_length]
        processed_channel = processed_channel[:min_length]
        # Также обновляем time_axis_ms до минимальной длины
        time_axis_ms = time_axis_ms[:min_length]
        logging.info(f"Новая длина сигналов: {min_length}")
    
    # Нормализуем входной сигнал, если он есть
    if len(in_channel) > 0:
        in_max = max(abs(np.max(in_channel)), abs(np.min(in_channel)))
        normalized_in = in_channel / in_max if in_max > 0 else in_channel
    else:
        normalized_in = in_channel
        
    # Рисуем входной сигнал
    plt_figure.plot(time_axis_ms, normalized_in, label='Исходный входной', alpha=0.7, color='green', linewidth=1.2)

    # Нормализуем обработанный сигнал
    if len(processed_channel) > 0:
        proc_max = max(abs(np.max(processed_channel)), abs(np.min(processed_channel)))
        normalized_proc = processed_channel / proc_max if proc_max > 0 else processed_channel
    else:
        normalized_proc = processed_channel
    
    # Сдвигаем нормализованный обработанный сигнал немного вниз для лучшей видимости
    plt_figure.plot(time_axis_ms, normalized_proc, label='Обработанный AEC', alpha=0.7, color='red', linewidth=1.2)
    
    # Добавляем аннотацию о нормализации
    plt_figure.annotate("Оба сигнала нормализованы для лучшей видимости",
                    xy=(0.02, 0.95), xycoords='axes fraction', 
                    color='black', fontsize=9, alpha=0.8)
    
    plt_figure.title(f'{subplot_position[-1]}. Сравнение исходного и обработанного сигналов')
    
    # Определяем максимальное значение для отображения и шаг
    max_display_ms = time_axis_ms[-1]
    
    # Определяем шаг деления оси X в зависимости от длительности
    if max_display_ms <= 10000:
        tick_step_ms = 250  # шаг 250мс для длительности до 10 секунд
    elif max_display_ms <= 20000:
        tick_step_ms = 500  # шаг 500мс для длительности от 10 до 20 секунд
    else:
        tick_step_ms = 2000  # шаг 2000мс для длительности более 20 секунд
        
    # Добавляем более детальные деления на оси X
    plt_figure.xticks(np.arange(0, max_display_ms + 1, tick_step_ms))
    plt_figure.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
    
    plt_figure.xlabel('Время (мс)')
    plt_figure.ylabel('Нормализованная амплитуда')
    plt_figure.grid(True)
    plt_figure.legend()
