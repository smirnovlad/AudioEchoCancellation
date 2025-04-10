import matplotlib.pyplot as plt
import numpy as np
import logging


def plot_original_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    in_channel,
    time_axis_ms,
    delay_ms,
    channel_num=1
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

def plot_reference_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    ref_by_micro_volumed_delayed_channel,
    sample_rate,
    ref_delay_ms=None,
    channel_num=1
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
    sample_rate,
    channel_num=1
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
    sample_rate,
    channel_num=1
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
    sample_rate,
    channel_num=1
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
    sample_rate,
    channel_num=1
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
    sample_rate,
    channel_num=1
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

def plot_corrected_signals(
    plt_figure,
    subplot_position,
    ref_channel,
    in_channel,
    time_axis_ms,
    lag,
    delay_ms,
    channel_num=1
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
    channels,
    channel_num=1
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
