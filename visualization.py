import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
from typing import Optional, Tuple, Dict, List, Any

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
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Вычисляем длительность в миллисекундах
            duration_ms = n_frames * 1000 / framerate
            
            return {
                'n_channels': n_channels,
                'sample_width': sample_width,
                'framerate': framerate,
                'n_frames': n_frames,
                'duration_ms': duration_ms
            }
    except Exception as e:
        logging.error(f"Ошибка при получении информации о WAV файле {file_path}: {e}")
        return {
            'n_channels': 1,
            'sample_width': 2,
            'framerate': 16000,
            'n_frames': 0,
            'duration_ms': 0
        }

def visualize_audio_processing(
    output_dir: str,
    reference_data: Optional[bytes] = None,
    input_data: Optional[bytes] = None,
    processed_data: Optional[bytes] = None,
    reference_delayed_data: Optional[bytes] = None,
    metrics: Optional[Dict[str, Any]] = None,
    sample_rate: int = 16000,
    output_prefix: str = "aec",
    max_delay_ms: int = 1000,
    reference_file_path: Optional[str] = None,
    input_file_path: Optional[str] = None,
    reference_delayed_file_path: Optional[str] = None,
    channels: int = 1
) -> Dict[str, Any]:
    """
    Единая функция для визуализации результатов обработки аудио
    
    Args:
        output_dir: Директория для сохранения результатов
        reference_data: Референсные данные (байты)
        input_data: Входные данные (байты)
        processed_data: Обработанные данные (байты)
        reference_delayed_data: Сдвинутые референсные данные (байты), обычно из reference_volumed_delayed.wav
        metrics: Метрики обработки
        sample_rate: Частота дискретизации
        output_prefix: Префикс для имен выходных файлов
        max_delay_ms: Максимальная задержка для поиска (мс)
        reference_file_path: Путь к референсному файлу (для анализа длительности)
        input_file_path: Путь к входному файлу (для анализа длительности)
        reference_delayed_file_path: Путь к задержанному референсному файлу (для анализа длительности)
        channels: Количество каналов (1 для моно, 2 для стерео)
        
    Returns:
        Dict[str, Any]: Словарь с метриками и статистикой
    """
    results = {}
    
    try:
        # Создаем директорию для результатов, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Логируем полученную частоту дискретизации и количество каналов
        logging.info(f"visualize_audio_processing: Получена частота дискретизации {sample_rate} Гц")
        logging.info(f"visualize_audio_processing: Начальное количество каналов: {channels}")
        
        # Если задан channels=1 (значение по умолчанию), пробуем определить из файлов
        if channels == 1:
            # Проверяем файлы для определения количества каналов
            if input_file_path and os.path.exists(input_file_path):
                try:
                    with wave.open(input_file_path, 'rb') as wf:
                        detected_channels = wf.getnchannels()
                        if detected_channels in [1, 2]:
                            channels = detected_channels
                            logging.info(f"visualize_audio_processing: Определено количество каналов из файла: {channels}")
                except Exception as e:
                    logging.warning(f"Не удалось определить количество каналов из файла: {e}")
        
        # Проверяем корректность значения
        if channels not in [1, 2]:
            logging.warning(f"Неверное количество каналов ({channels}), используем значение по умолчанию: 1")
            channels = 1
        
        # Логируем информацию о файлах
        if reference_file_path:
            logging.info(f"visualize_audio_processing: Референсный файл: {reference_file_path}")
            if os.path.exists(reference_file_path):
                file_info = get_wav_info(reference_file_path)
                logging.info(f"  Частота дискретизации: {file_info['framerate']} Гц")
                logging.info(f"  Длительность: {file_info['duration_ms']/1000:.3f} сек")
                logging.info(f"  Каналов: {file_info['n_channels']}")
                channels = file_info['n_channels']
        
        if input_file_path:
            logging.info(f"visualize_audio_processing: Входной файл: {input_file_path}")
            if os.path.exists(input_file_path):
                file_info = get_wav_info(input_file_path)
                logging.info(f"  Частота дискретизации: {file_info['framerate']} Гц")
                logging.info(f"  Длительность: {file_info['duration_ms']/1000:.3f} сек")
                logging.info(f"  Каналов: {file_info['n_channels']}")
                channels = file_info['n_channels']
        
        if reference_delayed_file_path and os.path.exists(reference_delayed_file_path):
            file_info = get_wav_info(reference_delayed_file_path)
            ref_delayed_duration_ms = file_info['duration_ms']
            logging.info(f"Задержанный референсный файл: длительность {ref_delayed_duration_ms:.2f} мс ({ref_delayed_duration_ms/1000:.2f} с), {file_info['framerate']} Гц")
            
            # Вычисляем разницу в длительности между референсным и задержанным референсным файлами
            if 'reference' in file_info:
                duration_diff_ms = file_info['duration_ms'] - file_info['reference']['duration_ms']
                logging.info(f"Разница длительностей между референсным и задержанным референсным файлами: {duration_diff_ms:.2f} мс")
        
        # Собираем информацию в результаты
        if file_info:
            results['file_info'] = file_info
        
        # Перед вызовом visualize_delay
        if reference_data is not None and input_data is not None:
            logging.info(f"visualize_audio_processing: Размер reference_data: {len(reference_data)} байт")
            logging.info(f"visualize_audio_processing: Размер input_data: {len(input_data)} байт")
            
            # Передаем количество каналов в функцию bytes_to_numpy
            ref_array = bytes_to_numpy(reference_data, sample_rate, channels)
            in_array = bytes_to_numpy(input_data, sample_rate, channels)
            
            # Конвертируем reference_delayed_data
            ref_delayed_array = bytes_to_numpy(reference_delayed_data, sample_rate, channels)
            logging.info(f"visualize_audio_processing: Размер ref_delayed_array: {ref_delayed_array.shape}")
            
            # Сравниваем с данными из create_volume_variants.py
            logging.info(f"visualize_audio_processing: Сравнение с данными из create_volume_variants.py:")
            logging.info(f"  Размер ref_array: {ref_array.shape}")
            logging.info(f"  Размер in_array: {in_array.shape}")
            logging.info(f"  Частота дискретизации: {sample_rate} Гц")
            
            # Вычисляем длительность в зависимости от формата данных
            if len(ref_array.shape) > 1:  # Стерео
                ref_duration = ref_array.shape[0] / sample_rate
            else:  # Моно
                ref_duration = len(ref_array) / sample_rate
                
            if len(in_array.shape) > 1:  # Стерео
                in_duration = in_array.shape[0] / sample_rate
            else:  # Моно
                in_duration = len(in_array) / sample_rate
                
            logging.info(f"  Длительность ref: {ref_duration:.3f} сек")
            logging.info(f"  Длительность in: {in_duration:.3f} сек")
            
            # Вычисляем задержку между сигналами напрямую, без вызова visualize_delay
            # Если данные стерео, берем только первый канал для анализа
            if len(ref_array.shape) > 1:
                ref_channel = ref_array[:, 0]  # Берем первый канал
            else:
                ref_channel = ref_array
                
            if len(in_array.shape) > 1:
                in_channel = in_array[:, 0]  # Берем первый канал
            else:
                in_channel = in_array
            
            # Вычисляем кросс-корреляцию
            correlation = signal.correlate(in_channel, ref_channel, mode='full')
            lags = signal.correlation_lags(len(in_channel), len(ref_channel), mode='full')
            
            # Находим индекс максимальной корреляции только с неотрицательной задержкой
            center_index = len(lags) // 2  # Индекс, соответствующий нулевой задержке
            # Рассматриваем только неотрицательные лаги (микрофон задержан относительно референса)
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
            
            # Вычисляем длительность сигналов в миллисекундах
            ref_duration_ms = len(ref_array) * 1000 / sample_rate
            in_duration_ms = len(in_array) * 1000 / sample_rate
            ref_delayed_duration_ms = len(ref_delayed_array) * 1000 / sample_rate
            
            # Логируем информацию о длительности
            logging.info(f"Длительность reference_volumed.wav: {ref_duration_ms:.2f} мс")
            logging.info(f"Длительность original_input.wav: {in_duration_ms:.2f} мс")
            logging.info(f"Длительность reference_volumed_delayed.wav: {ref_delayed_duration_ms:.2f} мс")
            
            # Для каждого массива определяем одноканальную версию для отображения
            if len(ref_array.shape) > 1:
                ref_channel_display = ref_array[:, 0]  # Берем первый канал для отображения
            else:
                ref_channel_display = ref_array

            if len(in_array.shape) > 1:
                in_channel_display = in_array[:, 0]  # Берем первый канал для отображения
            else:
                in_channel_display = in_array

            # Находим минимальную длину массивов для корректного отображения
            min_display_length = min(len(ref_channel_display), len(in_channel_display))
            logging.info(f"Минимальная длина массивов для отображения: {min_display_length} семплов")

            # Обрезаем массивы до минимальной длины
            ref_channel_display = ref_channel_display[:min_display_length]
            in_channel_display = in_channel_display[:min_display_length]

            # Создаем временную ось подходящей длины
            time_axis = np.arange(min_display_length) / sample_rate
            # Создаем временную ось в миллисекундах
            time_axis_ms = time_axis * 1000  # Преобразуем секунды в миллисекунды
            
            # Определяем количество графиков
            num_plots = 6  # Только 6 графиков
            
            # Создаем график
            plt.figure(figsize=(12, 4 * num_plots))
            
            # 1. График двух референсных сигналов (до и после задержки)
            plt.subplot(num_plots, 1, 1)
            
            # Приводим ref_delayed_array к той же форме, что и ref_array
            # Берем первый канал, если это стерео данные
            if len(ref_delayed_array.shape) > 1:
                ref_delayed_channel = ref_delayed_array[:, 0]
            else:
                ref_delayed_channel = ref_delayed_array
            
            if len(ref_array.shape) > 1:
                ref_channel_for_plot = ref_array[:, 0]
            else:
                ref_channel_for_plot = ref_array
            
            # Определяем длину для отображения до 8000 мс
            display_samples = int(8000 * sample_rate / 1000)  # Количество семплов для 8000 мс
            
            # Дополняем оба массива нулями до display_samples
            padded_ref = np.zeros(display_samples)
            padded_ref[:len(ref_channel_for_plot)] = ref_channel_for_plot
            
            padded_ref_delayed = np.zeros(display_samples)
            padded_ref_delayed[:len(ref_delayed_channel)] = ref_delayed_channel
            
            # Используем дополненные массивы
            ref_channel_for_plot = padded_ref
            ref_delayed_channel = padded_ref_delayed
            
            logging.info(f"Референсные сигналы дополнены нулями до 8000 мс ({display_samples} семплов)")
            logging.info(f"Исходная длина ref_channel: {len(ref_channel_for_plot)} семплов")
            logging.info(f"Исходная длина ref_delayed_channel: {len(ref_delayed_channel)} семплов")
            
            # Создаем расширенную временную ось для отображения
            display_time_ms = np.arange(display_samples) * 1000 / sample_rate
            
            # Строим графики используя подготовленные массивы с одинаковой длиной
            plt.plot(display_time_ms, ref_channel_for_plot, label=f'Исходный референс', alpha=0.7, color='blue')
            plt.plot(display_time_ms, ref_delayed_channel, label=f'Задержанный референс', alpha=0.7, color='red')
            
            # Если есть задержка, вычисленная по корреляции, отображаем её
            if 'ref_delay_ms' in locals():
                plt.axvline(x=ref_delay_ms, color='g', linestyle='--', 
                          label=f'Вычисленная задержка: {ref_delay_ms:.2f} мс')
            
            plt.title(f'1. Сравнение референсных сигналов (исходный и задержанный)')
            
            # Добавляем более детальные деления на оси X
            plt.xticks(np.arange(0, 8001, 250))  # Деления каждые 250 мс до 8000 мс
            plt.xlim([0, 8000])  # Фиксированный диапазон до 8000 мс
            plt.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
            
            plt.xlabel('Время (мс)')
            plt.ylabel('Амплитуда')
            plt.grid(True)
            plt.legend()
            
            # 2. График кросс-корреляции между обычным и сдвинутым референсными сигналами
            plt.subplot(num_plots, 1, 2)

            # Извлекаем первый канал из обоих сигналов, если это необходимо
            if len(ref_array.shape) > 1:
                ref_channel_corr = ref_array[:, 0]
            else:
                ref_channel_corr = ref_array

            if len(ref_delayed_array.shape) > 1:
                ref_delayed_channel_corr = ref_delayed_array[:, 0]
            else:
                ref_delayed_channel_corr = ref_delayed_array

            # Определяем минимальную длину для корреляции (до 5 секунд)
            correlation_window = int(5 * sample_rate)  # 5 секунд для корреляции
            actual_correlation_window = min(correlation_window, len(ref_channel_corr), len(ref_delayed_channel_corr))

            # Логируем информацию о корреляционном окне
            logging.info(f"Используем корреляционное окно: {actual_correlation_window} семплов ({actual_correlation_window/sample_rate:.1f} сек)")

            # Обрезаем оба сигнала до минимальной длины для корреляции
            ref_for_corr = ref_channel_corr[:actual_correlation_window]
            ref_delayed_for_corr = ref_delayed_channel_corr[:actual_correlation_window]

            # Вычисляем корреляцию
            corr = np.correlate(ref_delayed_for_corr, ref_for_corr, 'full')
            corr_time = np.arange(len(corr)) - (len(ref_for_corr) - 1)
            corr_time_ms = corr_time * 1000.0 / sample_rate

            # Находим максимальное значение корреляции и соответствующую задержку
            max_corr_idx = np.argmax(corr)
            ref_delay_samples = corr_time[max_corr_idx]
            ref_delay_ms = ref_delay_samples * 1000.0 / sample_rate

            # Логируем информацию о найденной задержке
            logging.info(f"Корреляция между референсными сигналами: макс.индекс={max_corr_idx}, задержка={ref_delay_samples} сэмплов ({ref_delay_ms:.2f} мс)")

            # Строим график корреляции
            plt.plot(corr_time_ms, corr, color='green')
            plt.axvline(x=ref_delay_ms, color='r', linestyle='--', 
                        label=f'Измеренная задержка: {ref_delay_ms:.2f} мс')

            plt.title(f'2. Кросс-корреляция между референсными сигналами')
            plt.xlabel('Задержка (мс)')
            plt.ylabel('Корреляция')
            plt.legend()

            # Устанавливаем диапазон по X
            plt.xlim([-100, 1500])  # Изменен диапазон от -100 до 1500 мс
            plt.xticks(np.arange(-100, 1501, 250))  # Деления каждые 250 мс
            # Добавляем сетку
            plt.grid(True, which='both', linestyle='--', alpha=0.7)

            # 3. Кросс-корреляция между входным и референсным сигналами
            plt.subplot(num_plots, 1, 3)
            # Преобразуем лаги в миллисекунды для оси X
            lags_ms = lags * 1000 / sample_rate
            plt.plot(lags_ms, np.abs(correlation), color='green')
            plt.axvline(x=delay_ms, color='r', linestyle='--', 
                        label=f'Измеренная задержка: {delay_ms:.2f} мс, уверенность: {confidence:.2f}')

            plt.title(f'3. Кросс-корреляция между входным и референсным сигналами')
            plt.xlabel('Задержка (мс)')
            plt.ylabel('Корреляция')
            plt.legend()

            # ИЗМЕНЕНО: Устанавливаем диапазон по X от -100 до 1500 мс
            plt.xlim([-100, 1500])
            plt.xticks(np.arange(-100, 1501, 250))  # Деления каждые 250 мс
            # ИЗМЕНЕНО: Добавляем сетку
            plt.grid(True, which='both', linestyle='--', alpha=0.7)

            # 4. Исходные сигналы с указанием задержки
            plt.subplot(num_plots, 1, 4)
            plt.plot(time_axis_ms, ref_channel_display, label='Референсный', alpha=0.7, color='blue')
            plt.plot(time_axis_ms, in_channel_display, label='Входной', alpha=0.7, color='green')
            plt.title(f'4. Оригинальные сигналы (задержка: {delay_ms:.2f} мс)')

            # Добавляем более детальные деления на оси X
            plt.xticks(np.arange(0, time_axis_ms[-1] + 1, 200))  # Деления каждые 200 мс
            plt.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X

            plt.xlabel('Время (мс)')
            plt.ylabel('Амплитуда')
            plt.legend()
            plt.grid(True)

            # 5. Сигналы после корректировки задержки
            plt.subplot(num_plots, 1, 5)
            
            # Определяем corrected_in_array и corrected_ref_array
            if lag >= 0:
                # Входной сигнал задержан относительно референсного
                corrected_in_array = np.roll(in_channel_display, -lag)
                corrected_ref_array = ref_channel_display
            else:
                # Референсный сигнал задержан относительно входного
                corrected_ref_array = np.roll(ref_channel_display, lag)
                corrected_in_array = in_channel_display
                
            # Используем те же цвета, что и на графике оригинальных сигналов
            plt.plot(time_axis_ms, corrected_ref_array, label='Референсный', alpha=0.7, color='blue')
            plt.plot(time_axis_ms, corrected_in_array, label='Входной', alpha=0.7, color='green')
            
            plt.title(f'5. Сигналы с учётом задержки ({delay_ms:.2f} мс)')
            
            # Добавляем более детальные деления на оси X
            plt.xticks(np.arange(0, time_axis_ms[-1] + 1, 200))  # Деления каждые 200 мс
            plt.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
            
            plt.xlabel('Время (мс)')
            plt.ylabel('Амплитуда')
            plt.grid(True)
            plt.legend()

            # 6. Сравнение исходного и обработанного сигналов
            plt.subplot(num_plots, 1, 6)
            
            # Нормализуем входной сигнал, если он есть
            if len(in_channel_display) > 0:
                in_max = max(abs(np.max(in_channel_display)), abs(np.min(in_channel_display)))
                normalized_in = in_channel_display / in_max if in_max > 0 else in_channel_display
            else:
                normalized_in = in_channel_display
                
            # Рисуем входной сигнал
            plt.plot(time_axis_ms, normalized_in, label='Исходный входной', alpha=0.7, color='green', linewidth=1.2)
            
            # Если есть обработанные данные, добавляем их
            if processed_data is not None:
                # Преобразуем processed_data
                processed_array = bytes_to_numpy(processed_data, sample_rate, channels)
                if len(processed_array.shape) > 1:
                    processed_channel = processed_array[:, 0]
                else:
                    processed_channel = processed_array
                
                # Обрезаем до минимальной длины для отображения
                min_proc_length = min(len(processed_channel), len(time_axis_ms))
                
                # Нормализуем обработанный сигнал
                proc_data = processed_channel[:min_proc_length]
                proc_max = max(abs(np.max(proc_data)), abs(np.min(proc_data)))
                normalized_proc = proc_data / proc_max if proc_max > 0 else proc_data
                
                # Сдвигаем нормализованный обработанный сигнал немного вниз для лучшей видимости
                plt.plot(time_axis_ms[:min_proc_length], normalized_proc,
                         label='Обработанный AEC', alpha=0.7, color='red', linewidth=1.2)
                
                # Добавляем аннотацию о нормализации
                plt.annotate("Оба сигнала нормализованы для лучшей видимости",
                             xy=(0.02, 0.95), xycoords='axes fraction', 
                             color='black', fontsize=9, alpha=0.8)
            
            plt.title('6. Сравнение исходного и обработанного сигналов')
            
            # Добавляем более детальные деления на оси X
            plt.xticks(np.arange(0, time_axis_ms[-1] + 1, 200))  # Деления каждые 200 мс
            plt.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
            
            plt.xlabel('Время (мс)')
            plt.ylabel('Нормализованная амплитуда')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            signals_file = os.path.join(output_dir, f'{output_prefix}_signals.png')
            plt.savefig(signals_file)
            plt.close()
            logging.info(f"Визуализация сигналов сохранена в {signals_file}")
            
            # Выводим статистику
            logging.info("Статистика сигналов:")
            in_rms = np.sqrt(np.mean(in_array**2))
            ref_rms = np.sqrt(np.mean(ref_array**2))
            
            logging.info(f"  RMS входного сигнала: {in_rms:.6f}")
            logging.info(f"  RMS референсного сигнала: {ref_rms:.6f}")
            logging.info(f"  Соотношение вход/референс: {in_rms/ref_rms if ref_rms > 0 else 0:.6f}")
            logging.info(f"  RMS разницы (вход - референс): {diff_rms:.6f}")
            logging.info(f"  Обнаруженная задержка: {delay_ms:.2f} мс")
            
            # Собираем статистику в результаты
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
            
            if processed_data is not None:
                proc_rms = np.sqrt(np.mean(corrected_proc_array**2))
                diff_proc_rms = np.sqrt(np.mean(diff_processed**2))
                logging.info(f"  RMS обработанного сигнала: {proc_rms:.6f}")
                logging.info(f"  RMS разницы (обработанный - референс): {diff_proc_rms:.6f}")
                
                # Вычисляем улучшение после обработки
                improvement = None
                if diff_rms > 0:
                    improvement = diff_rms / diff_proc_rms if diff_proc_rms > 0 else float('inf')
                    logging.info(f"  Улучшение после обработки: {improvement:.2f}x ({20*np.log10(improvement):.2f} дБ)")
                
                results["processed"] = {
                    "rms": proc_rms,
                    "diff_rms": diff_proc_rms,
                    "improvement": improvement
                }
            
            # Создаем отдельный файл с визуализацией кросс-корреляции между входным и референсным сигналом
            delay_results = visualize_delay(ref_array, in_array, output_dir, output_prefix, sample_rate, max_delay_ms)
            if delay_results and "delay" in delay_results:
                results.update(delay_results)

            # Создаем отдельный файл с визуализацией кросс-корреляции между входным и задержанным референсным сигналом
            if reference_delayed_data is not None:
                try:
                    delay_ref_delayed_results = visualize_delay(ref_delayed_array, in_array, output_dir, output_prefix + "_ref_delayed", sample_rate, max_delay_ms)
                    if delay_ref_delayed_results and "delay" in delay_ref_delayed_results:
                        results["delay_ref_delayed"] = delay_ref_delayed_results["delay"]
                except Exception as e:
                    logging.error(f"Ошибка при создании визуализации задержки для задержанного референса: {e}")
                    logging.exception("Подробная информация об ошибке:")
        
        return results
    
    except ImportError:
        logging.warning("Не удалось импортировать необходимые библиотеки для визуализации")
        return results
    except Exception as e:
        logging.error(f"Ошибка при визуализации: {e}")
        logging.exception("Подробная информация об ошибке:")
        return results

def visualize_delay(ref_array, in_array, output_dir, output_prefix="aec", sample_rate=16000, max_delay_ms=1000):
    """
    Визуализирует задержку между двумя сигналами с помощью кросс-корреляции.
    """
    try:
        # Логируем входные данные
        logging.info(f"visualize_delay: Размер ref_array: {ref_array.shape}, in_array: {in_array.shape}")
        logging.info(f"visualize_delay: Частота дискретизации: {sample_rate} Гц")
        
        # Если данные стерео, берем только первый канал для анализа
        if len(ref_array.shape) > 1:
            ref_channel = ref_array[:, 0]  # Берем первый канал
            logging.info(f"visualize_delay: Используем первый канал ref_array, новый размер: {ref_channel.shape}")
        else:
            ref_channel = ref_array
            
        if len(in_array.shape) > 1:
            in_channel = in_array[:, 0]  # Берем первый канал
            logging.info(f"visualize_delay: Используем первый канал in_array, новый размер: {in_channel.shape}")
        else:
            in_channel = in_array
        
        # Вычисляем длительность в секундах
        ref_duration = len(ref_channel) / sample_rate
        in_duration = len(in_channel) / sample_rate
        logging.info(f"visualize_delay: Длительность ref: {ref_duration:.3f} сек, in: {in_duration:.3f} сек")
        
        # Проверяем, что массивы не пустые
        if len(ref_channel) == 0 or len(in_channel) == 0:
            logging.error("Один из массивов пуст, невозможно вычислить задержку")
            return {"delay": None}
        
        # Вычисляем кросс-корреляцию
        correlation = signal.correlate(in_channel, ref_channel, mode='full')
        lags = signal.correlation_lags(len(in_channel), len(ref_channel), mode='full')
        
        # Находим индекс максимальной корреляции только с неотрицательной задержкой
        center_index = len(lags) // 2  # Индекс, соответствующий нулевой задержке
        # Рассматриваем только неотрицательные лаги (микрофон задержан относительно референса)
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
        
        # Определяем максимальную длину для визуализации
        max_samples = min(len(ref_channel), len(in_channel))
        
        # Создаем график
        plt.figure(figsize=(12, 8))
        
        # 1. Кросс-корреляция
        plt.subplot(2, 1, 1)
        # Преобразуем лаги в миллисекунды для оси X
        lags_ms = lags * 1000 / sample_rate
        plt.plot(lags_ms, np.abs(correlation))
        plt.axvline(x=delay_ms, color='r', linestyle='--', label=f'Задержка: {delay_ms:.2f} мс')
        plt.title(f'Кросс-корреляция (уверенность: {confidence:.2f})')
        
        # Устанавливаем диапазон от -100 до 1500 мс
        plt.xlim(-100, 1500)
        plt.xticks(np.arange(-100, 1501, 250))  # Метки каждые 250 мс
        # Добавляем сетку
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.xlabel('Задержка (мс)', fontsize=10)
        plt.ylabel('Корреляция')
        plt.legend()
        
        # 2. Сигналы с учетом задержки
        plt.subplot(2, 1, 2)
        
        # ВАЖНОЕ ИСПРАВЛЕНИЕ: Создаем временную ось с правильным масштабом
        # Используем точное количество семплов и частоту дискретизации
        time_axis_ms = np.arange(max_samples) * 1000 / sample_rate  # мс
        
        # Логируем информацию о временной оси
        logging.info(f"visualize_delay: Временная ось от {time_axis_ms[0]:.2f} до {time_axis_ms[-1]:.2f} мс")
        logging.info(f"visualize_delay: Шаг временной оси: {time_axis_ms[1] - time_axis_ms[0]:.3f} мс")
        logging.info(f"visualize_delay: Длина временной оси: {len(time_axis_ms)} точек")
        
        # Если задержка положительная, сдвигаем входной сигнал вправо
        # Если отрицательная, сдвигаем референсный сигнал вправо
        if lag >= 0:
            # Входной сигнал задержан относительно референсного
            plt.plot(time_axis_ms, ref_channel[:max_samples], label='Референсный', alpha=0.7)
            plt.plot(time_axis_ms, np.roll(in_channel, -lag)[:max_samples], label='Входной (скорректированный)', alpha=0.7)
        else:
            # Референсный сигнал задержан относительно входного
            plt.plot(time_axis_ms, np.roll(ref_channel, lag)[:max_samples], label='Референсный (скорректированный)', alpha=0.7)
            plt.plot(time_axis_ms, in_channel[:max_samples], label='Входной', alpha=0.7)
        
        plt.title(f'Сигналы с учетом задержки ({delay_ms:.2f} мс)')
        
        # Добавляем более детальные деления на оси X
        max_time_ms = time_axis_ms[-1]
        plt.xticks(np.arange(0, max_time_ms + 1, 200))  # Деления каждые 200 мс
        plt.grid(True, which='both', linestyle='--', alpha=0.7)  # Добавляем сетку
        
        # Устанавливаем диапазон оси X не менее 6000 мс
        min_view_window = 6000  # минимум 6000 мс
        view_window = max(min_view_window, max_time_ms)
        plt.xlim([0, min(view_window, max_time_ms * 1.1)])  # Ограничиваем максимальное значение
        
        plt.xlabel('Время (мс)')
        plt.ylabel('Амплитуда')
        plt.legend()
        
        plt.tight_layout()
        delay_file = os.path.join(output_dir, f'{output_prefix}_delay.png')
        plt.savefig(delay_file)
        plt.close()
        logging.info(f"Визуализация задержки сохранена в {delay_file}")
        
        return {
            "delay": {
                "samples": lag,
                "ms": delay_ms,
                "confidence": confidence
            }
        }
    
    except Exception as e:
        logging.error(f"Ошибка при визуализации задержки: {e}")
        logging.exception("Подробная информация об ошибке:")
        return {"delay": None}

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
        # Логируем размер входных данных в байтах
        logging.info(f"bytes_to_numpy: Получено {len(audio_bytes)} байт данных")
        
        # Преобразуем байты в numpy массив
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Преобразуем одномерный массив в двумерный для стерео данных
        if channels == 2:
            # Проверяем, что длина массива четная
            if len(audio_array) % 2 == 0:
                # Преобразуем в массив [n_samples, 2]
                audio_array = audio_array.reshape(-1, 2)
                logging.info(f"bytes_to_numpy: Преобразовано в стерео массив размером {audio_array.shape}")
            else:
                logging.warning(f"bytes_to_numpy: Нечетное количество элементов для стерео данных: {len(audio_array)}")
        
        # Логируем информацию о преобразованных данных
        logging.info(f"bytes_to_numpy: Частота дискретизации: {sample_rate} Гц")
        
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
        return None

def visualize_signals(ref_data, in_data, processed_data=None, output_dir=".", output_prefix="aec", sample_rate=16000):
    """
    Визуализирует входной, референсный и обработанный сигналы.
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Преобразуем данные в одноканальные, если они многоканальные
        if len(ref_data.shape) > 1:
            ref_channel = ref_data[:, 0]
        else:
            ref_channel = ref_data
            
        if len(in_data.shape) > 1:
            in_channel = in_data[:, 0]
        else:
            in_channel = in_data
        
        # Преобразуем processed_data из байтов в numpy массив, если он предоставлен
        if processed_data is not None:
            # Проверяем, не является ли processed_data уже numpy массивом
            if isinstance(processed_data, bytes):
                processed_array = bytes_to_numpy(processed_data, sample_rate, channels)
                logging.info(f"visualize_signals: Преобразованный processed_data имеет размер: {processed_array.shape}")
            else:
                processed_array = processed_data
                
            if len(processed_array.shape) > 1:
                processed_channel = processed_array[:, 0]
            else:
                processed_channel = processed_array
        else:
            processed_channel = None
        
        # Определяем максимальную длину для визуализации
        if processed_channel is not None:
            max_samples = min(len(ref_channel), len(in_channel), len(processed_channel))
        else:
            max_samples = min(len(ref_channel), len(in_channel))
        
        # Создаем временную ось в миллисекундах
        time_axis_ms = np.arange(max_samples) * 1000 / sample_rate
        
        # Строим графики
        plt.plot(time_axis_ms, ref_channel[:max_samples], label='Референсный', alpha=0.7)
        plt.plot(time_axis_ms, in_channel[:max_samples], label='Входной', alpha=0.7)
        
        if processed_channel is not None:
            plt.plot(time_axis_ms, processed_channel[:max_samples], label='Обработанный', alpha=0.7)
        
        plt.title('Сравнение сигналов')
        plt.xlabel('Время (мс)')
        plt.ylabel('Амплитуда')
        plt.grid(True)
        plt.legend()
        
        # Добавляем более детальные деления на оси X
        max_time_ms = time_axis_ms[-1]
        plt.xticks(np.arange(0, max_time_ms + 1, 250))  # Деления каждые 250 мс
        plt.grid(axis='x', which='both', linestyle='--', alpha=0.7)  # Сетка по оси X
        
        # Устанавливаем диапазон оси X
        plt.xlim([0, max_time_ms])
        
        plt.tight_layout()
        signals_file = os.path.join(output_dir, f'{output_prefix}_signals.png')
        plt.savefig(signals_file)
        plt.close()
        
        return signals_file
    except Exception as e:
        logging.error(f"Ошибка при визуализации сигналов: {e}")
        logging.exception("Подробная информация об ошибке:")
        return None
