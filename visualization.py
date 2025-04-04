import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Optional, Tuple, Dict, List, Any

def visualize_audio_processing(
    output_dir: str,
    reference_data: Optional[bytes] = None,
    input_data: Optional[bytes] = None,
    processed_data: Optional[bytes] = None,
    scaled_mic_data: Optional[bytes] = None,
    scaled_ref_data: Optional[bytes] = None,
    metrics: Optional[Dict[str, Any]] = None,
    sample_rate: int = 16000,
    output_prefix: str = "aec",
    max_delay_ms: int = 1000
) -> Dict[str, Any]:
    """
    Единая функция для визуализации результатов обработки аудио
    
    Args:
        output_dir: Директория для сохранения результатов
        reference_data: Референсные данные (байты)
        input_data: Входные данные (байты)
        processed_data: Обработанные данные (байты)
        scaled_mic_data: Масштабированные данные с микрофона (байты)
        scaled_ref_data: Масштабированные референсные данные (байты)
        metrics: Метрики обработки
        sample_rate: Частота дискретизации
        output_prefix: Префикс для имен выходных файлов
        max_delay_ms: Максимальная задержка для поиска (мс)
        
    Returns:
        Dict[str, Any]: Словарь с метриками и статистикой
    """
    results = {}
    
    try:
        # Создаем директорию для результатов, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Визуализация задержки, если предоставлены входной и референсный сигналы
        if reference_data is not None and input_data is not None:
            delay_results = visualize_delay(
                reference_data=reference_data,
                input_data=input_data,
                sample_rate=sample_rate,
                max_delay_ms=max_delay_ms,
                output_dir=output_dir,
                output_prefix=output_prefix
            )
            results.update(delay_results)
        
        # Визуализация метрик, если они доступны
        if metrics and all(key in metrics for key in ['frame_index', 'echo_return_loss', 'echo_return_loss_enhancement', 'echo_detected']):
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(metrics['frame_index'], metrics['echo_return_loss'], label='ERL')
            plt.plot(metrics['frame_index'], metrics['echo_return_loss_enhancement'], label='ERLE')
            plt.title('Echo Return Loss (ERL) и Echo Return Loss Enhancement (ERLE)')
            plt.xlabel('Номер фрейма')
            plt.ylabel('дБ')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(metrics['frame_index'], metrics['echo_detected'], label='Echo Detected')
            plt.title('Обнаружение эха')
            plt.xlabel('Номер фрейма')
            plt.ylabel('Эхо обнаружено (1=да, 0=нет)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            metrics_file = os.path.join(output_dir, f'{output_prefix}_metrics.png')
            plt.savefig(metrics_file)
            plt.close()
            logging.info(f"Визуализация метрик сохранена в {metrics_file}")
            
            # Добавляем метрики в результаты
            results['metrics'] = metrics
        else:
            logging.info("Метрики недоступны или неполные, пропускаем визуализацию метрик")
        
        # Визуализация сигналов, если они предоставлены
        if reference_data is not None and input_data is not None:
            # Преобразуем байты в numpy массивы
            ref_array = np.frombuffer(reference_data, dtype=np.int16).astype(np.float32) / 32768.0
            in_array = np.frombuffer(input_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Если есть обработанные данные, преобразуем их
            if processed_data is not None:
                proc_array = np.frombuffer(processed_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                proc_array = None
            
            # Если есть масштабированные данные, преобразуем их
            if scaled_mic_data is not None:
                scaled_mic_array = np.frombuffer(scaled_mic_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                scaled_mic_array = None
                
            if scaled_ref_data is not None:
                scaled_ref_array = np.frombuffer(scaled_ref_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                scaled_ref_array = None
            
            # Ограничиваем длину для визуализации (первые 5 секунд или меньше)
            max_samples = min(5 * sample_rate, len(ref_array), len(in_array))
            ref_array = ref_array[:max_samples]
            in_array = in_array[:max_samples]
            
            if proc_array is not None:
                proc_array = proc_array[:max_samples]
            if scaled_mic_array is not None:
                scaled_mic_array = scaled_mic_array[:max_samples]
            if scaled_ref_array is not None:
                scaled_ref_array = scaled_ref_array[:max_samples]
            
            # Вычисляем кросс-корреляцию для определения задержки
            correlation = signal.correlate(in_array, ref_array, mode='full')
            lags = signal.correlation_lags(len(in_array), len(ref_array), mode='full')
            
            # Ограничиваем поиск максимума заданным диапазоном задержки
            max_lag_samples = int(max_delay_ms * sample_rate / 1000)
            center_index = len(lags) // 2
            start_index = max(0, center_index - max_lag_samples)
            end_index = min(len(lags), center_index + max_lag_samples + 1)
            
            # Находим индекс максимальной корреляции в заданном диапазоне
            max_index = start_index + np.argmax(np.abs(correlation[start_index:end_index]))
            lag = lags[max_index]
            
            # Вычисляем задержку в миллисекундах
            delay_ms = lag * 1000 / sample_rate
            
            # Вычисляем уверенность в оценке
            confidence = np.abs(correlation[max_index]) / np.max(np.abs(correlation))
            
            # Создаем временную ось
            time_axis = np.arange(max_samples) / sample_rate
            
            # Определяем количество графиков
            num_plots = 6  # Исходные сигналы, кросс-корреляция, сигналы с учетом задержки, масштабированные, обработанные, разница
            
            # Создаем график
            plt.figure(figsize=(12, 4 * num_plots))
            
            # 1. Исходные сигналы
            plt.subplot(num_plots, 1, 1)
            plt.plot(time_axis, ref_array, label='Референсный', alpha=0.7)
            plt.plot(time_axis, in_array, label='Входной', alpha=0.7)
            plt.title(f'Исходные сигналы (задержка: {delay_ms:.2f} мс)')
            plt.xlabel('Время (с)')
            plt.ylabel('Амплитуда')
            plt.legend()
            plt.grid(True)
            
            # 2. Кросс-корреляция
            plt.subplot(num_plots, 1, 2)
            # Преобразуем лаги в миллисекунды для оси X
            lags_ms = lags * 1000 / sample_rate
            plt.plot(lags_ms, np.abs(correlation))
            plt.axvline(x=delay_ms, color='r', linestyle='--', label=f'Задержка: {delay_ms:.2f} мс')
            plt.title(f'Кросс-корреляция (уверенность: {confidence:.2f})')
            plt.xlabel('Задержка (мс)')
            plt.ylabel('Корреляция')
            plt.grid(True)
            plt.legend()
            
            # 3. Сигналы с учетом задержки
            plt.subplot(num_plots, 1, 3)
            # Если задержка положительная, сдвигаем входной сигнал влево
            # Если отрицательная, сдвигаем референсный сигнал влево
            if lag >= 0:
                # Входной сигнал задержан относительно референсного
                plt.plot(time_axis, ref_array, label='Референсный', alpha=0.7)
                plt.plot(time_axis, np.roll(in_array, -lag)[:max_samples], label='Входной (скорректированный)', alpha=0.7)
                
                # Сохраняем скорректированный входной сигнал для дальнейшего использования
                corrected_in_array = np.roll(in_array, -lag)[:max_samples]
                corrected_ref_array = ref_array
            else:
                # Референсный сигнал задержан относительно входного
                plt.plot(time_axis, np.roll(ref_array, lag)[:max_samples], label='Референсный (скорректированный)', alpha=0.7)
                plt.plot(time_axis, in_array, label='Входной', alpha=0.7)
                
                # Сохраняем скорректированный референсный сигнал для дальнейшего использования
                corrected_ref_array = np.roll(ref_array, lag)[:max_samples]
                corrected_in_array = in_array
            
            plt.title(f'Сигналы с учетом задержки ({delay_ms:.2f} мс)')
            plt.xlabel('Время (с)')
            plt.ylabel('Амплитуда')
            plt.grid(True)
            plt.legend()
            
            # 4. Масштабированные сигналы с учетом задержки
            plt.subplot(num_plots, 1, 4)
            
            if scaled_mic_array is not None and scaled_ref_array is not None:
                # Применяем ту же коррекцию задержки к масштабированным сигналам
                if lag >= 0:
                    # Входной сигнал задержан относительно референсного
                    plt.plot(time_axis, scaled_ref_array, label='Масштабированный референс', alpha=0.7)
                    plt.plot(time_axis, np.roll(scaled_mic_array, -lag)[:max_samples], label='Масштабированный микрофон (скорректированный)', alpha=0.7)
                    
                    # Сохраняем скорректированные масштабированные сигналы
                    corrected_scaled_mic_array = np.roll(scaled_mic_array, -lag)[:max_samples]
                    corrected_scaled_ref_array = scaled_ref_array
                else:
                    # Референсный сигнал задержан относительно входного
                    plt.plot(time_axis, np.roll(scaled_ref_array, lag)[:max_samples], label='Масштабированный референс (скорректированный)', alpha=0.7)
                    plt.plot(time_axis, scaled_mic_array, label='Масштабированный микрофон', alpha=0.7)
                    
                    # Сохраняем скорректированные масштабированные сигналы
                    corrected_scaled_ref_array = np.roll(scaled_ref_array, lag)[:max_samples]
                    corrected_scaled_mic_array = scaled_mic_array
                
                # Вычисляем RMS для масштабированных сигналов с учетом задержки
                scaled_mic_rms = np.sqrt(np.mean(corrected_scaled_mic_array**2))
                scaled_ref_rms = np.sqrt(np.mean(corrected_scaled_ref_array**2))
                scaled_ratio = scaled_mic_rms / scaled_ref_rms if scaled_ref_rms > 0 else 0
                
                plt.title(f'Масштабированные сигналы с учетом задержки (соотношение RMS: {scaled_ratio:.2f})')
            else:
                plt.title('Масштабированные сигналы недоступны')
            
            plt.xlabel('Время (с)')
            plt.ylabel('Амплитуда')
            plt.grid(True)
            plt.legend()
            
            # 5. Обработанный сигнал с учетом задержки
            plt.subplot(num_plots, 1, 5)
            
            if proc_array is not None:
                # Применяем ту же коррекцию задержки к обработанному сигналу
                if lag >= 0:
                    # Обработанный сигнал должен быть синхронизирован с входным, но нам нужно сдвинуть его
                    # так же, как входной, чтобы он соответствовал референсному
                    plt.plot(time_axis, corrected_ref_array, label='Референсный (скорректированный)', alpha=0.7)
                    corrected_proc_array = np.roll(proc_array, -lag)[:max_samples]
                    plt.plot(time_axis, corrected_proc_array, label='Обработанный (скорректированный)', alpha=0.7)
                else:
                    # Референсный сигнал задержан относительно входного и обработанного
                    plt.plot(time_axis, corrected_ref_array, label='Референсный (скорректированный)', alpha=0.7)
                    corrected_proc_array = proc_array  # Обработанный уже синхронизирован с входным
                    plt.plot(time_axis, corrected_proc_array, label='Обработанный', alpha=0.7)
                
                plt.title('Обработанный сигнал с учетом задержки')
            else:
                plt.title('Обработанный сигнал недоступен')
            
            plt.xlabel('Время (с)')
            plt.ylabel('Амплитуда')
            plt.grid(True)
            plt.legend()
            
            # 6. Разница между сигналами с учетом задержки
            plt.subplot(num_plots, 1, 6)
            
            # Разница между исходными сигналами с учетом задержки
            diff_original = corrected_in_array - corrected_ref_array
            plt.plot(time_axis, diff_original, label='Исходная разница (до обработки)', alpha=0.7, color='red')
            
            # Разница между масштабированными сигналами с учетом задержки
            if scaled_mic_array is not None and scaled_ref_array is not None:
                scaled_diff = corrected_scaled_mic_array - corrected_scaled_ref_array
                plt.plot(time_axis, scaled_diff, label='Разница после масштабирования', alpha=0.7, color='orange')
            
            # Разница между обработанным и референсным сигналами с учетом задержки
            if proc_array is not None:
                # corrected_proc_array уже задан в предыдущем блоке с учетом задержки
                diff_processed = corrected_proc_array - corrected_ref_array
                plt.plot(time_axis, diff_processed, label='Разница после AEC (меньше = лучше)', alpha=0.7, color='green')
            
            # Добавляем горизонтальную линию на нуле
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Вычисляем RMS разницы
            diff_rms = np.sqrt(np.mean(diff_original**2))
            proc_diff_rms = np.sqrt(np.mean(diff_processed**2)) if proc_array is not None else None
            
            # Добавляем аннотации с RMS для каждой линии
            plt.annotate(f'RMS исходной разницы: {diff_rms:.6f}', 
                         xy=(0.02, 0.95), xycoords='axes fraction', 
                         color='red', fontsize=9)

            if scaled_mic_array is not None and scaled_ref_array is not None:
                scaled_diff_rms = np.sqrt(np.mean(scaled_diff**2))
                plt.annotate(f'RMS после масштабирования: {scaled_diff_rms:.6f}', 
                             xy=(0.02, 0.90), xycoords='axes fraction', 
                             color='orange', fontsize=9)

            if proc_array is not None:
                plt.annotate(f'RMS после AEC: {proc_diff_rms:.6f}', 
                             xy=(0.02, 0.85), xycoords='axes fraction', 
                             color='green', fontsize=9)
            
            # Улучшенное название с информацией об улучшении
            if proc_array is not None and diff_rms > 0 and proc_diff_rms > 0:
                improvement = diff_rms / proc_diff_rms
                plt.title(f'Остаточное эхо (разница с референсом): улучшение после AEC {improvement:.2f}x ({20*np.log10(improvement):.2f} дБ)')
            else:
                plt.title(f'Остаточное эхо (разница с референсом): RMS исходной разницы = {diff_rms:.6f}')
            
            plt.xlabel('Время (с)')
            plt.ylabel('Амплитуда разницы')
            plt.legend()
            plt.grid(True)
            
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
            
            if scaled_mic_array is not None and scaled_ref_array is not None:
                scaled_diff = corrected_scaled_mic_array - corrected_scaled_ref_array
                scaled_diff_rms = np.sqrt(np.mean(scaled_diff**2))
                scaled_mic_rms = np.sqrt(np.mean(corrected_scaled_mic_array**2))
                scaled_ref_rms = np.sqrt(np.mean(corrected_scaled_ref_array**2))
                scaled_ratio = scaled_mic_rms / scaled_ref_rms if scaled_ref_rms > 0 else 0
                
                logging.info(f"  RMS масштабированного микрофона: {scaled_mic_rms:.6f}")
                logging.info(f"  RMS масштабированного референса: {scaled_ref_rms:.6f}")
                logging.info(f"  Соотношение после масштабирования: {scaled_ratio:.6f}")
                logging.info(f"  RMS разницы после масштабирования: {scaled_diff_rms:.6f}")
                
                results["scaled"] = {
                    "mic_rms": scaled_mic_rms,
                    "ref_rms": scaled_ref_rms,
                    "ratio": scaled_ratio,
                    "diff_rms": scaled_diff_rms
                }
            
            if proc_array is not None:
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
        
        return results
    
    except ImportError:
        logging.warning("Не удалось импортировать необходимые библиотеки для визуализации")
        return results
    except Exception as e:
        logging.error(f"Ошибка при визуализации: {e}")
        logging.exception("Подробная информация об ошибке:")
        return results

def visualize_delay(reference_data: bytes, input_data: bytes, 
                   sample_rate: int = 16000, max_delay_ms: int = 1000,
                   output_dir: str = "results", output_prefix: str = "aec") -> Dict[str, Any]:
    """
    Визуализирует задержку между референсным и входным сигналами
    
    Args:
        reference_data: Референсные данные (байты)
        input_data: Входные данные (байты)
        sample_rate: Частота дискретизации (Гц)
        max_delay_ms: Максимальная задержка для поиска (мс)
        output_dir: Директория для сохранения результатов
        output_prefix: Префикс для имен выходных файлов
        
    Returns:
        Dict[str, Any]: Словарь с метриками задержки
    """
    try:
        # Преобразуем байты в numpy массивы
        ref_array = np.frombuffer(reference_data, dtype=np.int16).astype(np.float32)
        in_array = np.frombuffer(input_data, dtype=np.int16).astype(np.float32)
        
        # Ограничиваем длину для анализа (первые 10 секунд или меньше)
        max_samples = min(10 * sample_rate, len(ref_array), len(in_array))
        ref_array = ref_array[:max_samples]
        in_array = in_array[:max_samples]
        
        # Вычисляем кросс-корреляцию
        correlation = signal.correlate(in_array, ref_array, mode='full')
        lags = signal.correlation_lags(len(in_array), len(ref_array), mode='full')
        
        # Ограничиваем поиск максимума заданным диапазоном задержки
        max_lag_samples = int(max_delay_ms * sample_rate / 1000)
        center_index = len(lags) // 2
        start_index = max(0, center_index - max_lag_samples)
        end_index = min(len(lags), center_index + max_lag_samples + 1)
        
        # Находим индекс максимальной корреляции в заданном диапазоне
        max_index = start_index + np.argmax(np.abs(correlation[start_index:end_index]))
        lag = lags[max_index]
        
        # Вычисляем задержку в миллисекундах
        delay_ms = lag * 1000 / sample_rate
        
        # Вычисляем уверенность в оценке
        confidence = np.abs(correlation[max_index]) / np.max(np.abs(correlation))
        
        # Создаем график
        plt.figure(figsize=(12, 8))
        
        # 1. Кросс-корреляция
        plt.subplot(2, 1, 1)
        # Преобразуем лаги в миллисекунды для оси X
        lags_ms = lags * 1000 / sample_rate
        plt.plot(lags_ms, np.abs(correlation))
        plt.axvline(x=delay_ms, color='r', linestyle='--', label=f'Задержка: {delay_ms:.2f} мс')
        plt.title(f'Кросс-корреляция (уверенность: {confidence:.2f})')
        plt.xlabel('Задержка (мс)')
        plt.ylabel('Корреляция')
        plt.grid(True)
        plt.legend()
        
        # 2. Сигналы с учетом задержки
        plt.subplot(2, 1, 2)
        # Создаем временную ось
        time_axis = np.arange(max_samples) / sample_rate
        
        # Если задержка положительная, сдвигаем входной сигнал вправо
        # Если отрицательная, сдвигаем референсный сигнал вправо
        if lag >= 0:
            # Входной сигнал задержан относительно референсного
            plt.plot(time_axis, ref_array, label='Референсный', alpha=0.7)
            plt.plot(time_axis, np.roll(in_array, -lag)[:max_samples], label='Входной (скорректированный)', alpha=0.7)
        else:
            # Референсный сигнал задержан относительно входного
            plt.plot(time_axis, np.roll(ref_array, lag)[:max_samples], label='Референсный (скорректированный)', alpha=0.7)
            plt.plot(time_axis, in_array, label='Входной', alpha=0.7)
        
        plt.title(f'Сигналы с учетом задержки ({delay_ms:.2f} мс)')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.grid(True)
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
