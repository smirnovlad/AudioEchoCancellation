"""
Простой инструмент тестирования WebRTC AEC без зависимостей от NumPy

Этот скрипт позволяет тестировать функциональность WebRTC AEC путем:
1. Воспроизведения аудиофайла через динамики
2. Одновременной записи аудио с микрофона
3. Применения обработки AEC к записанному аудио
4. Сохранения обработанного аудио в файл
"""

import os
import sys
import time
import wave
import threading
import argparse
import subprocess
import logging
from pathlib import Path
import numpy as np
import tempfile
from scipy import signal
from visualization import visualize_audio_processing

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # Показывать все сообщения уровня DEBUG и выше
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webrtc_aec_test.log", mode='w'),  # Перезапись файла
        logging.StreamHandler()  # Вывод в консоль
    ]
)

# Добавляем корень проекта в путь Python для импорта модулей проекта
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Импорт WebRTCAECSession
try:
    from aec.webrtc_aec_wrapper import WebRTCAECSession
    logging.info("WebRTCAECSession успешно импортирован")
except Exception as e:
    logging.error(f"Ошибка при импорте WebRTCAECSession: {e}")
    logging.exception("Подробная информация об ошибке:")
    raise

def play_audio_file(filename, start_event=None):
    """Воспроизведение аудиофайла с помощью системной команды"""
    try:
        # Если есть событие синхронизации, ждем готовности записи
        if start_event:
            logging.debug("Воспроизведение ожидает готовности записи...")
            start_event.wait()  # Ждем, пока запись будет готова
            logging.debug("Запись готова, начинаем воспроизведение")
            # Устанавливаем флаг начала записи
            start_event.recording_started = True
            
        logging.info(f"Воспроизведение файла: {filename}")
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['afplay', filename])
        else:  # Linux и другие
            subprocess.run(['aplay', filename])
        logging.info("Воспроизведение завершено")
        return True
    except Exception as e:
        logging.error(f"Ошибка при воспроизведении аудио: {e}")
        return False

def record_audio(filename, duration=10, sample_rate=16000, channels=1, start_event=None):
    """Запись аудио с микрофона с помощью системной команды"""
    try:
        logging.info(f"Подготовка к записи аудио в файл: {filename}, длительность: {duration}с")
        
        # Сигнализируем о готовности к записи
        if start_event:
            logging.debug("Запись готова к старту, ожидание сигнала...")
            start_event.set()
            # Ждем сигнала для начала записи
            while not hasattr(start_event, 'recording_started') or not start_event.recording_started:
                time.sleep(0.01)
        
        logging.info("Начало записи...")
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['rec', '-r', str(sample_rate), '-c', str(channels), filename, 'trim', '0', str(duration)])
        else:  # Linux и другие
            subprocess.run(['arecord', '-f', 'S16_LE', '-r', str(sample_rate), '-c', str(channels), '-d', str(duration), filename])
        logging.info("Запись завершена")
        return True
    except Exception as e:
        logging.error(f"Ошибка при записи аудио: {e}")
        return False

def process_audio_with_aec(input_file, output_file, reference_file, sample_rate=16000, channels=1, 
                          system_delay=0, visualize=True, output_dir="results", frame_size_ms=10.0):
    """Обработка аудиофайла с помощью WebRTC AEC"""
    try:
        logging.info(f"Обработка файла {input_file} с помощью AEC...")
        
        # Чтение референсного файла
        with wave.open(reference_file, 'rb') as ref_wf:
            ref_data = ref_wf.readframes(ref_wf.getnframes())
            
        # Чтение входного файла
        with wave.open(input_file, 'rb') as in_wf:
            in_data = in_wf.readframes(in_wf.getnframes())
        
        # Инициализация сессии AEC
        aec_session = WebRTCAECSession(
            session_id="test_session",
            sample_rate=sample_rate,
            channels=channels,
            batch_mode=False,
            frame_size_ms=frame_size_ms
        )
        
        # Логируем размер фрейма из AEC сессии
        frame_size = aec_session.frame_size
        frame_size_ms = frame_size / sample_rate * 1000
        frame_size_bytes = frame_size * 2 * channels  # 2 байта на сэмпл (16 бит)
        
        logging.info(f"Размер фрейма в AEC сессии: {frame_size} сэмплов, {frame_size_bytes} байт, {frame_size_ms:.2f} мс")
        
        # Оценка и установка задержки
        delay_samples, delay_ms, confidence = aec_session.auto_set_delay(ref_data, in_data)
        
        # Вычисляем задержку в фреймах
        delay_frames = int(delay_samples / frame_size)
        logging.info(f"Задержка в фреймах: {delay_frames} фреймов")
        
        # Разделение на фреймы
        ref_frames = []
        for i in range(0, len(ref_data), frame_size_bytes):
            frame = ref_data[i:i+frame_size_bytes]
            if len(frame) == frame_size_bytes:  # Только полные фреймы
                ref_frames.append(frame)
        
        in_frames = []
        for i in range(0, len(in_data), frame_size_bytes):
            frame = in_data[i:i+frame_size_bytes]
            if len(frame) == frame_size_bytes:  # Только полные фреймы
                in_frames.append(frame)
        
        # Логируем размеры первого фрейма для проверки
        if ref_frames:
            first_ref_frame = ref_frames[0]
            first_ref_samples = len(first_ref_frame) // (2 * channels)  # 2 байта на сэмпл
            first_ref_ms = first_ref_samples / sample_rate * 1000
            logging.info(f"Первый референсный фрейм: {first_ref_samples} сэмплов, {len(first_ref_frame)} байт, {first_ref_ms:.2f} мс")
        
        if in_frames:
            first_in_frame = in_frames[0]
            first_in_samples = len(first_in_frame) // (2 * channels)  # 2 байта на сэмпл
            first_in_ms = first_in_samples / sample_rate * 1000
            logging.info(f"Первый входной фрейм: {first_in_samples} сэмплов, {len(first_in_frame)} байт, {first_in_ms:.2f} мс")
        
        logging.info(f"Референсный файл разделен на {len(ref_frames)} фреймов")
        logging.info(f"Входной файл разделен на {len(in_frames)} фреймов")
        
        # Определяем количество фреймов для обработки
        min_frames = min(len(ref_frames), len(in_frames))
        logging.info(f"Будет обработано {min_frames} фреймов")
        
        # Предварительная буферизация с учетом задержки
        pre_buffer_size = min(delay_frames, min_frames)  # Задержка
        pre_buffer_ms = pre_buffer_size * frame_size_ms
        logging.info(f"Предварительная буферизация: {pre_buffer_size} фреймов ({pre_buffer_ms:.2f} мс)")
        
        for i in range(pre_buffer_size):
            ref_frame = ref_frames[i]
            ref_samples = len(ref_frame) // (2 * channels)
            ref_ms = ref_samples / sample_rate * 1000
            logging.info(f"Добавлен референсный фрейм {i+1}/{pre_buffer_size} для предварительной буферизации: {ref_samples} сэмплов, {len(ref_frame)} байт, {ref_ms:.2f} мс")
            aec_session.add_reference_frame(ref_frame)
        
        # Обработка фреймов в правильной последовательности
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
        
        # Получаем финальную статистику
        final_stats = aec_session.get_statistics()
        
        # Выводим информацию о фреймах с эхо
        total_frames = final_stats["processed_frames"]
        echo_frames = final_stats["echo_frames"]
        
        # Проверяем, что total_frames не равно 0 и echo_frames не превышает total_frames
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
        
        # Сохранение обработанного аудио
        with wave.open(output_file, 'wb') as out_wf:
            out_wf.setnchannels(channels)
            out_wf.setsampwidth(2)  # 16-bit audio = 2 bytes
            out_wf.setframerate(sample_rate)
            out_wf.writeframes(b''.join(processed_frames))
        
        logging.info(f"Обработка завершена, результат сохранен в {output_file}")
        
        # Расчет метрик качества
        metrics = calculate_metrics(ref_data, b''.join(processed_frames), in_data)
        metrics["echo_frames"] = echo_frames
        metrics["echo_frames_percentage"] = echo_percentage
        metrics["delay_samples"] = delay_samples
        metrics["delay_ms"] = delay_ms
        metrics["delay_confidence"] = confidence
        
        # Выводим метрики
        logging.info("\nМетрики качества AEC:")
        logging.info(f"  MSE (среднеквадратичная ошибка): {metrics['mse']:.6f}")
        logging.info(f"  NMSE (нормализованная MSE): {metrics['nmse']:.6f}")
        logging.info(f"  SNR (отношение сигнал/шум): {metrics['snr_db']:.2f} дБ")
        logging.info(f"  Корреляция (обработанный-референс): {metrics['correlation_proc_ref']:.4f}")
        logging.info(f"  Фреймов с эхо: {metrics['echo_frames']} ({metrics['echo_frames_percentage']:.2f}%)")
        logging.info(f"  Оцененная задержка: {metrics['delay_samples']} сэмплов ({metrics['delay_ms']:.2f} мс)")

        if 'correlation_orig_ref' in metrics:
            logging.info(f"  Корреляция (входной-референс): {metrics['correlation_orig_ref']:.4f}")
            logging.info(f"  Улучшение корреляции: {metrics['correlation_improvement']:.4f}")
            
        if 'erle_db' in metrics:
            logging.info(f"  ERLE (подавление эха): {metrics['erle_db']:.2f} дБ")
        
        # Получаем масштабированные сигналы
        scaled_mic_data, scaled_ref_data = aec_session.get_scaled_signals(in_data, ref_data)
        
        # В конце функции добавляем визуализацию
        if visualize:
            # Создаем директорию для результатов, если она не существует
            os.makedirs(output_dir, exist_ok=True)
            
            # Читаем обработанные данные
            with open(output_file, 'rb') as f:
                processed_data = f.read()
            
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
            metrics.update(visualization_results)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Ошибка при обработке аудио: {e}")
        logging.exception("Подробная информация об ошибке:")
        raise

def get_wav_duration(filename):
    """Получить длительность WAV файла в секундах"""
    try:
        with wave.open(filename, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        logging.error(f"Ошибка при определении длительности файла {filename}: {e}")
        return 5  # Возвращаем значение по умолчанию в случае ошибки

def convert_audio_format(input_file, output_file, sample_rate=16000, channels=1):
    """Конвертирует аудиофайл в нужный формат с помощью ffmpeg"""
    try:
        logging.info(f"Конвертация {input_file} в формат {sample_rate}Hz, {channels} каналов...")
        
        # Проверяем, установлен ли ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logging.error("ffmpeg не установлен. Установите его для конвертации аудио.")
            return False
        
        # Выполняем конвертацию
        cmd = [
            'ffmpeg', '-y',  # Перезаписывать существующие файлы
            '-i', input_file,  # Входной файл
            '-ar', str(sample_rate),  # Частота дискретизации
            '-ac', str(channels),  # Количество каналов
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            output_file  # Выходной файл
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            logging.info(f"Конвертация успешно завершена: {output_file}")
            return True
        else:
            logging.error(f"Ошибка при конвертации: {result.stderr.decode()}")
            return False
    except Exception as e:
        logging.error(f"Ошибка при конвертации аудио: {e}")
        return False

def calculate_metrics(reference_data, processed_data, original_data=None):
    """Рассчитывает метрики качества AEC"""
    
    # Преобразуем байтовые данные в массивы numpy
    def bytes_to_numpy(audio_bytes, dtype=np.int16):
        return np.frombuffer(audio_bytes, dtype=dtype)
    
    ref = bytes_to_numpy(reference_data)
    proc = bytes_to_numpy(processed_data)
    orig = bytes_to_numpy(original_data) if original_data is not None else None
    
    # Обрезаем до одинаковой длины
    min_len = min(len(ref), len(proc))
    if orig is not None:
        min_len = min(min_len, len(orig))
    
    ref = ref[:min_len]
    proc = proc[:min_len]
    if orig is not None:
        orig = orig[:min_len]
    
    # Нормализуем данные (приводим к диапазону [-1, 1])
    ref_norm = ref.astype(np.float32) / 32768.0
    proc_norm = proc.astype(np.float32) / 32768.0
    if orig is not None:
        orig_norm = orig.astype(np.float32) / 32768.0
    
    # Рассчитываем метрики
    metrics = {}
    
    # 1. Среднеквадратичная ошибка (MSE)
    mse = np.mean((proc_norm - ref_norm) ** 2)
    metrics['mse'] = mse
    
    # 2. Нормализованная среднеквадратичная ошибка (NMSE)
    # NMSE = MSE / среднее(ref^2)
    ref_power = np.mean(ref_norm ** 2)
    if ref_power > 0:
        nmse = mse / ref_power
    else:
        nmse = float('inf')
    metrics['nmse'] = nmse
    
    # 3. Отношение сигнал/шум (SNR) в дБ
    # SNR = 10 * log10(среднее(ref^2) / среднее((proc - ref)^2))
    if mse > 0:
        snr = 10 * np.log10(ref_power / mse)
    else:
        snr = float('inf')
    metrics['snr_db'] = snr
    
    # 4. Коэффициент корреляции Пирсона между обработанным и референсным сигналами
    # Показывает степень линейной зависимости между сигналами
    corr_proc_ref = np.corrcoef(ref_norm, proc_norm)[0, 1]
    metrics['correlation_proc_ref'] = corr_proc_ref
    
    # 5. Коэффициент корреляции между входным и референсным сигналами
    # Показывает, насколько сильно эхо присутствует во входном сигнале
    if orig is not None:
        corr_orig_ref = np.corrcoef(ref_norm, orig_norm)[0, 1]
        metrics['correlation_orig_ref'] = corr_orig_ref
        
        # 6. Улучшение корреляции (насколько AEC уменьшил корреляцию с референсным сигналом)
        # Чем больше разница, тем лучше работает AEC
        corr_improvement = abs(corr_orig_ref) - abs(corr_proc_ref)
        metrics['correlation_improvement'] = corr_improvement
    
    # 7. Подавление эха (Echo Return Loss Enhancement, ERLE) в дБ
    # Если есть оригинальный сигнал (до обработки AEC)
    if orig is not None:
        # ERLE = 10 * log10(среднее(orig^2) / среднее(proc^2))
        orig_power = np.mean(orig_norm ** 2)
        proc_power = np.mean(proc_norm ** 2)
        
        if proc_power > 0:
            erle = 10 * np.log10(orig_power / proc_power)
        else:
            erle = float('inf')
        metrics['erle_db'] = erle
    
    # Метрики эха будут добавлены в process_audio_with_aec
    metrics["echo_frames"] = 0
    metrics["echo_frames_percentage"] = 0.0
    
    return metrics

def estimate_delay_cross_correlation(reference_data, input_data, sample_rate=16000, max_delay_ms=1000):
    """
    Оценивает задержку между референсным и входным сигналами с помощью кросс-корреляции
    
    Args:
        reference_data: Референсный сигнал (байты)
        input_data: Входной сигнал (байты)
        sample_rate: Частота дискретизации (Гц)
        max_delay_ms: Максимальная задержка для поиска (мс)
        
    Returns:
        tuple: (задержка в сэмплах, задержка в мс, уверенность в оценке)
    """
    logging.info("Оценка задержки с помощью кросс-корреляции...")
    
    # Преобразуем байты в numpy массивы
    ref_signal = np.frombuffer(reference_data, dtype=np.int16)
    in_signal = np.frombuffer(input_data, dtype=np.int16)
    
    # Ограничиваем длину сигналов для ускорения вычислений
    # Используем первые 5 секунд или меньше, если сигналы короче
    max_samples = min(5 * sample_rate, len(ref_signal), len(in_signal))
    ref_signal = ref_signal[:max_samples]
    in_signal = in_signal[:max_samples]
    
    # Нормализуем сигналы для лучшей корреляции
    ref_signal = ref_signal.astype(np.float32) / 32768.0
    in_signal = in_signal.astype(np.float32) / 32768.0
    
    # Вычисляем максимальную задержку в сэмплах
    max_delay_samples = int(max_delay_ms * sample_rate / 1000)
    
    # Вычисляем кросс-корреляцию
    correlation = np.correlate(in_signal, ref_signal, mode='full')
    
    # Находим индекс максимальной корреляции
    max_index = np.argmax(correlation)
    
    # Вычисляем задержку (учитываем, что correlate возвращает сдвиг)
    delay_samples = max_index - (len(ref_signal) - 1)
    
    # Ограничиваем задержку
    delay_samples = max(0, min(delay_samples, max_delay_samples))
    
    # Вычисляем задержку в мс
    delay_ms = delay_samples * 1000 / sample_rate
    
    # Вычисляем уверенность в оценке (нормализованное значение максимума корреляции)
    max_correlation = correlation[max_index]
    confidence = max_correlation / (np.std(ref_signal) * np.std(in_signal) * len(ref_signal))
    confidence = min(1.0, max(0.0, confidence))
    
    logging.info(f"Оценка задержки: {delay_samples} сэмплов ({delay_ms:.2f} мс), уверенность: {confidence:.2f}")
    
    return delay_samples, delay_ms, confidence

def estimate_delay_phase_correlation(reference_data, input_data, sample_rate=16000, max_delay_ms=1000):
    """
    Оценивает задержку между референсным и входным сигналами с помощью фазовой корреляции
    
    Args:
        reference_data: Референсный сигнал (байты)
        input_data: Входной сигнал (байты)
        sample_rate: Частота дискретизации (Гц)
        max_delay_ms: Максимальная задержка для поиска (мс)
        
    Returns:
        tuple: (задержка в сэмплах, задержка в мс, уверенность в оценке)
    """
    logging.info("Оценка задержки с помощью фазовой корреляции...")
    
    # Преобразуем байты в numpy массивы
    ref_signal = np.frombuffer(reference_data, dtype=np.int16)
    in_signal = np.frombuffer(input_data, dtype=np.int16)
    
    # Ограничиваем длину сигналов
    max_samples = min(5 * sample_rate, len(ref_signal), len(in_signal))
    ref_signal = ref_signal[:max_samples]
    in_signal = in_signal[:max_samples]
    
    # Нормализуем сигналы
    ref_signal = ref_signal.astype(np.float32) / 32768.0
    in_signal = in_signal.astype(np.float32) / 32768.0
    
    # Вычисляем FFT
    ref_fft = np.fft.fft(ref_signal)
    in_fft = np.fft.fft(in_signal)
    
    # Вычисляем кросс-спектр
    cross_spectrum = ref_fft * np.conj(in_fft)
    
    # Нормализуем кросс-спектр
    cross_spectrum = cross_spectrum / np.abs(cross_spectrum)
    
    # Вычисляем обратное FFT
    correlation = np.fft.ifft(cross_spectrum)
    
    # Находим индекс максимальной корреляции
    max_index = np.argmax(np.abs(correlation))
    
    # Вычисляем задержку
    if max_index > len(correlation) / 2:
        delay_samples = max_index - len(correlation)
    else:
        delay_samples = max_index
    
    # Ограничиваем задержку
    max_delay_samples = int(max_delay_ms * sample_rate / 1000)
    delay_samples = max(0, min(delay_samples, max_delay_samples))
    
    # Вычисляем задержку в мс
    delay_ms = delay_samples * 1000 / sample_rate
    
    # Вычисляем уверенность в оценке
    confidence = np.abs(correlation[max_index]) / len(correlation)
    confidence = min(1.0, max(0.0, confidence))
    
    logging.info(f"Оценка задержки (фазовая корреляция): {delay_samples} сэмплов ({delay_ms:.2f} мс), уверенность: {confidence:.2f}")
    
    return delay_samples, delay_ms, confidence

def estimate_delay_combined(reference_data, input_data, sample_rate=16000, max_delay_ms=1000):
    """
    Комбинирует несколько методов оценки задержки для повышения надежности
    
    Args:
        reference_data: Референсный сигнал (байты)
        input_data: Входной сигнал (байты)
        sample_rate: Частота дискретизации (Гц)
        max_delay_ms: Максимальная задержка для поиска (мс)
        
    Returns:
        tuple: (задержка в сэмплах, задержка в мс)
    """
    # Получаем оценки от разных методов
    delay1, delay_ms1, conf1 = estimate_delay_cross_correlation(reference_data, input_data, sample_rate, max_delay_ms)
    delay2, delay_ms2, conf2 = estimate_delay_phase_correlation(reference_data, input_data, sample_rate, max_delay_ms)
    
    # Комбинируем оценки с учетом уверенности
    if conf1 > conf2:
        logging.info(f"Выбрана оценка кросс-корреляции (уверенность: {conf1:.2f})")
        return delay1, delay_ms1
    else:
        logging.info(f"Выбрана оценка фазовой корреляции (уверенность: {conf2:.2f})")
        return delay2, delay_ms2

def main():
    parser = argparse.ArgumentParser(description="Простой инструмент тестирования WebRTC AEC")
    parser.add_argument("--reference", "-r", default="reference_new.wav", 
                        help="Аудиофайл для воспроизведения через динамики (по умолчанию: reference.wav)")
    parser.add_argument("--output", "-o", default="processed_input.wav", 
                        help="Файл для сохранения обработанного аудио (по умолчанию: processed_input.wav)")
    parser.add_argument("--duration", "-d", type=int, default=None, 
                        help="Длительность записи в секундах (по умолчанию: длительность референсного файла + 1 секунда)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000, 
                        help="Частота дискретизации в Гц (по умолчанию: 16000)")
    parser.add_argument("--channels", "-ch", type=int, default=1, 
                        help="Количество аудиоканалов (по умолчанию: 1)")
    parser.add_argument("--convert", "-c", action="store_true",
                        help="Конвертировать референсный файл в нужный формат перед тестированием")
    parser.add_argument("--frame-size-ms", "-fs", type=float, default=10.0,
                        help="Размер фрейма в миллисекундах (по умолчанию: 10.0)")
    
    args = parser.parse_args()
    
    # Проверка наличия референсного файла
    if not os.path.exists(args.reference):
        logging.error(f"Ошибка: Референсный файл '{args.reference}' не найден")
        return
    
    # Конвертация референсного файла, если требуется
    reference_file = args.reference
    if args.convert:
        converted_file = "reference_converted.wav"
        if convert_audio_format(args.reference, converted_file, args.sample_rate, args.channels):
            reference_file = converted_file
        else:
            logging.warning("Не удалось конвертировать файл, используем оригинальный")
    
    # Определяем длительность записи
    if args.duration is None:
        ref_duration = get_wav_duration(reference_file)
        # Добавляем 1 секунду для гарантии полного захвата
        duration = int(ref_duration) + 1
        logging.info(f"Длительность референсного файла: {ref_duration:.2f} с, установлена длительность записи: {duration} с")
    else:
        duration = args.duration
    
    # Временный файл для записи
    recorded_file = "original_input.wav"
    
    # Шаг 1: Запись и воспроизведение одновременно
    logging.info("Шаг 1: Запись и воспроизведение одновременно")
    
    # Создаем событие для синхронизации начала записи и воспроизведения
    start_event = threading.Event()
    start_event.recording_started = False
    
    # Запускаем запись в отдельном потоке
    record_thread = threading.Thread(
        target=record_audio, 
        args=(recorded_file, duration, args.sample_rate, args.channels, start_event)
    )
    record_thread.start()
    
    # Воспроизводим референсный файл (будет ждать готовности записи)
    play_audio_file(reference_file, start_event)
    
    # Ждем завершения записи
    record_thread.join()
    
    # Шаг 2: Обработка записанного аудио с помощью AEC
    logging.info("\nШаг 2: Обработка записанного аудио с помощью AEC")
    
    # Проверяем формат записанного файла
    with wave.open(recorded_file, 'rb') as wf:
        rec_channels = wf.getnchannels()
        rec_rate = wf.getframerate()
        logging.info(f"Записанный файл имеет формат: {rec_rate}Hz, {rec_channels} каналов")
        logging.warning(f"Требуется: {args.sample_rate}Hz, {args.channels} каналов")
        if rec_rate != args.sample_rate or rec_channels != args.channels:            
            # Конвертируем записанный файл
            converted_input = "original_input_converted.wav"
            if convert_audio_format(recorded_file, converted_input, args.sample_rate, args.channels):
                logging.info(f"Записанный файл успешно конвертирован в {converted_input}")
                recorded_file = converted_input
            else:
                logging.warning("Не удалось конвертировать записанный файл, используем оригинальный")
    
    # Чтение файлов для визуализации
    with wave.open(reference_file, 'rb') as ref_wf:
        ref_data = ref_wf.readframes(ref_wf.getnframes())
    
    with wave.open(recorded_file, 'rb') as in_wf:
        in_data = in_wf.readframes(in_wf.getnframes())
    
    # Обработка аудио с AEC
    metrics = process_audio_with_aec(
        recorded_file, 
        args.output, 
        reference_file,
        args.sample_rate,
        args.channels,
        visualize=True,
        output_dir=os.path.dirname(os.path.abspath(args.output)),
        frame_size_ms=args.frame_size_ms
    )
    
    # Визуализация уже выполнена внутри process_audio_with_aec
    
    logging.info("\nТестирование AEC завершено!")
    logging.info(f"Оригинальная запись: {recorded_file}")
    logging.info(f"Обработанная запись: {args.output}")
    logging.info(f"Референсный файл: {reference_file}")

if __name__ == "__main__":
    main() 