import os
import shutil
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

"""
Скрипт для создания тестовых данных с различными уровнями громкости и задержками.

Скрипт ожидает следующую структуру директорий (создается скриптом prepare_tests.py):
directory_path/
├── clear_reference/           # Директория для чистого референсного сигнала
└── reference_by_micro/        # Директория для референса через микрофон

В каждой поддиректории скрипт создает директории для разных задержек:
reference_by_micro/
├── delay_0/            # Задержка 0 мс (без задержки)
├── delay_50/           # Задержка 50 мс
├── delay_100/          # Задержка 100 мс
├── delay_200/          # Задержка 200 мс
└── delay_300/          # Задержка 300 мс

Внутри каждой директории с задержкой создаются директории с разными уровнями громкости:
delay_100/
├── volume_01/          # 10% громкости
├── volume_04/          # 40% громкости
├── volume_07/          # 70% громкости
├── volume_10/          # 100% громкости (исходная)
└── volume_13/          # 130% громкости

В каждой поддиректории создаются следующие файлы:
1. reference.wav - оригинальный референсный сигнал без изменений
2. reference_by_micro_volumed.wav - референсный сигнал с измененной громкостью (без задержки)
3. reference_by_micro_volumed_delayed.wav - референсный сигнал с измененной громкостью и добавленной задержкой
   (соответствует сигналу, который использовался для микширования с голосом пользователя)
4. original_input.wav - микшированный сигнал, содержащий микс голоса пользователя и
   соответствующего референсного сигнала с примененным коэффициентом громкости 
   и задержкой (если указана)
5. my_voice.wav - чистый голос пользователя без примесей, для сравнения

ВАЖНО: Скрипт ожидает наличия следующих файлов в обрабатываемой директории:
- reference.wav - чистый референсный сигнал
- reference_by_micro.wav - запись референса через микрофон
- my_voice.wav или my_voice.mp3 - запись голоса пользователя

Процесс работы:
1. Скрипт проверяет наличие необходимых файлов в директории
2. Загружает оба референсных файла и файл с голосом пользователя
3. Для каждой задержки (только для reference_by_micro) создает поддиректории
4. Для каждого уровня громкости создает директории в соответствующей структуре
5. В каждой директории создает файлы reference.wav, reference_by_micro_volumed.wav, 
   reference_by_micro_volumed_delayed.wav, original_input.wav и my_voice.wav с заданными параметрами
"""

# Задержки в миллисекундах
DELAY_LEVELS = {
    "0": 0,     # 0 мс (без задержки)
    "50": 50,   # 50 мс
    "100": 100, # 100 мс
    "200": 200, # 200 мс
    "250": 250, # 250 мс
    # "300": 300, # 300 мс
    "1000": 1000, # 1000 мс (1 секунда)
}

# Коэффициенты изменения амплитуды
VOLUME_LEVELS = {
    "volume_01": 0.1,  # 10% от исходной громкости
    # "volume_04": 0.4,  # 40% от исходной громкости
    "volume_05": 0.5,  # 50% от исходной громкости
    "volume_07": 0.7,  # 70% от исходной громкости
    "volume_10": 1.0,  # 100% (исходная громкость)
    "volume_13": 1.3,  # 130% от исходной громкости
}

def create_directory_if_not_exists(directory_path):
    """Создает директорию, если она не существует."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Создана директория: {directory_path}")
    else:
        logging.info(f"Директория уже существует: {directory_path}")

def adjust_volume(input_file, output_file, volume_factor):
    """Изменяет громкость аудиофайла на заданный коэффициент."""
    # Чтение аудиофайла
    data, samplerate = sf.read(input_file)
    
    # Выводим информацию о входном файле
    is_stereo = len(data.shape) > 1
    channels = data.shape[1] if is_stereo else 1
    length_samples = len(data)
    duration = length_samples / samplerate
    logging.info(f"Входной файл {input_file}:")
    logging.info(f"  - Размер данных: {data.shape}")
    logging.info(f"  - Количество каналов: {channels}")
    logging.info(f"  - Длина в семплах: {length_samples}")
    logging.info(f"  - Длительность: {duration:.2f} секунд")
    logging.info(f"  - Частота дискретизации: {samplerate} Гц")
    
    # Изменение амплитуды (громкости)
    adjusted_data = data * volume_factor
    
    # Проверяем, не изменился ли размер данных после умножения
    adjusted_is_stereo = len(adjusted_data.shape) > 1
    adjusted_channels = adjusted_data.shape[1] if adjusted_is_stereo else 1
    adjusted_length = len(adjusted_data)
    logging.info(f"После изменения громкости:")
    logging.info(f"  - Размер данных: {adjusted_data.shape}")
    logging.info(f"  - Количество каналов: {adjusted_channels}")
    logging.info(f"  - Длина в семплах: {adjusted_length}")
    
    # Сохранение измененного аудиофайла с явным указанием параметров
    try:
        # Получаем информацию о формате исходного файла
        info = sf.info(input_file)
        logging.info(f"Информация о формате исходного файла:")
        logging.info(f"  - Формат: {info.for1}")
        logging.info(f"  - Subtype: {info.subtype}")
        
        # Сохраняем с теми же параметрами
        sf.write(output_file, adjusted_data, samplerate, format=info.format, subtype=info.subtype)
        logging.info(f"Создан файл с громкостью {volume_factor}: {output_file} (формат: {info.format}, subtype: {info.subtype})")
    except Exception as e:
        logging.info(f"Не удалось определить формат исходного файла: {e}")
        logging.info("Сохраняем файл с параметрами по умолчанию")
        sf.write(output_file, adjusted_data, samplerate)
        logging.info(f"Создан файл с громкостью {volume_factor}: {output_file}")
    
    # Проверяем размер сохраненного файла
    try:
        saved_data, saved_samplerate = sf.read(output_file)
        saved_is_stereo = len(saved_data.shape) > 1
        saved_channels = saved_data.shape[1] if saved_is_stereo else 1
        saved_length = len(saved_data)
        saved_duration = saved_length / saved_samplerate
        logging.info(f"Сохраненный файл {output_file}:")
        logging.info(f"  - Размер данных: {saved_data.shape}")
        logging.info(f"  - Количество каналов: {saved_channels}")
        logging.info(f"  - Длина в семплах: {saved_length}")
        logging.info(f"  - Длительность: {saved_duration:.2f} секунд")
        logging.info(f"  - Частота дискретизации: {saved_samplerate} Гц")
        
        # Проверяем, совпадают ли размеры исходных и сохраненных данных
        if saved_length != adjusted_length:
            logging.warning(f"ВНИМАНИЕ! Размер данных изменился после сохранения: было {adjusted_length}, стало {saved_length}")
        if saved_channels != adjusted_channels:
            logging.warning(f"ВНИМАНИЕ! Количество каналов изменилось после сохранения: было {adjusted_channels}, стало {saved_channels}")
    except Exception as e:
        logging.info(f"Ошибка при проверке сохраненного файла: {e}")
    
    return adjusted_data, samplerate

def convert_mp3_to_wav(mp3_file, target_wav_file, target_samplerate=None):
    """
    Конвертирует MP3 в WAV с использованием ffmpeg.
    Если указана целевая частота дискретизации, то конвертирует в неё.
    """
    try:
        ffmpeg_cmd = ['ffmpeg', '-i', mp3_file, '-acodec', 'pcm_s16le']
        if target_samplerate:
            ffmpeg_cmd.extend(['-ar', str(target_samplerate)])
        ffmpeg_cmd.extend([target_wav_file, '-y'])
        
        # Выполняем команду и перенаправляем вывод
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"Ошибка при конвертации MP3 в WAV: {stderr.decode()}")
            return False
        
        logging.info(f"Файл {mp3_file} успешно конвертирован в {target_wav_file}")
        return True
    except Exception as e:
        logging.error(f"Ошибка при конвертации MP3 в WAV: {e}")
        return False

def mix_audio_files(file1_data, file2_data, output_file, samplerate):
    """
    Микширует два аудиосигнала и сохраняет результат.
    Корректно обрабатывает разные форматы данных и сохраняет максимально возможную длину.
    """
    # Проверка на наличие данных
    if file1_data is None or file2_data is None:
        logging.warning("Не удалось произвести микширование: один из файлов отсутствует")
        return
    
    # Сохраняем исходную длину для логирования
    original_len1 = len(file1_data)
    original_len2 = len(file2_data)
    logging.info(f"Исходные длины файлов: {original_len1} семплов и {original_len2} семплов")
    
    # Проверка формата файлов и приведение к единому формату каналов
    is_stereo1 = len(file1_data.shape) > 1
    is_stereo2 = len(file2_data.shape) > 1
    logging.info(f"Форматы файлов: file1 - {'стерео' if is_stereo1 else 'моно'}, file2 - {'стерео' if is_stereo2 else 'моно'}")
    
    # Создаем копии данных, чтобы не модифицировать оригиналы
    file1_data_copy = file1_data.copy()
    file2_data_copy = file2_data.copy()
    
    # Приведение к одинаковому формату каналов (моно или стерео)
    if is_stereo1 and not is_stereo2:
        # file1 - стерео, file2 - моно
        logging.info("Преобразуем второй файл из моно в стерео")
        file2_data_copy = np.column_stack((file2_data_copy, file2_data_copy))
    elif not is_stereo1 and is_stereo2:
        # file1 - моно, file2 - стерео
        logging.info("Преобразуем первый файл из моно в стерео")
        file1_data_copy = np.column_stack((file1_data_copy, file1_data_copy))
    
    # Определяем итоговый формат (моно или стерео)
    final_is_stereo = len(file1_data_copy.shape) > 1 or len(file2_data_copy.shape) > 1
    logging.info(f"Итоговый формат: {'стерео' if final_is_stereo else 'моно'}")
    
    # Приведение массивов к одинаковой длине
    min_length = min(len(file1_data_copy), len(file2_data_copy))
    logging.info(f"Минимальная длина после преобразования форматов: {min_length} семплов")
    
    # Если есть значительная разница в длине, выводим предупреждение
    max_length = max(len(file1_data_copy), len(file2_data_copy))
    if min_length < max_length * 0.9:  # Если разница больше 10%
        logging.warning(f"Предупреждение: значительная разница в длине файлов ({min_length} vs {max_length})")
    
    # Обрезаем до минимальной длины
    file1_data_copy = file1_data_copy[:min_length]
    file2_data_copy = file2_data_copy[:min_length]
    
    # Микширование аудиосигналов (суммирование)
    mixed_data = file1_data_copy + file2_data_copy
    
    # Нормализация для предотвращения клиппинга
    max_amplitude = np.max(np.abs(mixed_data))
    if max_amplitude > 1.0:
        logging.info(f"Применяем нормализацию (максимальная амплитуда: {max_amplitude})")
        mixed_data = mixed_data / max_amplitude * 0.9
    
    # Сохранение результата
    sf.write(output_file, mixed_data, samplerate)
    logging.info(f"Создан микшированный файл: {output_file}, длина {len(mixed_data)} семплов, частота дискретизации {samplerate} Гц")

def add_delay(audio_data, sample_rate, delay_ms, output_dir=None):
    """
    Добавляет задержку к аудиоданным, вставляя тишину в начало.
    
    Args:
        audio_data: NumPy массив с аудиоданными
        sample_rate: Частота дискретизации в Гц
        delay_ms: Задержка в миллисекундах
        output_dir: Директория для сохранения визуализации (опционально)
        
    Returns:
        NumPy массив с аудиоданными с добавленной задержкой
    """
    if delay_ms <= 0:
        return audio_data
    
    # Используем полную запрошенную задержку
    actual_delay_ms = delay_ms
    
    # Вычисляем количество семплов для задержки
    delay_samples = int(sample_rate * actual_delay_ms / 1000)
    
    # Проверяем формат аудиоданных (моно или стерео)
    is_stereo = len(audio_data.shape) > 1
    
    # Подробное логирование формата данных
    logging.info(f"DEBUG: Формат аудиоданных перед добавлением задержки: {'стерео' if is_stereo else 'моно'}")
    logging.info(f"DEBUG: Форма массива аудиоданных: {audio_data.shape}")
    logging.info(f"DEBUG: Частота дискретизации: {sample_rate} Гц")
    logging.info(f"DEBUG: Запрошенная задержка: {delay_ms} мс")
    logging.info(f"DEBUG: Применяемая задержка: {actual_delay_ms} мс, что соответствует {delay_samples} семплам")
    
    if is_stereo:
        # Для стерео данных создаем тишину для каждого канала
        channels = audio_data.shape[1]
        logging.info(f"DEBUG: Стерео данные с {channels} каналами")
        silence = np.zeros((delay_samples, channels), dtype=audio_data.dtype)
        # Соединяем тишину и оригинальные данные
        delayed_data = np.vstack((silence, audio_data))
    else:
        # Для моно данных создаем тишину и соединяем
        logging.info(f"DEBUG: Моно данные")
        silence = np.zeros(delay_samples, dtype=audio_data.dtype)
        delayed_data = np.concatenate((silence, audio_data))
    
    logging.info(f"Добавлена задержка {actual_delay_ms:.2f} мс ({delay_samples} семплов) к аудиоданным")
    logging.info(f"  - Исходная длина: {len(audio_data)} семплов")
    logging.info(f"  - Новая длина: {len(delayed_data)} семплов")
    logging.info(f"  - Форма данных после добавления задержки: {delayed_data.shape}")
    
    return delayed_data

def verify_delay_by_length(original_data, delayed_data, sample_rate, expected_delay_ms):
    """
    Проверяет фактическую задержку по разнице в длине между оригинальным и задержанным сигналами.
    
    Args:
        original_data: Оригинальный сигнал (numpy array)
        delayed_data: Задержанный сигнал (numpy array)
        sample_rate: Частота дискретизации в Гц
        expected_delay_ms: Ожидаемая задержка в миллисекундах
        
    Returns:
        dict: Результаты проверки задержки
    """
    # Вычисляем разницу в длине в сэмплах
    length_diff_samples = len(delayed_data) - len(original_data)
    
    # Вычисляем разницу в длине в миллисекундах
    length_diff_ms = length_diff_samples * 1000 / sample_rate
    
    # Проверяем соответствие ожидаемой задержке
    delay_difference_ms = length_diff_ms - expected_delay_ms
    is_accurate = abs(delay_difference_ms) < 1  # допуск 1 мс
    
    return {
        "measured_delay_ms": length_diff_ms,
        "expected_delay_ms": expected_delay_ms,
        "delay_difference_ms": delay_difference_ms,
        "length_diff_samples": length_diff_samples,
        "is_accurate": is_accurate
    }

def verify_delay_between_files_by_length(file1, file2, expected_delay_ms):
    """
    Проверяет задержку между двумя аудиофайлами по разнице в их длине.
    
    Args:
        file1: Путь к первому файлу (референсный, короткий)
        file2: Путь к второму файлу (задержанный, длинный)
        expected_delay_ms: Ожидаемая задержка в миллисекундах
        
    Returns:
        dict: Результаты проверки задержки
    """
    import soundfile as sf
    
    try:
        # Получаем информацию о файлах
        info1 = sf.info(file1)
        info2 = sf.info(file2)
        
        # Проверяем соответствие частот дискретизации
        if info1.samplerate != info2.samplerate:
            logging.warning(f"ВНИМАНИЕ: Разные частоты дискретизации: {info1.samplerate} Гц и {info2.samplerate} Гц")
            return {"error": "sample_rate_mismatch"}
        
        # Вычисляем разницу в длине в сэмплах
        length_diff_samples = info2.frames - info1.frames
        
        # Вычисляем разницу в длине в миллисекундах
        length_diff_ms = length_diff_samples * 1000 / info1.samplerate
        
        # Проверяем соответствие ожидаемой задержке
        delay_difference_ms = length_diff_ms - expected_delay_ms
        is_accurate = abs(delay_difference_ms) < 1  # допуск 1 мс
        
        return {
            "measured_delay_ms": length_diff_ms,
            "expected_delay_ms": expected_delay_ms,
            "delay_difference_ms": delay_difference_ms,
            "length_diff_samples": length_diff_samples,
            "file1_frames": info1.frames,
            "file2_frames": info2.frames,
            "is_accurate": is_accurate
        }
    except Exception as e:
        logging.error(f"Ошибка при проверке задержки между файлами: {e}")
        return {"error": str(e)}

def process_directory(directory_path):
    """
    Обрабатывает директорию с аудиофайлами и создает структуру для тестирования.
    
    Структура исходной директории должна содержать:
    - reference.wav
    - reference_by_micro.wav
    - my_voice.wav или my_voice.mp3
    
    Создаваемая структура:
    - clear_reference/
      - volume_01/
        - original_input.wav  # Микшированный сигнал (референс + голос пользователя)
        - reference_volumed.wav   # Референсный сигнал с измененной громкостью
        - reference_volumed_delayed.wav # Тот же сигнал для clear_reference
        - my_voice.wav        # Оригинальный голос пользователя
      - volume_04/
        - original_input.wav
        - reference_volumed.wav
        - reference_volumed_delayed.wav
        - my_voice.wav
      - ...
    - reference_by_micro/
      - delay_0/
        - volume_01/
          - original_input.wav
          - reference_volumed.wav
          - reference_volumed_delayed.wav
          - my_voice.wav
        - volume_04/
          - original_input.wav
          - reference_volumed.wav
          - reference_volumed_delayed.wav
          - my_voice.wav
        - ...
      - delay_50/
        - volume_01/
          - ...
      - ...
    """
    # Проверяем наличие необходимых файлов в директории
    clear_reference_file = os.path.join(directory_path, "reference.wav")
    reference_by_micro_file = os.path.join(directory_path, "reference_by_micro.wav")
    my_voice_file = os.path.join(directory_path, "my_voice.wav")
    my_voice_mp3 = os.path.join(directory_path, "my_voice.mp3")
    
    if not os.path.exists(clear_reference_file):
        logging.warning(f"ВНИМАНИЕ: Файл clear_reference.wav не найден в {directory_path}")
    
    if not os.path.exists(reference_by_micro_file):
        logging.warning(f"ВНИМАНИЕ: Файл reference_by_micro.wav не найден в {directory_path}")
    
    # Проверяем наличие my_voice файлов
    if not os.path.exists(my_voice_file) and not os.path.exists(my_voice_mp3):
        logging.error(f"ОШИБКА: Не найдены файлы my_voice.wav или my_voice.mp3 в {directory_path}")
        return
        
    # Читаем reference файлы
    logging.info(f"\nОбработка директории: {directory_path}")
    
    # Читаем clear_reference.wav если он существует
    if os.path.exists(clear_reference_file):
        try:
            clear_reference_data, clear_reference_sr = sf.read(clear_reference_file)
            clear_duration = len(clear_reference_data) / clear_reference_sr
            clear_channels = 1 if len(clear_reference_data.shape) == 1 else clear_reference_data.shape[1]
            logging.info(f"clear_reference: частота {clear_reference_sr} Гц, длительность {clear_duration:.2f} с, {clear_channels} канал(а/ов)")
        except Exception as e:
            logging.error(f"ОШИБКА при чтении clear_reference.wav: {str(e)}")
            clear_reference_data, clear_reference_sr = None, None
    else:
        clear_reference_data, clear_reference_sr = None, None
    
    # Читаем reference_by_micro.wav если он существует
    if os.path.exists(reference_by_micro_file):
        try:
            reference_by_micro_data, reference_by_micro_sr = sf.read(reference_by_micro_file)
            micro_duration = len(reference_by_micro_data) / reference_by_micro_sr
            micro_channels = 1 if len(reference_by_micro_data.shape) == 1 else reference_by_micro_data.shape[1]
            logging.info(f"reference_by_micro: частота {reference_by_micro_sr} Гц, длительность {micro_duration:.2f} с, {micro_channels} канал(а/ов)")
            
            # Проверяем наличие NaN или бесконечных значений
            if np.any(np.isnan(reference_by_micro_data)) or np.any(np.isinf(reference_by_micro_data)):
                logging.warning("ВНИМАНИЕ: reference_by_micro.wav содержит NaN или бесконечные значения!")
            
            # Проверяем ведущие и завершающие нули
            epsilon = 1e-6  # Порог для определения "тишины"
            leading_zeros = 0
            trailing_zeros = 0
            
            if len(reference_by_micro_data.shape) == 1:  # Моно
                for i in range(len(reference_by_micro_data)):
                    if np.abs(reference_by_micro_data[i]) > epsilon:
                        break
                    leading_zeros += 1
                
                for i in range(len(reference_by_micro_data) - 1, -1, -1):
                    if np.abs(reference_by_micro_data[i]) > epsilon:
                        break
                    trailing_zeros += 1
            else:  # Стерео
                for i in range(len(reference_by_micro_data)):
                    if np.max(np.abs(reference_by_micro_data[i])) > epsilon:
                        break
                    leading_zeros += 1
                
                for i in range(len(reference_by_micro_data) - 1, -1, -1):
                    if np.max(np.abs(reference_by_micro_data[i])) > epsilon:
                        break
                    trailing_zeros += 1
            
            if leading_zeros > 0:
                lead_secs = leading_zeros / reference_by_micro_sr
                logging.warning(f"ВНИМАНИЕ: reference_by_micro.wav имеет {leading_zeros} ведущих нулей ({lead_secs:.3f} с)")
            
            if trailing_zeros > 0:
                trail_secs = trailing_zeros / reference_by_micro_sr
                logging.warning(f"ВНИМАНИЕ: reference_by_micro.wav имеет {trailing_zeros} завершающих нулей ({trail_secs:.3f} с)")
                
        except Exception as e:
            logging.error(f"ОШИБКА при чтении reference_by_micro.wav: {str(e)}")
            reference_by_micro_data, reference_by_micro_sr = None, None
    else:
        reference_by_micro_data, reference_by_micro_sr = None, None
    
    # Проверяем, совпадают ли sample rates у reference файлов
    if (clear_reference_sr is not None and reference_by_micro_sr is not None 
            and clear_reference_sr != reference_by_micro_sr):
        logging.warning(f"ВНИМАНИЕ: Частоты дискретизации reference файлов не совпадают: {clear_reference_sr} и {reference_by_micro_sr}")
        
        # Если частота reference_by_micro отличается, выполним ресемплинг для её исправления
        if reference_by_micro_sr != clear_reference_sr:
            logging.info(f"Выполняется ресемплинг reference_by_micro.wav с {reference_by_micro_sr} Гц до {clear_reference_sr} Гц...")
            
            # Выводим данные о файле до ресемплинга
            original_shape = reference_by_micro_data.shape
            original_length = len(reference_by_micro_data)
            original_duration = original_length / reference_by_micro_sr
            logging.info(f"До ресемплинга: форма={original_shape}, длина={original_length}, длительность={original_duration:.3f} с")
            
            # Вычисляем соотношение частот дискретизации
            ratio = clear_reference_sr / reference_by_micro_sr
            
            # Вычисляем новую длину данных
            new_length = int(np.round(original_length * ratio))
            logging.info(f"Соотношение частот: {ratio}, новая длина: {new_length}")
            
            # Создаем временные оси для оригинальных и новых данных
            original_time = np.arange(original_length) / reference_by_micro_sr
            new_time = np.arange(new_length) / clear_reference_sr
            
            # Выполняем ресемплинг с использованием линейной интерполяции
            if len(original_shape) == 1:  # Моно
                resampled_data = np.interp(new_time, original_time, reference_by_micro_data)
            else:  # Стерео или многоканальный
                resampled_data = np.zeros((new_length, original_shape[1]))
                for channel in range(original_shape[1]):
                    resampled_data[:, channel] = np.interp(new_time, original_time, reference_by_micro_data[:, channel])
            
            # Создаем временный файл для сохранения ресемплированных данных
            temp_file = os.path.join(directory_path, "temp_resampled.wav")
            sf.write(temp_file, resampled_data, clear_reference_sr)
            
            # Выводим данные о ресемплированном файле
            resampled_shape = resampled_data.shape
            resampled_length = len(resampled_data)
            resampled_duration = resampled_length / clear_reference_sr
            logging.info(f"После ресемплинга: форма={resampled_shape}, длина={resampled_length}, длительность={resampled_duration:.3f} с")
            
            # Проверяем, что длительность не изменилась больше, чем на 10%
            duration_change_percent = abs(resampled_duration - original_duration) / original_duration * 100
            if duration_change_percent > 10:
                logging.warning(f"ВНИМАНИЕ: длительность изменилась на {duration_change_percent:.2f}% после ресемплинга!")
            
            # Заменяем оригинальный файл ресемплированным
            os.replace(temp_file, reference_by_micro_file)
            logging.info(f"Файл reference_by_micro.wav успешно ресемплирован до {clear_reference_sr} Гц")
            
            # Обновляем данные и sample rate
            reference_by_micro_data, reference_by_micro_sr = resampled_data, clear_reference_sr
    
    # Подготавливаем данные my_voice
    if os.path.exists(my_voice_file):
        try:
            my_voice_data, my_voice_sr = sf.read(my_voice_file)
            voice_duration = len(my_voice_data) / my_voice_sr
            voice_channels = 1 if len(my_voice_data.shape) == 1 else my_voice_data.shape[1]
            logging.info(f"my_voice: частота {my_voice_sr} Гц, длительность {voice_duration:.2f} с, {voice_channels} канал(а/ов)")
        except Exception as e:
            logging.error(f"ОШИБКА при чтении my_voice.wav: {str(e)}")
            my_voice_data, my_voice_sr = None, None
    elif os.path.exists(my_voice_mp3):
        # Если WAV не найден, но найден MP3, конвертируем его
        logging.info(f"Файл my_voice.wav не найден, но найден my_voice.mp3. Конвертирование...")
        
        try:
            # Создаем временный файл
            temp_wav = os.path.join(directory_path, "temp_voice.wav")
            
            # Используем FFmpeg для конвертации (если установлен)
            import subprocess
            try:
                subprocess.run(["ffmpeg", "-y", "-i", my_voice_mp3, temp_wav], 
                              check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info(f"Успешно сконвертирован my_voice.mp3 в WAV через FFmpeg")
            except (subprocess.SubprocessError, FileNotFoundError):
                # Если FFmpeg не доступен, пробуем pydub
                try:
                    from pydub import AudioSegment
                    sound = AudioSegment.from_mp3(my_voice_mp3)
                    sound.export(temp_wav, format="wav")
                    logging.info(f"Успешно сконвертирован my_voice.mp3 в WAV через pydub")
                except ImportError:
                    logging.error("ОШИБКА: Не удалось конвертировать MP3 в WAV. Установите FFmpeg или pydub.")
                    return
            
            # Проверяем создался ли WAV файл
            if os.path.exists(temp_wav):
                # Перемещаем временный файл
                os.replace(temp_wav, my_voice_file)
                # Читаем конвертированный WAV файл
                my_voice_data, my_voice_sr = sf.read(my_voice_file)
                voice_duration = len(my_voice_data) / my_voice_sr
                voice_channels = 1 if len(my_voice_data.shape) == 1 else my_voice_data.shape[1]
                logging.info(f"Конвертированный my_voice: частота {my_voice_sr} Гц, длительность {voice_duration:.2f} с, {voice_channels} канал(а/ов)")
            else:
                logging.error("ОШИБКА: Не удалось создать WAV файл после конвертации")
                return
                
        except Exception as e:
            logging.error(f"ОШИБКА при конвертации my_voice.mp3: {str(e)}")
            return
    else:
        logging.error("ОШИБКА: Не найдены файлы my_voice.wav или my_voice.mp3")
        return
    
    # Проверяем наличие необходимых данных
    if clear_reference_data is None and reference_by_micro_data is None:
        logging.error("ОШИБКА: Не найдены данные reference файлов")
        return
    
    if my_voice_data is None:
        logging.error("ОШИБКА: Не найдены данные my_voice")
        return
    
    # Проверяем, совпадают ли sample rates у my_voice и reference файлов
    reference_sr = clear_reference_sr if clear_reference_sr is not None else reference_by_micro_sr
    
    if my_voice_sr != reference_sr:
        logging.warning(f"ВНИМАНИЕ: Частота my_voice ({my_voice_sr} Гц) не совпадает с reference ({reference_sr} Гц)")
        logging.info(f"Выполняется ресемплинг my_voice.wav...")
        
        # Выводим данные о файле до ресемплинга
        original_shape = my_voice_data.shape
        original_length = len(my_voice_data)
        original_duration = original_length / my_voice_sr
        logging.info(f"До ресемплинга: форма={original_shape}, длина={original_length}, длительность={original_duration:.3f} с")
        
        # Вычисляем соотношение частот дискретизации
        ratio = reference_sr / my_voice_sr
        
        # Вычисляем новую длину данных
        new_length = int(np.round(original_length * ratio))
        logging.info(f"Соотношение частот: {ratio}, новая длина: {new_length}")
        
        # Создаем временные оси для оригинальных и новых данных
        original_time = np.arange(original_length) / my_voice_sr
        new_time = np.arange(new_length) / reference_sr
        
        # Выполняем ресемплинг с использованием линейной интерполяции
        if len(original_shape) == 1:  # Моно
            resampled_data = np.interp(new_time, original_time, my_voice_data)
        else:  # Стерео или многоканальный
            resampled_data = np.zeros((new_length, original_shape[1]))
            for channel in range(original_shape[1]):
                resampled_data[:, channel] = np.interp(new_time, original_time, my_voice_data[:, channel])
        
        # Создаем временный файл для сохранения ресемплированных данных
        temp_file = os.path.join(directory_path, "temp_my_voice.wav")
        sf.write(temp_file, resampled_data, reference_sr)
        
        # Выводим данные о ресемплированном файле
        resampled_shape = resampled_data.shape
        resampled_length = len(resampled_data)
        resampled_duration = resampled_length / reference_sr
        logging.info(f"После ресемплинга: форма={resampled_shape}, длина={resampled_length}, длительность={resampled_duration:.3f} с")
        
        # Проверяем, что длительность не изменилась больше, чем на 10%
        duration_change_percent = abs(resampled_duration - original_duration) / original_duration * 100
        if duration_change_percent > 10:
            logging.warning(f"ВНИМАНИЕ: длительность изменилась на {duration_change_percent:.2f}% после ресемплинга!")
        
        # Заменяем оригинальный файл ресемплированным
        os.replace(temp_file, my_voice_file)
        logging.info(f"Файл my_voice.wav успешно ресемплирован до {reference_sr} Гц")
        
        # Обновляем данные и sample rate
        my_voice_data, my_voice_sr = resampled_data, reference_sr
    
    # Проверяем соотношение длительностей my_voice и reference_by_micro
    if reference_by_micro_data is not None:
        voice_duration = len(my_voice_data) / my_voice_sr
        micro_duration = len(reference_by_micro_data) / reference_by_micro_sr
        duration_ratio = voice_duration / micro_duration
        
        if duration_ratio < 0.9 or duration_ratio > 1.1:
            logging.warning(f"ВНИМАНИЕ: Значительное различие в длительности между my_voice ({voice_duration:.2f} с) и reference_by_micro ({micro_duration:.2f} с)!")
            logging.info(f"Соотношение: {duration_ratio:.2f}")
    
    # Создаем директории для обоих типов reference файлов
    if clear_reference_data is not None:
        process_clear_reference(clear_reference_data, clear_reference_sr, 
                              my_voice_data, my_voice_sr, 
                              directory_path, VOLUME_LEVELS)
    
    if reference_by_micro_data is not None:
        process_reference_by_micro(reference_by_micro_data, reference_by_micro_sr, 
                                 my_voice_data, my_voice_sr, 
                                 directory_path, DELAY_LEVELS, VOLUME_LEVELS)
    
    logging.info(f"\nОбработка директории {directory_path} завершена.")

def process_clear_reference(clear_reference_data, clear_reference_sr, my_voice_data, my_voice_sr, 
                          output_dir, volume_levels):
    """
    Обрабатывает clear_reference аудиофайл с разными уровнями громкости
    
    Args:
        clear_reference_data: Данные clear_reference.wav
        clear_reference_sr: Частота дискретизации clear_reference.wav
        my_voice_data: Данные my_voice.wav
        my_voice_sr: Частота дискретизации my_voice.wav
        output_dir: Директория, куда сохранять результаты
        volume_levels: Словарь с уровнями громкости
    """
    logging.info(f"\nОбработка clear_reference с {len(volume_levels)} уровнями громкости...")
    
    # Создаем основную директорию для clear_reference
    clear_reference_dir = os.path.join(output_dir, "clear_reference")
    os.makedirs(clear_reference_dir, exist_ok=True)
    
    # Путь к оригинальному reference.wav
    reference_file = os.path.join(output_dir, "reference.wav")
    
    # Обрабатываем каждый уровень громкости
    for vol_name, vol_level in volume_levels.items():
        logging.info(f"  Обработка уровня громкости {vol_name}: {vol_level}")
        
        # Создаем директорию для текущего уровня громкости
        vol_dir = os.path.join(clear_reference_dir, vol_name)
        os.makedirs(vol_dir, exist_ok=True)
        
        # Регулируем громкость
        clear_adjusted_data = clear_reference_data * vol_level
        
        # Проверяем наличие голоса в my_voice_data
        if np.all(np.abs(my_voice_data) < 1e-6):
            logging.warning("    ВНИМАНИЕ: my_voice.wav содержит только тишину или очень тихий звук!")
        
        # Микшируем аудио с регулированной громкостью с my_voice
        mixed_output_file = os.path.join(vol_dir, "original_input.wav")
        mix_audio_files(clear_adjusted_data, my_voice_data, mixed_output_file, clear_reference_sr)
        
        # Сохраняем оригинальный голос пользователя для сравнения
        sf.write(os.path.join(vol_dir, "my_voice.wav"), my_voice_data, my_voice_sr)
        
        # Сохраняем сигнал с измененной громкостью как reference_by_micro_volumed.wav
        sf.write(os.path.join(vol_dir, "reference_by_micro_volumed.wav"), clear_adjusted_data, clear_reference_sr)
        
        # Сохраняем также копию как reference_by_micro_volumed_delayed 
        # (для clear_reference задержка не применяется, поэтому это тот же файл, что и reference_by_micro_volumed.wav)
        sf.write(os.path.join(vol_dir, "reference_by_micro_volumed_delayed.wav"), clear_adjusted_data, clear_reference_sr)
        
        # Копируем оригинальный reference.wav в директорию с уровнем громкости
        if os.path.exists(reference_file):
            reference_copy_path = os.path.join(vol_dir, "reference.wav")
            shutil.copy2(reference_file, reference_copy_path)
            logging.info(f"    Скопирован оригинальный reference.wav в {reference_copy_path}")
        
        logging.info(f"    Сохранено в директорию: {vol_dir}")

def process_reference_by_micro(reference_by_micro_data, reference_by_micro_sr, my_voice_data, my_voice_sr, 
                         output_dir, delay_levels, volume_levels):
    """
    Обрабатывает reference_by_micro аудиофайл с разными задержками и уровнями громкости
    
    Args:
        reference_by_micro_data: Данные reference_by_micro.wav
        reference_by_micro_sr: Частота дискретизации reference_by_micro.wav
        my_voice_data: Данные my_voice.wav
        my_voice_sr: Частота дискретизации my_voice.wav
        output_dir: Директория, куда сохранять результаты
        delay_levels: Словарь с уровнями задержки
        volume_levels: Словарь с уровнями громкости
    """
    logging.info(f"\nОбработка reference_by_micro с {len(delay_levels)} задержками и {len(volume_levels)} уровнями громкости...")
    
    # Логирование формата входных данных
    ref_is_stereo = len(reference_by_micro_data.shape) > 1
    voice_is_stereo = len(my_voice_data.shape) > 1
    
    logging.info(f"DEBUG: Формат reference_by_micro_data: {'стерео' if ref_is_stereo else 'моно'}, форма: {reference_by_micro_data.shape}")
    logging.info(f"DEBUG: Формат my_voice_data: {'стерео' if voice_is_stereo else 'моно'}, форма: {my_voice_data.shape}")
    logging.info(f"DEBUG: Частота дискретизации reference_by_micro_sr: {reference_by_micro_sr} Гц")
    logging.info(f"DEBUG: Частота дискретизации my_voice_sr: {my_voice_sr} Гц")
    
    # Создаем основную директорию для reference_by_micro
    reference_by_micro_dir = os.path.join(output_dir, "reference_by_micro")
    os.makedirs(reference_by_micro_dir, exist_ok=True)
    
    # Путь к оригинальному reference.wav
    reference_file = os.path.join(output_dir, "reference.wav")
    
    # Обрабатываем каждую задержку
    for delay_name, delay_ms in delay_levels.items():
        logging.info(f"\nОбработка задержки {delay_name}: {delay_ms} мс")
        
        # Создаем директорию для текущей задержки
        # Исправляем: Имя директории должно соответствовать фактической задержке
        delay_dir = os.path.join(reference_by_micro_dir, f"delay_{delay_ms}")
        os.makedirs(delay_dir, exist_ok=True)
        
        # Добавляем задержку к reference_by_micro
        actual_delay_ms = delay_ms
        
        # Сохраняем длину оригинального файла до добавления задержки
        original_length = len(reference_by_micro_data)
        
        delayed_reference_data = add_delay(reference_by_micro_data, reference_by_micro_sr, actual_delay_ms, delay_dir)
        
        # Добавляем логирование для сравнения
        logging.info(f"DEBUG: Сравнение данных для визуализации:")
        logging.info(f"DEBUG: Размер reference_by_micro_data: {reference_by_micro_data.shape}")
        logging.info(f"DEBUG: Размер delayed_reference_data: {delayed_reference_data.shape}")
        logging.info(f"DEBUG: Частота дискретизации: {reference_by_micro_sr} Гц")
        logging.info(f"DEBUG: Длительность reference_by_micro_data: {len(reference_by_micro_data)/reference_by_micro_sr:.3f} сек")
        logging.info(f"DEBUG: Длительность delayed_reference_data: {len(delayed_reference_data)/reference_by_micro_sr:.3f} сек")
        
        # Сохраняем длину после добавления задержки
        delayed_length = len(delayed_reference_data)
        
        # Проверяем фактическую задержку по разнице длин
        delay_verification = verify_delay_by_length(
            reference_by_micro_data, 
            delayed_reference_data, 
            reference_by_micro_sr, 
            actual_delay_ms
        )
        
        # Выводим результаты проверки
        logging.info(f"  Проверка задержки по разнице длин:")
        logging.info(f"    Длина оригинала: {original_length} сэмплов ({original_length * 1000 / reference_by_micro_sr:.2f} мс)")
        logging.info(f"    Длина с задержкой: {delayed_length} сэмплов ({delayed_length * 1000 / reference_by_micro_sr:.2f} мс)")
        logging.info(f"    Разница в сэмплах: {delay_verification['length_diff_samples']} сэмплов")
        logging.info(f"    Ожидаемая задержка: {actual_delay_ms} мс ({int(actual_delay_ms * reference_by_micro_sr / 1000)} сэмплов)")
        logging.info(f"    Измеренная задержка: {delay_verification['measured_delay_ms']:.2f} мс")
        logging.info(f"    Точность: {'OK' if delay_verification['is_accurate'] else 'ОШИБКА'}")
        
        # В случае ошибки выводим предупреждение
        if not delay_verification['is_accurate']:
            logging.warning(f"    ВНИМАНИЕ: Фактическая задержка ({delay_verification['measured_delay_ms']:.2f} мс) отличается от ожидаемой ({actual_delay_ms} мс)!")
        
        # Визуализация сигналов (оригинал и с задержкой) - только один раз для каждой задержки
        try:
            # Создаем график
            plt.figure(figsize=(12, 6))
            
            # Определяем временную ось для каждого сигнала (в миллисекундах)
            time_original = np.arange(len(reference_by_micro_data)) * 1000 / reference_by_micro_sr
            time_delayed = np.arange(len(delayed_reference_data)) * 1000 / reference_by_micro_sr
            
            # Если данные стерео, берем только первый канал для визуализации
            if len(reference_by_micro_data.shape) > 1:
                original_plot_data = reference_by_micro_data[:, 0]
            else:
                original_plot_data = reference_by_micro_data
            
            if len(delayed_reference_data.shape) > 1:
                delayed_plot_data = delayed_reference_data[:, 0]
            else:
                delayed_plot_data = delayed_reference_data
            
            # Ограничиваем количество точек для отображения (для производительности)
            max_points = 10000
            if len(original_plot_data) > max_points:
                step = len(original_plot_data) // max_points
                original_plot_data = original_plot_data[::step]
                time_original = time_original[::step]
            
            if len(delayed_plot_data) > max_points:
                step = len(delayed_plot_data) // max_points
                delayed_plot_data = delayed_plot_data[::step]
                time_delayed = time_delayed[::step]
            
            # Рисуем оба сигнала
            plt.plot(time_original, original_plot_data, 'b-', alpha=0.7, label='Оригинальный сигнал')
            plt.plot(time_delayed, delayed_plot_data, 'r-', alpha=0.7, label='Сигнал с задержкой')
            
            # Добавляем вертикальную линию, показывающую ожидаемую задержку
            plt.axvline(x=actual_delay_ms, color='g', linestyle='--', 
                       label=f'Ожидаемая задержка: {actual_delay_ms} мс')
            
            # Настраиваем график
            plt.title(f'Сравнение референсных сигналов с задержкой {actual_delay_ms} мс')
            plt.xlabel('Время (мс)')
            plt.ylabel('Амплитуда')
            plt.legend()
            plt.grid(True)
            
            # Устанавливаем область просмотра минимум 7000 мс
            min_view_window = 7000  # минимум 7000 мс
            # Если задержка большая, показываем больше
            view_window = max(min_view_window, actual_delay_ms * 2 + 1000)
            plt.xlim([0, view_window])
            
            # Добавляем деления по 250 мс на оси X
            plt.xticks(np.arange(0, view_window + 250, 250))
            # Добавляем сетку для лучшей читаемости
            plt.grid(axis='x', which='major', linestyle='-', alpha=0.7)
            
            # Сохраняем график в корневую папку задержки
            delay_visualization_file = os.path.join(delay_dir, f"delay_visualization_{delay_ms}ms.png")
            plt.savefig(delay_visualization_file, dpi=150)
            plt.close()
            
            logging.info(f"  Визуализация сигналов сохранена в файл: {delay_visualization_file}")
        except Exception as e:
            logging.error(f"  ОШИБКА при создании визуализации: {str(e)}")
        
        # Обрабатываем каждый уровень громкости для текущей задержки
        for vol_name, vol_level in volume_levels.items():
            logging.info(f"  Обработка уровня громкости {vol_name}: {vol_level}")
            
            # Создаем директорию для текущего уровня громкости
            vol_dir = os.path.join(delay_dir, vol_name)
            os.makedirs(vol_dir, exist_ok=True)
            
            # Регулируем громкость с задержкой
            micro_adjusted_data = delayed_reference_data * vol_level
            
            # Проверяем наличие голоса в my_voice_data
            if np.all(np.abs(my_voice_data) < 1e-6):
                logging.warning("    ВНИМАНИЕ: my_voice.wav содержит только тишину или очень тихий звук!")
            
            # Логирование перед микшированием
            logging.info(f"DEBUG: Данные для микширования:")
            logging.info(f"DEBUG:   - micro_adjusted_data (форма): {micro_adjusted_data.shape}")
            logging.info(f"DEBUG:   - my_voice_data (форма): {my_voice_data.shape}")
            
            # Определяем имена файлов
            reference_volumed_file = os.path.join(vol_dir, "reference_by_micro_volumed.wav")
            reference_delayed_file = os.path.join(vol_dir, "reference_by_micro_volumed_delayed.wav")
            original_input_file = os.path.join(vol_dir, "original_input.wav")
            my_voice_output_file = os.path.join(vol_dir, "my_voice.wav")
            
            # Микшируем аудио с задержкой и регулированной громкостью с my_voice
            mix_audio_files(micro_adjusted_data, my_voice_data, original_input_file, reference_by_micro_sr)
            
            # Сохраняем оригинальный голос пользователя для сравнения
            sf.write(my_voice_output_file, my_voice_data, my_voice_sr)
            
            # Сохраняем измененный референсный сигнал без задержки как reference_by_micro_volumed.wav
            non_delayed_reference_data = reference_by_micro_data * vol_level
            sf.write(reference_volumed_file, non_delayed_reference_data, reference_by_micro_sr)
            
            # Сохраняем измененный и задержанный референсный сигнал как reference_by_micro_volumed_delayed
            sf.write(reference_delayed_file, micro_adjusted_data, reference_by_micro_sr)
            
            # Копируем оригинальный reference.wav в директорию с уровнем громкости
            if os.path.exists(reference_file):
                reference_copy_path = os.path.join(vol_dir, "reference.wav")
                shutil.copy2(reference_file, reference_copy_path)
                logging.info(f"    Скопирован оригинальный reference.wav в {reference_copy_path}")
            
            logging.info(f"    Сохранено в директорию: {vol_dir}")
            
            # Теперь проверяем задержку между файлами
            
            # 1. Проверка задержки между reference_by_micro_volumed.wav и reference_by_micro_volumed_delayed.wav
            logging.info(f"    Проверка задержки между файлами (по разнице длин):")
            delay_check = verify_delay_between_files_by_length(
                reference_volumed_file,
                reference_delayed_file,
                actual_delay_ms
            )
            
            if "error" not in delay_check:
                logging.info(f"      Файл reference_by_micro_volumed.wav: {delay_check['file1_frames']} сэмплов ({delay_check['file1_frames'] * 1000 / reference_by_micro_sr:.2f} мс)")
                logging.info(f"      Файл reference_by_micro_volumed_delayed.wav: {delay_check['file2_frames']} сэмплов ({delay_check['file2_frames'] * 1000 / reference_by_micro_sr:.2f} мс)")
                logging.info(f"      Разница в сэмплах: {delay_check['length_diff_samples']} сэмплов")
                logging.info(f"      Ожидаемая задержка: {actual_delay_ms} мс")
                logging.info(f"      Измеренная задержка: {delay_check['measured_delay_ms']:.2f} мс")
                logging.info(f"      Точность: {'OK' if delay_check['is_accurate'] else 'ОШИБКА'}")
                
                if not delay_check['is_accurate']:
                    logging.warning(f"      ВНИМАНИЕ: Задержка между файлами ({delay_check['measured_delay_ms']:.2f} мс) отличается от ожидаемой ({actual_delay_ms} мс)!")
            else:
                logging.warning(f"      Ошибка при проверке задержки: {delay_check['error']}")
            
            logging.info(f"    Создано {len(os.listdir(vol_dir))} файлов в {vol_dir}")

def main(directory_path=None):
    """
    Основная функция для обработки директории с аудиофайлами.
    Параметр directory_path позволяет указать конкретную директорию для обработки.
    Если параметр не указан, обрабатывается текущая директория скрипта.
    """
    if directory_path is None:
        # Если директория не указана, используем директорию скрипта
        directory_path = os.path.dirname(os.path.abspath(__file__))
    
    logging.info(f"Обработка директории: {directory_path}")
    process_directory(directory_path)

if __name__ == "__main__":
    import sys
    
    # Проверяем, передан ли путь к директории как аргумент командной строки
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        main(directory_path)
    else:
        # Если аргумент не передан, обрабатываем текущую директорию
        main() 