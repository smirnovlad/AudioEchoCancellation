import os
import shutil
import numpy as np
import soundfile as sf
import subprocess
import tempfile

"""
Скрипт для создания тестовых данных с различными уровнями громкости.

Скрипт ожидает следующую структуру директорий (создается скриптом prepare_tests.py):
directory_path/
├── clear_reference/           # Директория для чистого референсного сигнала
└── reference_by_micro/        # Директория для референса через микрофон

В каждой поддиректории скрипт создает директории для разных уровней громкости:
clear_reference/
├── reference_01/          # 10% громкости
├── reference_04/          # 40% громкости
├── reference_07/          # 70% громкости
├── reference_10/          # 100% громкости (исходная)
└── reference_13/          # 130% громкости

В каждой поддиректории создаются следующие файлы:
1. reference_new.wav - референсный сигнал с измененной громкостью
2. original_input.wav - микширование голоса пользователя (my_voice.wav) 
   и соответствующего референсного сигнала с примененным коэффициентом громкости

ВАЖНО: Скрипт ожидает наличия следующих файлов в обрабатываемой директории:
- reference.wav - чистый референсный сигнал
- reference_by_micro.wav - запись референса через микрофон
- my_voice.wav или my_voice.mp3 - запись голоса пользователя

Процесс работы:
1. Скрипт проверяет наличие поддиректорий clear_reference и reference_by_micro
2. Загружает оба референсных файла
3. Для каждого уровня громкости создает директории reference_XX в обеих поддиректориях
4. В каждой директории создает файлы reference_new.wav и original_input.wav
"""

# Коэффициенты изменения амплитуды
VOLUME_LEVELS = {
    "01": 0.1,  # 10% от исходной громкости
    "04": 0.4,  # 40% от исходной громкости
    "07": 0.7,  # 70% от исходной громкости
    "10": 1.0,  # 100% (исходная громкость)
    "13": 1.3,  # 130% от исходной громкости
}

def create_directory_if_not_exists(directory_path):
    """Создает директорию, если она не существует."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Создана директория: {directory_path}")
    else:
        print(f"Директория уже существует: {directory_path}")

def adjust_volume(input_file, output_file, volume_factor):
    """Изменяет громкость аудиофайла на заданный коэффициент."""
    # Чтение аудиофайла
    data, samplerate = sf.read(input_file)
    
    # Выводим информацию о входном файле
    is_stereo = len(data.shape) > 1
    channels = data.shape[1] if is_stereo else 1
    length_samples = len(data)
    duration = length_samples / samplerate
    print(f"Входной файл {input_file}:")
    print(f"  - Размер данных: {data.shape}")
    print(f"  - Количество каналов: {channels}")
    print(f"  - Длина в семплах: {length_samples}")
    print(f"  - Длительность: {duration:.2f} секунд")
    print(f"  - Частота дискретизации: {samplerate} Гц")
    
    # Изменение амплитуды (громкости)
    adjusted_data = data * volume_factor
    
    # Проверяем, не изменился ли размер данных после умножения
    adjusted_is_stereo = len(adjusted_data.shape) > 1
    adjusted_channels = adjusted_data.shape[1] if adjusted_is_stereo else 1
    adjusted_length = len(adjusted_data)
    print(f"После изменения громкости:")
    print(f"  - Размер данных: {adjusted_data.shape}")
    print(f"  - Количество каналов: {adjusted_channels}")
    print(f"  - Длина в семплах: {adjusted_length}")
    
    # Сохранение измененного аудиофайла с явным указанием параметров
    try:
        # Получаем информацию о формате исходного файла
        info = sf.info(input_file)
        print(f"Информация о формате исходного файла:")
        print(f"  - Формат: {info.format}")
        print(f"  - Subtype: {info.subtype}")
        
        # Сохраняем с теми же параметрами
        sf.write(output_file, adjusted_data, samplerate, format=info.format, subtype=info.subtype)
        print(f"Создан файл с громкостью {volume_factor}: {output_file} (формат: {info.format}, subtype: {info.subtype})")
    except Exception as e:
        print(f"Не удалось определить формат исходного файла: {e}")
        print("Сохраняем файл с параметрами по умолчанию")
        sf.write(output_file, adjusted_data, samplerate)
        print(f"Создан файл с громкостью {volume_factor}: {output_file}")
    
    # Проверяем размер сохраненного файла
    try:
        saved_data, saved_samplerate = sf.read(output_file)
        saved_is_stereo = len(saved_data.shape) > 1
        saved_channels = saved_data.shape[1] if saved_is_stereo else 1
        saved_length = len(saved_data)
        saved_duration = saved_length / saved_samplerate
        print(f"Сохраненный файл {output_file}:")
        print(f"  - Размер данных: {saved_data.shape}")
        print(f"  - Количество каналов: {saved_channels}")
        print(f"  - Длина в семплах: {saved_length}")
        print(f"  - Длительность: {saved_duration:.2f} секунд")
        print(f"  - Частота дискретизации: {saved_samplerate} Гц")
        
        # Проверяем, совпадают ли размеры исходных и сохраненных данных
        if saved_length != adjusted_length:
            print(f"ВНИМАНИЕ! Размер данных изменился после сохранения: было {adjusted_length}, стало {saved_length}")
        if saved_channels != adjusted_channels:
            print(f"ВНИМАНИЕ! Количество каналов изменилось после сохранения: было {adjusted_channels}, стало {saved_channels}")
    except Exception as e:
        print(f"Ошибка при проверке сохраненного файла: {e}")
    
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
            print(f"Ошибка при конвертации MP3 в WAV: {stderr.decode()}")
            return False
        
        print(f"Файл {mp3_file} успешно конвертирован в {target_wav_file}")
        return True
    except Exception as e:
        print(f"Ошибка при конвертации MP3 в WAV: {e}")
        return False

def mix_audio_files(file1_data, file2_data, output_file, samplerate):
    """
    Микширует два аудиосигнала и сохраняет результат.
    Корректно обрабатывает разные форматы данных и сохраняет максимально возможную длину.
    """
    # Проверка на наличие данных
    if file1_data is None or file2_data is None:
        print("Не удалось произвести микширование: один из файлов отсутствует")
        return
    
    # Сохраняем исходную длину для логирования
    original_len1 = len(file1_data)
    original_len2 = len(file2_data)
    print(f"Исходные длины файлов: {original_len1} семплов и {original_len2} семплов")
    
    # Проверка формата файлов и приведение к единому формату каналов
    is_stereo1 = len(file1_data.shape) > 1
    is_stereo2 = len(file2_data.shape) > 1
    print(f"Форматы файлов: file1 - {'стерео' if is_stereo1 else 'моно'}, file2 - {'стерео' if is_stereo2 else 'моно'}")
    
    # Создаем копии данных, чтобы не модифицировать оригиналы
    file1_data_copy = file1_data.copy()
    file2_data_copy = file2_data.copy()
    
    # Приведение к одинаковому формату каналов (моно или стерео)
    if is_stereo1 and not is_stereo2:
        # file1 - стерео, file2 - моно
        print("Преобразуем второй файл из моно в стерео")
        file2_data_copy = np.column_stack((file2_data_copy, file2_data_copy))
    elif not is_stereo1 and is_stereo2:
        # file1 - моно, file2 - стерео
        print("Преобразуем первый файл из моно в стерео")
        file1_data_copy = np.column_stack((file1_data_copy, file1_data_copy))
    
    # Определяем итоговый формат (моно или стерео)
    final_is_stereo = len(file1_data_copy.shape) > 1 or len(file2_data_copy.shape) > 1
    print(f"Итоговый формат: {'стерео' if final_is_stereo else 'моно'}")
    
    # Приведение массивов к одинаковой длине
    min_length = min(len(file1_data_copy), len(file2_data_copy))
    print(f"Минимальная длина после преобразования форматов: {min_length} семплов")
    
    # Если есть значительная разница в длине, выводим предупреждение
    max_length = max(len(file1_data_copy), len(file2_data_copy))
    if min_length < max_length * 0.9:  # Если разница больше 10%
        print(f"Предупреждение: значительная разница в длине файлов ({min_length} vs {max_length})")
    
    # Обрезаем до минимальной длины
    file1_data_copy = file1_data_copy[:min_length]
    file2_data_copy = file2_data_copy[:min_length]
    
    # Микширование аудиосигналов (суммирование)
    mixed_data = file1_data_copy + file2_data_copy
    
    # Нормализация для предотвращения клиппинга
    max_amplitude = np.max(np.abs(mixed_data))
    if max_amplitude > 1.0:
        print(f"Применяем нормализацию (максимальная амплитуда: {max_amplitude})")
        mixed_data = mixed_data / max_amplitude * 0.9
    
    # Сохранение результата
    sf.write(output_file, mixed_data, samplerate)
    print(f"Создан микшированный файл: {output_file}, длина {len(mixed_data)} семплов, частота дискретизации {samplerate} Гц")

def process_directory(directory_path):
    """
    Обрабатывает директорию, создавая структуру поддиректорий для двух типов
    референсных файлов с разными уровнями громкости.
    
    Структура директорий:
    directory_path/
    ├── clear_reference/           # Директория для чистого референсного сигнала
    │   ├── reference_01/          # 10% громкости
    │   ├── reference_04/          # 40% громкости
    │   ...
    │
    └── reference_by_micro/        # Директория для референса через микрофон
        ├── reference_01/
        ├── reference_04/
        ...
    """
    # Пути к референсным файлам
    clear_reference_file = os.path.join(directory_path, "reference.wav")
    reference_by_micro_file = os.path.join(directory_path, "reference_by_micro.wav")
    my_voice_file = os.path.join(directory_path, "my_voice.wav")
    my_voice_mp3 = os.path.join(directory_path, "my_voice.mp3")
    
    # Проверяем наличие основных файлов
    if not os.path.exists(clear_reference_file):
        print(f"Файл {clear_reference_file} не найден!")
        return
    
    if not os.path.exists(reference_by_micro_file):
        print(f"Файл {reference_by_micro_file} не найден!")
        return
    
    # Получаем параметры референсных файлов
    try:
        clear_ref_data, clear_ref_samplerate = sf.read(clear_reference_file)
        clear_ref_duration = len(clear_ref_data) / clear_ref_samplerate
        clear_ref_format = 'стерео' if len(clear_ref_data.shape) > 1 else 'моно'
        print(f"Чистый референсный файл: {clear_reference_file}")
        print(f"  - Частота дискретизации: {clear_ref_samplerate} Гц")
        print(f"  - Формат: {clear_ref_format}")
        print(f"  - Длительность: {clear_ref_duration:.2f} секунд ({len(clear_ref_data)} семплов)")
        if len(clear_ref_data.shape) > 1:
            print(f"  - Количество каналов: {clear_ref_data.shape[1]}")
    except Exception as e:
        print(f"Ошибка при чтении файла {clear_reference_file}: {e}")
        return
    
    try:
        ref_by_micro_data, ref_by_micro_samplerate = sf.read(reference_by_micro_file)
        ref_by_micro_duration = len(ref_by_micro_data) / ref_by_micro_samplerate
        ref_by_micro_format = 'стерео' if len(ref_by_micro_data.shape) > 1 else 'моно'
        print(f"Референс через микрофон: {reference_by_micro_file}")
        print(f"  - Частота дискретизации: {ref_by_micro_samplerate} Гц")
        print(f"  - Формат: {ref_by_micro_format}")
        print(f"  - Длительность: {ref_by_micro_duration:.2f} секунд ({len(ref_by_micro_data)} семплов)")
        if len(ref_by_micro_data.shape) > 1:
            print(f"  - Количество каналов: {ref_by_micro_data.shape[1]}")
            
        # Проверяем наличие некорректных данных
        if np.isnan(ref_by_micro_data).any():
            print("  - ПРЕДУПРЕЖДЕНИЕ: файл содержит NaN значения!")
        if np.isinf(ref_by_micro_data).any():
            print("  - ПРЕДУПРЕЖДЕНИЕ: файл содержит бесконечные значения!")
            
        # Проверяем, есть ли нулевые участки в начале или конце
        if len(ref_by_micro_data) > 0:
            start_zeros = np.sum(np.abs(ref_by_micro_data[:min(1000, len(ref_by_micro_data))]) < 1e-6)
            end_zeros = np.sum(np.abs(ref_by_micro_data[-min(1000, len(ref_by_micro_data)):]) < 1e-6)
            if start_zeros > 100:
                print(f"  - ПРЕДУПРЕЖДЕНИЕ: обнаружено {start_zeros} нулевых/почти нулевых значений в начале файла")
            if end_zeros > 100:
                print(f"  - ПРЕДУПРЕЖДЕНИЕ: обнаружено {end_zeros} нулевых/почти нулевых значений в конце файла")
    except Exception as e:
        print(f"Ошибка при чтении файла {reference_by_micro_file}: {e}")
        return
    
    # Проверяем соответствие частоты дискретизации
    if ref_by_micro_samplerate != clear_ref_samplerate:
        print(f"ПРЕДУПРЕЖДЕНИЕ: частота дискретизации reference_by_micro ({ref_by_micro_samplerate} Гц) отличается от reference ({clear_ref_samplerate} Гц)")
        print("Для корректной обработки они должны иметь одинаковую частоту дискретизации.")
        print("Выполняется ресэмплинг reference_by_micro.wav с правильной частотой дискретизации...")
        
        # Пересохраняем reference_by_micro с нужной частотой дискретизации
        temp_wav_file = os.path.join(directory_path, "reference_by_micro_temp.wav")
        try:
            # Вывод диагностической информации перед пересохранением
            is_stereo = len(ref_by_micro_data.shape) > 1
            channels = ref_by_micro_data.shape[1] if is_stereo else 1
            length_samples = len(ref_by_micro_data)
            duration = length_samples / ref_by_micro_samplerate
            print(f"Данные файла reference_by_micro.wav перед ресэмплингом:")
            print(f"  - Размер данных: {ref_by_micro_data.shape}")
            print(f"  - Количество каналов: {channels}")
            print(f"  - Длина в семплах: {length_samples}")
            print(f"  - Длительность: {duration:.2f} секунд")
            print(f"  - Частота дискретизации: {ref_by_micro_samplerate} Гц")
            
            # Выполняем ресэмплинг с помощью встроенных средств numpy
            # Сначала определяем соотношение частот
            ratio = clear_ref_samplerate / ref_by_micro_samplerate
            
            # Вычисляем новую длину массива данных
            new_length = int(len(ref_by_micro_data) * ratio)
            print(f"Соотношение частот дискретизации: {ratio}")
            print(f"Новая длина массива после ресэмплинга: {new_length} семплов")
            
            # Создаем новый массив данных с правильной длиной
            if is_stereo:
                # Для стерео файла обрабатываем каждый канал отдельно
                resampled_data = np.zeros((new_length, channels))
                for c in range(channels):
                    # Создаем временные индексы для исходного массива
                    orig_indices = np.arange(len(ref_by_micro_data))
                    # Создаем новые индексы с учетом соотношения частот
                    new_indices = np.linspace(0, len(ref_by_micro_data) - 1, new_length)
                    # Интерполируем данные
                    resampled_data[:, c] = np.interp(new_indices, orig_indices, ref_by_micro_data[:, c])
            else:
                # Для моно файла обрабатываем один канал
                orig_indices = np.arange(len(ref_by_micro_data))
                new_indices = np.linspace(0, len(ref_by_micro_data) - 1, new_length)
                resampled_data = np.interp(new_indices, orig_indices, ref_by_micro_data)
            
            # Сохраняем ресэмплированные данные
            sf.write(temp_wav_file, resampled_data, clear_ref_samplerate)
            
            # Загружаем пересохраненный файл
            temp_data, temp_samplerate = sf.read(temp_wav_file)
            
            # Вывод диагностической информации после ресэмплинга
            is_stereo = len(temp_data.shape) > 1
            channels = temp_data.shape[1] if is_stereo else 1
            length_samples = len(temp_data)
            duration = length_samples / temp_samplerate
            print(f"Данные после ресэмплинга:")
            print(f"  - Размер данных: {temp_data.shape}")
            print(f"  - Количество каналов: {channels}")
            print(f"  - Длина в семплах: {length_samples}")
            print(f"  - Длительность: {duration:.2f} секунд")
            print(f"  - Частота дискретизации: {temp_samplerate} Гц")
            
            # Проверяем длительность
            original_duration = len(ref_by_micro_data) / ref_by_micro_samplerate
            new_duration = len(temp_data) / temp_samplerate
            duration_ratio = new_duration / original_duration
            
            if duration_ratio > 1.1 or duration_ratio < 0.9:
                print(f"ВНИМАНИЕ! Длительность файла значительно изменилась после ресэмплинга!")
                print(f"Было: {original_duration:.2f} сек, стало: {new_duration:.2f} сек (соотношение: {duration_ratio:.2f})")
            else:
                print(f"Длительность сохранена: было {original_duration:.2f} сек, стало {new_duration:.2f} сек")
            
            # Сохраняем обратно в исходный файл
            sf.write(reference_by_micro_file, temp_data, temp_samplerate)
            print(f"Файл reference_by_micro.wav пересохранен с частотой {temp_samplerate} Гц")
            
            # Обновляем данные
            ref_by_micro_data = temp_data
            ref_by_micro_samplerate = temp_samplerate
            
            # Обновляем информацию о длительности
            ref_by_micro_duration = len(ref_by_micro_data) / ref_by_micro_samplerate
            print(f"  - Новая длительность: {ref_by_micro_duration:.2f} секунд ({len(ref_by_micro_data)} семплов)")
            
            # Удаляем временный файл
            if os.path.exists(temp_wav_file):
                os.remove(temp_wav_file)
                
        except Exception as e:
            print(f"Ошибка при пересохранении reference_by_micro.wav: {e}")
            if os.path.exists(temp_wav_file):
                os.remove(temp_wav_file)
    
    # Проверка на наличие MP3 вместо WAV и конвертация с правильной частотой
    if not os.path.exists(my_voice_file) and os.path.exists(my_voice_mp3):
        print("Найден файл my_voice.mp3 вместо my_voice.wav. Конвертация с нужной частотой дискретизации...")
        temp_wav_file = os.path.join(directory_path, "my_voice_temp.wav")
        
        if convert_mp3_to_wav(my_voice_mp3, temp_wav_file, clear_ref_samplerate):
            # Проверяем сконвертированный файл
            try:
                temp_data, temp_samplerate = sf.read(temp_wav_file)
                print(f"Сконвертированный файл: частота дискретизации {temp_samplerate} Гц")
                
                # Сохраняем как my_voice.wav
                sf.write(my_voice_file, temp_data, temp_samplerate)
                os.remove(temp_wav_file)
                print(f"Файл сохранен как {my_voice_file}")
            except Exception as e:
                print(f"Ошибка при проверке сконвертированного файла: {e}")
                my_voice_file = None
        else:
            my_voice_file = None
    
    # Загружаем файл с голосом, если он существует
    my_voice_data = None
    my_voice_samplerate = None
    if os.path.exists(my_voice_file):
        try:
            my_voice_data, my_voice_samplerate = sf.read(my_voice_file)
            my_voice_duration = len(my_voice_data) / my_voice_samplerate
            my_voice_format = 'стерео' if len(my_voice_data.shape) > 1 else 'моно'
            print(f"Голос пользователя: {my_voice_file}")
            print(f"  - Частота дискретизации: {my_voice_samplerate} Гц")
            print(f"  - Формат: {my_voice_format}")
            print(f"  - Длительность: {my_voice_duration:.2f} секунд ({len(my_voice_data)} семплов)")
            if len(my_voice_data.shape) > 1:
                print(f"  - Количество каналов: {my_voice_data.shape[1]}")
            
            # Проверяем соответствие частоты дискретизации
            if my_voice_samplerate != clear_ref_samplerate:
                print(f"ПРЕДУПРЕЖДЕНИЕ: частота дискретизации my_voice ({my_voice_samplerate} Гц) отличается от reference ({clear_ref_samplerate} Гц)")
                print("Пересохраняем my_voice с нужной частотой дискретизации...")
                
                # Пересохраняем файл с помощью soundfile
                temp_wav_file = os.path.join(directory_path, "my_voice_temp.wav")
                try:
                    # Пересохраняем с помощью soundfile
                    sf.write(temp_wav_file, my_voice_data, clear_ref_samplerate)
                    # Загружаем пересохраненный файл
                    my_voice_data, my_voice_samplerate = sf.read(temp_wav_file)
                    # Сохраняем обратно
                    sf.write(my_voice_file, my_voice_data, my_voice_samplerate)
                    os.remove(temp_wav_file)
                    print(f"Файл my_voice.wav пересохранен с частотой {my_voice_samplerate} Гц")
                    
                    # Обновляем информацию о длительности
                    my_voice_duration = len(my_voice_data) / my_voice_samplerate
                    print(f"  - Новая длительность: {my_voice_duration:.2f} секунд ({len(my_voice_data)} семплов)")
                except Exception as e:
                    print(f"Ошибка при пересохранении my_voice.wav: {e}")
                
        except Exception as e:
            print(f"Ошибка при чтении файла с голосом {my_voice_file}: {e}")
            my_voice_data = None
    else:
        print("Файл с голосом не найден. Будут созданы только файлы с измененной громкостью.")
        
    # Проверяем соотношение длительностей
    if my_voice_data is not None and ref_by_micro_data is not None:
        len_ratio = len(my_voice_data) / len(ref_by_micro_data)
        if len_ratio < 0.9 or len_ratio > 1.1:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Значительная разница в длительности my_voice ({len(my_voice_data)} семплов) и reference_by_micro ({len(ref_by_micro_data)} семплов)")
            print(f"Соотношение длин: {len_ratio:.2f}")
    
    # Создаем базовые директории для двух типов референсных файлов
    clear_reference_dir = os.path.join(directory_path, "clear_reference")
    reference_by_micro_dir = os.path.join(directory_path, "reference_by_micro")
    
    create_directory_if_not_exists(clear_reference_dir)
    create_directory_if_not_exists(reference_by_micro_dir)
    
    # Для каждого уровня громкости создаем поддиректории в обеих базовых директориях
    for suffix, volume_factor in VOLUME_LEVELS.items():
        print(f"\nОбработка уровня громкости: {volume_factor} (суффикс {suffix})")
        
        # 1. Создаем поддиректорию для clear_reference
        volume_dir_name = f"reference_{suffix}"
        clear_volume_dir = os.path.join(clear_reference_dir, volume_dir_name)
        create_directory_if_not_exists(clear_volume_dir)
        
        # Создаем reference_new.wav с измененной громкостью для clear_reference
        clear_output_file = os.path.join(clear_volume_dir, "reference_new.wav")
        clear_adjusted_data, clear_samplerate = adjust_volume(clear_reference_file, clear_output_file, volume_factor)
        
        # Если есть файл с голосом, создаем original_input.wav для clear_reference
        if my_voice_data is not None and my_voice_samplerate is not None:
            clear_mixed_file = os.path.join(clear_volume_dir, "original_input.wav")
            print(f"Микширование для clear_reference с уровнем громкости {volume_factor}:")
            mix_audio_files(clear_adjusted_data, my_voice_data, clear_mixed_file, clear_samplerate)
            print(f"Создан original_input для clear_reference: {clear_mixed_file}")
        
        # 2. Создаем поддиректорию для reference_by_micro
        micro_volume_dir = os.path.join(reference_by_micro_dir, volume_dir_name)
        create_directory_if_not_exists(micro_volume_dir)
        
        # Создаем reference_new.wav с измененной громкостью для reference_by_micro
        micro_output_file = os.path.join(micro_volume_dir, "reference_new.wav")
        micro_adjusted_data, micro_samplerate = adjust_volume(reference_by_micro_file, micro_output_file, volume_factor)
        
        # Если есть файл с голосом, создаем original_input.wav для reference_by_micro
        if my_voice_data is not None and my_voice_samplerate is not None:
            micro_mixed_file = os.path.join(micro_volume_dir, "original_input.wav")
            print(f"Микширование для reference_by_micro с уровнем громкости {volume_factor}:")
            
            # Специальная проверка для reference_by_micro
            print("Проверка данных перед микшированием для reference_by_micro:")
            print(f"  - Размер массива micro_adjusted_data: {len(micro_adjusted_data)} семплов")
            print(f"  - Размер массива my_voice_data: {len(my_voice_data)} семплов")
            
            # Если длины массивов сильно отличаются, используем специальный метод микширования
            if len(micro_adjusted_data) < len(my_voice_data) * 0.9 or len(micro_adjusted_data) > len(my_voice_data) * 1.1:
                print("Значительная разница в длине файлов. Используем специальный метод микширования.")
                
                # Копируем данные для обработки
                ref_data_copy = micro_adjusted_data.copy()
                voice_data_copy = my_voice_data.copy()
                
                # Проверяем форматы (моно/стерео)
                ref_is_stereo = len(ref_data_copy.shape) > 1
                voice_is_stereo = len(voice_data_copy.shape) > 1
                
                # Приводим к одинаковому формату
                if ref_is_stereo and not voice_is_stereo:
                    voice_data_copy = np.column_stack((voice_data_copy, voice_data_copy))
                elif not ref_is_stereo and voice_is_stereo:
                    ref_data_copy = np.column_stack((ref_data_copy, ref_data_copy))
                
                # Используем более длинный массив и заполняем короткий тишиной
                target_length = max(len(ref_data_copy), len(voice_data_copy))
                print(f"Целевая длина: {target_length} семплов")
                
                # Создаем массивы нужной длины
                if len(ref_data_copy.shape) > 1:  # Стерео
                    channels = ref_data_copy.shape[1]
                    extended_ref = np.zeros((target_length, channels))
                    extended_voice = np.zeros((target_length, channels))
                    
                    # Копируем данные в новые массивы
                    extended_ref[:len(ref_data_copy)] = ref_data_copy
                    extended_voice[:len(voice_data_copy)] = voice_data_copy
                else:  # Моно
                    extended_ref = np.zeros(target_length)
                    extended_voice = np.zeros(target_length)
                    
                    # Копируем данные в новые массивы
                    extended_ref[:len(ref_data_copy)] = ref_data_copy
                    extended_voice[:len(voice_data_copy)] = voice_data_copy
                
                # Микшируем
                mixed_data = extended_ref + extended_voice
                
                # Нормализация для предотвращения клиппинга
                max_amplitude = np.max(np.abs(mixed_data))
                if max_amplitude > 1.0:
                    print(f"Применяем нормализацию (максимальная амплитуда: {max_amplitude})")
                    mixed_data = mixed_data / max_amplitude * 0.9
                
                # Сохраняем результат
                sf.write(micro_mixed_file, mixed_data, micro_samplerate)
                print(f"Создан микшированный файл для reference_by_micro: {micro_mixed_file}, длина {len(mixed_data)} семплов")
            else:
                # Используем стандартное микширование
                mix_audio_files(micro_adjusted_data, my_voice_data, micro_mixed_file, micro_samplerate)
                print(f"Создан original_input для reference_by_micro: {micro_mixed_file}")

def main(directory_path=None):
    """
    Основная функция для обработки директории с аудиофайлами.
    Параметр directory_path позволяет указать конкретную директорию для обработки.
    Если параметр не указан, обрабатывается текущая директория скрипта.
    """
    if directory_path is None:
        # Если директория не указана, используем директорию скрипта
        directory_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Обработка директории: {directory_path}")
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