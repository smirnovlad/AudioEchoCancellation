#!/usr/bin/env python3
"""
Скрипт для подготовки тестовых данных для аудио тестов.
Создает копии референсных аудиофайлов с различной громкостью в соответствующих директориях.
Также создает смешанные файлы, имитирующие запись микрофоном одновременно голоса пользователя
и воспроизводимого звука.
"""

import os
import sys
import subprocess
import importlib.util
import shutil
import wave
import numpy as np

# Импортируем общий модуль для создания вариантов с разной громкостью
try:
    # Пытаемся импортировать напрямую
    import create_volume_variants
except ImportError:
    # Если не удалось, пробуем импортировать по пути
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        import create_volume_variants
    except ImportError:
        print("Ошибка: Не удалось импортировать модуль create_volume_variants.py")
        sys.exit(1)

def check_ffmpeg():
    """Проверяет наличие ffmpeg в системе."""
    try:
        process = subprocess.Popen(
            ['ffmpeg', '-version'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("FFmpeg установлен и готов к использованию.")
            return True
        else:
            print("FFmpeg не найден. Конвертация MP3 в WAV будет недоступна.")
            return False
    except Exception:
        print("FFmpeg не найден. Конвертация MP3 в WAV будет недоступна.")
        return False

def clean_test_directories(test_dir):
    """
    Очищает директории с тестами, созданные в предыдущих запусках.
    
    Удаляет следующие директории и их содержимое:
    - clear_reference/
    - reference_by_micro/
    """
    directories_to_clean = [
        os.path.join(test_dir, "clear_reference"),
        os.path.join(test_dir, "reference_by_micro")
    ]
    
    for directory in directories_to_clean:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"Очистка директории: {directory}")
            try:
                shutil.rmtree(directory)
                print(f"Директория {directory} успешно удалена")
            except Exception as e:
                print(f"Ошибка при удалении директории {directory}: {e}")

def convert_wav_channels(input_file, output_file, target_channels):
    """
    Конвертирует WAV файл в формат с заданным числом каналов.
    
    Args:
        input_file: Путь к входному WAV файлу
        output_file: Путь для сохранения результата
        target_channels: Целевое количество каналов (1 или 2)
    
    Returns:
        bool: True если конвертация успешна, False в противном случае
    """
    try:
        with wave.open(input_file, 'rb') as wf:
            # Получаем параметры входного файла
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Читаем все фреймы
            raw_data = wf.readframes(n_frames)
            
            # Если уже нужное количество каналов, просто копируем файл
            if n_channels == target_channels:
                shutil.copy(input_file, output_file)
                return True
                
            # Конвертируем данные в numpy массив
            if sample_width == 2:  # 16-bit
                data = np.frombuffer(raw_data, dtype=np.int16)
            elif sample_width == 3:  # 24-bit
                # Для 24-бит нужно дополнительное преобразование, упрощаем до 16-бит
                data = np.frombuffer(raw_data, dtype=np.int8)
                data = np.reshape(data, (n_frames, n_channels, 3))
                data = (data[:,:,0].astype(np.int32) + 
                        (data[:,:,1].astype(np.int32) << 8) + 
                        (data[:,:,2].astype(np.int32) << 16))
                data = np.reshape(data, -1).astype(np.int16)
            else:
                print(f"Неподдерживаемая битность аудио: {sample_width * 8} бит")
                return False
            
            # Выполняем конвертацию каналов
            if n_channels == 1 and target_channels == 2:
                # Mono to Stereo: дублируем канал
                stereo_data = np.empty(len(data) * 2, dtype=np.int16)
                stereo_data[0::2] = data  # Левый канал
                stereo_data[1::2] = data  # Правый канал
                output_data = stereo_data
            elif n_channels == 2 and target_channels == 1:
                # Stereo to Mono: усредняем каналы
                mono_data = np.mean(data.reshape(-1, 2), axis=1).astype(np.int16)
                output_data = mono_data
            else:
                print(f"Неподдерживаемая конвертация: из {n_channels} в {target_channels} каналов")
                return False
            
            # Записываем результат
            with wave.open(output_file, 'wb') as wf_out:
                wf_out.setnchannels(target_channels)
                wf_out.setsampwidth(2)  # Всегда 16-бит для простоты
                wf_out.setframerate(framerate)
                wf_out.writeframes(output_data.tobytes())
            
            print(f"Успешно сконвертирован файл из {n_channels} в {target_channels} каналов: {output_file}")
            return True
    except Exception as e:
        print(f"Ошибка при конвертации файла {input_file}: {e}")
        return False

def prepare_channel_variant(source_dir, target_dir, channel_type):
    """
    Создает вариант директории с аудио файлами заданного количества каналов.
    
    Args:
        source_dir: Исходная директория с оригинальными файлами
        target_dir: Целевая директория для сохранения результатов
        channel_type: Тип канала ('mono' или 'stereo')
    
    Returns:
        bool: True если подготовка успешна, False в противном случае
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Определяем нужное количество каналов
    target_channels = 1 if channel_type == 'mono' else 2
    
    # Список файлов для конвертации
    files_to_convert = ['reference.wav', 'reference_by_micro.wav', 'my_voice.wav']
    
    success = True
    for file_name in files_to_convert:
        input_file = os.path.join(source_dir, file_name)
        # Проверяем наличие файла (my_voice.wav может отсутствовать)
        if os.path.exists(input_file):
            output_file = os.path.join(target_dir, file_name)
            file_success = convert_wav_channels(input_file, output_file, target_channels)
            success = success and file_success
        elif file_name != 'my_voice.wav':  # my_voice.wav необязателен
            print(f"Критическая ошибка: файл {file_name} не найден в {source_dir}")
            success = False
            
    # Проверяем, есть ли my_voice.mp3, если нет my_voice.wav
    if not os.path.exists(os.path.join(source_dir, 'my_voice.wav')) and os.path.exists(os.path.join(source_dir, 'my_voice.mp3')):
        print(f"Предупреждение: Найден my_voice.mp3, но не найден my_voice.wav в {source_dir}")
        print(f"Для директории {channel_type} необходимо вручную сконвертировать MP3 в WAV с {target_channels} каналами")
    
    return success

def prepare_all_test_directories():
    """
    Подготавливает все директории с тестами.
    
    Создает следующую структуру директорий:
    test_dir/
    ├── mono/                     # Вариант с моно аудио
    │   ├── clear_reference/          # Для чистого референсного сигнала
    │   │   ├── volume_01/            # 10% громкости
    │   │   ├── volume_04/            # 40% громкости
    │   │   └── ...
    │   └── reference_by_micro/       # Для референса через микрофон
    │       ├── delay_0/              # Задержка 0 мс (без задержки)
    │       │   ├── volume_01/        # 10% громкости
    │       │   ├── volume_04/        # 40% громкости
    │       │   └── ...
    │       ├── delay_50/             # Задержка 50 мс
    │       │   ├── volume_01/
    │       │   └── ...
    │       └── ...
    │
    └── stereo/                   # Вариант со стерео аудио
        ├── clear_reference/          # Аналогично mono
        └── reference_by_micro/
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Проверка наличия ffmpeg
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("Предупреждение: ffmpeg не установлен. Для конвертации MP3 в WAV требуется ffmpeg.")
        print("Установите ffmpeg или сконвертируйте файлы вручную.")
    
    # Список директорий с тестами
    test_dirs = [
        os.path.join(base_dir, "agent_speech"),
        os.path.join(base_dir, "agent_user_speech"),
        os.path.join(base_dir, "agent_speech_30_sec"),
    ]
    
    # Директория music обрабатывается по-старому (без разделения на mono/stereo)
    music_dir = os.path.join(base_dir, "music")
    if os.path.exists(music_dir):
        process_directory(music_dir)
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\nПодготовка директории: {test_dir}")
            
            # Проверяем наличие файла reference.wav
            reference_file = os.path.join(test_dir, "reference.wav")
            if not os.path.exists(reference_file):
                print(f"Предупреждение: Файл {reference_file} не найден, пропускаем директорию")
                continue
                
            # Проверяем наличие файла reference_by_micro.wav
            reference_by_micro_file = os.path.join(test_dir, "reference_by_micro.wav")
            if not os.path.exists(reference_by_micro_file):
                print(f"Предупреждение: Файл {reference_by_micro_file} не найден, пропускаем директорию")
                continue
            
            # Создаем поддиректории mono и stereo
            mono_dir = os.path.join(test_dir, "mono")
            stereo_dir = os.path.join(test_dir, "stereo")
            
            if not os.path.exists(mono_dir):
                os.makedirs(mono_dir)
            if not os.path.exists(stereo_dir):
                os.makedirs(stereo_dir)
            
            # Готовим mono и stereo варианты базовых файлов
            print(f"Создание mono варианта для {test_dir}")
            mono_success = prepare_channel_variant(test_dir, mono_dir, 'mono')
            
            print(f"Создание stereo варианта для {test_dir}")
            stereo_success = prepare_channel_variant(test_dir, stereo_dir, 'stereo')
            
            # Обрабатываем mono директорию
            if mono_success:
                process_directory(mono_dir)
            
            # Обрабатываем stereo директорию
            if stereo_success:
                process_directory(stereo_dir)
        else:
            print(f"Директория {test_dir} не найдена, пропускаем")

def process_directory(test_dir):
    """
    Обрабатывает директорию с тестами, создавая варианты с различной громкостью.
    
    Args:
        test_dir: Путь к директории с тестами
    """
    print(f"Обработка директории {test_dir}")
    
    # Проверяем наличие файла reference.wav
    reference_file = os.path.join(test_dir, "reference.wav")
    if not os.path.exists(reference_file):
        print(f"Предупреждение: Файл {reference_file} не найден, пропускаем директорию")
        return
            
    # Проверяем наличие файла reference_by_micro.wav
    reference_by_micro_file = os.path.join(test_dir, "reference_by_micro.wav")
    if not os.path.exists(reference_by_micro_file):
        print(f"Предупреждение: Файл {reference_by_micro_file} не найден, пропускаем директорию")
        return
    
    # Очищаем существующие директории перед созданием новых
    clean_test_directories(test_dir)
    
    # Создаем необходимые поддиректории
    clear_reference_dir = os.path.join(test_dir, "clear_reference")
    reference_by_micro_dir = os.path.join(test_dir, "reference_by_micro")
    
    # Создаем базовые директории, если они не существуют
    if not os.path.exists(clear_reference_dir):
        os.makedirs(clear_reference_dir)
        print(f"Создана директория {clear_reference_dir}")
    
    if not os.path.exists(reference_by_micro_dir):
        os.makedirs(reference_by_micro_dir)
        print(f"Создана директория {reference_by_micro_dir}")
    
    # Запускаем обработку директории через общий модуль
    try:
        create_volume_variants.main(test_dir)
        print(f"Успешно созданы варианты громкости для {test_dir}")
    except Exception as e:
        print(f"Ошибка при обработке директории {test_dir}: {e}")

if __name__ == "__main__":
    prepare_all_test_directories()
    print("\nПодготовка тестов завершена!") 