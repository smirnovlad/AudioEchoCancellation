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

def prepare_all_test_directories():
    """
    Подготавливает все директории с тестами.
    
    Создает следующую структуру директорий:
    test_dir/
    ├── clear_reference/             # Для чистого референсного сигнала
    │   ├── volume_01/            # 10% громкости
    │   ├── volume_04/            # 40% громкости
    │   └── ...
    │
    └── reference_by_micro/          # Для референса через микрофон
        ├── delay_0/                 # Задержка 0 мс (без задержки)
        │   ├── volume_01/        # 10% громкости
        │   ├── volume_04/        # 40% громкости
        │   └── ...
        ├── delay_50/                # Задержка 50 мс
        │   ├── volume_01/
        │   └── ...
        └── ...
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Проверка наличия ffmpeg
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("Предупреждение: ffmpeg не установлен. Для конвертации MP3 в WAV требуется ffmpeg.")
        print("Установите ffmpeg или сконвертируйте файлы вручную.")
    
    # Список директорий с тестами
    test_dirs = [
        os.path.join(base_dir, "music"),
        os.path.join(base_dir, "agent_speech"),
        os.path.join(base_dir, "agent_user_speech"),
    ]
    
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
            
            # Проверяем наличие файла my_voice (WAV или MP3)
            my_voice_wav = os.path.join(test_dir, "my_voice.wav")
            my_voice_mp3 = os.path.join(test_dir, "my_voice.mp3")
            if not os.path.exists(my_voice_wav) and not os.path.exists(my_voice_mp3):
                print(f"Предупреждение: Файл my_voice.wav или my_voice.mp3 не найден в {test_dir}")
                print("Будут созданы только файлы с различным уровнем громкости без микширования с голосом")
            elif not os.path.exists(my_voice_wav) and os.path.exists(my_voice_mp3) and not has_ffmpeg:
                print(f"Найден файл my_voice.mp3, но ffmpeg не установлен для конвертации.")
                print("Рекомендуется установить ffmpeg или вручную конвертировать MP3 в WAV.")
            else:
                print(f"Найден файл с голосом в {test_dir}")
            
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
                print(f"Обработка директории {test_dir}")
                create_volume_variants.main(test_dir)
                print(f"Успешно созданы варианты громкости для {test_dir}")
            except Exception as e:
                print(f"Ошибка при обработке директории {test_dir}: {e}")
        else:
            print(f"Директория {test_dir} не найдена, пропускаем")

if __name__ == "__main__":
    prepare_all_test_directories()
    print("\nПодготовка тестов завершена!") 