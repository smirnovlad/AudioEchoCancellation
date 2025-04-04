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

def prepare_all_test_directories():
    """Подготавливает все директории с тестами."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Проверка наличия ffmpeg
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("Предупреждение: ffmpeg не установлен. Для конвертации MP3 файлов требуется ffmpeg.")
        print("Установите ffmpeg или сконвертируйте файлы вручную.")
    
    # Список директорий с тестами
    test_dirs = [
        os.path.join(base_dir, "music"),
        os.path.join(base_dir, "agent_speech")
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\nПодготовка директории: {test_dir}")
            
            # Проверяем наличие файла reference.wav
            reference_file = os.path.join(test_dir, "reference.wav")
            if not os.path.exists(reference_file):
                print(f"Предупреждение: Файл {reference_file} не найден, пропускаем директорию")
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