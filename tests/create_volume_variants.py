import os
import shutil
import numpy as np
import soundfile as sf
import subprocess
import tempfile

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
    
    # Изменение амплитуды (громкости)
    adjusted_data = data * volume_factor
    
    # Сохранение измененного аудиофайла
    sf.write(output_file, adjusted_data, samplerate)
    print(f"Создан файл с громкостью {volume_factor}: {output_file}")
    
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
    """Микширует два аудиосигнала и сохраняет результат."""
    # Проверка на наличие данных
    if file1_data is None or file2_data is None:
        print("Не удалось произвести микширование: один из файлов отсутствует")
        return
    
    # Приведение к одинаковому формату каналов (моно или стерео)
    if len(file1_data.shape) > 1 and len(file2_data.shape) == 1:
        # file1 - стерео, file2 - моно
        file2_data = np.column_stack((file2_data, file2_data))
    elif len(file1_data.shape) == 1 and len(file2_data.shape) > 1:
        # file1 - моно, file2 - стерео
        file1_data = np.column_stack((file1_data, file1_data))
    
    # Определяем количество каналов после преобразования
    is_stereo = len(file1_data.shape) > 1
    
    # Приведение массивов к одинаковой длине (берем минимальную длину)
    min_length = min(len(file1_data), len(file2_data))
    file1_data = file1_data[:min_length]
    file2_data = file2_data[:min_length]
    
    # Микширование аудиосигналов (суммирование)
    mixed_data = file1_data + file2_data
    
    # Нормализация для предотвращения клиппинга
    if np.max(np.abs(mixed_data)) > 1.0:
        mixed_data = mixed_data / np.max(np.abs(mixed_data)) * 0.9
    
    # Сохранение результата
    sf.write(output_file, mixed_data, samplerate)
    print(f"Создан микшированный файл: {output_file}")

def process_directory(directory_path):
    """Обрабатывает директорию, создавая поддиректории с разными уровнями громкости."""
    reference_file = os.path.join(directory_path, "reference.wav")
    my_voice_file = os.path.join(directory_path, "my_voice.wav")
    my_voice_mp3 = os.path.join(directory_path, "my_voice.mp3")
    
    # Проверяем, существует ли файл reference.wav
    if not os.path.exists(reference_file):
        print(f"Файл {reference_file} не найден!")
        return
    
    # Получаем параметры референсного файла
    ref_data, ref_samplerate = sf.read(reference_file)
    print(f"Референсный файл: частота дискретизации {ref_samplerate} Гц, формат: {'стерео' if len(ref_data.shape) > 1 else 'моно'}")
    
    # Проверка на наличие MP3 вместо WAV и конвертация с правильной частотой
    if not os.path.exists(my_voice_file) and os.path.exists(my_voice_mp3):
        print("Найден файл my_voice.mp3 вместо my_voice.wav. Конвертация с нужной частотой дискретизации...")
        temp_wav_file = os.path.join(directory_path, "my_voice_temp.wav")
        
        if convert_mp3_to_wav(my_voice_mp3, temp_wav_file, ref_samplerate):
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
            print(f"Загружен файл с голосом: {my_voice_file} - частота дискретизации {my_voice_samplerate} Гц, формат: {'стерео' if len(my_voice_data.shape) > 1 else 'моно'}")
            
            # Проверяем соответствие частоты дискретизации
            if my_voice_samplerate != ref_samplerate:
                print(f"Предупреждение: частота дискретизации my_voice ({my_voice_samplerate} Гц) отличается от reference ({ref_samplerate} Гц)")
                print("Пересохраняем my_voice с нужной частотой дискретизации...")
                
                # Пересохраняем файл с помощью ffmpeg
                temp_wav_file = os.path.join(directory_path, "my_voice_temp.wav")
                if convert_mp3_to_wav(my_voice_file, temp_wav_file, ref_samplerate):
                    try:
                        my_voice_data, my_voice_samplerate = sf.read(temp_wav_file)
                        sf.write(my_voice_file, my_voice_data, my_voice_samplerate)
                        os.remove(temp_wav_file)
                        print(f"Файл пересохранен с частотой {my_voice_samplerate} Гц")
                    except Exception as e:
                        print(f"Ошибка при пересохранении файла: {e}")
                
        except Exception as e:
            print(f"Ошибка при чтении файла с голосом {my_voice_file}: {e}")
            my_voice_data = None
    else:
        print("Файл с голосом не найден. Будут созданы только файлы с измененной громкостью.")
    
    # Для каждого уровня громкости создаем поддиректорию
    for suffix, volume_factor in VOLUME_LEVELS.items():
        # Создаем имя поддиректории
        subdirectory_name = f"reference_{suffix}"
        subdirectory_path = os.path.join(directory_path, subdirectory_name)
        
        # Создаем поддиректорию
        create_directory_if_not_exists(subdirectory_path)
        
        # Создаем имя выходного файла для аудио с измененной громкостью
        output_reference_file = os.path.join(subdirectory_path, "reference_new.wav")
        
        # Изменяем громкость и сохраняем файл
        adjusted_data, samplerate = adjust_volume(reference_file, output_reference_file, volume_factor)
        
        # Если есть файл с голосом, создаем смешанный файл
        if my_voice_data is not None and my_voice_samplerate is not None:
            # Создаем имя выходного файла для смешанного аудио
            output_mixed_file = os.path.join(subdirectory_path, "original_input.wav")
            
            # Микшируем аудиофайлы
            mix_audio_files(adjusted_data, my_voice_data, output_mixed_file, samplerate)

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