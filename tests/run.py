#!/usr/bin/env python3
"""
Скрипт для последовательного запуска обработки аудио, расчета метрик и генерации отчета.

Запускает последовательно:
1. process_audio.py - обработка аудио и расчет базовых метрик
2. calculate_metrics.py - расчет дополнительных метрик (ERLE и др.)
3. generate_report.py - генерация отчетов и графиков

Скрипт должен находиться в директории tests и запускаться оттуда.

Использование:
cd tests
python run.py --test-dir music
"""

import os
import sys
import argparse
import subprocess
import logging
import time

def setup_logging(log_dir):
    """
    Настраивает логирование с учетом директории.
    
    Args:
        log_dir: Директория для сохранения лог-файла
    """
    log_file = os.path.join(log_dir, "run.log")
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Логирование настроено. Лог-файл: {log_file}")

def run_script(script_name, args, env=None):
    """
    Запускает указанный скрипт с переданными аргументами.
    
    Args:
        script_name: Имя скрипта для запуска
        args: Список аргументов командной строки
        env: Словарь с переменными окружения (опционально)
    
    Returns:
        bool: Успешность выполнения скрипта
    """
    cmd = [sys.executable, script_name] + args
    cmd_str = " ".join(cmd)
    
    logging.info(f"Запуск команды: {cmd_str}")
    print("-" * 80)
    print(f"Запуск: {script_name}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Запускаем процесс и передаем его вывод в текущую консоль
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )
        
        # Выводим stdout в реальном времени
        for line in process.stdout:
            print(line, end='')
        
        # Ждем завершения процесса
        process.wait()
        
        duration = time.time() - start_time
        
        if process.returncode == 0:
            logging.info(f"Скрипт {script_name} успешно выполнен за {duration:.2f} секунд")
            print(f"\nСкрипт {script_name} успешно выполнен за {duration:.2f} секунд")
            return True
        else:
            logging.error(f"Скрипт {script_name} завершился с ошибкой (код {process.returncode})")
            print(f"\nОшибка: Скрипт {script_name} завершился с ошибкой (код {process.returncode})")
            return False
            
    except Exception as e:
        logging.error(f"Ошибка при запуске скрипта {script_name}: {e}")
        print(f"\nОшибка при запуске скрипта {script_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Последовательный запуск обработки аудио, метрик и отчетов")
    parser.add_argument("--test-dir", "-d", default=".",
                      help="Директория с тестовыми данными относительно текущей директории (по умолчанию: текущая директория)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                      help="Подробный вывод")
    parser.add_argument("--skip-process", action="store_true", default=False,
                      help="Пропустить этап обработки аудио (process_audio.py), используя существующие результаты")
    
    args = parser.parse_args()
    
    # Получаем абсолютный путь к тестовой директории
    test_dir = os.path.abspath(args.test_dir)
    
    # Проверяем наличие указанной директории
    if not os.path.exists(test_dir):
        print(f"Директория {test_dir} не существует. Создаем...")
        try:
            os.makedirs(test_dir, exist_ok=True)
            print(f"Директория {test_dir} успешно создана")
        except Exception as e:
            print(f"Ошибка при создании директории {test_dir}: {e}")
            sys.exit(1)
    
    # Настраиваем логирование
    setup_logging(test_dir)
    
    # Определяем имена скриптов - скрипты находятся в той же директории, что и run.py
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    process_audio_script = os.path.join(scripts_dir, "process_audio.py")
    calculate_metrics_script = os.path.join(scripts_dir, "calculate_metrics.py")
    generate_report_script = os.path.join(scripts_dir, "generate_report.py")
    
    # Проверяем наличие скриптов
    for script in [process_audio_script, calculate_metrics_script, generate_report_script]:
        if not os.path.exists(script):
            logging.error(f"Не найден скрипт: {script}")
            print(f"Ошибка: Не найден скрипт: {script}")
            sys.exit(1)
    
    # Общий таймер
    total_start_time = time.time()
    
    # Шаг 1: Запуск process_audio.py (пропускаем, если указан флаг --skip-process)
    if not args.skip_process:
        process_audio_args = ["--test-dir", test_dir]
        if args.verbose:
            process_audio_args.append("--verbose")
        
        if not run_script(process_audio_script, process_audio_args):
            logging.error("Прерывание выполнения из-за ошибки в process_audio.py")
            sys.exit(1)
    else:
        logging.info("Этап обработки аудио пропущен (--skip-process)")
        print("Этап обработки аудио пропущен (--skip-process)")
    
    # Шаг 2: Запуск calculate_metrics.py
    calculate_metrics_args = ["--test-dir", test_dir]
    if args.verbose:
        calculate_metrics_args.append("--verbose")
    
    if not run_script(calculate_metrics_script, calculate_metrics_args):
        logging.error("Прерывание выполнения из-за ошибки в calculate_metrics.py")
        sys.exit(1)
    
    # Шаг 3: Запуск generate_report.py
    # Теперь generate_report.py использует --test-dir вместо --results-dir
    # и автоматически создает отчет в report_output внутри test-dir
    generate_report_args = ["--test-dir", test_dir]
    if args.verbose:
        generate_report_args.append("--verbose")
    
    if not run_script(generate_report_script, generate_report_args):
        logging.error("Прерывание выполнения из-за ошибки в generate_report.py")
        sys.exit(1)
    
    # Выводим общее время выполнения
    total_duration = time.time() - total_start_time
    logging.info(f"Все скрипты успешно выполнены за {total_duration:.2f} секунд")
    print(f"\nВсе скрипты успешно выполнены за {total_duration:.2f} секунд")
    print(f"\nРезультаты обработки сохранены в директории {test_dir} и папке {test_dir}/report_output")

if __name__ == "__main__":
    main() 