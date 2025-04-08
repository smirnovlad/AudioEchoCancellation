"""
WebRTC AEC Wrapper Module for audio processing with echo cancellation based on WebRTC.
Optimized for use with websocket sessions.
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple, List, Union, Any
import logging
from scipy import signal
from scipy.fftpack import fft
import time

# Check availability of webrtc_audio_processing library
try:
    import webrtc_audio_processing as webrtc
    WEBRTC_AVAILABLE = True
    logging.info("WebRTC is available")
except ImportError:
    WEBRTC_AVAILABLE = False
    logging.warning("webrtc_audio_processing не установлен. AEC будет недоступен.")

class WebRTCAECSession:
    """
    Класс для обработки аудио с помощью WebRTC AEC (Acoustic Echo Cancellation)
    
    Этот класс обеспечивает интерфейс для использования WebRTC AEC для подавления эха
    в аудиопотоке. Он поддерживает два режима работы:
    1. Классический режим реального времени - поочередная обработка референсных и входных фреймов
    2. Пакетный режим (устаревший) - сначала обрабатываются все референсные фреймы, затем все входные
    """
    
    def __init__(self, session_id: str, sample_rate: int = 16000, channels: int = 1, 
                 webrtc_mode: bool = True, batch_mode: bool = False, system_delay: int = 0,
                 input_scale_factor: float = 1.0, reference_scale_factor: float = 1.0, scaling_enabled=False,
                 frame_size_ms: float = 10.0):
        """
        Инициализация сессии WebRTC AEC
        
        Args:
            session_id: Уникальный идентификатор сессии
            sample_rate: Частота дискретизации в Гц (по умолчанию 16000)
            channels: Количество аудиоканалов (по умолчанию 1)
            webrtc_mode: Использовать WebRTC AEC (True) или заглушку (False)
            batch_mode: Использовать пакетный режим обработки (устаревший)
            system_delay: Системная задержка в сэмплах (по умолчанию 0)
            input_scale_factor: Коэффициент масштабирования входного сигнала (по умолчанию 1.0)
            reference_scale_factor: Коэффициент масштабирования референсного сигнала (по умолчанию 1.0)
            scaling_enabled: Включить автоматическое масштабирование сигналов (по умолчанию False)
            frame_size_ms: Размер фрейма в миллисекундах (по умолчанию 10.0 мс)
        """
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_scale_factor = input_scale_factor
        self.reference_scale_factor = reference_scale_factor
        self.scaling_enabled = scaling_enabled
        self.frame_size_ms = frame_size_ms
        
        # Логируем входные параметры
        logging.info(f"Инициализация WebRTCAECSession: session_id={session_id}, sample_rate={sample_rate}, channels={channels}, webrtc_mode={webrtc_mode}, batch_mode={batch_mode}, system_delay={system_delay}")
        logging.info(f"WEBRTC_AVAILABLE = {WEBRTC_AVAILABLE}")
        logging.info(f"Автоматическое масштабирование: {'ВКЛЮЧЕНО' if scaling_enabled else 'ВЫКЛЮЧЕНО'}")
        logging.info(f"Размер фрейма: {frame_size_ms} мс")
        
        # Устанавливаем webrtc_mode с учетом доступности библиотеки
        self.webrtc_mode = webrtc_mode and WEBRTC_AVAILABLE
        logging.info(f"Итоговый webrtc_mode = {self.webrtc_mode}")
        
        self.batch_mode = batch_mode
        self.system_delay = system_delay
        
        # Статистика
        self.stats = {
            "processed_frames": 0,
            "reference_frames": 0,
            "processing_time": 0,
            "echo_frames": 0,  # Счетчик фреймов с эхо (только для входных фреймов)
            "signal_metrics": {
                "echo_detected": False,
                "echo_return_loss": 0,
                "echo_return_loss_enhancement": 0
            }
        }
        
        # Буфер для референсных фреймов (используется только в пакетном режиме)
        self.reference_frames = []
        
        # Размер фрейма в сэмплах (frame_size_ms мс при заданной частоте дискретизации)
        self.frame_size = int(self.sample_rate * self.frame_size_ms / 1000)
        logging.info(f"Инициализация WebRTC AEC сессии {session_id}, размер фрейма: {self.frame_size} сэмплов ({self.frame_size_ms} мс)")
        
        if self.webrtc_mode:
            try:
                logging.info("Начинаем инициализацию WebRTC APM...")
                
                # Инициализация WebRTC APM (Audio Processing Module) в соответствии с audio_processing.h
                # aec_type: 1 - AECM, 2 - AEC, 3 - AEC3 (не используется в audio_processing.h)
                # enable_ns: включить подавление шума
                # agc_type: 0 - отключено, 1 - адаптивный цифровой AGC, 2 - адаптивный аналоговый AGC
                # enable_vad: включить обнаружение голоса
                
                aec_type = 2  # Используем AEC (не AECM)
                enable_ns = True  # Включаем подавление шума
                agc_type = 0  # Отключаем AGC
                enable_vad = False  # Отключаем VAD
                
                logging.info(f"Параметры APM: aec_type={aec_type}, enable_ns={enable_ns}, agc_type={agc_type}, enable_vad={enable_vad}")
                
                self.apm = webrtc.AudioProcessingModule(
                    aec_type=aec_type,
                    enable_ns=enable_ns,
                    agc_type=agc_type,
                    enable_vad=enable_vad
                )
                
                # Настройка параметров в соответствии с audio_processing.h
                logging.info("Настройка параметров AEC...")
                
                # Установка уровня подавления AEC (0 - низкий, 1 - средний, 2 - высокий)
                self.apm.set_aec_level(2)  # Высокий уровень подавления
                
                # Установка задержки системы (по умолчанию 0)
                self.apm.set_system_delay(0)
                
                # Настройка NS (уровень подавления шума: 0 - низкий, 1 - средний, 2 - высокий, 3 - очень высокий)
                self.apm.set_ns_level(1)  # Средний уровень подавления шума
                
                # Установка формата потока (частота дискретизации, количество каналов)
                self.apm.set_stream_format(
                    self.sample_rate,
                    self.channels,
                    self.sample_rate,
                    self.channels
                )
                
                # Установка формата обратного потока (референсный сигнал)
                self.apm.set_reverse_stream_format(
                    self.sample_rate,
                    self.channels
                )
                
                # Проверяем доступные возможности APM
                self.apm_capabilities = self.inspect_apm_capabilities()
                
                # Устанавливаем системную задержку, если она задана
                if system_delay > 0:
                    self.set_system_delay(system_delay)
                
                logging.info(f"WebRTC APM успешно инициализирован для сессии {session_id}")
            except Exception as e:
                logging.error(f"Ошибка при инициализации WebRTC APM: {e}")
                logging.exception("Подробная информация об ошибке:")
                self.webrtc_mode = False
                logging.warning("webrtc_mode установлен в False из-за ошибки инициализации")
        
    def add_reference_frame(self, frame_bytes: bytes) -> None:
        """
        Добавление референсного фрейма (звук, воспроизводимый через динамики)
        
        Args:
            frame_bytes: Байтовые данные фрейма (16-бит PCM)
        """
        if not self.webrtc_mode:
            logging.warning("webrtc_mode=False, референсный фрейм не будет обработан")
            return
        
        try:
            # Логируем размер фрейма
            frame_size_bytes = len(frame_bytes)
            frame_size_samples = frame_size_bytes // (2 * self.channels)  # 2 байта на сэмпл (16 бит)
            frame_size_ms = frame_size_samples / self.sample_rate * 1000
            
            logging.debug(f"Референсный фрейм: {frame_size_samples} сэмплов, {frame_size_bytes} байт, {frame_size_ms:.2f} мс")
            
            # Проверяем, нужно ли масштабировать референсный сигнал
            ref_array = np.frombuffer(frame_bytes, dtype=np.int16)
            
            # Применяем коэффициент масштабирования из настроек
            if self.reference_scale_factor != 1.0:
                ref_array = (ref_array * self.reference_scale_factor).astype(np.int16)
                frame_bytes = ref_array.tobytes()
                logging.debug(f"Применен коэффициент масштабирования референсного сигнала: {self.reference_scale_factor}")
            
            # Дополнительная проверка на слишком низкую амплитуду
            max_amplitude = np.max(np.abs(ref_array))
            if max_amplitude > 0 and max_amplitude < 327 and self.scaling_enabled:  # Если амплитуда меньше 1% от максимальной (32768)
                # Вычисляем коэффициент масштабирования (увеличиваем в 10 раз, но не больше максимума)
                scale_factor = min(10.0, 32767 / max_amplitude)
                logging.debug(f"Масштабирование референсного сигнала: амплитуда {max_amplitude}, коэффициент {scale_factor}")
                
                # Масштабируем сигнал
                scaled_array = (ref_array * scale_factor).astype(np.int16)
                frame_bytes = scaled_array.tobytes()
            elif max_amplitude > 0 and max_amplitude < 327 and not self.scaling_enabled:
                logging.debug(f"Обнаружена низкая амплитуда референсного сигнала ({max_amplitude}), но масштабирование ВЫКЛЮЧЕНО")
            
            if self.batch_mode:
                # В пакетном режиме сохраняем фрейм для последующей обработки
                self.reference_frames.append(np.frombuffer(frame_bytes, dtype=np.int16))
                self.stats["reference_frames"] += 1
            else:
                # В режиме реального времени сразу обрабатываем фрейм
                # Проверяем размер фрейма
                expected_size = self.frame_size * 2 * self.channels  # в байтах
                
                if frame_size_bytes != expected_size:
                    logging.warning(f"Размер референсного фрейма ({frame_size_bytes} байт, {frame_size_ms:.2f} мс) не соответствует ожидаемому ({expected_size} байт, {self.frame_size_ms} мс)")
                    
                    # Если фрейм слишком большой, разбиваем его на части
                    if frame_size_bytes > expected_size:
                        chunks_count = 0
                        for i in range(0, frame_size_bytes, expected_size):
                            if i + expected_size <= frame_size_bytes:
                                chunk = frame_bytes[i:i+expected_size]
                                chunk_samples = len(chunk) // (2 * self.channels)
                                chunk_ms = chunk_samples / self.sample_rate * 1000
                                logging.debug(f"Обработка части референсного фрейма: {chunk_samples} сэмплов, {len(chunk)} байт, {chunk_ms:.2f} мс")
                                
                                # Используем process_reverse_stream из audio_processing.h
                                self.apm.process_reverse_stream(chunk)
                                
                                self.stats["reference_frames"] += 1
                                chunks_count += 1
                        logging.debug(f"Референсный фрейм разделен на {chunks_count} частей")
                    else:
                        # Если фрейм слишком маленький, дополняем его нулями
                        padding_size = expected_size - frame_size_bytes
                        padded_frame = frame_bytes + b'\x00' * padding_size
                        logging.debug(f"Референсный фрейм дополнен нулями до {expected_size} байт, {self.frame_size_ms} мс")
                        
                        # Используем process_reverse_stream из audio_processing.h
                        self.apm.process_reverse_stream(padded_frame)
                        
                        self.stats["reference_frames"] += 1
                else:
                    # Если размер фрейма соответствует ожидаемому, обрабатываем его напрямую
                    logging.debug(f"Обработка референсного фрейма стандартного размера: {frame_size_samples} сэмплов, {frame_size_bytes} байт, {frame_size_ms:.2f} мс")
                    
                    # Используем process_reverse_stream из audio_processing.h
                    self.apm.process_reverse_stream(frame_bytes)
                    
                    self.stats["reference_frames"] += 1
                
                # Обновляем только статистику, но не счетчик фреймов с эхо
                if self.webrtc_mode and not self.batch_mode:
                    try:
                        # Проверяем, обнаружено ли эхо
                        has_echo = self.apm.has_echo()
                        self.stats["signal_metrics"]["echo_detected"] = has_echo
                        # Не увеличиваем счетчик echo_frames здесь
                    except Exception as e:
                        logging.error(f"Ошибка при проверке наличия эха: {e}")
        
        except Exception as e:
            logging.error(f"Ошибка при добавлении референсного фрейма: {e}")
            logging.exception("Подробная информация об ошибке:")
    
    def process_frame(self, frame_bytes: bytes) -> bytes:
        """
        Обработка входного фрейма (звук, записанный с микрофона)
        
        Args:
            frame_bytes: Байтовые данные фрейма (16-бит PCM)
            
        Returns:
            bytes: Обработанные байтовые данные фрейма
        """
        if not self.webrtc_mode:
            return frame_bytes
        
        # Если это первый входной фрейм, сохраняем его для калибровки
        if self.stats["processed_frames"] == 0 and len(frame_bytes) > 1000:
            # Сохраняем копию фрейма для калибровки
            self.first_input_frame = frame_bytes
        
        # Если у нас уже есть первый входной и референсный фреймы, выполняем калибровку
        if hasattr(self, 'first_input_frame') and hasattr(self, 'first_ref_frame'):
            # Выполняем калибровку уровней
            if self.scaling_enabled:
                self.calibrate_levels(self.first_input_frame, self.first_ref_frame)

            # Удаляем сохраненные фреймы, чтобы не выполнять калибровку повторно
            delattr(self, 'first_input_frame')
            delattr(self, 'first_ref_frame')
        
        start_time = time.time()
        
        try:
            # Логируем размер фрейма
            frame_size_bytes = len(frame_bytes)
            frame_size_samples = frame_size_bytes // (2 * self.channels)  # 2 байта на сэмпл (16 бит)
            frame_size_ms = frame_size_samples / self.sample_rate * 1000
            
            logging.debug(f"Входной фрейм: {frame_size_samples} сэмплов, {frame_size_bytes} байт, {frame_size_ms:.2f} мс")
            
            # Нормализация входного сигнала
            input_array = np.frombuffer(frame_bytes, dtype=np.int16)
            
            # Применяем коэффициент масштабирования из настроек
            if self.input_scale_factor != 1.0:
                input_array = (input_array * self.input_scale_factor).astype(np.int16)
                frame_bytes = input_array.tobytes()
                logging.debug(f"Применен коэффициент масштабирования входного сигнала: {self.input_scale_factor}")
            
            # Дополнительная проверка на слишком высокую амплитуду
            max_amplitude = np.max(np.abs(input_array))
            if max_amplitude > 3276 and self.scaling_enabled:  # Если амплитуда больше 10% от максимальной (32768)
                # Вычисляем коэффициент масштабирования (уменьшаем в 10 раз)
                scale_factor = 0.1
                logging.debug(f"Масштабирование входного сигнала: амплитуда {max_amplitude}, коэффициент {scale_factor}")
                
                # Масштабируем сигнал
                scaled_array = (input_array * scale_factor).astype(np.int16)
                frame_bytes = scaled_array.tobytes()
            elif max_amplitude > 3276 and not self.scaling_enabled:
                logging.debug(f"Обнаружена высокая амплитуда входного сигнала ({max_amplitude}), но масштабирование ВЫКЛЮЧЕНО")
            
            if self.batch_mode:
                # В пакетном режиме сначала обрабатываем все референсные фреймы
                input_array = np.frombuffer(frame_bytes, dtype=np.int16)
                self._process_reference_frames_batch(input_array)
                
                # Затем обрабатываем входной фрейм
                expected_size = self.frame_size * 2 * self.channels  # в байтах
                
                if frame_size_bytes != expected_size:
                    # Обработка фреймов нестандартного размера
                    # ... (код для обработки нестандартных фреймов)
                    processed_bytes = frame_bytes  # Заглушка
                else:
                    # Используем process_stream из audio_processing.h
                    processed_bytes = self.apm.process_stream(frame_bytes)
            else:
                # В режиме реального времени обрабатываем фрейм напрямую
                expected_size = self.frame_size * 2 * self.channels  # в байтах
                
                if frame_size_bytes != expected_size:
                    logging.warning(f"Размер входного фрейма ({frame_size_bytes} байт, {frame_size_ms:.2f} мс) не соответствует ожидаемому ({expected_size} байт, {self.frame_size_ms} мс)")
                    
                    # Если фрейм слишком большой, разбиваем его на части
                    if frame_size_bytes > expected_size:
                        processed_chunks = []
                        for i in range(0, frame_size_bytes, expected_size):
                            if i + expected_size <= frame_size_bytes:
                                chunk = frame_bytes[i:i+expected_size]
                                chunk_samples = len(chunk) // (2 * self.channels)
                                chunk_ms = chunk_samples / self.sample_rate * 1000
                                logging.debug(f"Обработка части входного фрейма: {chunk_samples} сэмплов, {len(chunk)} байт, {chunk_ms:.2f} мс")
                                
                                # Используем process_stream из audio_processing.h
                                processed_chunk = self.apm.process_stream(chunk)
                                
                                processed_chunks.append(processed_chunk)
                        
                        # Объединяем обработанные чанки
                        processed_bytes = b''.join(processed_chunks)
                    else:
                        # Если фрейм слишком маленький, дополняем его нулями
                        padding_size = expected_size - frame_size_bytes
                        padded_frame = frame_bytes + b'\x00' * padding_size
                        logging.debug(f"Входной фрейм дополнен нулями до {expected_size} байт, {self.frame_size_ms} мс")
                        
                        # Используем process_stream из audio_processing.h
                        processed_padded = self.apm.process_stream(padded_frame)
                        
                        # Обрезаем результат до исходного размера
                        processed_bytes = processed_padded[:frame_size_bytes]
                else:
                    # Если размер фрейма соответствует ожидаемому, обрабатываем его напрямую
                    logging.debug(f"Обработка входного фрейма стандартного размера: {frame_size_samples} сэмплов, {frame_size_bytes} байт, {frame_size_ms:.2f} мс")
                    
                    # Используем process_stream из audio_processing.h
                    processed_bytes = self.apm.process_stream(frame_bytes)
            
            # Обновляем статистику и счетчик фреймов с эхо
            if self.webrtc_mode:
                try:
                    # Увеличиваем счетчик обработанных фреймов
                    self.stats["processed_frames"] += 1
                    
                    # Проверяем, обнаружено ли эхо
                    has_echo = self.apm.has_echo()
                    self.stats["signal_metrics"]["echo_detected"] = has_echo
                    
                    # Если обнаружено эхо, увеличиваем счетчик
                    if has_echo:
                        self.stats["echo_frames"] += 1
                        logging.debug(f"Обнаружено эхо во входном фрейме {self.stats['processed_frames']}")
                except Exception as e:
                    logging.error(f"Ошибка при проверке наличия эха: {e}")
            
            return processed_bytes
        
        except Exception as e:
            logging.error(f"Ошибка при обработке фрейма: {e}")
            logging.exception("Подробная информация об ошибке:")
            return frame_bytes
        
        finally:
            # Обновляем время обработки
            self.stats["processing_time"] += time.time() - start_time
    
    def _process_reference_frames_batch(self, input_array: np.ndarray) -> None:
        """
        Обработка всех референсных фреймов в пакетном режиме
        
        Args:
            input_array: Входной массив для определения размера чанков
        """
        if not self.reference_frames:
            return
        
        frame_size = self.frame_size
        input_len = len(input_array)
        
        logging.info(f"Пакетная обработка {len(self.reference_frames)} референсных фреймов")
        
        # ОПТИМИЗАЦИЯ: Сначала обрабатываем все референсные фреймы для всех позиций
        for ref_idx, ref_frame in enumerate(self.reference_frames):
            for i in range(0, input_len, frame_size):
                if i + frame_size <= input_len:
                    try:
                        # Получаем соответствующий кусок из референсного фрейма
                        ref_len = len(ref_frame)
                        if i < ref_len:
                            ref_chunk = ref_frame[i:min(i+frame_size, ref_len)]
                            if len(ref_chunk) < frame_size:
                                padding = np.zeros(frame_size - len(ref_chunk), dtype=np.int16)
                                ref_chunk = np.concatenate([ref_chunk, padding])
                        else:
                            ref_chunk = np.zeros(frame_size, dtype=np.int16)
                        
                        # Логируем размер чанка
                        chunk_bytes = ref_chunk.tobytes()
                        chunk_ms = len(ref_chunk) / self.sample_rate * 1000
                        
                        if ref_idx == 0 and i == 0:  # Логируем только первый чанк для уменьшения объема логов
                            logging.debug(f"Пакетный режим: референсный чанк {ref_idx+1}/{len(self.reference_frames)}, позиция {i}: {len(ref_chunk)} сэмплов, {len(chunk_bytes)} байт, {chunk_ms:.2f} мс")
                        
                        # Обрабатываем референсный поток
                        self.apm.process_reverse_stream(chunk_bytes)
                        
                    except Exception as e:
                        logging.error(f"Ошибка при обработке референсного фрейма в пакетном режиме: {e}")
        
        logging.info(f"Завершена пакетная обработка референсных фреймов")
        
        # Очищаем буфер референсных фреймов
        self.reference_frames = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики обработки
        
        Returns:
            Dict: Словарь со статистикой
        """
        if self.webrtc_mode:
            try:
                # Обновляем статистику из APM
                self.stats["signal_metrics"]["echo_detected"] = self.apm.has_echo()
                
                # Проверяем, что echo_frames не превышает processed_frames
                if self.stats["echo_frames"] > self.stats["processed_frames"] and self.stats["processed_frames"] > 0:
                    logging.warning(f"Количество фреймов с эхо ({self.stats['echo_frames']}) превышает количество обработанных фреймов ({self.stats['processed_frames']}). Корректируем.")
                    self.stats["echo_frames"] = self.stats["processed_frames"]
                
                # Добавляем информацию о задержке
                self.stats["system_delay"] = {
                    "samples": self.system_delay,
                    "ms": self.system_delay / self.sample_rate * 1000
                }
                
                # Дополнительная статистика, если доступна
                if hasattr(self.apm, 'has_voice'):
                    self.stats["signal_metrics"]["voice_detected"] = self.apm.has_voice()
                
                if hasattr(self.apm, 'vad_level'):
                    self.stats["signal_metrics"]["vad_level"] = self.apm.vad_level()
                
                if hasattr(self.apm, 'agc_level'):
                    self.stats["signal_metrics"]["agc_level"] = self.apm.agc_level()
                
                if hasattr(self.apm, 'ns_level'):
                    self.stats["signal_metrics"]["ns_level"] = self.apm.ns_level()
                
                if hasattr(self.apm, 'aec_level'):
                    self.stats["signal_metrics"]["aec_level"] = self.apm.aec_level()
            except Exception as e:
                logging.error(f"Ошибка при получении статистики: {e}")
        
        return self.stats
    
    def reset(self) -> None:
        """
        Сброс состояния сессии
        """
        if self.webrtc_mode:
            try:
                # Сохраняем текущую задержку
                current_delay = self.system_delay
                
                # Пересоздаем APM с теми же параметрами
                aec_type = 2  # Используем AEC (не AECM)
                enable_ns = True  # Включаем подавление шума
                agc_type = 0  # Отключаем AGC
                enable_vad = False  # Отключаем VAD
                
                self.apm = webrtc.AudioProcessingModule(
                    aec_type=aec_type,
                    enable_ns=enable_ns,
                    agc_type=agc_type,
                    enable_vad=enable_vad
                )
                
                # Настройка параметров
                self.apm.set_aec_level(2)  # Высокий уровень подавления
                self.apm.set_system_delay(0)
                self.apm.set_ns_level(1)  # Средний уровень подавления шума
                
                # Установка формата потока
                self.apm.set_stream_format(
                    self.sample_rate,
                    self.channels,
                    self.sample_rate,
                    self.channels
                )
                
                # Установка формата обратного потока
                self.apm.set_reverse_stream_format(
                    self.sample_rate,
                    self.channels
                )
                
                # Восстанавливаем задержку
                if current_delay > 0:
                    self.set_system_delay(current_delay)
                    logging.info(f"Восстановлена системная задержка после сброса: {current_delay} сэмплов")
                
                logging.info(f"Сессия {self.session_id} успешно сброшена")
            except Exception as e:
                logging.error(f"Ошибка при сбросе сессии: {e}")
                logging.exception("Подробная информация об ошибке:")
        
        # Сбрасываем статистику, но сохраняем задержку
        self.stats = {
            "processed_frames": 0,
            "reference_frames": 0,
            "processing_time": 0,
            "echo_frames": 0,
            "signal_metrics": {
                "echo_detected": False,
                "echo_return_loss": 0,
                "echo_return_loss_enhancement": 0
            }
        }
        
        # Очищаем буфер референсных фреймов
        self.reference_frames = []

    def set_system_delay(self, delay_samples: int) -> bool:
        """
        Устанавливает системную задержку для AEC
        
        Args:
            delay_samples: Задержка в сэмплах
            
        Returns:
            bool: True, если задержка успешно установлена, иначе False
        """
        if not self.webrtc_mode:
            logging.warning("webrtc_mode=False, невозможно установить системную задержку")
            return False
        
        try:
            # Проверяем, что задержка неотрицательная
            if delay_samples < 0:
                logging.warning(f"Задержка не может быть отрицательной: {delay_samples}. Устанавливаем в 0.")
                delay_samples = 0
            
            # Преобразуем задержку в целое число (int)
            delay_samples = int(delay_samples)
            
            # Устанавливаем задержку
            logging.info(f"Попытка установить системную задержку: {delay_samples} сэмплов")
            self.apm.set_system_delay(delay_samples)
            self.system_delay = delay_samples
            
            # Вычисляем задержку в миллисекундах для логирования
            delay_ms = delay_samples * 1000 / self.sample_rate
            
            logging.info(f"Установлена системная задержка: {delay_samples} сэмплов ({delay_ms:.2f} мс)")
            return True
        
        except TypeError as e:
            logging.error(f"Ошибка типа при установке системной задержки: {e}")
            logging.info("Пробуем альтернативный подход с преобразованием типа...")
            
            try:
                # Некоторые реализации могут требовать определенный тип
                # Пробуем разные варианты преобразования
                delay_int32 = np.int32(delay_samples)
                self.apm.set_system_delay(delay_int32)
                self.system_delay = delay_samples
                logging.info(f"Установлена системная задержка (int32): {delay_samples} сэмплов")
                return True
            except Exception as e2:
                logging.error(f"Альтернативный подход также не сработал: {e2}")
                
                return False
        
        except Exception as e:
            logging.error(f"Ошибка при установке системной задержки: {e}")
            logging.exception("Подробная информация об ошибке:")
            
            return False

    @staticmethod
    def get_delay_correlation_data(reference_data: bytes, input_data: bytes, sample_rate: int = 16000, channels: int = 1, max_delay_ms: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисляет данные корреляции между референсным и входным сигналами
        
        Args:
            reference_data: Референсный сигнал (байты)
            input_data: Входной сигнал (байты)
            sample_rate: Частота дискретизации (Гц)
            channels: Количество каналов (1 для моно, 2 для стерео)
            max_delay_ms: Максимальная задержка для поиска (мс)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (ref_signal_norm, in_signal_norm, correlation, lags) - 
                нормализованные сигналы, массив корреляции и массив задержек в отсчетах
        """
        logging.info(f"Вычисление корреляции для определения задержки (каналов: {channels})...")
        
        # Преобразуем байты в numpy массивы
        ref_signal = np.frombuffer(reference_data, dtype=np.int16)
        in_signal = np.frombuffer(input_data, dtype=np.int16)
        
        # Если у нас стерео-данные, преобразуем их в правильную форму и берем один канал
        if channels == 2:
            # Проверяем, что длина массивов четная для стерео данных
            if len(ref_signal) % 2 == 0:
                ref_signal = ref_signal.reshape(-1, 2)[:, 0]  # Берем только левый канал
                logging.debug(f"Стерео референсный сигнал преобразован в моно: {len(ref_signal)} сэмплов")
            else:
                logging.warning(f"Нечетное количество элементов ({len(ref_signal)}) в стерео референсном сигнале, используем как есть")
                
            if len(in_signal) % 2 == 0:
                in_signal = in_signal.reshape(-1, 2)[:, 0]  # Берем только левый канал
                logging.debug(f"Стерео входной сигнал преобразован в моно: {len(in_signal)} сэмплов")
            else:
                logging.warning(f"Нечетное количество элементов ({len(in_signal)}) в стерео входном сигнале, используем как есть")
        
        # Ограничиваем длину сигналов для ускорения вычислений
        # Используем первые 5 секунд или меньше, если сигналы короче
        max_samples = min(5 * sample_rate, len(ref_signal), len(in_signal))
        ref_signal = ref_signal[:max_samples]
        in_signal = in_signal[:max_samples]
        
        # Нормализуем сигналы для лучшей корреляции
        ref_signal_norm = ref_signal.astype(np.float32) / 32768.0
        in_signal_norm = in_signal.astype(np.float32) / 32768.0
        
        # Вычисляем кросс-корреляцию - ЕДИНСТВЕННЫЙ ВЫЗОВ np.correlate!
        correlation = np.correlate(in_signal_norm, ref_signal_norm, mode='full')
        
        # Создаем массив задержек в отсчетах
        lags = np.arange(len(correlation)) - (len(ref_signal_norm) - 1)
        
        return ref_signal_norm, in_signal_norm, correlation, lags

    @staticmethod
    def estimate_delay(reference_data: bytes, input_data: bytes, sample_rate: int = 16000, channels: int = 1, max_delay_ms: int = 1000) -> Tuple[int, float, float]:
        """
        Оценивает задержку между референсным и входным сигналами с помощью кросс-корреляции
        
        Args:
            reference_data: Референсный сигнал (байты)
            input_data: Входной сигнал (байты)
            sample_rate: Частота дискретизации (Гц)
            channels: Количество каналов (1 для моно, 2 для стерео)
            max_delay_ms: Максимальная задержка для поиска (мс)
            
        Returns:
            Tuple[int, float, float]: (задержка в сэмплах, задержка в мс, уверенность в оценке)
        """
        # Используем метод get_delay_correlation_data для получения корреляции
        ref_signal_norm, in_signal_norm, correlation, lags = WebRTCAECSession.get_delay_correlation_data(
            reference_data, input_data, sample_rate, channels, max_delay_ms
        )
        
        # Вычисляем максимальную задержку в сэмплах
        max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        
        # Находим индекс максимальной корреляции
        max_index = np.argmax(correlation)
        
        # Вычисляем задержку (учитываем, что correlate возвращает сдвиг)
        delay_samples = lags[max_index]
        
        # Ограничиваем задержку
        delay_samples = max(0, min(delay_samples, max_delay_samples))
        
        # Вычисляем задержку в мс
        delay_ms = delay_samples * 1000 / sample_rate
        
        # Вычисляем уверенность в оценке (нормализованное значение максимума корреляции)
        max_correlation = correlation[max_index]
        confidence = max_correlation / (np.std(ref_signal_norm) * np.std(in_signal_norm) * len(ref_signal_norm))
        confidence = min(1.0, max(0.0, confidence))
        
        logging.info(f"Оценка задержки: {delay_samples} сэмплов ({delay_ms:.2f} мс), уверенность: {confidence:.2f}")
        
        return delay_samples, delay_ms, confidence

    def auto_set_delay(self, reference_data: bytes, input_data: bytes, max_delay_ms: int = 1000) -> Tuple[int, float, float]:
        """
        Автоматически оценивает и устанавливает задержку между референсным и входным сигналами
        
        Args:
            reference_data: Референсный сигнал (байты)
            input_data: Входной сигнал (байты)
            max_delay_ms: Максимальная задержка для поиска (мс)
            
        Returns:
            Tuple[int, float, float]: (задержка в сэмплах, задержка в мс, уверенность в оценке)
        """
        # Получаем данные корреляции напрямую, чтобы сохранить их для последующего использования
        ref_signal_norm, in_signal_norm, correlation, lags = self.get_delay_correlation_data(
            reference_data, input_data, self.sample_rate, self.channels, max_delay_ms
        )
        
        # Вычисляем максимальную задержку в сэмплах
        max_delay_samples = int(max_delay_ms * self.sample_rate / 1000)
        
        # Находим индекс максимальной корреляции
        max_index = np.argmax(correlation)
        
        # Вычисляем задержку (учитываем, что correlate возвращает сдвиг)
        delay_samples = lags[max_index]
        
        # Ограничиваем задержку
        delay_samples = max(0, min(delay_samples, max_delay_samples))
        
        # Вычисляем задержку в мс
        delay_ms = delay_samples * 1000 / self.sample_rate
        
        # Вычисляем уверенность в оценке (нормализованное значение максимума корреляции)
        max_correlation = correlation[max_index]
        confidence = max_correlation / (np.std(ref_signal_norm) * np.std(in_signal_norm) * len(ref_signal_norm))
        confidence = min(1.0, max(0.0, confidence))
        
        logging.info(f"Оценка задержки: {delay_samples} сэмплов ({delay_ms:.2f} мс), уверенность: {confidence:.2f}")
        
        # Устанавливаем задержку в AEC
        self.set_system_delay(delay_samples)
        
        # Сохраняем данные корреляции для последующего использования
        self._last_correlation_data = {
            'correlation': correlation,
            'lags': lags,
            'delay_samples': delay_samples,
            'delay_ms': delay_ms,
            'confidence': confidence
        }
        
        return delay_samples, delay_ms, confidence

    def get_last_correlation_data(self) -> Dict[str, Any]:
        """
        Возвращает данные последнего расчета корреляции
        
        Returns:
            Dict[str, Any]: Словарь с данными корреляции или пустой словарь, если корреляция еще не вычислялась
        """
        if hasattr(self, '_last_correlation_data'):
            return self._last_correlation_data
        return {}

    def optimize_for_best_quality(self):
        """
        Оптимизирует настройки AEC для наилучшего качества подавления эха
        
        Returns:
            bool: True, если оптимизация успешна, иначе False
        """
        if not self.webrtc_mode:
            logging.warning("webrtc_mode=False, невозможно оптимизировать настройки")
            return False
        
        try:
            # Используем информацию о доступных возможностях
            capabilities = getattr(self, 'apm_capabilities', {})
            if not capabilities:
                capabilities = self.inspect_apm_capabilities()
            
            # Установка высокого уровня подавления эха
            if capabilities.get('set_suppression_level', False):
                self.apm.set_suppression_level(2)  # Высокий уровень
                logging.info("Установлен высокий уровень подавления эха")
            elif capabilities.get('set_echo_suppression_level', False):
                self.apm.set_echo_suppression_level(2)  # Высокий уровень
                logging.info("Установлен высокий уровень подавления эха")
            
            # Включение расширенного фильтра
            if capabilities.get('enable_extended_filter', False):
                self.apm.enable_extended_filter(True)
                logging.info("Включен расширенный фильтр AEC")
            
            # Включение задержки-независимого режима
            if capabilities.get('enable_delay_agnostic', False):
                self.apm.enable_delay_agnostic(True)
                logging.info("Включен режим, не зависящий от задержки")
            elif capabilities.get('set_aec_delay_agnostic', False):
                self.apm.set_aec_delay_agnostic(True)
                logging.info("Включен режим, не зависящий от задержки")
            
            # Включение подавления остаточного эха
            if capabilities.get('enable_residual_echo_suppression', False):
                self.apm.enable_residual_echo_suppression(True)
                logging.info("Включено подавление остаточного эха")
            
            # Включение высококачественного режима обработки
            if capabilities.get('set_processing_mode', False):
                self.apm.set_processing_mode(2)  # Высокое качество
                logging.info("Установлен режим обработки высокого качества")
            
            logging.info("Настройки AEC оптимизированы для наилучшего качества")
            return True
        
        except Exception as e:
            logging.error(f"Ошибка при оптимизации настроек AEC: {e}")
            logging.exception("Подробная информация об ошибке:")
            return False

    def inspect_apm_capabilities(self) -> Dict[str, bool]:
        """
        Проверяет доступные методы и свойства APM
        
        Returns:
            Dict[str, bool]: Словарь с информацией о доступных методах и свойствах
        """
        if not self.webrtc_mode:
            logging.warning("webrtc_mode=False, невозможно проверить возможности APM")
            return {}
        
        capabilities = {}
        
        # Список методов и свойств для проверки
        methods_to_check = [
            'set_system_delay',
            'set_stream_delay_ms',
            'set_delay_offset_ms',
            'set_aec_delay_agnostic',
            'enable_delay_agnostic',
            'set_suppression_level',
            'set_echo_suppression_level',
            'enable_extended_filter',
            'enable_residual_echo_suppression',
            'set_processing_mode',
            'has_echo',
            'has_voice',
            'is_saturated'
        ]
        
        for method in methods_to_check:
            capabilities[method] = hasattr(self.apm, method)
        
        # Логируем результаты
        logging.info("Доступные методы и свойства APM:")
        for method, available in capabilities.items():
            logging.info(f"  {method}: {'Доступен' if available else 'Недоступен'}")
        
        return capabilities

    def auto_balance_levels(self, mic_data: bytes, ref_data: bytes) -> Tuple[float, float]:
        """
        Автоматически определяет коэффициенты масштабирования для балансировки уровней
        
        Args:
            mic_data: Данные с микрофона (байты)
            ref_data: Референсные данные (байты)
            
        Returns:
            Tuple[float, float]: (коэффициент для микрофона, коэффициент для референса)
        """
        try:
            # Преобразуем байты в numpy массивы
            mic_array = np.frombuffer(mic_data, dtype=np.int16)
            ref_array = np.frombuffer(ref_data, dtype=np.int16)
            
            # Вычисляем RMS (среднеквадратичное значение) для обоих сигналов
            mic_rms = np.sqrt(np.mean(mic_array.astype(np.float32)**2))
            ref_rms = np.sqrt(np.mean(ref_array.astype(np.float32)**2))
            
            logging.info(f"RMS микрофона: {mic_rms:.2f}, RMS референса: {ref_rms:.2f}")
            
            # Если один из сигналов слишком слабый, устанавливаем минимальное значение
            mic_rms = max(mic_rms, 1.0)
            ref_rms = max(ref_rms, 1.0)
            
            # Вычисляем соотношение
            ratio = mic_rms / ref_rms
            
            # Определяем коэффициенты масштабирования
            if ratio > 10.0:
                # Микрофон слишком громкий
                mic_scale = 1.0 / ratio
                ref_scale = 1.0
            elif ratio < 0.1:
                # Референс слишком громкий
                mic_scale = 1.0
                ref_scale = ratio
            else:
                # Уровни примерно сбалансированы
                mic_scale = 1.0
                ref_scale = 1.0
            
            logging.info(f"Автоматическая балансировка: соотношение={ratio:.2f}, mic_scale={mic_scale:.2f}, ref_scale={ref_scale:.2f}")
            
            # Обновляем коэффициенты масштабирования
            self.input_scale_factor = mic_scale
            self.reference_scale_factor = ref_scale
            
            return mic_scale, ref_scale
        
        except Exception as e:
            logging.error(f"Ошибка при автоматической балансировке уровней: {e}")
            logging.exception("Подробная информация об ошибке:")
            return 1.0, 1.0

    def calibrate_levels(self, mic_data: bytes, ref_data: bytes, visualize: bool = False, output_file: str = "delay_visualization.png") -> Tuple[float, float]:
        """
        Калибрует уровни сигналов для оптимальной работы AEC
        
        Args:
            mic_data: Данные с микрофона (байты)
            ref_data: Референсные данные (байты)
            visualize: Визуализировать результаты калибровки (не используется)
            output_file: Имя выходного файла для визуализации (не используется)
            
        Returns:
            Tuple[float, float]: Коэффициенты масштабирования (input_scale_factor, reference_scale_factor)
        """
        try:
            # Проверяем размеры данных
            if len(mic_data) < 1000 or len(ref_data) < 1000:
                logging.warning(f"Данные слишком маленькие для калибровки: mic={len(mic_data)} байт, ref={len(ref_data)} байт")
                return self.input_scale_factor, self.reference_scale_factor
            
            if not self.scaling_enabled:
                logging.info("Калибровка уровней пропущена: автоматическое масштабирование ВЫКЛЮЧЕНО")
                return self.input_scale_factor, self.reference_scale_factor
            
            # Преобразуем байты в numpy массивы
            mic_array = np.frombuffer(mic_data, dtype=np.int16)
            ref_array = np.frombuffer(ref_data, dtype=np.int16)
            
            # Вычисляем RMS для обоих сигналов
            mic_rms = np.sqrt(np.mean(mic_array.astype(np.float64) ** 2))
            ref_rms = np.sqrt(np.mean(ref_array.astype(np.float64) ** 2))
            
            # Вычисляем коэффициенты масштабирования
            if mic_rms > 0 and ref_rms > 0:
                # Целевое соотношение RMS (микрофон / референс) = 1.0
                target_ratio = 1.0
                current_ratio = mic_rms / ref_rms
                
                # Если текущее соотношение сильно отличается от целевого, корректируем
                if abs(current_ratio - target_ratio) > 0.1:  # Допуск 10%
                    if current_ratio > target_ratio:
                        # Микрофон громче референса, уменьшаем его
                        self.input_scale_factor = target_ratio / current_ratio
                        self.reference_scale_factor = 1.0
                    else:
                        # Референс громче микрофона, уменьшаем его
                        self.input_scale_factor = 1.0
                        self.reference_scale_factor = current_ratio / target_ratio
                    
                    logging.info(f"Калибровка уровней: исходное соотношение={current_ratio:.2f}, "
                               f"input_scale={self.input_scale_factor:.2f}, ref_scale={self.reference_scale_factor:.2f}")
                else:
                    logging.info(f"Калибровка уровней: соотношение={current_ratio:.2f} в пределах допуска, масштабирование не требуется")
        
            # Возвращаем обновленные коэффициенты масштабирования
            return self.input_scale_factor, self.reference_scale_factor
            
        except Exception as e:
            logging.error(f"Ошибка при калибровке уровней: {e}")
            logging.exception("Подробная информация об ошибке:")
            return self.input_scale_factor, self.reference_scale_factor

    def get_scaled_signals(self, mic_data: bytes, ref_data: bytes) -> Tuple[bytes, bytes]:
        """
        Возвращает масштабированные версии входного и референсного сигналов
        
        Args:
            mic_data: Данные с микрофона (байты)
            ref_data: Референсные данные (байты)
            
        Returns:
            Tuple[bytes, bytes]: Масштабированные данные (микрофон, референс)
        """
        try:
            # Преобразуем байты в numpy массивы
            mic_array = np.frombuffer(mic_data, dtype=np.int16)
            ref_array = np.frombuffer(ref_data, dtype=np.int16)
            
            # Вычисляем RMS для обоих сигналов
            mic_rms = np.sqrt(np.mean(mic_array.astype(np.float64) ** 2))
            ref_rms = np.sqrt(np.mean(ref_array.astype(np.float64) ** 2))
            current_ratio = mic_rms / ref_rms if ref_rms > 0 else 0
            
            logging.info(f"Исходное соотношение RMS: {current_ratio:.4f}")
            logging.info(f"Текущие коэффициенты масштабирования: input={self.input_scale_factor:.4f}, ref={self.reference_scale_factor:.4f}")
            
            # Если коэффициенты масштабирования равны 1.0, но соотношение RMS отличается от 1.0,
            # устанавливаем коэффициенты для выравнивания уровней
            if self.input_scale_factor == 1.0 and self.reference_scale_factor == 1.0 and abs(current_ratio - 1.0) > 0.1:
                if self.scaling_enabled:
                    # Автоматически калибруем уровни
                    self.input_scale_factor, self.reference_scale_factor = self.calibrate_levels(mic_data, ref_data)
                    logging.info(f"Автоматическая калибровка: input={self.input_scale_factor:.4f}, ref={self.reference_scale_factor:.4f}")
                else:
                    logging.info(f"Автоматическая калибровка пропущена: соотношение={current_ratio:.4f}, масштабирование ВЫКЛЮЧЕНО")
            
            # Масштабируем сигналы
            scaled_mic_array = (mic_array * self.input_scale_factor).astype(np.int16)
            scaled_ref_array = (ref_array * self.reference_scale_factor).astype(np.int16)
            
            # Вычисляем RMS для масштабированных сигналов
            scaled_mic_rms = np.sqrt(np.mean(scaled_mic_array.astype(np.float64) ** 2))
            scaled_ref_rms = np.sqrt(np.mean(scaled_ref_array.astype(np.float64) ** 2))
            scaled_ratio = scaled_mic_rms / scaled_ref_rms if scaled_ref_rms > 0 else 0
            
            logging.info(f"Масштабированное соотношение RMS: {scaled_ratio:.4f}")
            
            # Преобразуем обратно в байты
            scaled_mic_data = scaled_mic_array.tobytes()
            scaled_ref_data = scaled_ref_array.tobytes()
            
            return scaled_mic_data, scaled_ref_data
        
        except Exception as e:
            logging.error(f"Ошибка при масштабировании сигналов: {e}")
            logging.exception("Подробная информация об ошибке:")
            return mic_data, ref_data  # Возвращаем исходные данные в случае ошибки

class WebRTCAECManager:
    """
    Manager for handling multiple WebRTC AEC sessions
    """
    def __init__(self):
        self.sessions: Dict[str, WebRTCAECSession] = {}
        self.default_sample_rate = 16000
        self.default_channels = 1
        logging.info("WebRTCAECManager: initialized")
    
    def get_or_create_session(self, session_id: str, sample_rate: int = None, channels: int = None,
                              input_scale_factor: float = 1.0, reference_scale_factor: float = 1.0,
                              scaling_enabled: bool = False, frame_size_ms: float = 10.0) -> WebRTCAECSession:
        """
        Gets an existing or creates a new WebRTC AEC session
        
        Args:
            session_id: Session identifier
            sample_rate: Sample rate
            channels: Number of channels
            input_scale_factor: Scale factor for input signal
            reference_scale_factor: Scale factor for reference signal
            scaling_enabled: Enable automatic signal scaling
            frame_size_ms: Frame size in milliseconds
            
        Returns:
            WebRTCAECSession: WebRTC AEC session
        """
        if session_id in self.sessions:
            # Если сессия уже существует, обновляем коэффициенты масштабирования
            session = self.sessions[session_id]
            if input_scale_factor != 1.0:
                session.input_scale_factor = input_scale_factor
                logging.info(f"Обновлен коэффициент масштабирования входного сигнала для сессии {session_id}: {input_scale_factor}")
            if reference_scale_factor != 1.0:
                session.reference_scale_factor = reference_scale_factor
                logging.info(f"Обновлен коэффициент масштабирования референсного сигнала для сессии {session_id}: {reference_scale_factor}")
            return session
            
        # Create a new session
        sample_rate = sample_rate or self.default_sample_rate
        channels = channels or self.default_channels
        
        session = WebRTCAECSession(
            session_id=session_id,
            sample_rate=sample_rate,
            channels=channels,
            input_scale_factor=input_scale_factor,
            reference_scale_factor=reference_scale_factor,
            scaling_enabled=scaling_enabled,
            frame_size_ms=frame_size_ms
        )
        
        self.sessions[session_id] = session
        logging.info(f"WebRTCAECManager: created new session {session_id} with input_scale={input_scale_factor}, ref_scale={reference_scale_factor}, scaling_enabled={scaling_enabled}, frame_size_ms={frame_size_ms}")
        return session
    
    def remove_session(self, session_id: str):
        """
        Removes a WebRTC AEC session
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id].reset()
            del self.sessions[session_id]
            logging.info(f"WebRTCAECManager: removed session {session_id}")
    
    def get_all_stats(self) -> List[Dict]:
        """
        Returns statistics for all sessions
        
        Returns:
            List[Dict]: List of session statistics
        """
        return [session.get_statistics() for session in self.sessions.values()]
    
    def cleanup(self):
        """
        Cleans up all sessions
        """
        session_count = len(self.sessions)
        for session_id in list(self.sessions.keys()):
            self.remove_session(session_id)
        logging.info(f"WebRTCAECManager: cleaned up all sessions (total: {session_count})")

# Create global manager instance
aec_manager = WebRTCAECManager() 