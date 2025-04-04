# AudioEchoCancellation

Python version: 3.11

```sh
# Установка необходимых зависимостей
sudo apt-get install libtool autoconf automake build-essential pkg-config libglib2.0-dev
sudo apt-get install swig python3-dev python3.11-dev alsa-utils
sudo apt-get install pulseaudio
python3.11 -m pip install --upgrade pip setuptools wheel

# Добавление пользователя в группу audio для доступа к звуковым устройствам
sudo usermod -a -G audio $USER
# Важно: после этой команды нужно перезайти в систему, чтобы изменения вступили в силу

git clone https://github.com/xiongyihui/python-webrtc-audio-processing.git
cd python-webrtc-audio-processing
git submodule init && git submodule update
cd webrtc-audio-processing
sudo ./autogen.sh

# Очистка предыдущей сборки (если есть)
sudo make clean

# Важно: используем флаг --with-pic и устанавливаем CFLAGS и CXXFLAGS
CFLAGS="-fPIC" CXXFLAGS="-fPIC" ./configure --with-pic
sudo make CFLAGS="-fPIC" CXXFLAGS="-fPIC"

cd ../src

# Исправление Makefile для использования Python 3
sed -i 's/python-config/python3-config/g' Makefile

sudo make
```

```sh
# Установка пакета
python3.11 -m pip install webrtc-audio-processing==0.1.3
```
