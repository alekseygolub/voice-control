# Project Alina
[![University: HSE](https://img.shields.io/badge/University-HSE-blue?&style=for-the-badge)](https://www.hse.ru/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
<img alt="Raspberry Pi" src="https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi"/>
<img alt="Arduino" src="https://img.shields.io/badge/-Arduino-00979D?style=for-the-badge&logo=Arduino&logoColor=white"/>

This project is a home assistant that allows you to control smart devices by voice. Implementation based on GRU model for keyword spotting, Yandex.Cloud SpeechKit for voice recognition, Raspberry Pi as a base station, and an esp32 microcontroller as a controlled device. 

## Installation
To install the repository with all requirements:
1. Clone repository
2. Run `install.sh` and enter password

## Dataset
Dataset contains 2 files in `.npy` format:
1. `X_n.npy` is a numpy array of numpy arrays with a splitted negative words
2. `X_p.npy` is a numpy array of numpy arrays with a splitted positive words without gaps

All words should have `16000 Hz` sample rate

Dataset can be downloaded from [here](https://disk.yandex.ru/d/agsVS6BdSYHzwA). Put files in the `data` folder, which created during installation. 

## Training
To start training, use the following command:
```
python3 main.py training
```
model will be saved in `data/model.txt`. 

Using `--epoch_count` and `--model_path` attributes you can set the number of epochs and path to the initialized model

After training, you should set best `threshold` in `config.py`

## Configuring Yandex.Cloud
Use [this](https://cloud.yandex.com/en/docs/iam/concepts/authorization/oauth-token) instruction to get `OAuth`-token and [this](https://cloud.yandex.com/en/docs/speechkit/stt/streaming) to get `folder-id`

## Configuring MQTT broker
Install `mosquitto` on web server or device in local network with following command:
```
sudo apt-get install mosquitto mosquitto-clients
```
mosquitto demon will be started automatically. 

Configure IP-address and Port of MQTT-broker in `config.py`

## Configuring ESP-32
You can find instructions [here](https://github.com/alekseygolub/voice-control/blob/main/esp32/README.md)

## Run it!
You can start detection by following command:
```
python3 main.py detection --oauth <OAUTH_TOKEN> --folder_id <FOLDER_ID>
```
`data/model.txt` should contain [pretrained model](https://disk.yandex.ru/d/agsVS6BdSYHzwA).
