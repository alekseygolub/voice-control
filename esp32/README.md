# ESP32 Firmware
This firmware can be used to create smart lamp or led

## Configuring
1. You need to define a SSID name and a password to access the WiFi
2. Set IP-address of MQTT-broker

## Uploading
Use [Arduiono IDE](https://www.arduino.cc/en/software) or [PlatformIo](https://platformio.org) plugin for VsCode to upload this firmware on the microcontoller. 

Library dependencies you can find in `platformio.ini`

## IO
You can use `13` pin as a signal pin or set its number in `LED_PIN`
