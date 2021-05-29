import paho.mqtt.client as mqtt
import bin.config as config

mqttc = mqtt.Client()

def establish_connection():
    global mqttc
    mqttc.connect(config.MQTT_SERVER, config.MQTT_PORT)

def disconnect():
    global mqttc
    mqttc.disconnect()

def publish(topic, message):
    global mqttc
    establish_connection()
    mqttc.publish(topic, payload=message)
    disconnect()