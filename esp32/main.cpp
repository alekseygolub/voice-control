#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h>

#define LED_PIN 13
#define BLINK_DELAY 200

const char* ssid = "";
const char* password = "";
const char* mqtt_server = "";
const char* mqtt_client_id = "ESP-32-1";

WiFiClient espClient;
PubSubClient client(espClient);

void mqttconnect() {
    while (!client.connected()) {
        Serial.print("MQTT connecting ...");
        if (client.connect(mqtt_client_id)) {
            Serial.println("connected");
            // Init subscriptions
            client.subscribe("AlinaSmartHome/led1");
        } else {
            Serial.print("failed, status code =");
            Serial.print(client.state());
            Serial.println("try again in 2 seconds");
            delay(2000);
        }
    }
}

void callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Message arrived [");
    Serial.print(topic);
    Serial.print("] ");
    for (int i = 0; i < length; i++) {
        Serial.print((char)payload[i]);
    }
    Serial.println();
    if ((char)payload[0] == '1') {
        digitalWrite(LED_PIN, HIGH);
    } else {
        digitalWrite(LED_PIN, LOW);
    }
}

void initWiFi() {
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        Serial.print('.');
        delay(1000);
    }
    Serial.println();
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
}

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
    initWiFi();
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
}

void loop() {
    if (!client.connected()) {
        mqttconnect();
    }
    client.loop();
}