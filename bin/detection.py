import torch
import numpy as np

import torch.nn.functional as F
import pyaudio
import argparse
import threading

import bin.config as config
import bin.mqtt as mqtt

from bin.config import Net
from bin.config import mfcc
from bin.cloudapi import recognize_phrase
from tools.helpers import load_model_txt
from collections import deque

mutex = threading.Lock()

def process_phrase(phrase):
    phrase = phrase.lower()
    if "включи лампочку" in phrase:
        mqtt.publish(config.TOPIC_PATH, '1')
        print(config.TOPIC_PATH, ": 1")
    if "выключи лампочку" in phrase:
        mqtt.publish(config.TOPIC_PATH, '0')
        print(config.TOPIC_PATH, ": 0")

def run(oauth, folder_id):
    global mutex

    model = Net()
    load_model_txt(model, 'model.txt')
    model.eval()

    audio = pyaudio.PyAudio()
    streamf32 = audio.open(
        format=pyaudio.paFloat32,
        rate=config.SAMPLE_RATE,
        frames_per_buffer=config.CHUNK_SIZE,
        channels=1,
        input=True,
    )

    h = torch.zeros(1, config.HIDDEN_SIZE, dtype=torch.float32)
    arr = []
    samples = []
    cnt = 10

    recordData = deque()

    while 1:
        with torch.no_grad():
            mutex.acquire()
            rawX = streamf32.read(config.CHUNK_SIZE, exception_on_overflow = False)
            mutex.release()
            recordData.append(rawX)
            if len(recordData) > config.MAXLENGTHOFRECORD:
                recordData.popleft()
            X = np.frombuffer(rawX, dtype="float32")
            xb = mfcc(torch.from_numpy(X).float())
            _cnt = cnt
            for j in range(xb.shape[1]):
                y, h = model(xb[:, j:j+1].T, h)
                y_ = F.softmax(y, dim=1)
                if y_[0, 1] > config.THRESHOLD and cnt > 5:
                    cnt = 0
                    print('Start listening command')
                    phrase = recognize_phrase(folder_id, streamf32, mutex, oauth, recordData)
                    print("Command =", phrase)
                    process_phrase(phrase)
                    break
            cnt += 1
