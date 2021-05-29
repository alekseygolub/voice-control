import torchaudio as audio
import torch
import librosa
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------

# Audio Settings
WINDOW_TIME = 0.01 # seconds
SAMPLE_RATE = 16000 # 16 kHz
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_TIME)
HOP_LENGTH = WINDOW_SIZE // 2
audio.set_audio_backend('sox_io')

# Model Settings
N_FEATURES = 40
HIDDEN_SIZE = 128
THRESHOLD = 0.25 # Congigure after training

TRAIN_RATIO = 0.85

# Detection config
CHUNK_SIZE = 1024
MAXLENGTHOFRECORD = SAMPLE_RATE // CHUNK_SIZE # 1 second
KEYWORD = 'алина'
MAXWORDINRECORD = 2
LANGUAGE_CODE = 'ru-RU'

# Mqtt config
TOPIC_PATH = "AlinaSmartHome/led1"
MQTT_SERVER = "" # ip address of mqtt broker
MQTT_PORT = 1883 # default port

# CloudApi config
CLOUD_API_URL = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'

# ---------------------------------------------------------

def mfcc(data):
    res = torch.from_numpy(librosa.feature.mfcc(data.numpy(), sr=SAMPLE_RATE, n_mfcc=N_FEATURES, n_fft=WINDOW_SIZE, win_length=WINDOW_SIZE, hop_length=HOP_LENGTH, power=2))
    return res / torch.mean(res, 0)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.rnn = torch.nn.GRUCell(input_size=40, hidden_size=HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.fc2 = nn.Linear(HIDDEN_SIZE // 2, HIDDEN_SIZE // 4)
        self.fc3 = nn.Linear(HIDDEN_SIZE // 4, 2)


    def forward(self, x, h=None):
        h = self.rnn(x, h)
        return self.fc3( F.leaky_relu( self.fc2( (F.leaky_relu(self.fc1( h ))))) ), h
