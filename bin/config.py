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

TRAIN_RATIO = 0.85

# ---------------------------------------------------------

def mfcc(data, n_features=40):
    res = torch.from_numpy(librosa.feature.mfcc(data.numpy(), sr=SAMPLE_RATE, n_mfcc=N_FEATURES,
                           n_fft=WINDOW_SIZE, win_length=WINDOW_SIZE, hop_length=(WINDOW_SIZE // 2), power=2))
    for i in range(res.shape[1]):
        S = sum(res[:, i])
        S /= len(res)
        for j in range(len(res)):
            res[j, i] /= S
    return res

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