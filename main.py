from config import Env
from config import mfcc
from config import Net

from tools.dataloader import DataLoader

from bin.training import train

model = Net()
dataloader = DataLoader()

train(model, dataloader, mfcc, Env, 1)
