from bin.config import Net

from tools.dataloader import DataLoader

from bin.training import train

model = Net()
dataloader = DataLoader()

train(model, dataloader, 1)
