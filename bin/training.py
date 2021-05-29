from torch import nn
from sklearn.metrics import precision_recall_curve

import torch.nn.functional as F
import numpy as np
import random
import torch
import sys

import bin.config as config
from bin.config import mfcc

from tools.helpers import save_model_txt

def print_accuracy(model, dataloader):
    X_p = dataloader.getTestPositives()
    X_n = dataloader.getTestNegatives()

    with torch.no_grad():
        y_test = []
        probs = []
        for i in range(len(X_p)):
            xb = mfcc(torch.from_numpy(X_p[i]).float())
            y_test.append(1)
            h = torch.zeros(1, config.HIDDEN_SIZE, dtype=torch.float32)
            res = 0
            for j in range(xb.shape[1]):
                y, h = model(xb[:, j:j + 1].T, h)
                y = F.softmax(y)
                res = max(res, y[0, 1])
            probs.append(res)

        for i in range(len(X_n)):
            xb = mfcc(torch.from_numpy(X_n[i]).float())
            y_test.append(0)
            h = torch.zeros(1, config.HIDDEN_SIZE, dtype=torch.float32)
            
            res = 0
            for j in range(xb.shape[1]):
                y, h = model(xb[:, j:j + 1].T, h)
                y = F.softmax(y, dim=-1)
                res = max(res, y[0, 1])

            probs.append(res)
        y_test, probs = np.array(y_test), np.array(probs)
        p, r, tres = precision_recall_curve(y_test, probs)
        p = p[1:]
        r = r[1:]
        f = 2 * p * r / (p + r)
        best_idx = np.argmax(f)
        print("Precision = ", p[best_idx])
        print("Recall = ", r[best_idx])
        print("Best threshold = ", tres[best_idx])


def train(model, dataloader, epochCount=20):
    opt = torch.optim.Adam(model.parameters(), lr=0.00001)
    w = torch.tensor([1.0, 1000.0])
    loss_func = nn.CrossEntropyLoss(weight=w)

    BATCH_SIZE = 5
    PN_RATIO = 2
    
    print("Training started")

    positives = dataloader.getTrainPositives()
    negatives = dataloader.getTrainNegatives()
    
    for epoch in range(epochCount):
        samples_count = 0
        running_loss = 0.0
        perm_p = list(torch.randperm(len(positives)))
        perm_n = list(torch.randperm(len(negatives)))
        p = n = 0
        while p < len(positives) and n < len(negatives):
            xb = []
            yb = []
            while p < len(positives) and n < len(negatives) and len(xb) < BATCH_SIZE:
                if random.randint(0, PN_RATIO):
                    xb.append(mfcc(torch.from_numpy(negatives[perm_n[n]]).float()))
                    yb.append(torch.zeros(xb[-1].shape[1]).long())
                    n += 1
                else:
                    xb.append(mfcc(torch.from_numpy(positives[perm_p[p]]).float()))
                    yb.append(torch.cat([ torch.zeros(xb[-1].shape[1] - 15), torch.ones(10), torch.zeros(5)]).long())
                    p += 1
            xb = torch.cat(xb, axis=1)
            yb = torch.cat(yb)

            h = torch.zeros(1, config.HIDDEN_SIZE, dtype=torch.float32)
            
            opt.zero_grad()

            error = 0
            y = None
            for j in range(xb.shape[1]):
                y, h = model(xb[:, j:j+1].T, h)
                error += loss_func(y, torch.tensor([yb[j]]))

            error.backward()
            opt.step()

            running_loss += error.item()
            samples_count += 1
        print("Epoch #" + str(epoch) + " ended")
        print("Average loss", running_loss / samples_count)
        print_accuracy(model, dataloader)
        save_model_txt(model, 'model.txt')
        running_loss = 0.0
    
    print('Training completed')
