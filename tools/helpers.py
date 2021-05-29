import torch
import numpy
import sys

import numpy as np

def save_model_txt(model, path):
    fout = open(path, 'w')
    for k, v in model.state_dict().items():
        fout.write(str(k) + '\n')
        fout.write(str(v.tolist()) + '\n')
    fout.close()


def load_model_txt(model, path):
    print('Loading model...')
    data_dict = {}
    fin = open(path, 'r')
    i = 0
    odd = 1
    prev_key = None
    while True:
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            prev_key = s
        else:
            print('Iter', i)
            val = eval(s)
            if type(val) != type([]):
                data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
            else:
                data_dict[prev_key] = torch.FloatTensor(eval(s))
            i += 1
        odd = (odd + 1) % 2

    own_state = model.state_dict()
    print('Items:', len(own_state.items()))
    for k, v in data_dict.items():
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        else:
            try:
                own_state[k].copy_(v)
            except:
                print('Key:', k)
                print('Old:', own_state[k])
                print('New:', v)
                sys.exit(0)
    print('Model loaded')


def convertBinFloat32ToBinInt16(data):
    maxInt16 = 32768.
    tmp = np.frombuffer(data, dtype="float32")
    tmp = tmp * maxInt16
    tmp = tmp.astype("int16")
    return tmp.tobytes()

