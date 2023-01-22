# Utilities (mostly for saving datasets)

import numpy as np
import os
import pandas as pd

def condMakeDir(path):
    if path not in os.listdir(os.getcwd()):
        os.mkdir(path)

def save_numpy_to_json(nparray, filename):
    with open(f'{filename}.json', 'w') as outfile:
        outfile.write(pd.DataFrame(nparray).to_json(orient='values'))

def read_json_to_numpy(filename):
    return pd.read_json(f'{filename}.json').to_numpy()


def RMSE(prediction, target):
    n = len(prediction)
    return np.linalg.norm(prediction - target)/np.sqrt(n)