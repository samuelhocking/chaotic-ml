'''
Consolidated utility methods and classes for machine learning on chaotic systems
`cml_core.py` contains the core components, models, etc.
'''

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import os

def condMakeDir(path):
    '''
    conditional mkdir: makes dir w/ specified path if it does not exist in the cwd, otherwise does nothing
    '''
    if path not in os.listdir(os.getcwd()):
        os.mkdir(path)

def save_numpy_to_json(nparray, filename):
    with open(f'{filename}.json', 'w') as outfile:
        outfile.write(pd.DataFrame(nparray).to_json(orient='values'))

def read_json_to_numpy(filename):
    return pd.read_json(f'{filename}.json').to_numpy()

def RMSE(prediction, target):
    '''
    root mean squared error (model evaluation metric)
    '''
    n = len(prediction)
    return np.linalg.norm(prediction - target)/np.sqrt(n)

def NRMSE(prediction, target):
    maxnorm = max(np.array([np.linalg.norm(x) for x in target]))
    minnorm = min(np.array([np.linalg.norm(x) for x in target]))
    return RMSE(prediction, target)/(maxnorm-minnorm)

def RK4(func, y0, t_array, args=None):
    d = len(y0)
    n = len(t_array)
    y_array = np.zeros((n,d))
    y_array[0,:] = y0
    for i in range(1,n):
        y = y_array[i-1,:]
        t = t_array[i-1]
        h = t_array[i] - t_array[i-1]
        k1 = func(t, y, *args)
        k2 = func(t+h/2, y+h/2*k1, *args)
        k3 = func(t+h/2, y+h/2*k2, *args)
        k4 = func(t+h, y+h*k3, *args)
        ystep = y+h/6*(k1+2*k2+2*k3+k4)
        y_array[i,:] = ystep
    return OptimizeResult(t=t_array, y=y_array.T)

def dlorenz(t, X, a, b, c, a_eps, b_eps, c_eps):
    x, y, z = X
    a = (1+a_eps)*a
    b = (1+b_eps)*b
    c = (1+c_eps)*c
    return np.array([
        a*(y-x),
        x*(b-z)-y,
        x*y-c*z
        ])

def makeIndivWeightDF(weightMatrix, symbolicLabels):
    cols = [f'x_{i}' for i in range(weightMatrix.shape[1])]
    df = pd.DataFrame(weightMatrix, columns=cols)
    df['feature'] = symbolicLabels[:,0]
    df['feature_type'] = symbolicLabels[:,1]
    return df

def makeCombinedWeightDF(weightMatrix, symbolicLabels):
    feature_norms = np.array([np.linalg.norm(x) for x in weightMatrix])
    df = pd.DataFrame(feature_norms, columns=['feature_norm'])
    df['feature'] = symbolicLabels[:,0]
    df['feature_type'] = symbolicLabels[:,1]
    return df

def plotWeights(df):
    ax = df.plot(
        kind='bar',
        figsize=(25,8),
        xlabel='Feature',
        ylabel='Weight',
        title='W entries',
        width=0.85
        )
    ax.set_xticklabels(df['feature'])
    ax.legend()

def plotRecursiveComparison(recursiveOut, data, t0, t_forward, labels=None, figsize=(22,8)):
    if labels == None:
        labels = [f'x_{i}' for i in range(recursiveOut.shape[1])]
    fig, axs = plt.subplots(2, figsize=figsize)
    for i in range(len(labels)):
        axs[0].plot(np.arange(t0,t0+t_forward), recursiveOut[:t_forward,i], label=f'{labels[i]} prediction')
        axs[0].plot(np.arange(t0,t0+t_forward), data[t0:t0+t_forward,i], label=f'{labels[i]} target')
        axs[1].plot(np.arange(t0,t0+t_forward), np.abs(recursiveOut[:t_forward,i]-data[t0:t0+t_forward,i]), label=f'{labels[i]} error')
    axs[0].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    axs[1].legend(loc="right", ncol=1, bbox_to_anchor=(1.12,0.5))
    axs[0].set_title(f'Lorenz: recursive prediction vs. target [{t0},{t0+t_forward}]')
    plt.subplots_adjust(hspace=0.2)
    plt.show()

def plotTestComparison(testOut, testTarget, t0, t_forward, labels=None, figsize=(22,8)):
    if labels == None:
        labels = [f'x_{i}' for i in range(testOut.shape[1])]
    fig, axs = plt.subplots(2, figsize=figsize)
    for i in range(len(labels)):
        axs[0].plot(np.arange(t0,t0+t_forward), testOut[:t_forward,i], linewidth=0.75, label=f'{labels[i]} prediction')
        axs[0].plot(np.arange(t0,t0+t_forward), testTarget[t0:t0+t_forward,i], linewidth=0.75, label=f'{labels[i]} target')
        axs[1].plot(np.arange(t0,t0+t_forward), np.abs(testOut[:t_forward,i]-testTarget[t0:t0+t_forward,i]), linewidth=0.75, label=f'{labels[i]} error')
    axs[0].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    axs[1].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    axs[0].set_title(f'Lorenz: one-step ahead test vs. target [{t0},{t0+t_forward}]')
    plt.subplots_adjust(hspace=0.2)
    plt.show()