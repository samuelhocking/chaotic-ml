'''
Consolidated utility methods and classes for machine learning on chaotic systems
`cml_core.py` contains the core components, models, etc.
'''

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from math import factorial

# -----------------------------------------------------
# ..................... UTILITIES .....................
# -----------------------------------------------------

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
    
def backDiffMatrix(T, dt):
    D = np.zeros((T,T-1))
    for i in range(T-1):
        D[i,i] = -1/dt
        D[i+1,i] = 1/dt
    return D

def firstCentralDiffMatrix(T, dt):
    D = np.zeros((T,T-2))
    for i in range(T-2):
        D[i,i] = -1/(2*dt)
        D[i+2,i] = 1/(2*dt)
    return D

def secondCentralDiffMatrix(T, dt):
    D = np.zeros((T,T-2))
    for i in range(T-2):
        D[i,i] = 1/dt**2
        D[i+1,i] = -2/(dt**2)
        D[i+2,i] = 1/dt**2
    return D

def n_choose_k(n,k):
    return factorial(n)/(factorial(n-k)*factorial(k))

def n_multichoose_k(n,k):
    return n_choose_k(n+k-1,k)

def n_multichoose_l_levels(n,l):
    return sum([n_multichoose_k(n,k) for k in range(1,l+1)])

def basisVectRn(i,n):
    e = np.zeros(n)
    e[i] = 1
    return e

# -----------------------------------------------------
# ....................... METRICS .....................
# -----------------------------------------------------

def NormSq(prediction, target):
    return (np.linalg.norm(prediction - target)**2)

def MSE(prediction, target):
    n = len(prediction)
    return (np.linalg.norm(prediction - target)**2)/n

def RMSE(prediction, target):
    '''
    root mean squared error (model evaluation metric)
    '''
    n = len(prediction)
    return np.linalg.norm(prediction - target)/np.sqrt(n)

def NRMSE(prediction, target):
    '''
    normalized root mean squared error (model evaluation metric)
    '''
    maxnorm = max(np.array([np.linalg.norm(x) for x in target]))
    minnorm = min(np.array([np.linalg.norm(x) for x in target]))
    return RMSE(prediction, target)/(maxnorm-minnorm)

def meanNorm(target):
    n = len(target)
    return sum(np.array([np.linalg.norm(x) for x in target]))/n

def meanNormSq(target):
    n = len(target)
    return np.linalg.norm(target)**2/n

def NMSE(prediction, target):
    return MSE(prediction,target)/meanNormSq(target)

# closure!
# needs to be re-worked since norm of denominator could well be zero
def validTime(threshold):
    threshold = threshold
    def inner_func(prediction, target):
        valid_idx = len(target)
        arr = [np.linalg.norm(prediction[i]-target[i])/np.linalg.norm(target[i]) for i in range(len(target))]
        for i in range(len(arr)):
            if arr[i] >= threshold:
                valid_idx = i
                break
        return valid_idx
    return inner_func

# -----------------------------------------------------
# ................... INTEGRATION .....................
# -----------------------------------------------------

def RK4(func, y0, t_array, args=None, method=None, max_step=None):
    # method and max_step are unused dummy variables to facilitate compatibility with solve_ivp calls
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

def euler(func, x0, tf, dt, out_dt=None):
    pers = int((tf/dt))
    tt = dt*np.arange(0, pers+1)
    out = np.zeros((pers+1, len(x0)))
    out[0] = x0
    for i in range(0,pers):
        out[i+1] = out[i] + dt*func(tt[i],out[i])
    if out_dt != None:
        s = int(out_dt/dt)
        tt = tt[::s]
        out = out[::s]
    return OptimizeResult(t=tt, y=out.T)

def dlorenz(t, X, a, b, c, a_eps, b_eps, c_eps):
    # t is an unused dummy variable to facilitate compatibility with solve_ivp calls
    x, y, z = X
    a = (1+a_eps)*a
    b = (1+b_eps)*b
    c = (1+c_eps)*c
    return np.array([
        a*(y-x),
        x*(b-z)-y,
        x*y-c*z
        ])

def simple_dlorenz(a=10, b=28, c=8/3):
    a = a
    b = b
    c = c
    def innerFunc(X):
        x, y, z = X
        return np.array([
            a*(y-x),
            x*(b-z)-y,
            x*y-c*z
        ])
    return innerFunc

def simple_ddlorenz(a=10, b=28, c=8/3):
    a = a
    b = b
    c = c
    def innerFunc(X):
        x, y, z = X
        return np.array([
            a*(x*(b-z)-y)-a**2*(y-x),
            a*b*(y-x)-a*z*(y-x)-x*(x*y-c*z)-(x*(b-z)-y),
            a*y*(y-x)+x*(x*(b-z)-y)-c*(x*y-c*z)
        ])
    return innerFunc

# closure!
def solveIVPCompatibilityFunc(method, max_step):
    '''
    compatible callable with same input space as RK4, for use in CompatibleStepper class for hybrid prediction
    '''
    method = method
    max_step = max_step
    def inner_func(func, y0, t_array, args):
        return solve_ivp(
            func,
            (t_array[0], t_array[-1]+(t_array[-1]-t_array[-2])),
            y0,
            method=method,
            t_eval=t_array,
            max_step=max_step,
            args=args,
            vectorized=True
        )
    return inner_func

class CompatibleStepper():
    def __init__(self, func, args, callable, dt):
        self.func = func
        self.args = args
        self.callable = callable
        self.dt = dt
    def step(self, y0):
        sol = self.callable(self.func, y0, np.array([0, self.dt]), self.args)
        return sol.y[:,1]

# -----------------------------------------------------
# ..................... ANALYSIS ......................
# -----------------------------------------------------

def makeIndivWeightDF(weightMatrix, symbolicLabels):
    cols = [f'x_{i}' for i in range(weightMatrix.shape[1])]
    df = pd.DataFrame(weightMatrix, columns=cols)
    if len(symbolicLabels.shape) == 1:
        df['feature'] = symbolicLabels
    else:
        df['feature'] = symbolicLabels[:,0]
    if len(symbolicLabels.shape)>1 and symbolicLabels.shape[1]>0:
        df['feature_type'] = symbolicLabels[:,1]
    return df

def makeCombinedWeightDF(weightMatrix, symbolicLabels):
    feature_norms = np.array([np.linalg.norm(x) for x in weightMatrix])
    df = pd.DataFrame(feature_norms, columns=['feature_norm'])
    if len(symbolicLabels.shape) == 1:
        df['feature'] = symbolicLabels
    else:
        df['feature'] = symbolicLabels[:,0]
    if len(symbolicLabels.shape)>1 and symbolicLabels.shape[1]>0:
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

def plotRecursiveComparison(recursiveOut, data, t0, t_forward, labels=None, figsize=(22,8), vline_x=None):
    if labels == None:
        labels = [f'x_{i}' for i in range(recursiveOut.shape[1])]
    fig, axs = plt.subplots(2, figsize=figsize)
    for i in range(len(labels)):
        axs[0].plot(np.arange(t0,t0+t_forward), recursiveOut[:t_forward,i], label=f'{labels[i]} prediction')
        axs[0].plot(np.arange(t0,t0+t_forward), data[t0:t0+t_forward,i], label=f'{labels[i]} target')
        axs[1].plot(np.arange(t0,t0+t_forward), np.abs(recursiveOut[:t_forward,i]-data[t0:t0+t_forward,i]), label=f'{labels[i]} error')
    axs[0].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    axs[1].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    if vline_x:
        axs[0].axvline(vline_x, color='r', linestyle='dashed')
        axs[1].axvline(vline_x, color='r', linestyle='dashed')
    axs[0].set_title(f'Recursive prediction vs. target [{t0},{t0+t_forward}]')
    axs[1].set_title(f'Coordinate-wise |error|')
    fig.tight_layout()
    plt.show()

def plotEnhancedRecursiveComparison(recursiveOut, data, t0, t_forward, labels=None, figsize=(22,12), vline_x=None):
    if labels == None:
        labels = [f'x_{i}' for i in range(recursiveOut.shape[1])]
    fig, axs = plt.subplots(3, figsize=figsize)
    for i in range(len(labels)):
        axs[0].plot(np.arange(t0,t0+t_forward), recursiveOut[:t_forward,i], label=f'{labels[i]} prediction')
        axs[0].plot(np.arange(t0,t0+t_forward), data[t0:t0+t_forward,i], label=f'{labels[i]} target')
        axs[1].plot(np.arange(t0,t0+t_forward), np.abs(recursiveOut[:t_forward,i]-data[t0:t0+t_forward,i]), label=f'{labels[i]} error')
    axs[2].plot(np.arange(t0,t0+t_forward), [np.linalg.norm(recursiveOut[i]-data[t0+i])/np.linalg.norm(data[t0+i]) for i in range(t_forward)], label=f'error norm/data norm')
    axs[0].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    axs[1].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    if vline_x:
        axs[0].axvline(vline_x, color='r', linestyle='dashed')
        axs[1].axvline(vline_x, color='r', linestyle='dashed')
        axs[2].axvline(vline_x, color='r', linestyle='dashed')
    axs[0].set_title(f'Recursive prediction vs. target [{t0},{t0+t_forward}]')
    axs[1].set_title(f'Coordinate-wise |error|')
    axs[2].set_title('Total ||error||/||x||')
    fig.tight_layout()
    plt.show()

def plotTestComparison(testOut, testTarget, t0, t_forward, labels=None, figsize=(22,8), vline_x=None):
    if labels == None:
        labels = [f'x_{i}' for i in range(testOut.shape[1])]
    fig, axs = plt.subplots(2, figsize=figsize)
    for i in range(len(labels)):
        axs[0].plot(np.arange(t0,t0+t_forward), testOut[:t_forward,i], linewidth=0.75, label=f'{labels[i]} prediction')
        axs[0].plot(np.arange(t0,t0+t_forward), testTarget[t0:t0+t_forward,i], linewidth=0.75, label=f'{labels[i]} target')
        axs[1].plot(np.arange(t0,t0+t_forward), np.abs(testOut[:t_forward,i]-testTarget[t0:t0+t_forward,i]), linewidth=0.75, label=f'{labels[i]} error')
    axs[0].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    axs[1].legend(loc="right", ncol=1, bbox_to_anchor=(1.1,0.5))
    if vline_x:
        axs[0].axvline(vline_x, color='r', linestyle='dashed')
        axs[1].axvline(vline_x, color='r', linestyle='dashed')
    axs[0].set_title(f'One-step ahead test vs. target [{t0},{t0+t_forward}]')
    axs[1].set_title(f'Coordinate-wise |error|')
    fig.tight_layout()
    plt.show()
