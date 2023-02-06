'''
Consolidated core model classes, methods, etc. for machine learning on chaotic systems
`cml_utils.py` contains ancillary utility objects, etc.
'''

import numpy as np
import sympy as sp
import pandas as pd
from cmlUtils import RMSE
from IPython.display import clear_output, display

def get_symbolic_state_labels(k, d, nonlinearFunc=None, extraNonlinearFunc=None):
    bias = np.array([sp.symbols("1")])
    lin_entries = []
    for i in range(k):
        for j in range(d):
            if i == 0:
                lin_entries.append(sp.symbols(f'x(t)_{j+1}'))
            elif i == 1:
                lin_entries.append(sp.symbols(f'x(t-s)_{j+1}'))
            else:
                lin_entries.append(sp.symbols(f'x(t-{i}s)_{j+1}'))
    linarray = np.array(lin_entries)
    lin = np.reshape(linarray, (linarray.shape[0],1))
    zipped = []
    if nonlinearFunc != None and extraNonlinearFunc != None:
        nonlin = nonlinearFunc(lin)
        extranonlin = extraNonlinearFunc(np.vstack([lin, nonlin]))
        arr = np.vstack(
            [
                1,
                lin,
                nonlin,
                extranonlin
            ]
        )
        for i in range(len(arr)):
            if i < 1:
                zipped.append([arr[i,0], 'bias'])
            elif i < 1 + len(lin):
                zipped.append([arr[i,0], 'linear']) 
            elif i < 1 + len(lin) + len(nonlin):
                zipped.append([arr[i,0], 'nonlinear'])
            elif i < 1 + len(lin) + len(nonlin) + len(extranonlin):
                zipped.append([arr[i,0], 'extra-nonlinear'])
        zipped = np.array(zipped)
        return zipped
    elif nonlinearFunc != None:
        nonlin = nonlinearFunc(lin)
        arr = np.vstack(
            [
                1,
                lin,
                nonlin
            ]
        )
        for i in range(len(arr)):
            if i < 1:
                zipped.append([arr[i,0], 'bias'])
            elif i < 1 + len(lin):
                zipped.append([arr[i,0], 'linear']) 
            elif i < 1 + len(lin) + len(nonlin):
                zipped.append([arr[i,0], 'nonlinear'])
        zipped = np.array(zipped)
        return zipped
    else:
        arr = np.vstack(
            [
                1,
                lin
            ]
        )
        for i in range(len(arr)):
            if i < 1:
                zipped.append([arr[i,0], 'bias'])
            elif i < 1 + len(lin):
                zipped.append([arr[i,0], 'linear'])
        zipped = np.array(zipped)
        return zipped

def get_norm_df(symb_state_arr, model):
    dim = model.w.shape[1]
    dict_list = []
    for i in range(dim):
        d = {x : np.linalg.norm(model.w[symb_state_arr[:,1] == x,i])**2 for x in np.unique(symb_state_arr[:,1])}
        d['total'] = np.linalg.norm(model.w[:,i])**2
        dict_list.append(d.copy())
    d = {x : np.linalg.norm(model.w[symb_state_arr[:,1] == x,:])**2 for x in np.unique(symb_state_arr[:,1])}
    d['total'] = np.linalg.norm(model.w)**2
    dict_list.append(d)
    df = pd.DataFrame(
        dict_list,
        index = [f'x_{i+1}' for i in range(dim)] + ['total'],
        # index = ['total', 'x_1', 'x_2', 'x_3'],
        columns=list(np.unique(symb_state_arr[:,1])) + ['total']
        ).transpose()
    return df

def nvar_rows(d, k):
    return int(1 + k*d + k*d*(k*d + 1)/2)

def quadraticCombination(x):
    return np.vstack([x[i]*x[i:,:] for i in range(len(x))])

def pureLinearFunc(data, k, s, idx):
    d = data.shape[1]
    return np.reshape(np.array(data[idx:idx-k*s:-s]), (k*d,1))

# def hybridLinearFunc()
#     pass

# class HybridIntegrator():
#     def __init__(self, )

def make_NVAR_state_vector(data, k, s, idx, linearFunc=pureLinearFunc, nonlinearFunc=None, extraNonlinearFunc=None):
    # nonlinearFunc should:
    #   accept a single argument of the linear portion of the state vector
    #   and return a numpy array
    lin = linearFunc(data, k, s, idx)
    if nonlinearFunc != None and extraNonlinearFunc != None:
        nonlin = nonlinearFunc(lin)
        extranonlin = extraNonlinearFunc(np.vstack([lin, nonlin]))
        return np.vstack(
            [
                1,
                lin,
                nonlin,
                extranonlin
            ]
        )
    elif nonlinearFunc != None:
        nonlin = nonlinearFunc(lin)
        return np.vstack(
            [
                1,
                lin,
                nonlin
            ]
        )
    else:
        return np.vstack(
            [
                1,
                lin
            ]
        )

def make_NVAR_state_matrix(data, k, s, indices, linearFunc=pureLinearFunc, nonlinearFunc=None, extraNonlinearFunc=None):
    return np.column_stack(
        [make_NVAR_state_vector(data, k, s, idx, linearFunc, nonlinearFunc, extraNonlinearFunc) for idx in indices]
    )

class NVARModel():
    def __init__(self, k, s, reg, linearFunc=pureLinearFunc, nonlinearFunc=None, extraNonlinearFunc=None):
        self.k = k
        self.s = s
        self.reg = reg
        self.linearFunc = linearFunc
        self.nonlinearFunc = nonlinearFunc
        self.extraNonlinearFunc = extraNonlinearFunc

    def train(self, data, target, train_indices):
        self.training_target = target[train_indices]
        self.state = make_NVAR_state_matrix(data=data, k=self.k, s=self.s, indices=train_indices, linearFunc=self.linearFunc, nonlinearFunc=self.nonlinearFunc, extraNonlinearFunc=self.extraNonlinearFunc)
        # self.w = np.linalg.lstsq(self.state.dot(self.state.T) + self.reg * np.eye(self.state.shape[0]), self.state.dot(self.training_target), rcond=None)[0]
        self.w = np.linalg.lstsq(self.state @ self.state.T + self.reg * np.eye(self.state.shape[0]), self.state @ self.training_target, rcond=None)[0]

    def evaluate(self, data, target, test_indices):
        self.test_target = target[test_indices]
        self.test_state = make_NVAR_state_matrix(data=data, k=self.k, s=self.s, indices=test_indices, linearFunc=self.linearFunc, nonlinearFunc=self.nonlinearFunc, extraNonlinearFunc=self.extraNonlinearFunc)
        self.test_out = self.test_state.T @ self.w
        self.test_RMSE = RMSE(self.test_out, self.test_target)
    
    def recursive_predict(self, data, start, end, t_forward):
        running_data = data[start:end,]
        recursive_out = []
        for i in range(t_forward):
            state_vect = make_NVAR_state_vector(data=running_data, k=self.k, s=self.s, idx=-1, nonlinearFunc=self.nonlinearFunc, extraNonlinearFunc=self.extraNonlinearFunc)
            y = state_vect.T @ self.w
            recursive_out.append(y[0])
            running_data = np.vstack((running_data, y))
        recursive_out = np.array(recursive_out)
        return recursive_out

class ESNModel():
    def __init__(self, inputUnits, inputConnGen, internalUnits, internalConnGen, outputUnits, leakRate, spectralRadius, regularization, activation=np.tanh):
        self.inputUnits = inputUnits
        self.inputConnGen = inputConnGen
        self.internalUnits = internalUnits
        self.internalConnGen = internalConnGen
        self.outputUnits = outputUnits
        self.leakRate = leakRate
        self.spectralRadius = spectralRadius
        self.regularization = regularization
        self.activation = activation
        self.inputRandomState = np.random.get_state()
        self.W_in = self.makeInputMatrix()
        self.internalRandomState = np.random.get_state()
        self.W = self.makeInternalMatrix()
    
    def makeInputMatrix(self):
        return self.inputConnGen(self.internalUnits, self.inputUnits)
        
    def makeInternalMatrix(self):
        maxeig = 0
        while maxeig == 0:
            self.A = self.internalConnGen(self.internalUnits, self.internalUnits)
            vals, vects = np.linalg.eig(self.A)
            maxeig = max(abs(vals))
        return self.spectralRadius/maxeig*self.A

    def train(self, data, target, train_indices):
        # should scan during pre-training and record pre-train inputs states into pre-train history
        # compute when to switch into active training and start recording inputs and states into training history
        # x(n+1) = (1-a)x(n) + a*f(W_in u(n+1) + W x(n))
        state = np.zeros(self.internalUnits)
        pretrain_indices = np.delete(np.arange(train_indices[-1]+1), train_indices)
        self.pretrainInputs = data[pretrain_indices]
        self.trainingInputs = data[train_indices]
        self.trainingTarget = target[train_indices]
        self.pretrainStates = np.zeros((train_indices[0],self.internalUnits))
        self.trainingStates = np.zeros((len(train_indices),self.internalUnits))
        for i in range(train_indices[-1]):
            state = (1-self.leakRate)*state + self.leakRate*self.activation(self.W_in @ data[i] + self.W @ state)
            if i < train_indices[0]:
                # pre-training
                self.pretrainStates[i] = state
            else:
                # training
                self.trainingStates[i-train_indices[0]] = state
        self.trainingConcat = np.column_stack([self.trainingInputs, self.trainingStates])
        self.W_out = np.linalg.lstsq(self.trainingConcat.T @ self.trainingConcat + self.regularization * np.eye(self.trainingConcat.shape[1]), self.trainingConcat.T @ self.trainingTarget, rcond=None)[0].T

    def recursive_predict(self, seedData, t_forward):
        recursive_out = np.zeros((t_forward, self.outputUnits))
        prePredictionStates = np.zeros((len(seedData)-1,self.internalUnits))
        predictionStates = np.zeros((t_forward, self.internalUnits))
        state = np.zeros(self.internalUnits)
        # run through seedData to achieve a running state
        for i in range(len(seedData)-1):
            state = (1-self.leakRate)*state + self.leakRate*self.activation(self.W_in @ seedData[i] + self.W @ state)
            prePredictionStates[i] = state
        input = seedData[-1]
        for j in range(t_forward):
            state = (1-self.leakRate)*state + self.leakRate*self.activation(self.W_in @ input + self.W @ state)
            predictionStates[j] = state
            y = self.W_out @ np.concatenate((input, state))
            recursive_out[j] = y
            input = y
        return recursive_out

def testRecursiveNVARParams(k, s_grid, reg_grid, data, target, train_start, train_end, test_start, test_end, linearFunc, nonlinearFunc, extranonlinearFunc):

    test_target = target[test_start:test_end]
    train_indices = np.arange(train_start,train_end)

    model_arr = []
    recursive_rmse_arr = []
    ctr = 0
    total_scens = len(s_grid)*len(reg_grid)
    for s in s_grid:
        for r in reg_grid:
            clear_output(wait=True)
            print(f'done: scen {ctr}/{total_scens-1} ({100*ctr/(total_scens-1):.02f}%) |  s={s} r={r}')
            model = NVARModel(k, s, r, linearFunc, nonlinearFunc, extranonlinearFunc)
            model.train(data, target, train_indices)
            recursive_rmse_arr.append(RMSE(test_target,model.recursive_predict(data, train_start, train_end, test_end-test_start)))
            model_arr.append(model)
            ctr += 1
    recursive_rmse_arr = np.array(recursive_rmse_arr)
    best_recursive_idx = np.argmin(recursive_rmse_arr[~np.isnan(recursive_rmse_arr)])
    best_recursive_model = [model_arr[i] for i in range(len(model_arr)) if ~np.isnan(recursive_rmse_arr)[i]][best_recursive_idx]

    print(f'best recursive params:')
    print(f'k  :{best_recursive_model.k}')
    print(f's  :{best_recursive_model.s}')
    print(f'reg:{best_recursive_model.reg}')