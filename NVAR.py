# NVAR Classes & Methods

import numpy as np
from utils import *

def nvar_rows(d, k):
    return int(1 + k*d + k*d*(k*d + 1)/2)

def quadraticCombination(x):
    return np.vstack([x[i]*x[i:,:] for i in range(len(x))])

def make_NVAR_state_vector(data, k, s, idx, nonlinearFunc=None, extraNonlinearFunc=None):
    # nonlinearFunc should:
    #   accept a single argument of the linear portion of the state vector
    #   and return a numpy array
    d = data.shape[1]
    lin = np.reshape(np.array(data[idx:idx-k*s:-s]), (k*d,1))
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

def make_NVAR_state_matrix(data, k, s, indices, nonlinearFunc=None, extraNonlinearFunc=None):
    return np.column_stack(
        [make_NVAR_state_vector(data, k, s, idx, nonlinearFunc, extraNonlinearFunc) for idx in indices]
    )

class NVARModel():
    def __init__(self, k, s, reg, nonlinearFunc=None, extraNonlinearFunc=None):
        self.k = k
        self.s = s
        self.reg = reg
        self.nonlinearFunc = nonlinearFunc
        self.extraNonlinearFunc = extraNonlinearFunc

    def train(self, data, target, train_indices):
        self.training_target = target[train_indices]
        self.state = make_NVAR_state_matrix(data=data, k=self.k, s=self.s, indices=train_indices, nonlinearFunc=self.nonlinearFunc, extraNonlinearFunc=self.extraNonlinearFunc)
        # self.w = np.linalg.lstsq(self.state.dot(self.state.T) + self.reg * np.eye(self.state.shape[0]), self.state.dot(self.training_target), rcond=None)[0]
        self.w = np.linalg.lstsq(self.state @ self.state.T + self.reg * np.eye(self.state.shape[0]), self.state @ self.training_target, rcond=None)[0]

    def evaluate(self, data, target, test_indices):
        self.test_target = target[test_indices]
        self.test_state = make_NVAR_state_matrix(data=data, k=self.k, s=self.s, indices=test_indices, nonlinearFunc=self.nonlinearFunc, extraNonlinearFunc=self.extraNonlinearFunc)
        self.test_out = self.test_state.T @ self.w
        return RMSE(self.test_out, self.test_target)
    
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