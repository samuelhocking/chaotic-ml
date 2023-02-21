'''
Consolidated core model classes, methods, etc. for machine learning on chaotic systems
`cml_utils.py` contains ancillary utility objects, etc.
'''

import numpy as np
import sympy as sp
import pandas as pd
from cmlUtils import * 
from IPython.display import clear_output, display

# -----------------------------------------------------
# ........................ NVAR .......................
# -----------------------------------------------------

def quadraticCombination(x):
    return np.vstack([x[i]*x[i:,:] for i in range(len(x))])

def pureLinearFunc(data, k, s, idx):
    d = data.shape[1]
    return np.reshape(np.array(data[idx:idx-k*s:-s]), (k*d,1))

def hybridLinearFunc(data, k, s, idx, stepper):
    d = data.shape[1]
    y0 = data[idx]
    hybridstep = stepper.step(y0)
    purelin = np.reshape(np.array(data[idx:idx-k*s:-s]), (k*d,1))
    return np.vstack([np.array([stepper.step(y0)]).T,purelin])

class NVARModel():
    def __init__(self, k, s, reg, nonlinearFunc=None, extraNonlinearFunc=None, hybridCallable=None, natural_dt=None):
        self.k = k
        self.s = s
        self.reg = reg
        self.nonlinearFunc = nonlinearFunc
        self.extraNonlinearFunc = extraNonlinearFunc
        self.hybridCallable = hybridCallable
        self.natural_dt = natural_dt

    def make_NVAR_state_vector(self, data, idx):
        d = data.shape[1]
        pure = np.reshape(np.array(data[idx:idx-self.k*self.s:-self.s]), (self.k*d,1))
        if self.hybridCallable == None:
            lin = pure
        else:
            y0 = data[idx]
            hybridstep = self.hybridCallable.step(y0)
            lin = np.vstack([np.array([hybridstep]).T,pure])
        if self.nonlinearFunc != None and self.extraNonlinearFunc != None:
            nonlin = self.nonlinearFunc(lin)
            extranonlin = self.extraNonlinearFunc(np.vstack([lin, nonlin]))
            return np.vstack(
                [
                    1,
                    lin,
                    nonlin,
                    extranonlin
                ]
            )
        elif self.nonlinearFunc != None:
            nonlin = self.nonlinearFunc(lin)
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

    def make_NVAR_state_matrix(self, data, indices):
        return np.column_stack(
            [self.make_NVAR_state_vector(data, idx) for idx in indices]
        ).T

    # def train(self, data, target, train_indices):
    #     self.training_target = target[train_indices]
    #     self.state = self.make_NVAR_state_matrix(data=data, indices=train_indices)
    #     # self.w = np.linalg.lstsq(self.state @ self.state.T + self.reg * np.eye(self.state.shape[0]), self.state @ self.training_target, rcond=None)[0]
    #     self.w = np.linalg.lstsq(self.state.T @ self.state + self.reg * np.eye(self.state.shape[1]), self.state.T @ self.training_target, rcond=None)[0]

    def train(self, data, target, train_indices, dataLossFactor=1, ODEFunc=None, ODELossFactor=0, D_ODEFunc=None, D_ODELossFactor=0, printResults=True):
        ODE_indices = train_indices[1:-1] # central difference
        D_ODE_indices = train_indices[1:-1] # central difference
        self.training_target = target[train_indices]
        self.state = self.make_NVAR_state_matrix(data=data, indices=train_indices)
        len_h = self.state.shape[1]
        T = len(train_indices)
        d = data.shape[1]
        if ODEFunc != None:
            self.ODE_training_target = np.array([ODEFunc(x) for x in target[ODE_indices]]) # T-2 x d matrix
            self.D1 = firstCentralDiffMatrix(T, self.natural_dt) # T x T-2 matrix -> D1 @ ODE_training_target -> T x d matrix
            ODE_LHS_term = self.state.T @ self.D1 @ self.D1.T @ self.state # |h|x|h|
            ODE_RHS_term = self.D1 @ self.ODE_training_target # Txd
        else:
            self.ODE_training_target = None
            self.D1 = None
            ODE_LHS_term = np.zeros((len_h,len_h))
            ODE_RHS_term = np.zeros((T, d))
        if D_ODEFunc != None:
            self.D_ODE_training_target = np.array([D_ODEFunc(x) for x in target[D_ODE_indices]]) # T-2 x d matrix
            self.D2 = secondCentralDiffMatrix(T, self.natural_dt) # T x T-2 matrix -> D2 @ ODE_training_target -> T x d matrix
            D_ODE_LHS_term = self.state.T @ self.D2 @ self.D2.T @ self.state # |h|x|h|
            D_ODE_RHS_term = self.D2 @ self.D_ODE_training_target # Txd
        else:
            self.D_ODE_training_target = None
            self.D2 = None
            D_ODE_LHS_term = np.zeros((len_h,len_h))
            D_ODE_RHS_term = np.zeros((T, d))
        # all products on LHS are |h|x|h| where |h| is length of state vector
        # all products on RHS that multiply self.state.T on the right are Txd, and entire product is |h|xd -> looking for w |h|xd to multiply |h|x|h| on right
        self.w = np.linalg.lstsq(
            dataLossFactor*self.state.T @ self.state + ODELossFactor*ODE_LHS_term + D_ODELossFactor*D_ODE_LHS_term + self.reg * np.eye(len_h),
            self.state.T @ (dataLossFactor*self.training_target + ODELossFactor*ODE_RHS_term + D_ODELossFactor*D_ODE_RHS_term), rcond=None
            )[0]
        # self.w = np.linalg.lstsq(dataLossFactor*self.state.T @ self.state + ODELossFactor*self.state.T @ self.D @ self.D.T @ self.state + self.reg * np.eye(self.state.shape[1]), self.state.T @ (dataLossFactor*self.training_target + ODELossFactor*self.D @ self.ODE_training_target), rcond=None)[0]
        
        loss_info = []
        # -- Data
        self.training_pred_data = self.state @ self.w
        self.training_data_loss = NormSq(self.training_pred_data, self.training_target)
        self.training_data_MSE = MSE(self.training_pred_data, self.training_target)
        self.training_data_NRMSE = NRMSE(self.training_pred_data, self.training_target)
        self.weighted_training_data_loss = dataLossFactor * self.training_data_loss
        data_loss_info = {
            'Component' : 'Data',
            'MSE' : self.training_data_MSE,
            'NRMSE' : self.training_data_NRMSE,
            'Loss' : self.training_data_loss,
            'Weighted Loss' : self.weighted_training_data_loss
        }
        loss_info.append(data_loss_info)
        # -- Regularization
        self.w_norm_sq = np.linalg.norm(self.w)**2
        self.w_MSE = np.linalg.norm(self.w)**2/(self.w.size)
        self.w_NRMSE = np.linalg.norm(self.w)/np.sqrt(self.w.size)
        self.reg_penalty = self.reg*self.w_norm_sq
        reg_loss_info = {
            'Component' : 'Regularization',
            'MSE' : self.w_MSE,
            'NRMSE' : self.w_NRMSE,
            'Loss' : self.w_norm_sq,
            'Weighted Loss' : self.reg_penalty
        }
        ['Regularization', 'N/A', self.w_norm_sq, self.reg_penalty]
        # -- ODE
        if ODEFunc != None:
            self.training_pred_ODE = self.D1.T @ self.training_pred_data
            self.training_ODE_loss = NormSq(self.training_pred_ODE, self.ODE_training_target)
            self.training_ODE_MSE = MSE(self.training_pred_ODE, self.ODE_training_target)
            self.training_ODE_NRMSE = NRMSE(self.training_pred_ODE, self.ODE_training_target)
            self.weighted_training_ODE_loss = ODELossFactor * self.training_ODE_loss
            ODE_loss_info = {
            'Component' : 'ODE',
            'MSE' : self.training_ODE_MSE,
            'NRMSE' : self.training_ODE_NRMSE,
            'Loss' : self.training_ODE_loss,
            'Weighted Loss' : self.weighted_training_ODE_loss
            }
            loss_info.append(ODE_loss_info)
        else:
            self.training_pred_ODE = None
            self.training_ODE_loss = 0
            self.weighted_training_ODE_loss = 0
        # -- D_ODE
        if D_ODEFunc != None:
            self.training_pred_D_ODE = self.D2.T @ self.training_pred_data
            self.training_D_ODE_loss = NormSq(self.training_pred_D_ODE, self.D_ODE_training_target)
            self.training_D_ODE_MSE = MSE(self.training_pred_D_ODE, self.D_ODE_training_target)
            self.training_D_ODE_NRMSE = NRMSE(self.training_pred_D_ODE, self.D_ODE_training_target)
            self.weighted_training_D_ODE_loss = D_ODELossFactor * self.training_D_ODE_loss
            D_ODE_loss_info = {
            'Component' : "ODE'",
            'MSE' : self.training_D_ODE_MSE,
            'NRMSE' : self.training_D_ODE_NRMSE,
            'Loss' : self.training_D_ODE_loss,
            'Weighted Loss' : self.weighted_training_D_ODE_loss
            }
            loss_info.append(D_ODE_loss_info)
        else:
            self.training_pred_D_ODE = None
            self.training_D_ODE_loss = 0
            self.weighted_training_D_ODE_loss = 0
        loss_info.append(reg_loss_info)
        total_loss_info = {
            'Component' : 'Total',
            'MSE' : sum([x['MSE'] for x in loss_info]),
            'NRMSE' : sum([x['NRMSE'] for x in loss_info]),
            'Loss' : sum([x['Loss'] for x in loss_info]),
            'Weighted Loss' : sum([x['Weighted Loss'] for x in loss_info])
            }
        loss_info.append(total_loss_info)
        self.training_total_weighted_loss = total_loss_info['Weighted Loss']
        self.training_df = pd.DataFrame.from_records(loss_info)
        self.training_df.set_index('Component', inplace=True)
        if printResults:
            print(
                self.training_df.to_string(
                formatters= {
                    "MSE": "{:.6f}".format,
                    "NRMSE": "{:.6f}".format,
                    "Loss": "{:.6f}".format,
                    "Weighted Loss": "{:.6f}".format
                }
                )
            )
        # if printResults:
        #     print(f"Training results:")
        #     for x in loss_info:
        #         for y in x:
        #             if isinstance(y, str):
        #                 print(y.rjust(25), end='')
        #             else:
        #                 print(str('%.6f' % y).rjust(25), end='')
        #         print()

    def test(self, data, target, test_indices, printResults=True):
        self.test_target = target[test_indices]
        self.test_state = self.make_NVAR_state_matrix(data=data, indices=test_indices)
        self.test_out = self.test_state @ self.w
        self.test_MSE = MSE(self.test_out, self.test_target)
        self.test_RMSE = RMSE(self.test_out, self.test_target)
        self.test_NRMSE = NRMSE(self.test_out, self.test_target)

    def predict(self, data, indices):
        predict_state = self.make_NVAR_state_matrix(data=data, indices=indices)
        predict_out = predict_state @ self.w
        return predict_out
    
    def recursive_predict(self, data, start, end, t_forward):
        running_data = data[start:end,]
        recursive_out = []
        for i in range(t_forward):
            state_vect = self.make_NVAR_state_vector(data=running_data, idx=-1)
            y = state_vect.T @ self.w
            recursive_out.append(y[0])
            running_data = np.vstack((running_data, y))
        recursive_out = np.array(recursive_out)
        return recursive_out

def get_symbolic_state_labels(k, d, nonlinearFunc=None, extraNonlinearFunc=None, hybrid=False):
    bias = np.array([sp.symbols("1")])
    lin_entries = []
    if hybrid:
        for j in range(d):
            lin_entries.append(sp.symbols(f'H[x(t)_{j+1}]'))
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

# -----------------------------------------------------
# ......................... ESN .......................
# -----------------------------------------------------

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

def testRecursiveNVARParams(
    k,
    s_grid,
    reg_grid,
    data,
    target,
    train_start,
    train_end, 
    test_start,
    test_end,
    metricFunc,
    objective=np.argmin,
    nonlinearFunc=None,
    extranonlinearFunc=None,
    hybridCallable=None,
    returnFlag=False,
    ODEFunc=None,
    ODELossFactor=1,
    natural_dt=None
    ):

    test_target = target[test_start:test_end]
    train_indices = np.arange(train_start,train_end)

    model_arr = []
    metric_arr = []
    ctr = 0
    total_scens = len(s_grid)*len(reg_grid)
    for s in s_grid:
        for r in reg_grid:
            model = NVARModel(k, s, r, nonlinearFunc, extranonlinearFunc, hybridCallable, natural_dt)
            if ODEFunc == None:
                model.train(data, target, train_indices)
            else:
                model.trainWithODE(data, target, train_indices, ODEFunc, ODELossFactor)
            recursive_out = model.recursive_predict(data, train_start, train_end, test_end-test_start)
            metric = metricFunc(recursive_out,test_target)
            metric_arr.append(metric)
            model_arr.append(model)
            clear_output(wait=True)
            print(f'done: scen {ctr}/{total_scens-1} ({100*ctr/(total_scens-1):.02f}%) |  s={s} r={r} metric={metric}')
            ctr += 1
    metric_arr = np.array(metric_arr)
    best_idx = objective(metric_arr[~np.isnan(metric_arr)])
    best_model = [model_arr[i] for i in range(len(model_arr)) if ~np.isnan(metric_arr)[i]][best_idx]
    best_metric = [metric_arr[i] for i in range(len(metric_arr)) if ~np.isnan(metric_arr)[i]][best_idx]

    print(f'best recursive params:')
    print(f'k        :{best_model.k}')
    print(f's        :{best_model.s}')
    print(f'reg      :{best_model.reg}')
    print(f'objective:{best_metric}')

    if returnFlag:
        return {
            'k' : best_model.k,
            's' : best_model.s,
            'reg' : best_model.reg,
            'size' : best_model.w.shape[0],
            'metric' : best_metric
        }

