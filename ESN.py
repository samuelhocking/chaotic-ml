# Echo State Network Classes & Methods

import numpy as np

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
        # self.state = np.zeros(self.internalUnits)
    
    def makeInputMatrix(self):
        return self.inputConnGen(self.internalUnits, self.inputUnits)
        
    def makeInternalMatrix(self):
        A = self.internalConnGen(self.internalUnits, self.internalUnits)
        vals, vects = np.linalg.eig(A)
        maxeig = max(abs(vals))
        return self.spectralRadius/maxeig*A

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