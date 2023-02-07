# Machine Learning for Chaotic Dynamical Systems

## Overview

This repo contains classes, methods, functions, and notebooks to study machine learning techniques as applied to chaotic dynamical systems. The primary focus is on reservoir computing (e.g. Echo State Networks, "ESNs", and their derivatives) and adjacent methods (e.g. Nonlinear Vector Autoregression, "NVAR".

While the repo's contents are very much in flux, a basic roadmap is below:

- `cmlCore.py`: consolidated core ML models
- `cmlUtils.py`: consolidated ancillary tools
- `Lorenz-NVAR.ipynb`: notebook examining various aspects of predicting the Lorenz attractor with NVAR models
- `chaotic_datasets.ipynb`: notebook to assist generation of Mackey-Glass and Lorenz time-series

Miscellaneous others:
- `Gated-RNN-Prediction.ipynb`: examination of the Mackey-Glass and Lorenz systems gated recurrent neural networks, specifically Long Short-term Memory (LSTM)
- `Matrix-iterative-NVAR.ipynb`: experiments and scratchwork seeking closed-form matrix expressions for recursive NVAR prediction 

