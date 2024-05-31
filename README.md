# Masters Thesis
Code for the Masters Thesis titled "Financed Informed Auto-Encoder Market Models" developed by FQL193 and FKG199

The code is structured as follows:

Preprocess.ipynb
  - module responsable for loading data, aswell as processing data for model training and testing.

Research.ipynb
  - module responsable for creating yield curves from swaps and vice versa. (This module only works for numpy not torch)

Mymodels.ipynb
  - module where all our models and the training functions of our models. 

Main.ipynb
  - module for calling training functions of models, also here you choose hyperparameters.

Postprocess.ipynb
  - module for postprocessing of data, loads a model a creates plots, loss averages etc.

All the code is written in Python 3.9 and is orchestrated in notebooks, i.e. .ipynb files. There is dependecy between the notebooks described above. 
The packages used are as follows:
- import import_ipynb
- import pandas as pd
- import matplotlib.pyplot as plt
- import numpy as np
- import torch 
- import torch.nn as nn 
- import torch.optim as optim 
- from torchvision import datasets
- from torchvision import transforms
- from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
- from datetime import datetime, timedelta
- from scipy.interpolate import CubicSpline
- from scipy.optimize import newton,least_squares
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- import os
- from torch.func import hessian
- import math
- from torch.func import jacfwd
- from torch.func import vmap, vjp


