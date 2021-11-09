import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import copy
import random
import cvxpy as cp
from random import randrange
import time
from abc import ABC, abstractmethod
import scipy.linalg as la

from linearkernelmdp import algorithms, envs, featuremaps as *

# To change the experiment, only modify "dongruo_dim" or "gamma"
# Everything else can stay the same
dongruo_dim = 8
gamma = 0.5

ACTION = np.power(2,dongruo_dim-1)
STATE = 2
thetastar = np.append(0.05*np.ones(dongruo_dim-1), 1)

new_dongruotoy = DongRuoToy(dongruo_dim, ACTION, STATE, gamma, thetastar)
new_dongruophi = DongRuoPhi(dongruo_dim, gamma)
dongruo_UCLKNoConv = UCLKNoConv(new_dongruotoy, dongruo_dim, ACTION, STATE, gamma, new_dongruophi)

dongruo_UCLKNoConv.train(200)
