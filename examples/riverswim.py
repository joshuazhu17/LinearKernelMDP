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

# To change the experiment, only modify "states" or "river_gamma"
# Everything else can stay the same
river_states = 5
river_gamma = 0.5

new_river = RiverSwim(river_states)
new_river_phi = TwoActionPhi(river_states)
swim_UCLK = UCLK(new_river, river_states*2*river_states, 2, river_states, river_gamma, new_river_phi)
swim_UCLKNoConv = UCLKNoConv(new_river, river_states*2*river_states, 2, river_states, river_gamma, new_river_phi)

swim_UCLK.train(200)
swim_UCLKNoConv.train(200)
