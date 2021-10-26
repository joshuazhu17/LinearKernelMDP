import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import copy
import random
import cvxpy as cp
from random import randrange
import time
import scipy as sp
from abc import ABC, abstractmethod
import scipy.linalg as la

from linearkernelmdp import algorithms, envs, featuremaps as *

# To change the experiment, only modify "quarter_leaves" or "tree_gamma"
# Everything else can stay the same
quarter_leaves = 1
tree_gamma = 0.5

tree_states = 3 + quarter_leaves*4

new_tree = WideTreeInfinite(quarter_leaves)
new_tree_phi = TwoActionPhi(tree_states)
tree_UCLK = UCLK(new_tree, tree_states*2*tree_states, 2, tree_states, tree_gamma, new_tree_phi)
tree_UCLKNoConv = UCLKNoConv(new_tree, tree_states*2*tree_states, 2, tree_states, tree_gamma, new_tree_phi)

tree_UCLK.train(200)
tree_UCLKNoConv.train(200)
