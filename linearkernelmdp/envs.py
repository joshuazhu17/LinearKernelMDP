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

class Env(ABC):

    @abstractmethod
    def transition_prob(self, state, action, next_state):
        pass

    @abstractmethod
    def reward(self, state, action):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

class LinearKernelMDPEnv(Env):
    def __init__(self, d, A, S, gamma, theta):
        self.dim = d
        self.num_actions = A
        self.num_states = S
        self.gamma = gamma
        self.theta = theta

    # this is the phi function
    @abstractmethod
    def embed(self, state, action, next_state):
        pass

    # this is the phiv function
    @abstractmethod
    def weighted_value_embedding(self, state, action, values):
        pass

class DongRuoToy(LinearKernelMDPEnv):
    def __init__(self, d, A, S, gamma, theta):
        super().__init__(d, A, S, gamma, theta)
        self.delta = 1 - gamma
        self.cur_state = 0

    def trans_action(self, a, d):###d is dimension, dimension of a is d-1
        tt = np.zeros(d-1)-1
        bb = bin(a)
        for i in range(len(bb)-2):
            tt[i] = 2*np.double(bb[i+2])-1
        return(tt)

    def embed(self, state, action, next_state):
        tt = np.zeros(self.dim)
        aa = self.trans_action(action,self.dim)
        if state==0:
            if next_state == 0:
                for i in range(self.dim-1):
                    tt[i] = -aa[i]
                tt[self.dim-1] = 1-self.delta
        if state==0:
            if next_state == 1:
                for i in range(self.dim-1):
                    tt[i] = aa[i]
                tt[self.dim-1] = self.delta
        if state==1:
            if next_state == 0:
                for i in range(self.dim-1):
                    tt[i] = 0
                tt[self.dim-1] = self.delta
        if state==1:
            if next_state == 1:
                for i in range(self.dim-1):
                    tt[i] = 0
                tt[self.dim-1] = 1-self.delta
        return tt

    def weighted_value_embedding(self, state, action, values):
        return (self.embed(state,action,0)*values[0] + self.embed(state,action,1)*values[1])

    def transition_prob(self, state, action, next_state):
        return np.dot(self.embed(state,action,next_state), self.theta)

    def reward(self, state, action):
        return state

    def step(self, action):
        self.cur_state = np.random.binomial(1, self.transition_prob(self.cur_state, action, 1))
        return self.cur_state

    def reset(self):
        self.cur_state = 0

class RiverSwim(Env):
    def __init__(self, S, start_reward = 0.005, end_reward = 1, seed = None):
        self.S = S
        self.start_reward = start_reward
        self.end_reward = end_reward

        # The states are numbered 0 through S-1
        self.cur_state = 0

        # The actions are 0 (left) and 1 (right)
        self.num_actions = 2

        self.rng = np.random.default_rng(seed = seed)

    def transition_prob(self, state, action, next_state):
        # Assumes S >= 2

        if state == 0:
            if next_state == 0:
                if action == 0:
                    return 1
                else:
                    return 0.1
            elif next_state == state + 1:
                if action == 1:
                    return 0.9
                else:
                    return 0

        elif state == self.S-1:
            if next_state == self.S-1:
                if action == 1:
                    return 0.95
                else:
                    return 0
            elif next_state == self.S-2:
                if action == 0:
                    return 1
                else:
                    return 0.05

        else:
            if next_state == state:
                if action == 1:
                    return 0.05
                else:
                    return 0
            elif next_state == state - 1:
                if action == 0:
                    return 1
                else:
                    return 0.05
            elif next_state == state + 1:
                if action == 1:
                    return 0.9
                else:
                    return 0

        return 0


    def reward(self, state, action):
        # The original River Run seems to give a reward for landing on a state, not performing an action
        # I've changed it so that taking an action from the right states gives the reward

        if state == 0:
            return self.start_reward
        elif state == self.S-1:
            return self.end_reward
        else:
            return 0

    def step(self, action):
        num = self.rng.random()

        if self.cur_state == 0:
            if action == 0:
                pass
            else:
                if num <= 0.9:
                    self.cur_state = self.cur_state + 1

        elif self.cur_state == self.S-1:
            if action == 0:
                self.cur_state = self.cur_state - 1
            else:
                if num >= 0.95:
                    self.cur_state = self.cur_state - 1

        else:
            if action == 0:
                self.cur_state = self.cur_state - 1
            else:
                if num <= 0.9:
                    self.cur_state = self.cur_state + 1
                elif num >= 0.95:
                    self.cur_state = self.cur_state - 1

        return self.cur_state

    def reset(self):
        self.cur_state = 0

class WideTreeInfinite(Env):
    def __init__(self, quarter_leaves, reward_val = 1, seed = None):
        self.quarter_leaves = quarter_leaves
        self.reward_val = reward_val

        # There are self.quarter_leaves*4 + 3 states
        # State 0 is the starting state, with 2 children
        # States 1 and 2 both have self.quarter_leaves*2 leaves
        # All of state 2's leaves give a reward, all of state 1's leaves give nothing
        self.cur_state = 0

        # Left or right, per say
        # From state 0, action is deterministic, 0 goes to state 1 and 1 goes to state 2
        # From states 1 or 2, action determines which quarter_leaves the next state is, then probability is even among the leaves
        # From a leaf, any action takes back to state 0
        self.num_actions = 2

        self.rng = np.random.default_rng(seed = seed)

    def transition_prob(self, state, action, next_state):
        pass

    def reward(self, state, action):
        if state >= (3 + self.quarter_leaves*2):
            return self.reward_val
        else:
            return 0

    def step(self, action):
        if self.cur_state == 0:
            if action == 0:
                self.cur_state = 1
            else:
                self.cur_state = 2
        elif self.cur_state == 1:
            next = self.rng.integers(0, self.quarter_leaves)
            if action == 0:
                self.cur_state = 3 + next
            else:
                self.cur_state = 3 + self.quarter_leaves + next
        elif self.cur_state == 2:
            next = self.rng.integers(0, self.quarter_leaves)
            if action == 0:
                self.cur_state = 3 + self.quarter_leaves*2 + next
            else:
                self.cur_state = 3 + self.quarter_leaves*3 + next
        else:
            self.cur_state = 0

        return self.cur_state

    def reset(self):
        self.cur_state = 0
