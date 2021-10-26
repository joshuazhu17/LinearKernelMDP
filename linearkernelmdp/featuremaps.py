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

def DongRuoPhi(d, gamma):
    def outputphi(state, action, next_state):
        tt = np.zeros(d)
        aa = np.zeros(d-1)-1
        bbb = bin(a)
        for i in range(len(bbb)-2):
            aa[i] = 2*np.double(bbb[i+2])-1

        if state==0:
            if next_state == 0:
                for i in range(d-1):
                    tt[i] = -aa[i]
                tt[d-1] = gamma
        if state==0:
            if next_state == 1:
                for i in range(d-1):
                    tt[i] = aa[i]
                tt[d-1] = 1-gamma
        if state==1:
            if next_state == 0:
                for i in range(d-1):
                    tt[i] = 0
                tt[d-1] = 1-gamma
        if state==1:
            if next_state == 1:
                for i in range(d-1):
                    tt[i] = 0
                tt[d-1] = gamma
        return tt

def TwoActionPhi(S):
    def outputphi(state, action, next_state):
        embedding = np.zeros(S*2*S)
        embedding[state*2*S + action*S + next_state] = 1
        return embedding
    return outputphi
