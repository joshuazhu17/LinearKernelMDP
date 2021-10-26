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

def EVI(ite, U, hattheta, beta, phi, phiv, reward, S, A, d, gamma):
    #here U should be sigma^{1/2}
    qq = np.ones((S, A))/(1-gamma)
    vv = np.ones(S)/(1-gamma)

    x = cp.Variable(d)

    soc_constraints = [cp.norm(U@(x-hattheta), 2) <= beta]
    # The following loops add the following contraints:
    # Constraint 1: 0 <= phi(s'| s, a) * x <= 1
    # Constraint 2: Sum over s' of phi(s' | s, a) * x = 1

    for s in range(S):
        for a in range(A):
            running_matrix = []
            running_sum = 0
            for s_prime in range(S):
                running_sum += phi(s, a, s_prime)
                running_matrix = np.concatenate((running_matrix, phi(s, a, s_prime)))
            running_matrix = running_matrix.reshape((-1, d))
            soc_constraints += [running_matrix@x <= 1, running_matrix@x >= 0, running_sum@x == 1]]

    for i in range(ite):
        for ss in range(S):
            for aa in range(A):
                phi_v = phiv(ss,aa,vv)
                f = -phi_v
                prob = cp.Problem(cp.Minimize(f@x), soc_constraints)
                prob.solve()

                while prob.status != cp.OPTIMAL:
                    beta *= 2
                    soc_constraints = [cp.norm(U@(x-hattheta), 2) <= beta] + soc_constraints[1:]
                    prob = cp.Problem(cp.Minimize(f@x), soc_constraints)
                    prob.solve()

                qq[ss,aa] = reward(ss,aa)+gamma*phi_v@x.value
            vv[ss] = max(qq[ss])\
    return qq,vv

def BVI(ite, sigma, hattheta, beta, phiv, reward, S, A, d, gamma):
    q0 = np.ones((S, A))/(1-gamma)
    running_v = np.ones(S)/(1-gamma)
    q_history = [q0]
    sigma_sqrt_inv = la.inv(la.sqrtm(sigma))

    for i in range(ite):
        qnext = np.ones((S, A))


        #compute new q values
        for s in range(S):
            for a in range(A):
                phi_v = phiv(s, a, running_v)
                term1 = (gamma*(np.dot(hattheta, phi_v)))
                term2 = (gamma*beta*(la.norm( sigma_sqrt_inv@phi_v, 2 )) )
                ans = reward(s, a) + term1 + term2
                qnext[s][a] = min(1.0/(1-gamma), ans)

        #update v
        for s in range(S):
            running_v[s] = max(qnext[s])

        #store new q values at the front of the array
        q_history = [qnext] + q_history

    return q_history

#-------------------------------------------------------------

class Algorithm(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def train(self, timesteps):
        pass

class UCLK(Algorithm):
    def __init__(self, env, d, A, S, gamma, phi):
        super().__init__(env)
        self.d = d
        self.A = A
        self.S = S
        self.gamma = gamma
        self.phi = phi
        #self.thetastar = thetastar

        self.qvals = np.ones((S, A))/(1-gamma)
        self.vvals = np.ones(S)/(1-gamma)
        self.SIGMA = np.diag(np.ones(d))
        self.TSIGMA = np.diag(np.ones(d))
        self.BB = np.zeros((d, 1))

    def phiv(self, s, a, v):
        total = 0
        for i in range(self.S):
            total += self.phi(s, a, i)*v[i]
        return total

    def train(self, timesteps, plot = True):
        REWARD = 0
        self.env.reset()
        t = 1
        reward_history = []
        cumulative_reward_history = []
        trajectory = []

        TotalStartTime = time.time()

        for k in range(timesteps):
            hattheta = np.linalg.solve(self.SIGMA, self.BB)
            hattheta.shape = (self.d)
            self.TSIGMA = copy.deepcopy(self.SIGMA)
            #bonus = 1*np.log(t+1) + np.linalg.norm(self.thetastar, 2)
            #bonus = 1*np.log(t+1) + 2
            bonus = 1.6
            UU = scipy.linalg.sqrtm(self.SIGMA)

            EVIStartTime = time.time()
            qq, vv = EVI(math.ceil(k**(1.0/2))+1, UU, hattheta, bonus, self.phi, self.phiv, self.env.reward, self.S, self.A, self.d, self.gamma)
            EVIExecutionTime = str(time.time() - EVIStartTime)

            UpdateStartTime = time.time()
            for ss in range(self.S):
                for aa in range(self.A):
                    self.qvals[ss][aa] = qq[ss][aa]
                self.vvals[ss] = vv[ss]
            while (np.linalg.det(self.SIGMA)<3*np.linalg.det(self.TSIGMA)):
                mmax = max(self.qvals[self.env.cur_state])
                tt = []
                for aa in range(self.A):
                    if self.qvals[self.env.cur_state][aa] == mmax:
                        tt.append(aa)
                a_cur = tt[randrange(len(tt))]

                REWARD += self.env.reward(self.env.cur_state, a_cur)
                reward_history.append(self.env.reward(self.env.cur_state, a_cur))
                cumulative_reward_history.append(REWARD)

                phi_v = self.phiv(self.env.cur_state, a_cur, self.vvals)
                phi_v.shape = (self.d,1)
                self.SIGMA += np.dot(phi_v, phi_v.T)

                next_state = self.env.step(a_cur)
                trajectory.append(next_state)

                self.BB += phi_v*self.vvals[self.env.cur_state]
                t = t+1

                if t > timesteps:
                    break

            UpdateExecutionTime = str(time.time() - UpdateStartTime)

            print("##########")
            print("EVI Run Time: ", EVIExecutionTime)
            print("Update Run Time: ", UpdateExecutionTime)
            print("t: ", t)
            print("##########")

            if t > timesteps:
                break

        if plot:
            plt.subplot(121)
            plt.plot(cumulative_reward_history)
            plt.subplot(122)
            plt.plot(trajectory)

            plt.show()

        TotalExecutionTime = str(time.time() - TotalStartTime)
        print("#################")
        print("Total Run Time: ", TotalExecutionTime)
        print("Total Reward: ", REWARD)

class UCLKNoConv(Algorithm):
    def __init__(self, env, d, A, S, gamma, phi):
        super().__init__(env)
        self.d = d
        self.A = A
        self.S = S
        self.gamma = gamma
        self.phi = phi
        #self.thetastar = thetastar

        self.qvals = np.ones((S, A))/(1-gamma)
        self.vvals = np.ones(S)/(1-gamma)
        self.SIGMA = np.diag(np.ones(d))
        self.TSIGMA = np.diag(np.ones(d))
        self.BB = np.zeros((d, 1))

    def phiv(self, s, a, v):
        total = 0
        for i in range(self.S):
            total += self.phi(s, a, i)*v[i]
        return total

    def train(self, timesteps, plot = True):
        REWARD = 0
        self.env.reset()
        t = 1
        reward_history = []
        cumulative_reward_history = []
        trajectory = []

        TotalStartTime = time.time()

        for k in range(timesteps):
            hattheta = np.linalg.solve(self.SIGMA, self.BB)
            hattheta.shape = (self.d)
            self.TSIGMA = copy.deepcopy(self.SIGMA)
            #bonus = 1*np.log(t+1) + np.linalg.norm(self.thetastar, 2)
            #bonus = 1*np.log(t+1) + 2
            bonus = 1.6

            BVIStartTime = time.time()
            q_history  = BVI(timesteps-k, self.SIGMA, hattheta, bonus, self.phiv, self.env.reward, self.S, self.A, self.d, self.gamma)
            BVIExecutionTime = str(time.time() - BVIStartTime)

            UpdateStartTime = time.time()
            steps = 0
            while (np.linalg.det(self.SIGMA)<3*np.linalg.det(self.TSIGMA)):
                #update q and v for the new timestep
                for ss in range(self.S):
                  for aa in range(self.A):
                      self.qvals[ss][aa] = q_history[steps][ss][aa]
                  self.vvals[ss] = max(self.qvals[ss])

                #select an action
                mmax = max(self.qvals[self.env.cur_state])
                tt = []
                for aa in range(self.A):
                  if self.qvals[self.env.cur_state][aa] == mmax:
                    tt.append(aa)
                a_cur = tt[randrange(len(tt))]

                #receive reward
                REWARD += self.env.reward(self.env.cur_state, a_cur)
                reward_history.append(self.env.reward(self.env.cur_state, a_cur))
                cumulative_reward_history.append(REWARD)

                #update sigma, bb, environment
                phi_v = self.phiv(self.env.cur_state, a_cur, self.vvals)
                phi_v.shape = (self.d,1)
                self.SIGMA += np.dot(phi_v, phi_v.T)

                next_state = self.env.step(a_cur)
                trajectory.append(next_state)

                self.BB += phi_v*self.vvals[self.env.cur_state]
                t = t+1
                steps = steps + 1

                if t > timesteps:
                    break

            UpdateExecutionTime = str(time.time() - UpdateStartTime)

            print("##########")
            print("BVI Run Time: ", BVIExecutionTime)
            print("Update Run Time: ", UpdateExecutionTime)
            print("t: ", t)
            print("##########")

            if t > timesteps:
                break

        if plot:
            plt.subplot(121)
            plt.plot(cumulative_reward_history)
            plt.subplot(122)
            plt.plot(trajectory)

            plt.show()

        TotalExecutionTime = str(time.time() - TotalStartTime)
        print("#################")
        print("Total Run Time: ", TotalExecutionTime)
        print("Total Reward: ", REWARD)
