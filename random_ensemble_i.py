import numpy as np
import scipy.io as sio
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import time

from L0Obj import L0Obj
from ProjCSimplex import ProjCSimplex
from minConf.minConf_PQN import minConF_PQN
import random


tStart = time.process_time()

# Generate Synthetic Data
nInstances = 500
nVars = 5000
k = 20
pho = 0.2
SNR = 1
AccPQN = np.zeros((len([nInstances]), 10))

T = toeplitz(pho**np.arange(1, nVars+1))
# T = np.linalg.toeplitz(pho**np.arange(1, nVars+1))
# AL: random permutate the index and take the first k elements --> random selection of k features
RandIndtmp = np.random.permutation(nVars)
RandInd = np.sort(RandIndtmp[:k])
print(RandInd)
# AL: randomly select  +1 or -1 for each selected k features
RandInt = 2 * np.random.randint(0, 2, k) - 1
# RandInt = 2 * np.random.randint(0, 2, size=(k, 1)) - 1
# X = multivariate_normal.rvs(mean=np.zeros(nVars), cov=T, size=nInstances)
X = np.random.multivariate_normal(np.zeros(nVars), T, nInstances)
w = np.zeros((nVars, 1))
w[RandInd] = RandInt.reshape(-1, 1)
utrue = w #.copy()

print("Check!!!")
tEnd = time.process_time() - tStart
print("Execution time(Before):", tEnd)

tStart = time.process_time()
utrue[RandInd] = 1.0
NoiseT = np.random.normal(0, 1, (nInstances, 1))
Noise = ((np.linalg.norm(X @ w) / np.sqrt(SNR)) / np.linalg.norm(NoiseT)) * NoiseT
y = X @ w + Noise
pho = np.sqrt(nInstances)

print("Check!!!")
tEnd = time.process_time() - tStart
print("Execution time(Before):", tEnd)

# Initial guess of parameters
uSimplex = np.ones((nVars, 1)) * (1 / nVars)


# Set up Objective Function
funObj = lambda w: L0Obj(w, X, y, pho)

# Set up Simplex Projection Function
funProj = lambda w: ProjCSimplex(w, k)

tEnd = time.process_time() - tStart
print("Execution time(Before):", tEnd)
print("start!!!")
# Solve with PQN
options = {'maxIter': 50}
tStart = time.process_time()
uout, obj, _ = minConF_PQN(funObj, uSimplex, funProj, options)
print(f"uout: {uout}")
tEnd = time.process_time() - tStart

B = np.sort(-uout.flatten())
Ranktmp = np.argsort(-uout.flatten())
Rank = np.sort(Ranktmp[:k])
# uout[Ranktmp[:k[0][0]]]

Indtrue = np.where(utrue)
# print(Rank)
# print("IndTrue", Indtrue)
# print(f"Indtrue: {Indtrue}, Rank: {Rank}")
C = np.intersect1d(Rank, Indtrue)
# Convert numpy arrays to tuples, if necessary
# Rank_tuple = [tuple(r) for r in Rank]
# Indtrue_tuple = [tuple(i) for i in Indtrue]

# Find the intersection
AccPQN = len(C) / k

print("Execution time:", tEnd)
print("Accuracy PQN:", AccPQN)
