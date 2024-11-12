import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# from crfChain.crfChain_decode import crfChain_decode
# from crfChain.crfChain_error import crfChain_error
# from crfChain.crfChain_genSynthetic import crfChain_genSynthetic
# from crfChain.crfChain_infer import crfChain_infer
# from crfChain.crfChain_initSentences import crfChain_initSentences
# from crfChain.crfChain_initWeights import initWeights
# from crfChain.crfChain_loss import crfChain_loss
# from crfChain.crfChain_makePotentials import crfChain_makePotentials
# from crfChain.crfChain_sample import crfChain_sample
# from crfChain.crfChain_splitWeights import crfChain_splitWeights
# from KPM.drawGraph import drawGraph
# from KPM.setdiag import setDiag
# from KPM.standardizeCols import standardizeCols
# from lossFuncs.dualSVMLoss import dualSVMLoss
# from lossFuncs.kernelLinear import kernelLinear
# from lossFuncs.logdetFunction import logdetFunction
# from lossFuncs.LogisticLoss import LogisticLoss
# from lossFuncs.MeanFieldGibbsFreeEnergyLoss import MeanFieldGibbsFreeEnergyLoss
# from lossFuncs.overLassoLoss import overLassoLoss
# from lossFuncs.SimultaneousLogisticLoss import SimultaneousLogisticLoss
from lossFuncs.SimultaneousSquaredError import SimultaneousSquaredError
# from lossFuncs.SoftmaxLoss2 import SoftmaxLoss2
from lossFuncs.SquaredError import SquaredError
# from lossFuncs.SSVMLoss import SSVMLoss
# from minConf.boundProject import boundProject
from minConf.lbfgsHvFunc2 import lbfgsHvFunc2
# from minConf.linearProject import linearProject
from minConf.minConf_PQN import minConF_PQN
from minConf.projectSimplex import projectSimplex
from mexAll import projectRandom2C
from project.auxGroupLoss import auxGroupLoss
from project.auxGroupLinfProject import auxGroupLinfProject
from project.complexProject import complexProject
from project.groupl1_makeGroupPointers import groupl1_makeGroupPointers
from project.groupLinfProj import groupLinfProj
from misc.sampleDiscrete import sampleDiscrete

# Linear Regression on the Simplex
def LinearRegressionOnSimplex():
    # We will solve min_w ||Xw-y||^2, s.t. w >= 0, sum(w)=1
    #
    # Projection onto the simplex can be computed in O(n log n), this is
    # described in (among other places):
    # Michelot.  <http://www.springerlink.com/content/q1636371674m36p1 A Finite Algorithm for Finding the Projection of a
    # Point onto the Canonical Simplex of R^n>.  Journal of Optimization Theory
    # and Applications (1986).

    # Generate Synthetic Data
    nInstances = 50
    nVars = 10
    X = np.random.randn(nInstances, nVars)
    w = np.random.rand(nVars, 1) * (np.random.rand(nVars, 1) > 0.5)
    y = X @ w + np.random.randn(nInstances, 1)
    # data = sio.loadmat('../PQNexamples/var_data.mat')
    # X, w, y = data["X"], data["w"], data["y"]

    # Initial guess of parameters
    wSimplex = np.zeros((nVars, 1))

    # Set up Objective Function
    funObj = lambda w: SquaredError(w, X, y, 2)

    # Set up Simplex Projection Function
    funProj = lambda w: projectSimplex(w)

    # Solve with PQN
    print("Computing optimal linear regression parameters on the simplex...")
    wSimplex = minConF_PQN(funObj, wSimplex, funProj)

    # Check if variable lie in simplex
    wSimplex[0].conj().T
    print(f"Min value of wSimplex: {np.min(wSimplex[0]):.3f}")
    print(f"Max value of wSimplex: {np.max(wSimplex[0]):.3f}")
    print(f"Sum value of wSimplex: {np.sum(wSimplex[0]):.3f}")

LinearRegressionOnSimplex()
# Lasso regression
def LassoRegression():
    # We will solve min_w ||Xw-y||^2 s.t. sum_i |w_i| <= tau
    #
    # Projection onto the L1-Ball can be computed in O(n), see:
    # Duchi, Shalev-Schwartz, Singer, and Chandra.  <http://icml2008.cs.helsinki.fi/papers/361.pdf Efficient Projections onto
    # the L1-Ball for Learning in High Dimensions>.  ICML (2008).

    # Generate Synthetic Data
    nInstances = 500
    nVars = 50
    # X = np.random.randn(nInstances, nVars)
    # w = np.random.randn(nVars, 1) * (np.random.rand(nVars, 1) > 0.5)
    # y = X @ w + np.random.randn(nInstances, 1)
    data = sio.loadmat('../PQNexamples/var_data2.mat')
    X, w, y = data["X"], data["w"], data["y"]
    # print(f"X: {X}\n w: {w}\n y: {y}")
    # input()

    # Initial guess of parameters
    wL1 = np.zeros((nVars, 1))

    # Set up Objective Function
    funObj = lambda w: SquaredError(w, X, y, 2)

    # Set up L1-Ball Projection
    tau = 2
    funProj = lambda w: np.sign(w) * projectRandom2C(np.abs(w), tau)

    # Solve with PQN
    print("Computing optimal Lasso parameters...")
    wL1 = minConF_PQN(funObj, wL1, funProj)
    wL1[0][np.abs(wL1[0]) < 1e-4] = 0
    
    # Check sparsity of solution
    wL1[0].conj().T
    print(f"Number of non-zero variables in solution:: {np.count_nonzero(wL1[0])} (of {len(wL1[0])})")

    plt.subplot(1, 2, 1)
    plt.imshow(wL1[0], cmap='gray', aspect='auto', extent=[0.5, wL1[0].shape[1]+0.5, 0.5, wL1[0].shape[0]+0.5])
    plt.title('Weights')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(wL1[0] != 0, cmap='gray', aspect='auto', extent=[0.5, wL1[0].shape[1]+0.5, 0.5, wL1[0].shape[0]+0.5])
    plt.title('Sparsity of wL1')
    plt.colorbar()

    plt.tight_layout()  
    plt.show()    

LassoRegression()  
input("aaaaaaaaaaaaa") 

# Lasso with Complex Variables
def LassoComplex():
    # We will solve min_w ||Xz-y||^2, s.t. sum_i |z_i| <= tau,
    # where z and y are complex, and |z| represents the complex modulus
    # 
    # Efficient projection onto this complex L1-Ball is described in:
    # van den Berg and Friedlander.  <http://www.optimization-online.org/DB_FILE/2008/01/1889.pdf Probing the Pareto Frontier for Basis
    # Pursuit Solutions>.  SIAM Journal of Scientific Computing (2008).
    #
    # The calculation of the projection can be reduced from the O(n log n) 
    # required in the above to O(n) by using a linear-time median finding
    # algorithm

    # Generate Synthetic Data
    nInstances = 500
    nVars = 50
    # X = np.random.randn(nInstances, nVars)
    # # Generate random real and imaginary parts
    # real_part = np.random.randn(nVars, 1)
    # imaginary_part = np.random.randn(nVars, 1)

    # # Generate random mask
    # mask = np.random.rand(nVars, 1) > 0.5

    # # Combine real and imaginary parts to form complex numbers
    # z = (real_part + 1j * imaginary_part) * mask
    # # z = np.complex(randn(nVars, 1), randn(nVars, 1))*(rand(nVars, 1)>0.5)
    # y = X @ z
    data = sio.loadmat('../PQNexamples/var_data3.mat')
    X, w, y = data["X"], data["w"], data["y"]

    # Initial guess of parameters
    zReal = np.zeros((nVars, 1))
    zImag = np.zeros((nVars, 1))

    # Set up Objective Function
    funObj = lambda zRealImag: SquaredError(zRealImag, np.block([[X, np.zeros((nInstances, nVars))], [np.zeros((nInstances, nVars)), X]]), np.concatenate((np.real(y), np.imag(y))), nargout=2)

    # Set up Complex L1-Ball Projection
    tau = 2
    funProj = lambda zRealImag: complexProject(zRealImag, tau)

    # Solve with PQN
    print("Computing optimal Lasso parameters...")
    # Ensure zReal and zImag are numpy arrays of type float
    zReal = np.array(zReal, dtype=float)
    zImag = np.array(zImag, dtype=float)

    # Concatenate zReal and zImag vertically (as in MATLAB [zReal; zImag])
    zRealImag_concat = np.concatenate((zReal, zImag), axis=0)

    # Assuming minConF_PQN is defined and works with numpy arrays
    zRealImag = minConF_PQN(funObj, zRealImag_concat, funProj)

    # Split the result back into zReal and zImag
    zReal = zRealImag[0][:nVars]
    zImag = zRealImag[0][nVars:]

    # Combine zReal and zImag into complex numbers
    zL1 = zReal + 1j * zImag

    # Set values close to zero to exactly zero
    zL1[0][np.abs(zL1[0]) < 1e-4] = 0
    print(f"zL1: {zL1}")
    input()
    # zRealImag = minConF_PQN(funObj, np.concatenate((zReal, zImag)), funProj)
    # zRealImag = zRealImag[:nVars]
    # zImag = zRealImag[nVars+1:]
    # zL1 = zReal + 1j * zImag
    # # zL1 = np.complex(zReal, zImag)
    # zL1[np.abs(zL1) < 1e-4] = 0

    # Create figure
    # plt.figure(f)
    # f += 1

    # First subplot: Real Weights
    plt.subplot(1, 3, 1)
    plt.imshow(zReal, cmap='gray', aspect='auto', extent=[0.5, zReal.shape[1]+0.5, 0.5, zReal.shape[0]+0.5])
    plt.title('Real Weights')
    plt.colorbar()

    # Second subplot: Imaginary Weights
    plt.subplot(1, 3, 2)
    plt.imshow(zImag, cmap='gray', aspect='auto', extent=[0.5, zImag.shape[1]+0.5, 0.5, zImag.shape[0]+0.5])
    plt.title('Imaginary Weights')
    plt.colorbar()

    # Third subplot: Sparsity of zL1
    plt.subplot(1, 3, 3)
    plt.imshow(zL1 != 0, cmap='gray', aspect='auto', extent=[0.5, zL1.shape[1]+0.5, 0.5, zL1.shape[0]+0.5])
    plt.title('Sparsity of zL1')
    plt.colorbar()

    # Display the plot
    plt.show()

    # Check sparsity of solution
    zL1.conj().T
    print(f"Number of non-zero variables in solution: {np.count_nonzero(zL1)} (of {len(zL1)})")

# Group-Sparse Linear Regression with Categorical Features
def GroupSparseLinearRegressionCategoricalFeatures():
    # We will solve min_w ||Xw-y||^2, s.t. sum_g ||w_g||_inf <= tau,
    # where X uses binary indicator variables to represent a set of categorical
    # features, and we use the 'groups' g to encourage sparsity in terms of the
    # original categorical variables
    #
    # Using the L_1,inf mixed-norm for group-sparsity is described in:
    # Turlach, Venables, and Wright.  <http://pages.cs.wisc.edu/~swright/papers/tvw.pdf Simultaneous Variable Selection>.  Technometrics
    # (2005).
    #
    # Using group sparsity to select for categorical variables encoded with
    # indicator variables is described in:
    # Yuan and Lin.  <http://www.stat.wisc.edu/Department/techreports/tr1095.pdf Model Selection and Estimation in Regression with Grouped
    # Variables>.  JRSSB (2006).
    #
    # Projection onto the L_1,inf mixed-norm ball can be computed in O(n log n), 
    # this is described in:
    # Quattoni, Carreras, Collins, and Darell.  <http://www.cs.mcgill.ca/~icml2009/papers/475.pdf An Efficient Projection for
    # l_{1,\infty} Regularization>.  ICML (2009).

    # Generate categorical features
    nInstances = 100
    nStates = [3, 3, 3, 3, 5, 4, 5, 5, 6, 10, 3]
    # X = np.zeros((nInstances, len(nStates)))
    offset = 0
    # for i in range(nInstances):
    #     for s in range(len(nStates)):
    #         prob_s = np.random.rand(nStates[s], 1)
    #         prob_s /= np.sum(prob_s)
    #         X[i, s] = sampleDiscrete(prob_s)

    # # Make indicator variable encoding of categorical features
    # X_ind = np.zeros((nInstances, np.sum(nStates)))
    # w = np.zeros((sum(nStates), 1))
    # for s in range(len(nStates)):
    #     for i in range(nInstances):
    #         X_ind[i, offset + int(X[i, s])] = 1
    #     w[offset:offset + nStates[s], 0] = (np.random.rand() > 0.75) * np.random.randn(nStates[s])
    #     offset += nStates[s]

    # y = X_ind @ w + np.random.randn(nInstances, 1)

    # w_ind = np.zeros((sum(nStates), 1))
    data = sio.loadmat('../PQNexamples/var_data4.mat')
    X, w, y, X_ind, w_ind = data["X"], data["w"], data["y"], data["X_ind"], data["w_ind"]
    # print(f"X_ind: {X_ind}\n y: {y}\n w: {w}\n w_ind: {w_ind}")
    # input()

    # Set up Objective Function
    funObj = lambda w: SquaredError(w, X_ind, y, 2)

    # Set up groups
    offset = 0
    groups = np.zeros(w_ind.shape)
    # print(f"groups: {groups}")
    for s in range(len(nStates)):
        groups[offset:offset+nStates[s], 0] = s+1
        offset = offset + nStates[s]
    # print(f"groups_final: {groups}\n offset: {offset}")
    # input()

    # Set up L_1, inf Projection Function
    tau = 0.05
    funProj = lambda w: groupLinfProj(w, tau, groups)

    # Solve with PQN
    print("Computing Group-Sparse Linear Regression with Categorical Features Parameters...")
    w_ind = minConF_PQN(funObj, w_ind, funProj)
    w_ind[0][np.abs(w_ind[0]) < 1e-4] = 0
    # print(f"w_ind: {w_ind}")
    # input()

    # Check selected variables
    w_ind[0].conj().T
    for s in range(len(nStates)):
        print(f"Number of non-zero variables associated with categorical variable {s}: {np.count_nonzero(w_ind[0][groups==s])} (of {np.sum(groups==s)})")
    print(f"Total number of categorical variables selected: {np.count_nonzero(np.bincount(groups.flatten().astype(int), weights=np.abs(w_ind[0]).flatten()))} (of {len(nStates)})")

# Group-Sparse Simultaneous Regression
def GroupSparseSimultaneousRegression():
    # We will solve min_W ||XW-Y||^2 + lambda * sum_g ||W_g||_inf,
    # where we use the 'groups' g to encourage that we select variables that
    # are relevant across the output variables
    #
    # We solve this non-differentiable problem by transforming it into the
    # equivalent problem: 
    # min_w ||XW-Y||^2 + lambda * sum_g alpha_g, s.t. forall_g alpha_g >= ||W_g||_inf
    #
    # Using group-sparsity to select variables that are relevant across regression
    # tasks is described in:
    # Turlach, Venables, and Wright.  <http://pages.cs.wisc.edu/~swright/papers/tvw.pdf Simultaneous Variable Selection>.  Technometrics
    # (2005).
    #
    # The auxiliary variable formulation is described in:
    # Schmidt, Murphy, Fung, and Rosales.  <http://www.cs.ubc.ca/~murphyk/Papers/cvpr08.pdf Structure Learning in Random Field for
    # Heart Motion Abnormality Detection>.  CVPR (2008).
    # 
    # Computing the projection in the auxiliary variable formulation can be
    # done in O(n log n), this is described in the
    # <http://www.cs.ubc.ca/~murphyk/Software/L1CRF/cvpr08_extra.pdf addendum>
    # of the above paper.

    # Generate synthetic data
    nInstances = 100
    nVars = 25
    nOutputs = 10
    X = np.random.randn(nInstances,nVars)
    W = np.diag(np.random.rand(nVars,1) > .75) * np.random.randn(nVars,nOutputs)
    Y = X @ W + np.random.randn(nInstances,nOutputs)

    # Initial guess of parameters
    W_groupSparse = np.zeros((nVars,nOutputs))

    # Set up Objective Function
    funObj = lambda W: SimultaneousSquaredError(W,X,Y)

    # Set up Groups
    groups = np.tile(np.arange(1, nVars + 1).reshape(-1, 1), (1, nOutputs))
    groups = groups.flatten()
    nGroups = np.max(groups)

    # Initialize auxiliary variables that will bound norm
    lamda = 250
    alpha = np.zeros((nGroups,1))
    penalizedFunObj = lambda W: auxGroupLoss(W,groups,lamda,funObj)

    # Set up L_1,inf Projection Function
    groupStart, groupPtr = groupl1_makeGroupPointers(groups)
    funProj = lambda W: auxGroupLinfProject(W,nVars*nOutputs,groupStart,groupPtr)

    # Solve with PQN
    print(f'Computing group-sparse simultaneous regression parameters...')
    Walpha = minConF_PQN(penalizedFunObj,np.concatenate([W_groupSparse.flatten(), alpha]),funProj)

    # Extract parameters from augmented vector
    W_groupSparse = Walpha[:nVars*nOutputs].reshape(nVars, nOutputs)  # Reshape
    W_groupSparse[np.abs(W_groupSparse) < 1e-4] = 0                     # Thresholding

    # PLOTTING
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Sparsity Pattern
    axs[0].imshow(W_groupSparse != 0, cmap='gray', aspect='auto')
    axs[0].set_title('Sparsity Pattern')
    axs[0].set_ylabel('Variable')
    axs[0].set_xlabel('Output Target')

    # Variable weights
    im = axs[1].imshow(W_groupSparse, cmap='gray', aspect='auto')
    axs[1].set_title('Variable weights')
    axs[1].set_ylabel('Variable')
    axs[1].set_xlabel('Output Target')

    # Adding color bar for variable weights
    fig.colorbar(im, ax=axs[1])

    # Display the plots
    plt.tight_layout()
    plt.show()

    # Check selected variables
    for s in range(nVars):
        print(f"Number of tasks where variable {s} was selected: {np.count_nonzero(W_groupSparse[s:])} (of {nOutputs})")
    print(f"Total number of variables selected: {np.count_nonzero(np.sum(W_groupSparse, 2))} (of {nVars})")

GroupSparseSimultaneousRegression()

# Group-Sparse Multinomial Logistic Regression
def GroupSparseMultinomialLogisticRegression():
    pass

# Group-Sparse Multi-Task Classification
def GroupSparseMultiTaskClassification():
    pass

# Low-Rank Multi-Task Classification
def LowRankMultiTaskClassification():
    pass

# L_1, inf Blockwise-Sparse Graphical Lasso
def L1infBlockwiseSparseGraphicalLasso():
    pass

# L_1, 2 Blockwise-Sparse Graphical Lasso
def L12BlockwiseSparseGraphicalLasso():
    pass

# Linear Regression with the Over-Lasso
def LinearRegressionWithOverLasso():
    pass

# Kernelized dual form of support vector machines
def KernelizedDualSVM():
    pass

# Smooth (Primal) Support Vector Machine with Multiple Kernel Learning
def SmoothSVMWithMultipleKernelLearning():
    pass

# Conditional Random Field Feature Selection
def ConditionalRandomFieldFeatureSelection():
    pass

# Approximating node marginals in undirected graphical models with variational mean field
def ApproximatingNodeMarginals():
    pass

# Multi-State Markov Random Field Structure Learning
def MultiStateMarkovRandomField():
    pass

# Conditional Random Field Structure Learning with Pseudo-Likelihood
def ConditionalRandomField():
    # We will solve min_{w,v} nll(w,v,x,y) + lambda * sum_e ||v_e||_inf,
    # where nll(w,v,x,y) is the negative log-likelihood for a log-linear 
    # conditional random field and each 'group' e is the set of parameters
    # associated with an edge, leading to sparsity in the graph
    #
    # To solve the problem, we use a pseudo-likelihood approximation of the
    # negative log-likelihood, and convert the non-differentiable problem to a
    # differentiable one by introducing auxiliary variables
    #
    # Using group-sparsity to select edges in a conditional random field
    # trained with pseudo-likelihood is discussed in:
    # Schmidt, Murphy, Fung, and Rosales.  <http://www.cs.ubc.ca/~murphyk/Papers/cvpr08.pdf Structure Learning in Random Field for
    # Heart Motion Abnormality Detection>.  CVPR (2008).

    # Generate Data
    nInstances = 250
    nFeatures = 10
    nNodes = 20
    edgeDensity = 0.25
    nStates = 2
    ising = 0
    tied = 0
    useMex = 0
    y, adj, X = UGM_generate(nInstances, nFeatures, nNodes, edgeDensity, nStates, ising, tied)

    # Set up CRF
    adj = fullAdjMatrix(nNodes)
    edgeStruct = UGM_makeEdgeStruct(adj, nStates, useMex)

    # Make edge features
    Xedge = UGM_makeEdgeFeatures(X, edgeStruct.edgeEnds)
    nodeMap, edgeMap = UGM_makeCRFmaps(X, Xedge, edgeStruct, ising, tied)

    # Initialize Variables
    nNodeParams = max(nodeMap.flatten())
    nVars = max(edgeMap.flatten())
    w = np.zeros((nVars, 1))

    # Make Groups
    groups = np.zeros((nVars, 1))
    for e in range(edgeStruct.nEdges):
        pass