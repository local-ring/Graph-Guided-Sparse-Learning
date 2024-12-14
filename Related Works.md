### TL;DR
We care about related research on lasso with two main characteristics:
- They incorporate graphical information and handle general graphs.
- They address sparse learning.

In this sense, we usually frame these two purposes in the language of the following optimization problem:

$$
\arg\min_{\beta \in \mathbb{R}^p} \frac{1}{2} \| y - X\beta \|_2^2 + \lambda_1 \| \beta \|_{l_1} + \lambda_2 \sum_{(i,j) \in \mathcal{E}} \|\beta_i - \beta_j\|_{l_2}
$$

The difference lies in how norms $l_1$ and $l_2$ are chosen. In the following survey, we have:
- $l_1 = 1$, $l_2 = 1$.
- $l_1 = 1$, $l_2 = 2$.
For comparison, our study differs slightly. We take $l_1 = 0$, $l_2 = 0$, and add an extra 2-norm term for convexity. 

Additionally, some works included here address the generalized lasso, which can cover the problem we are working on:

$$
\arg\min_{\beta \in \mathbb{R}^p} \frac{1}{2} \| y - X\beta \|_2^2 + \lambda \| A\beta \|_1,
$$

We include work on the fused lasso signal approximation problem ($X = I$), because their algorithms can be easily extended to the general case (most are based on ADMM) and have connections to proximal methods.

There are several existing algorithms to solve the problem:
- ADMM and its variants.
- Proximal methods.
- Path algorithms.


### Related Papers
[Network-constrained regularization and variable selection for analysis of genomic data](https://academic.oup.com/bioinformatics/article/24/9/1175/206444)
**Year**: 
2008
**Journal**: 
Bioinformatics
**Questions**: 
Network-Constrained Regularization Criterion: for any fixed non-negative $\lambda_1$ and $\lambda_2$, we define the network-constrained regularization criterion:

$$
L(\lambda_1, \lambda_2, \beta) = (y - X\beta)^T(y - X\beta) + \lambda_1 |\beta|_1 + \lambda_2 \beta^T L \beta,
$$

where $|\beta|_1 = \sum_{j=1}^p |\beta_j|$ is the $L_1$-norm, the second term $\beta^T L \beta$ induces a smooth solution of $\beta$ on the network, and $L$ is the normalized Laplacian.
Since we have
$$
\beta^T L \beta = \sum_{u \sim v} \left( \frac{\beta_u}{\sqrt{d_u}} - \frac{\beta_v}{\sqrt{d_v}} \right)^2 w(u, v),
$$

the equation can be rewritten as:
$$
L(\lambda_1, \lambda_2, \beta) = (y - X\beta)^T(y - X\beta) + \lambda_1 \sum_{j=1}^p |\beta_j|
+ \lambda_2 \sum_{u \sim v} \left( \frac{\beta_u}{\sqrt{d_u}} - \frac{\beta_v}{\sqrt{d_v}} \right)^2 w(u, v).
$$
**Simulation**:
Suppose we have 200 transcription factors (TFs), each regulating 10 genes. The resulting network includes 2,200 genes and edges between each TF and its regulated genes. Assume four TFs and their regulated genes are related to response $Y$. For the first model, the data are simulated from:

- $y = X\beta + \epsilon$ and

$$
\beta = \left(5, \frac{5}{\sqrt{10}}, \ldots, \frac{5}{\sqrt{10}}, -5, \frac{-5}{\sqrt{10}}, \ldots, \frac{-5}{\sqrt{10}}, 
3, \frac{3}{\sqrt{10}}, \ldots, \frac{3}{\sqrt{10}}, -3, \frac{-3}{\sqrt{10}}, \ldots, \frac{-3}{\sqrt{10}}, 0, \ldots, 0 \right),
$$

where $\epsilon \sim N(0, \sigma^2)$.

- The expression levels for the 200 TFs follow a standard normal distribution, $X_{TF_j} \sim N(0, 1)$.
- The expression levels of the TF and the gene it regulates are jointly distributed as a bivariate normal with a correlation of 0.7. Conditioning on the expression level of the TF, the expression level of the gene it regulates follows $N(0.7 \ast X_{TF_j}, 0.51)$.

Variant models differ based on how $\beta$ is chosen. For each of these four models, the noise variance was chosen as $\sigma_\epsilon^2 = \left(\sum_j \beta_j^2\right) / 4$ so that the signal-to-noise ratio was 21.68, 7.34, 10.70, and 5.82 for Models 1, 2, 3, and 4, respectively.

- A training set and an independent test set, each with 100 samples, were simulated.
- A **10-fold cross-validation (10-CV)** was conducted on the training dataset to select the tuning parameters, and then parameter estimates were obtained using the entire training dataset.
- Each model simulation was repeated 50 times.
**Algorithm**:  
Convert this to a lasso-type optimization problem by creating an augmented dataset, "absorbing the Laplacian term into the quadratic term" with $(1+\lambda_2)$-rescaling.
**Dataset:**
Identifying age-dependent molecular modules based on gene expression data measured in human brains of individuals at different ages using KEGG regulatory network information.
**Code:** 
Not found, but we can code by ourself
**Memo**:
Of particular interest are gene-regulatory pathways that provide regulatory relationships between genes or gene products. These pathways are often interconnected and form a network, which can be represented as graphs, where the vertices of the graphs are genes or gene products and the edges of the graphs indicate some regulatory relationship between the genes.
A network-constrained regularization procedure for fitting linear-regression models and for variable selection, where the predictors in the regression model are genomic data with graphical structures. The goal of such a procedure is to identify genes and subnetworks that are related to diseases or disease outcomes
[Comment on ‘network-constrained regularization and variable selection for analysis of genomic data’](https://academic.oup.com/bioinformatics/article/24/21/2566/190893?login=true)


[Variable selection and regression analysis for graph-structured covariates with an application to genomics](https://arxiv.org/pdf/1011.3360)
**Year:** 
2010
**Journal:** 
The Annals of Applied Statistics
**Question:** 
Graphs and networks are commonly used to represent biological information. In biology, various processes such as regulatory networks, metabolic pathways, and protein–protein interaction networks are effectively depicted as graphs. The integration of graph-based structures provides valuable supplementary information to standard numerical datasets, such as microarray gene expression data. This paper addresses the problem of regression analysis and variable selection when covariates are linked via a graph. Through simulations and real datasets, the authors demonstrate that the proposed procedure achieves superior variable selection and prediction performance compared to existing methods that ignore graph-related covariate information.

The problem studied in this paper is closely related to prior work, with a key distinction: this approach can handle scenarios where explanatory variables within the same cluster have opposite signs (i.e., it allows for _negative correlations_ within clusters). This extension makes the method more versatile for biological applications.
**Algorithm:**
a cyclical coordinate descent algorithm 
**Code:**
Not found
**Memo:**
Among these procedures, the Enet regularization and the fused Lasso are particularly appropriate for the analysis of genomic data, where the former encourages a grouping effect and the latter often leads to smoothness of the coefficient profiles for ordered covariates.
it is weird why the AAAI paper didn't mention this paper since why they study is very similar.


[The solution path of the generalized lasso](https://projecteuclid.org/journals/annals-of-statistics/volume-39/issue-3/The-solution-path-of-the-generalized-lasso/10.1214/11-AOS878.full)
**Year:** 
2011
**Journal:**
The Annals of Statistics
**Questions:**
It provides a framework to solve the generalized lasso
$$ \min_{\beta \in \mathbb{R}^p} \frac{1}{2} \| y - X\beta \|_2^2 + \lambda \| D\beta \|_1. $$
In particular, it can deal with the optimization problem with an $\ell_2$ penalty is: $$ \min_{\beta \in \mathbb{R}^p} \frac{1}{2} \| y - X\beta \|_2^2 + \lambda \| D\beta \|_1 + \epsilon \| \beta \|_2^2, $$ which can be rewritten as: $$ \min_{\beta} \frac{1}{2} \| y^* - (X^*)\beta \|_2^2 + \lambda \| D\beta \|_1, $$ where: $$ y^* = \begin{bmatrix} y \\ 0 \end{bmatrix}, \quad X^* = \begin{bmatrix} X \\ \sqrt{\epsilon} I \end{bmatrix}. $$
the matrix $D$ can be the corresponding fused graph matrix.
**Algorithm:** 
Inspired by LARS, the approach can even compute the optimal choice of \(\lambda\). 
Efficient implementations are detailed in [Efficient Implementations of the Generalized Lasso Dual Path Algorithm](https://www.stat.cmu.edu/~ryantibs/papers/fastgl.pdf). 
**Code:** 
Available [here](https://github.com/ryantibs/genlasso). 
**Memo:** 
The incidence matrix plays a central role in the methodology.

[Smoothing proximal gradient method for general structured sparse regression](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-6/issue-2/Smoothing-proximal-gradient-method-for-general-structured-sparse-regression/10.1214/11-AOAS514.full)
**Year:** 
2012
**Journal:** 
Annals of Applied Statistics
**Question:**
It studies

$$\min_{\beta \in \mathbb{R}^J} f(\beta) \equiv g(\beta) + \Omega(\beta) + \lambda \|\beta\|_1.$$
where the graph-guided fused lasso:
$$\Omega(\beta) = \gamma \sum_{e=(m,l) \in E, m<l} \tau(r_{ml})|\beta_m - \text{sign}(r_{ml})\beta_l|,$$
where $\tau(r_{ml})$ represents a general weight function that enforces a fusion effect over coefficients $\beta_m$ and $\beta_l$ of relevant features. In this paper, we consider $\tau(r) = |r|$, but any monotonically increasing function of the absolute values of correlations can be used.
**Algorithm:** 
Smoothing Proximal Gradient (SPG) Algorithm: to make the nonseparable and non-smooth $\Omega(\beta)$ tractable, the authors use auxiliary variables to decouple its nonseparability. To handle the non-smooth penalties efficiently, the authors apply **Nesterov’s smoothing technique** to $\Omega(\beta)$. The idea is to approximate $\Omega(\beta)$ with a smooth version $g_\mu(\beta)$, where $\mu > 0$ controls the smoothness. 
The smoothed penalty $g_\mu(\beta)$ is differentiable, and its gradient can be computed efficiently.
Then use Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
**Memo:**
The paper seems to have been published twice:  [An Efficient Proximal Gradient Method for General Structured Sparse Learning](https://www.ml.cmu.edu/research/dap-papers/dap_chen.pdf) (JMLR). Interestingly, these two publications do not cite each other, suggesting duplication.`

[On the robustness of the generalized fused lasso to prior specifications](https://link.springer.com/article/10.1007/s11222-014-9497-6)
**Year:** 
2014
**Journal:**
Statistics and Computing
**Question:** 
This study examines: 
- The robustness of the adaptive fused lasso in the presence of noisy graphs. 
- The performance of the generalized fused lasso when connected coefficients are not exactly equal.
**Algorithm:** 
The adaptive generalized fused lasso is solved using a coordinate-wise optimization algorithm based on the method of Höfling et al. (2010). 
**Code:**
The implementation is available in the **FusedLasso R package**, which is distributed through the CRAN repository.
**Memo:** 
For a more detailed exposition, refer to: [https://hal.science/hal-00813281/](https://hal.science/hal-00813281/)

[Efficient Generalized Fused Lasso and Its Applications](https://dl.acm.org/doi/abs/10.1145/2847421)
**Year:** 
2015
**Journal:** 
[ACM Transactions on Intelligent Systems and Technology (TIST)](https://dl.acm.org/toc/tist/2016/7/4)
**Simulation:** 
The paper does not introduce any novel simulations, nor does it provide details on how the graph structure was generated.
**Algorithm:** 
The study employs a **parametric flow algorithm** to recursively solve the graph-cutting problem. 
**Code:**
The code is not available as the author's homepage is no longer active. However, some code is referenced in the supplementary materials.
**Memo:** 
Key observations include:   
- Selected critical voxels were well-structured, connected, consistent across cross-validation, and aligned with prior pathological knowledge.   
- The paper notes that the “genlasso” algorithm cannot solve the problem when $N < d$, warranting further investigation into this limitation.`

[Network Lasso: Clustering and Optimization in Large Graphs](https://stanford.edu/~boyd/papers/pdf/network_lasso.pdf)
**Year:** 
2015
**Conference:** 
KDD
**Question:** 
It focus on optimization problems posed on graphs. The optimization problem is defined as: $$ \min_{\{x_i\}_{i \in \mathcal{V}}} \quad \sum_{i \in \mathcal{V}} f_i(x_i) + \sum_{(j, k) \in \mathcal{E}} g_{jk}(x_j, x_k), $$They more focus on the special case: the cost functions $f_i$ are convex, and the edge penalties $g_{jk}$ take the form: $$ g_{jk}(x_j, x_k) = \lambda w_{jk} \|x_j - x_k\|_2, $$where $\lambda \geq 0$ is an overall scaling parameter,  $w_{jk} \geq 0$ are user-defined weights for edges $(j, k)$. The resulting optimization problem is: $$ \min_{\{x_i\}_{i \in \mathcal{V}}} \quad \sum_{i \in \mathcal{V}} f_i(x_i) + \lambda \sum_{(j, k) \in \mathcal{E}} w_{jk} \|x_j - x_k\|_2. $$
**Simulation:** 
The authors performed experiments where: 
- Each node in the graph represents a data point. 
- The signal associated with each node is a **vector** (not a scalar). 
- The objective is to estimate the noise-free signal vector for each node.
While the experiments focus on estimating vector signals, our problem involves a different objective. However, our problem aligns with the broader framework presented in this work. Therefore, the algorithm proposed in this paper can be adapted and used to address our specific requirements.
**Algorithm:** 
ADMM
**Code:** 
python package available [here](https://snap.stanford.edu/snapvx/)

[A Fast and Flexible Algorithm for the Graph-Fused Lasso](https://arxiv.org/pdf/1505.06475)
**Year:** 
2015
**Question:** 
This paper addresses the Graph-Fused Lasso, focusing on the graph denoising problem. However, the approach can be easily adapted for other use cases, including ours. Given a graph decomposed into a set of trails $T = \{t_1, t_2, \dots, t_k\}$, the optimization problem is formulated as: $$ \min_{\beta \in \mathbb{R}^n} \ell(y, \beta) + \lambda \sum_{t \in T} \sum_{(r,s) \in t} |\beta_r - \beta_s|. $$ **Slack Variables:** For each trail $t$ (where $|t| = m$), $m+1$ slack variables are introduced, one for each vertex along the trail. If a vertex is visited more than once in a trail, additional slack variables are added for each visit. This modifies the problem to: $$ \min_{\beta \in \mathbb{R}^n} \ell(y, \beta) + \lambda \sum_{t \in T} \sum_{(r,s) \in t} |z_r - z_s|, $$ subject to: $$ \beta_r = z_r, \quad \beta_s = z_s. $$  **ADMM Routine:** The problem is solved efficiently using ADMM with the following updates: 
1. **$\beta$ Update:** $$ \beta^{k+1} = \arg \min_\beta \left( \ell(y, \beta) + \frac{\alpha}{2} \|A \beta - z^k + u^k\|^2 \right), $$
2. **$z$ Update:** $$ z_t^{k+1} = \arg \min_z \left( w \sum_{r \in t} (\tilde{y}_r - z_r)^2 + \sum_{(r,s) \in t} |z_r - z_s| \right), \quad t \in T, $$
3. **$u$ Update:** $$ u^{k+1} = u^k + A \beta^{k+1} - z^{k+1}. $$
**Parameters:** 
- $u$: Scaled dual variable. 
- $\alpha$: Scalar penalty parameter. 
- $w = \frac{\alpha}{2}$, $\tilde{y}_r = \beta_r - u_r$. 
- $A$: Sparse binary matrix encoding the appropriate $\beta_i$ for each $z_j$. 
**Closed-Form Solution for Squared-Error Loss:** For $\ell(y, \beta) = \sum_i (y_i - \beta_i)^2$, the $\beta$ updates simplify to: $$ \beta_i^{k+1} = \frac{2y_i + \alpha \sum_{j \in \mathcal{J}} (z_j - u_j)}{2 + \alpha |\mathcal{J}|}, $$
where $\mathcal{J}$ represents the set of dual variable indices mapping to $\beta_j$. 
*This approach may yield a similar closed-form solution for our case.* 
**Algorithm:** 
The method uses **ADMM via trail decomposition** to convert the graph problem into a fused lasso problem. This decomposition reduces the problem to one-dimensional fused lasso subproblems, which are solved in linear time using an efficient dynamic programming routine ([Johnson, 2013]). 
**Code:** 
The implementation is available on GitHub: [https://github.com/tansey/gfl](https://github.com/tansey/gfl).

[An augmented ADMM algorithm with application to the generalized lasso problem](https://www.asc.ohio-state.edu/zhu.219/manuscript/gen_lasso_jcgs.pdf)
**Year**: 
2015
**Journal**: 
Journal of Computational and Graphical Statistics
**Questions**: 
The generalized lasso problem is formulated as:  

$$
\text{minimize}_{\beta \in \mathbb{R}^p} \frac{1}{2} \| y - X\beta \|_2^2 + \lambda \| A\beta \|_1,
$$
The problem generalizes the classical lasso ($A = I$) to a broader class of regularization frameworks such as fused lasso, convex clustering, and trend filtering.  

In the numerical experiment part, it mainly deal with the sparse fused Lasso over a graph: 
Let $G = (\mathcal{V}, \mathcal{E})$ be a graph, where $\mathcal{V}$ is the node set and $\mathcal{E}$ is the edge set. The node set $\mathcal{V}$ encodes the features/covariates, and the edge set $\mathcal{E}$ encodes their relationships. 
Based on such a graph, the following optimization problem is defined: $$ \text{minimize}_{\beta \in \mathbb{R}^p} \frac{1}{2} \| y - X\beta \|_2^2 + \lambda \sum_{(i,j) \in \mathcal{E}} |\beta_i - \beta_j| + \nu \cdot \lambda \| \beta \|_1, $$**Simulations**:
- The network contains 200 subnetworks, each represented by a transcription factor (TF). Each subnetwork is a complete graph with 11 nodes (1 TF + 10 genes it regulates, totaling 55 edges). Additional random erroneous edges are added between nodes from different subnetworks.
- Predictors for transcription factors (TFs) are generated from a standard normal distribution $N(0, 1)$. 
- Predictors for each target gene are constructed to have a bivariate normal distribution with the corresponding TF, having a correlation of $0.7$. Target genes are conditionally independent given the TF. 
- The true regression coefficients are defined as: $$ \beta_0 = ( \underbrace{1, \ldots, 1}_{11}, \underbrace{-1, \ldots, -1}_{11}, \underbrace{2, \ldots, 2}_{11}, \underbrace{-2, \ldots, -2}_{11}, \underbrace{0, \ldots, 0}_{p-44} )^\top, $$
- The response vector is defined as $y = X\beta_0 + \epsilon$, where $\epsilon_1, \ldots, \epsilon_n \overset{\text{iid}}{\sim} N(0, \sigma_e^2)$ and the error variance $\sigma_e^2 = 0.1$.

They did running time comparison, convergence rate (use suboptimality)
either for a fixed tuning parameter or for a sequence of tuning parameter
**Dataset**:
No real-world datasets were used; all experiments relied on synthetic data.
**Algorithm**: 
variant of the standard ADMM
**Code**: 
Code is provided in the supplementary materials and implemented in R and MATLAB. 
It includes implementations for standard ADMM, fGFL, and genlasso.

[The DFS Fused Lasso: Linear-Time Denoising over General Graphs](https://www.jmlr.org/papers/volume18/16-532/16-532.pdf)
**Year:** 
2018 (2016)
**Journal:** 
Journal of Machine Learning Research
**Question:**
The fused lasso estimate is defined as the solution of the following convex optimization problem: $$ \hat{\theta}_G = \arg \min_{\theta \in \mathbb{R}^n} \frac{1}{2} \|y - \theta\|_2^2 + \lambda \|\nabla_G \theta\|_1, $$ where $\nabla_G \in \mathbb{R}^{m \times n}$ is the **edge incidence matrix** of the graph $G$.

Edge Incidence Matrix $\nabla_G$:  The edge incidence matrix $\nabla_G$ and the fused lasso solution $\hat{\theta}_G$ are defined with respect to the graph $G$.  
- For each edge $e \in E$, an **arbitrary orientation** is assigned to the edge.   
- One vertex is selected as the **head** ($e^+$).   
- The other vertex is selected as the **tail** ($e^-$).  
- The corresponding row $(\nabla_G)_e$ of $\nabla_G$, for edge $e$, is defined as:   $$   (\nabla_G)_{e,e^+} = 1, \quad (\nabla_G)_{e,e^-} = -1, \quad (\nabla_G)_{e,v} = 0 \; \text{for all } v \not= e^+, e^-.   $$
- Total Variation Over the Graph:  For an arbitrary $\theta \in \mathbb{R}^n$, the $\ell_1$-norm of the gradient $\nabla_G \theta$ is given by: $$ \|\nabla_G \theta\|_1 = \sum_{e \in E} |\theta_{e^+} - \theta_{e^-}|. $$
- Key Observation: The particular choice of orientation for the edges does not affect the value of $\|\nabla_G \theta\|_1$, which is referred to as the **total variation** of $\theta$ over the graph $G$.`
**Algorithm:** 
The algorithm primarily targets signal denoising (and it might be possible to convert it to our problem).
Given a general graph, if we run the standard depth-first search (DFS) traversal algorithm, then the total variation of any signal over the chain graph induced by DFS is no more than twice its total variation over the original graph.
Finally, several related results also hold—for example, the analogous result holds for a roughness measure defined by the $\ell_0$ norm of differences across edges in place of the total variation metric.
**Code:** 
The original implementation was not found, possible alternatives include: 
- [Parallel Cut Pursuit Issues on GitHub](https://github.com/1a7r0ch3/parallel-cut-pursuit/issues/1) 
- [Parallel Cut Pursuit Repository on GitLab](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit)


[Generalized fused group lasso regularized multi-task feature learning for predicting cognitive outcomes in Alzheimers disease](https://www.sciencedirect.com/science/article/abs/pii/S0169260717312142?via%3Dihub)
**Year:**
2018
**Journal:** 
Computer Methods and Programs in Biomedicine
**Question:** 
The objective function of group guided fused Laplacian sparse group Lasso (GFL-SGL) is given in the following optimization problem: $$\min_{\Theta} \frac{1}{2} \|\Lambda \odot (Y - X\Theta)\|_F^2 + R_{\lambda_1}^{\lambda_2}(\Theta) + \lambda_3 \|\Theta \mathcal{D}\|_{G_{2,1}^F},$$ 
where $R_{\lambda_1}^{\lambda_2}(\Theta) = \lambda_1 \|\Theta\|_1 + \lambda_2 \|\Theta\|_{G_{2,1}^c}$ ,$\lambda_1, \lambda_2, \lambda_3$ are the regularization parameters, weights are figured out with the help of a Gaussian kernel: $$w_{\ell,t} = \frac{\exp(-((\ell - t)^2 / \sigma^2))}{\sum_{\ell' = 1, \ell' \neq t}^k \exp(-((\ell' - t)^2 / \sigma^2))}, \quad \ell, \ell' = 1, \dots, k, \ell \neq t$$
and 
$$\|\Theta \mathcal{D}\|_{G_{2,1}^F} = \sum_{t=1}^k \sum_{l=1}^q w_l \| \theta_{\mathcal{G}_l^t} - \sum_{\ell=1, \ell \neq t}^k w_{\ell,t} \theta_{\mathcal{G}_l^\ell} \|_2.$$ 
**Algorithm:** 
The paper employs the **Alternating Direction Method of Multipliers (ADMM)** to solve the optimization problem. 
**Code:**
The code link provided in the paper is dead.
**Memo:**
Useful theoretical analysis
This paper formulates the problem as a multitask learning framework with three key objectives: 
- **Smoothing:** Capturing temporal structure using fused lasso. 
- **Sparsity:** Selecting relevant features through sparse group lasso. 
- **MSE Minimization:** Reducing prediction error. The underlying graph in this problem represents **time**, where nodes correspond to time points. The method also addresses missing values in clinical data.

The clinical score data are incomplete at some time points for many patients, i.e., there may be no values in the target vector $y_i \in \mathbb{R}^k$. In order not to reduce the number of samples significantly, we use a matrix $\Lambda \in \mathbb{R}^{n \times k}$ to indicate incomplete target vectors instead of simply removing all the patients with missing values. Let $\Lambda_{i,j} = 0$ if the target value of sample $i$ is missing at the $j$-th time point, and $\Lambda_{i,j} = 1$ otherwise. We use the componentwise operator $\odot$ as follows: $Z = A \odot B$ denotes $z_{i,j} = a_{i,j} b_{i,j}$ for all $i, j$. 

[A graph decomposition-based approach for the graph-fused lasso](https://www.math.mcgill.ca/yyang/resources/papers/JSPI_GFLasso.pdf)
**Year:** 
2019
**Journal:** 
Journal of Statistical Planning and Inference
**Question:** 
The paper studies the following optimization problem: $$ \min_{\{x_i\}_{i \in \mathcal{V}} \subset \mathbb{R}^p} \sum_{i \in \mathcal{V}} f_i(x_i) + \lambda \sum_{(s, t) \in \mathcal{E}} \|x_t - x_s\|, $$This is essentially a **network lasso**, where the algorithm improves upon the Alternating Direction Method of Multipliers (ADMM) in terms of computational cost. - For $p > 1$, the approach generalizes to cases where each node corresponds to a vector rather than a scalar. - It can still handle simpler cases like ours.
The original optimization problem can be rewritten as:

$$
\{\hat{\mathbf{x}}_i\}_{i \in \mathcal{V}} = \arg\min_{\{\mathbf{x}_i\}_{i \in \mathcal{V}}, \{\mathbf{z}_{st}, \mathbf{z}_{ts}\}_{(s, t) \in \mathcal{E}}} \sum_{i \in \mathcal{V}} f_i(\mathbf{x}_i) + \lambda \sum_{(s, t) \in \mathcal{E}} \|\mathbf{z}_{st} - \mathbf{z}_{ts}\|,
$$

subject to $\mathbf{x}_s = \mathbf{z}_{st} \quad \text{and} \quad \mathbf{x}_t = \mathbf{z}_{ts}$,  for all $(s, t) \in \mathcal{E}$. 
This can allow for starting iteration of ADMM.
**Simulations**:
The simulations claim to follow Zhu (2017), but the details appear quite different. The specifics of the simulation design are not entirely clear.
**Algorithm:**
The approach divides the graph into two parts such that one does not contain any neighboring edges. This decomposition allows for efficient updates during the optimization process.
**Code:** 
Available [here](https://github.com/alexfengg/novelGFL)
**Memo:** 
The paper provides an extensive survey and summary of methods to solve the network lasso. Additional resources: 
- [https://www.math.mcgill.ca/yyang/soft.html](https://www.math.mcgill.ca/yyang/soft.html) 
- [Distributed Optimization and Statistical Learning via ADMM](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) 
- [An Efficient Algorithm for a Class of Fused Lasso Problems](https://dl.acm.org/doi/10.1145/1835804.1835847)

[Coordinate Descent Algorithm of Generalized Fused Lasso Logistic Regression for Multivariate Trend Filtering](https://link-springer-com.proxyiub.uits.iu.edu/article/10.1007/s42081-022-00162-2)
**Year:** 
2022
**Journal:**  
Japanese Journal of Statistics and Data Science
**Simulation:** 
The simulation methodology is based on [Spatio-Temporal Adaptive Fused Lasso for Proportion Data](https://link.springer.com/chapter/10.1007/978-981-16-2765-1_40), but the content of this reference is inaccessible.
**Algorithm:**
The paper introduces a **coordinate descent algorithm** with the following structure: 
1. **Descent Cycle:** Updates the model parameters by minimizing the objective function iteratively for each coordinate. 
2. **Fusion Cycle:** Adjusts parameters to enforce smoothness or fusion between neighboring variables, based on predefined structure. 
This algorithm builds on the approach presented in [A Coordinate-Wise Optimization Algorithm for the Fused Lasso](https://arxiv.org/pdf/1011.6409).
**Memo:**
This work builds on prior research involving Generalized Linear Models (GLMs). Related references include: 
- [Generalized Fused Lasso for Grouped Data in Generalized Linear Models](https://link.springer.com/article/10.1007/s11222-024-10433-5) (Code:[https://github.com/ohishim/GFLglm](https://github.com/ohishim/GFLglm) (R package)).

[Fused lasso for feature selection using structural information](https://arxiv.org/pdf/1902.09947)
**Year:** 
2021
**Journal:** 
Pattern Recognition
**Memo:** 
This paper proposes a novel **negative term** in the fused lasso formulation to address redundancy in feature selection. By penalizing redundant features, the method can enhance the interpretability of the model. 
This approach could potentially help in solving the **dilution issue** in our context when we do random rounding.

[Sparse Laplacian Shrinkage with the Graphical Lasso Estimator for Regression Problems](https://link.springer.com/article/10.1007/s11749-021-00779-7)
**Year:** 
2021(2019)
**Journal:** 
Test
Question: it mainly focus on this 
$$
\hat{\beta} = \arg \min_\beta \left\{ \frac{1}{2} \|y - X\beta\|_2^2 + P_{\lambda_1}(\beta) + \frac{\lambda_2}{2} \sum_{1 \leq j < j' \leq p} |\hat{\Theta}_{jj'}| (\beta_j - \beta_{j'})^2 \right\}.
$$
More precisely, on this 
$$ \hat{\beta} = \arg \min_\beta \left\{ \frac{1}{2} \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \frac{\lambda_2}{2} \beta^\top \hat{\Gamma} \beta \right\}. $$

The second term is a Laplacian quadratic penalty (Chung, 1997) and the assigned weights to the $\ell_2$ norm come from the adjacency matrix. In this paper, we construct the adjacency matrix by the estimated precision matrix $\hat{\Theta}$. The estimated Laplacian matrix is denoted by: $$ \hat{\Gamma} = \hat{D} - \hat{\Theta}, $$where $\hat{D} = \text{diag}(\hat{d}_1, \dots, \hat{d}_p)$ and $\hat{d}_j = \sum_{j'=1}^p |\hat{\Theta}_{jj'}|$. 
Note: The difference is the graph is formed/estimated directly from the data, not prior information.
**Simulation:** 
It is a study base on the paper on Annals of Statistics (2011): [The sparse Laplacian shrinkage estimator for high-dimensional regression](https://projecteuclid.org/journals/annals-of-statistics/volume-39/issue-4/The-sparse-Laplacian-shrinkage-estimator-for-high-dimensional-regression/10.1214/11-AOS897.full)
The data generation in this paper involves simulating covariates and response variables with predefined characteristics. 
- The dataset includes 500 covariates grouped into 100 clusters, each containing 5 covariates.
	- **Structure I:** Covariates within the same cluster have correlation coefficients $\rho^{|i-j|}$, while covariates across clusters are independent. 
	- **Structure II:** All covariates have correlation coefficients $\rho^{|i-j|}$. 
- Covariates follow a normal distribution with mean 0 and variance 1. 
- Among the 500 covariates, the first 25 (5 clusters) are assigned nonzero regression coefficients. 
- Two settings are used for these coefficients: 
	1. All nonzero coefficients are set to 0.5. 
	2. Nonzero coefficients are drawn uniformly from $[0.25, 0.75]$. 
- Sample size ($n$) is set to 100. 
- Random error terms are added to simulate the response variable.
**Algorithm:**
Coordinate descent algorithm
**Memo:**
A potential useful transformation: given dataset $(y, X)$, the estimation from the first step $\hat{\Theta}$ and parameter $\lambda_2$, define an artificial dataset $(y^*, X^*)$ by: $$ X^*_{(n+p)\times p} = (1 + \lambda_2)^{-1/2} \begin{pmatrix} X \\ \sqrt{\lambda_2} \hat{L} \end{pmatrix}, \quad y^*_{(n+p)} = \begin{pmatrix} y \\ 0 \end{pmatrix}, $$ where $\hat{L}$ is a $p \times p$ matrix that $\hat{L}^\top \hat{L} = \hat{\Gamma}$. The problem criterion can be written as: $$ \hat{\beta}^* := \arg \min_\beta \left\{ \frac{1}{2n} \|y^* - X^* \beta \|_2^2 + P_{\lambda_1}(\beta) \right\}, $$ then: $$ \hat{\beta} = \frac{1}{\sqrt{1 + \lambda_2}} \hat{\beta}^*. $$
 [An Alternating Direction Method of Multipliers Algorithm for the Weighted Fused LASSO Signal Approximator](https://arxiv.org/pdf/2407.18077) 
 **Year:** 
 2024
 **Questions:** 
 wFLSA(Weighted Fused LASSO Signal Approximator)
 **Algorithm:** 
This work builds upon [An Augmented ADMM Algorithm with Application to the Generalized Lasso Problem](https://www.asc.ohio-state.edu/zhu.219/manuscript/gen_lasso_jcgs.pdf). 
Key contributions include: 
- A novel choice of the **matrix $Q$** in the $\beta$-update step, designed to accelerate convergence. 
- The design matrix used in their problem setup is the **identity matrix**, which may limit direct applicability to more complex setups (e.g., non-identity design matrices). 
The algorithmic framework remains within the ADMM paradigm, leveraging weighted penalties for improved performance in signal approximation tasks.
 **Code**: 
 https://github.com/bips-hb/wflsa

[GFLASSO-LR: Logistic Regression with Generalized Fused LASSO for Gene Selection in High-Dimensional Cancer Classification](https://www.mdpi.com/2073-431X/13/4/93)
**Year:** 
2024
**Journal:** 
Computers
**Question:** 
It add logistic function to get classification, and the difference is weighted
$$ h_{\lambda_1, \lambda_2}(\beta) = g(\beta) + \lambda_1 \sum_{j=0}^N |\beta_j| + \lambda_2 \sum_{(k,l) \in \mathcal{E}, k < l} w_{k,l} |\beta_k - \beta_l|, \quad \forall \beta \in \mathbb{R}^{N+1}. $$
where the function $g(\beta)$ is defined as: $$ g(\beta) = -\sum_{i=1}^M \left( y_i x_i' \beta - \ln\left( 1 + e^{x_i' \beta} \right) \right), $$ which can be rewritten as: $$ g(\beta) = -\sum_{i=1}^M y_i x_i' \beta + \sum_{i=1}^M \ln\left( 1 + e^{x_i' \beta} \right), \quad \forall \beta \in \mathbb{R}^{N+1}. $$
**Algorithm:**
The method uses: 
1. **Subgradient Optimization:** For solving the fused lasso with logistic regression. 
2. **Pearson Filter:** Preprocessing step to select the best genes before applying fused lasso, which reduces the dimensionality of the data.
**Code:** 
Code availability is uncertain, and the method may not be directly relevant to current work. It can be revisited later if necessary.
**Memo:** 
The paper addresses challenges in high-dimensional cancer classification, where microarray experiments generate datasets with numerous genes but few samples. This imbalance leads to computational instability and the curse of dimensionality, making **gene selection** a critical but difficult task.

### (Maybe) Related Paper
https://scholar.google.com/scholar?as_ylo=2024&q=sparse+generalized+fused+lasso+over+graph&hl=en&as_sdt=0,15

[Learning with Structured Sparsity](https://icml.cc/Conferences/2009/papers/452.pdf)  
**Conference:** ICML 2009   
Develops a **general theory** for learning with structured sparsity, based on coding complexity associated with the structure.  

Proposes a **structured greedy algorithm** to efficiently solve structured sparsity problems.  
Experiments demonstrate the advantages of structured sparsity over standard sparsity in practical applications.

[Optimization with Sparsity-Inducing Penalties](https://www.di.ens.fr/~fbach/bach_jenatton_mairal_obozinski_FOT.pdf)  
Monograph (108 pages)  
Provides a **comprehensive review** of optimization techniques with sparsity-inducing penalties.  
Covers theoretical foundations and applications in structured sparsity optimization problems.

[Sparse-Group Lasso for Graph Learning From Multi-Attribute Data](https://ieeexplore.ieee.org/abstract/document/9350215)  
Applies **sparse-group lasso** for graph learning with multi-attribute data.  
Nodes in the graph represent **vectors**, enabling applications to higher-dimensional problems.

[The Generalized Lasso Problem and Uniqueness](https://www.stat.berkeley.edu/~ryantibs/papers/genlasuni.pdf)  
Explores the **uniqueness of solutions** for the generalized lasso problem.  
Proves that solutions are almost surely unique, providing stability and interpretability benefits.

[Spatial Smoothing and Hot Spot Detection for CGH Data Using the Fused Lasso](https://tibshirani.su.domains/ftp/fusedLassoCGH.pdf)  
**Journal:** Biostatistics (2007)  
An **early fused lasso formulation** with constraints, focusing on smoothing and hot spot detection in CGH (Comparative Genomic Hybridization) data.  
Discusses methods to determine the sparsity level and balance between smoothness and sparsity.

[Multi-block Linearized Alternating Direction Method for Sparse Fused Lasso Modeling Problems](https://www.sciencedirect.com/science/article/pii/S0307904X24004475)
Focuses on **sparse fused lasso** without graph structure.  
**Code:**
[https://github.com/xfwu1016/LADMM-for-qfLasso](https://github.com/xfwu1016/LADMM-for-qfLasso).  
**Memo:** 
Explores **quantile loss**, which may have applications in robust regression.

[Linearized Alternating Direction Method of Multipliers for Separable Convex Optimization of Real Functions in the Complex Domain](http://www.jaac-online.com/article/doi/10.11948/20180256)  
**Journal:** 
Journal of Applied Analysis and Computation (2019)  
**Memo:**  
Introduces **complex analysis** (VI, Wirtinger Calculus) to establish ADMM convergence.  The approach seems adaptable to generalized fused lasso (GFL) problems.

[A Systematic Review of Structured Sparse Learning](https://link.springer.com/article/10.1631/FITEE.1601489)  
**Year:** 2017  
Provides an **overview table** summarizing formulations, algorithms, and software for lasso and its structured extensions.  
Offers a comprehensive review of optimization methods in structured sparse learning.
