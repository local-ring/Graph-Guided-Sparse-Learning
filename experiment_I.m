clc;
clear;

% Timing start
tStart = tic;

% Generate synthetic data
n = 1000; % Number of samples
d = 500;  % Number of features
k = 20;   % Number of non-zero features
h_total = 40; % Number of clusters in the graph
h = 5;        % Number of selected clusters
nVars = d * h; % Number of Boolean variables in m
inter_cluster = 1; % Probability of inter-cluster edges in the graph
outer_cluster = 0.05; % Probability of outer-cluster edges in the graph
gamma = 1.5; % Noise standard deviation
pho = sqrt(n);
mu = 0.1;

SNR = 1;

fixed_seed = false;
random_rounding = false;
connected = false;
correlated = true;
random_graph = true;
visualize = true;

% Generate or read synthetic data
if fixed_seed
    file_path = "data/synthetic_data.mat";
    load(file_path, 'X', 'w_true', 'y', 'adj_matrix', 'L', 'clusters_true', 'selected_features_true');
else
    % Generate synthetic data
    [X, w_true, y, adj_matrix, L, clusters_true, k] = generateSyntheticDataWithGraph( ...
        n, d, h_total, h, inter_cluster, outer_cluster, gamma, visualize, connected, correlated, random_graph);
    % Save synthetic data
    file_path = "synthetic_data.mat";
    save(file_path, 'X', 'w_true', 'y', 'adj_matrix', 'L', 'clusters_true', 'selected_features_true');
end

% Define constants
clusters_size = cellfun(@length, clusters_true);
clusters_size = sort(clusters_size);
C = (2 * clusters_size(end) + outer_cluster * (d - clusters_size(1) - clusters_size(2))) + 2 * d;

% Modify X for objective function
X_hat = repmat(X, 1, h);

fprintf('Check!\n');
fprintf('Execution time (data generation): %.2f seconds\n', toc(tStart));

% Initial guess of parameters
m_initial = ones(nVars, 1) * (k / nVars);

% Objective function
funObj = @(m) L0Obj(X_hat, m, y, L, pho, mu, d, h, n, C);

% Projection function (Gurobi or other solver)
funProj = @(m) ProjOperatorGurobi(m, k, d, h);

% Solve with PQN
options.maxIter = 100;
options.verbose = 3;

[mout, obj, fun_evals] = minConF_PQN(funObj, m_initial, funProj, options);

fprintf('Solution:\n');
disp(mout);

% Save results
save('mout.mat', 'mout');

% Additional analysis and metrics (to be implemented based on MATLAB capabilities)