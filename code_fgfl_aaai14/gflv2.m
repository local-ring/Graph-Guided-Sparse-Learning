% Load data from .mat file
data = load('gfl_data.mat');

X = data.X;            % Design matrix
y = data.y;            % Response vector
AdjMat = data.AdjMat;  % Adjacency matrix
rho1 = data.pho;       % Tuning parameter rho1
rho2 = data.lamb;      % Tuning parameter rho2

% Convert adjacency matrix to graph structure
[nE, E_in, E_out, E_w] = adj_matrix_to_graph(AdjMat);

% Graph structure required by fast_gfl
Graph = {nE, E_w, E_in, E_out};

% Options for fast_gfl
opts.maxIter = 1000;
opts.tol = 1e-4;

% Call the fast_gfl function
[beta, funcVal] = fast_gfl(X, y, Graph, rho1, rho2, opts);

% Display results
disp('Optimized beta:');
disp(beta);
disp('Function values:');
disp(funcVal);

% Function to process adjacency matrix
function [nE, E_in, E_out, E_w] = adj_matrix_to_graph(AdjMat)
    [rows, cols] = find(AdjMat > 0); % Find nonzero entries
    nE = length(rows);              % Number of edges
    E_in = rows;                    % Starting nodes
    E_out = cols;                   % Ending nodes
    E_w = AdjMat(sub2ind(size(AdjMat), rows, cols)); % Edge weights
end

stem(beta, 'g', 'filled'); % Estimate from fast_gfl
title('GFL Estimate on corr (beta)');
xlabel('Index');
ylabel('Value');

save('beta.mat', 'beta');
