function solution = ProjOperator_Gurobi(m, k, d, h)
    % Projects the input vector m onto the simplex using Gurobi, ensuring constraints.
    
    % Inputs:
    %   m: Input vector of size (d * h).
    %   k: Sparsity level (number of non-zero entries allowed).
    %   d: Number of features.
    %   h: Number of clusters.
    
    % Outputs:
    %   solution: Projected vector of size (d * h).
    
        % Define the constraint matrices
        A = ones(1, d * h);       % Sparsity constraint
        b = k;                    % Sparsity constraint bound
    
        % Feature assignment constraints
        B = zeros(d, d * h);
        for i = 1:d
            B(i, (i - 1) * h + 1:i * h) = 1;
        end
        c = ones(d, 1);
    
        % Combine all constraints
        C = [A; B];
        Cb = [b; c];
    
        % Create Gurobi model
        model.modelsense = 'min';  % Minimization problem
        model.Q = sparse(2 * eye(d * h));  % Quadratic terms (2 for scaling)
        model.obj = -2 * m(:)';  % Linear terms
        model.A = sparse(C);
        model.rhs = Cb(:);
        model.sense = '<';  % All constraints are inequalities
        model.lb = zeros(d * h, 1);  % Lower bound for x
        model.ub = ones(d * h, 1);   % Upper bound for x
    
        % Gurobi options
        params.outputflag = 0;  % Suppress solver output
    
        % Solve using Gurobi
        result = gurobi(model, params);
    
        % Handle infeasibility
        if strcmp(result.status, 'INFEASIBLE')
            warning('Model is infeasible. Returning default solution.');
            solution = zeros(d * h, 1);
            return;
        end
    
        % Return the solution
        solution = result.x;
    end

function D = construct_difference_matrix(m, n)
    % Constructs the difference matrix D for variables indexed by (i, j),
    % representing pairwise differences (x_{i,j} - x_{i,j-1}).
    
    % Inputs:
    %   m: Number of rows (features or dimensions, "d").
    %   n: Number of columns (clusters, "h").
    
    % Outputs:
    %   D: Sparse matrix representing pairwise differences.
    
        % Number of rows in D is m * (n-1)
        % Number of columns in D is m * n
        num_rows = m * (n - 1);
        num_cols = m * n;
    
        % Initialize sparse matrix
        D = spalloc(num_rows, num_cols, 2 * num_rows);
    
        % Fill the matrix with +1 and -1 for each difference x_{i,j} - x_{i,j-1}
        for i = 1:m
            for j = 2:n
                row_idx = (i - 1) * (n - 1) + (j - 1);  % Row in D
                col_plus = (i - 1) * n + j;            % x_{i,j} position (+1)
                col_minus = (i - 1) * n + (j - 1);     % x_{i,j-1} position (-1)
    
                D(row_idx, col_plus) = 1;
                D(row_idx, col_minus) = -1;
            end
        end
    end

