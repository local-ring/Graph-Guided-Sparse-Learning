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

