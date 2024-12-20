function [f, g] = L0Obj(X, m, y, L, pho, mu, d, h, n, C)
    % Computes the objective function value and gradient for L0 regularized least squares.
    
    % Inputs:
    %   X: n x dh matrix of features
    %   m: dh x 1 vector (flattened assignment matrix)
    %   y: n x 1 vector of labels
    %   L: d x d graph Laplacian matrix
    %   pho: regularization parameter for L2 penalty
    %   mu: regularization parameter for L0 graph penalty
    %   d: Number of features
    %   h: Number of clusters
    %   n: Number of samples
    %   C: Regularization parameter for correction term (default = 1)
    
    % Outputs:
    %   f: Objective function value
    %   g: Gradient vector
    
        % Ensure dimensions match
        [nRows, dh] = size(X);
        if d * h ~= dh
            error('The dimensions of X and d*h do not match.');
        end
    
        % Construct sparse diagonal matrix
        SpDiag = spdiags(m(:), 0, dh, dh);
    
        % Compute B matrix
        B = inv((1 / pho) * X * SpDiag * X' + n * eye(nRows));
    
        % Generate assignment matrix
        assignment_matrix = reshape(m, d, h);
    
        % Graph penalty term
        graph_penalty = mu * trace(assignment_matrix' * L * assignment_matrix);
    
        % Precision penalty term
        precision_penalty = y' * B * y;
    
        % Compute correction term
        MTM = assignment_matrix' * assignment_matrix;
        correction_term = sum(MTM(:)) - sum(diag(MTM));
    
        % Compute objective function value
        f = precision_penalty + graph_penalty + correction_term;
    
        % Gradients
        A_grad = -(1 / pho) * ((X' * B * y).^2);
        B_grad = 2 * mu * (L * assignment_matrix);
        row_sums = sum(assignment_matrix, 2);
        gradient = 2 * (row_sums - assignment_matrix);
        g = A_grad(:) + B_grad(:) + C * gradient(:);
    end