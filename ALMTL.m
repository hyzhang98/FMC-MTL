function [W, e, Sigma, alpha, funcVals] = ALMTL(X, Y, sigma, alpha, lambda, ksi, max_iter)
    % Written by Hongyuan Zhang, 2019.
    % This is for FMC-MTL(special case, all tasks share a common training set).
    % -------------- input: --------------
    %   ###### Notations ######
    %   n: no. of samples
    %   d: dimensions of samples
    %   #######################
    % 
    %   X: d * n, training samples
    %   Y: n * k, training labels
    %   sigma, lambda, ksi, alpha: the details about these parameters can be found in the paper.
    %   max_iter: maximum iteration
    % 
    %   In particular, we set "lambda = 10^-2" and "ksi=10^-3".
    % 
    % -------------- output: --------------
    %   W: d * k, learned projection coefficents
    %   e: k * 1, learn bias
    %   Sigma - d * d
    %   alpha: it will be activated when alpha is treated as a learned parameter.
    %   funcVals: objective values with iterations

    [d, ~] = size(X);
    [~, k] = size(Y);

    if ~exist('lambda', 'var')
        max_iter = 10^-2;
    end

    if ~exist('ksi', 'var')
        max_iter = 10^-3;
    end

    if ~exist('max_iter', 'var')
        max_iter = 100;
    end

    epsilon = 5 * 10^-4;

    % init
    W = rand(d, k);
    b = rand(k, 1);
    Sigma = eye(d) * d;
    % alpha = 0.02;
    D = 0;
    funcVals = zeros(max_iter, 1);
    
    % old = 0;
    for i = 1:max_iter
        old = D;
        % compute D, D: n * 1, old version: n * n
        D = cal_D(X, W / alpha, b / alpha, Y, sigma);

        if norm(D - old) < epsilon
            funcVals = funcVals(1:i-1);
            break;
        end

        % X_tilde = X * H' which save the memory
        X_tilde = X * D / sum(D); % d * 1
        X_tilde = X - X_tilde;

        Y_tilde = D' * Y / sum(D);
        Y_tilde = Y - Y_tilde;


        % calculate b = alpha * e
        b = (alpha * Y' - W' * X) * D / sum(D);


        % calculate alpha adaptively
        % alpha = trace(W' * X * H' * D * H * Y) / trace(Y' * H' * D * H * Y);
        % alpha = trace(W' * X * H' * D * H * Y) / (trace(Y' * H' * D * H * Y) + lambda * ksi * trace(Sigma));
        % alpha = trace((W' * X_tilde .* D') * Y_tilde) / (trace((Y_tilde' .* D') * Y_tilde) + lambda * ksi * trace(Sigma));
        % alpha = max(abs(alpha), 10^-1) * sign(alpha);
        % alpha = 0.02;


        % update Sigma
        T = W * W' + alpha^2 * ksi * eye(d);
        T = max(T, T');
        T = T^(1/2);
        T = max(T, T');
        Sigma = T^(-1) * trace(T);


        % update W
        T = ((X_tilde .* D') * X_tilde' + lambda * Sigma);
        T = max(T, T');
        T = T^(-1/2);
        M = T * (X_tilde .* D') * Y_tilde;
        [U, ~, V] = svd(M);
        % Z: k * d
        if k < d 
            Z = [eye(k), zeros(k, d - k)];
        else
            Z = [eye(d); zeros(k - d, d)];
        end

        W = T * U * Z' * V';

        funcVals(i) = cal_obj(X, W, b, Y, Sigma, alpha, sigma, lambda, ksi);

    end

    % rescale back
    e = b / alpha;
    W = W / alpha;
end

function D = cal_D(X, W, e, Y, sigma)
    % D: n * n
    [n, ~] = size(Y);
    T = X' * W + repmat(e', n, 1) - Y;
    t = sqrt(sum(T.^2, 2));
    t = (t + 2 * sigma) ./ ((t + sigma).^2);
    D = ((1 + sigma) / 2) * (t); % save the memory
end

function val = cal_obj(X, W, b, Y, Sigma, alpha, sigma, lambda, xi)
    W = W / alpha;
    e = b / alpha;
    T = X' * W + e' - Y;
    val = adaptive_loss(T, sigma);
    val = val /  size(X, 2) + lambda * (trace(W' * Sigma * W) + xi * trace(Sigma));
end
