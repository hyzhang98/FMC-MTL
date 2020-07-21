function [Xtrain, Ytrain, Xtest, Ytest] = select_sample(X, Y, n)
    [d, N] = size(X);
    [~, k] = size(Y);
    indeces = randperm(N, n + 5000);
    Xtrain = zeros(d, n);
    Ytrain = zeros(n, k);
    Xtest = zeros(d, 5000);
    Ytest = zeros(5000, k);
    for i = 1: n + 5000
        ind = indeces(i);
        if i <= n 
            Xtrain(:, i) = X(:, ind);
            Ytrain(i, :) = Y(ind, :);
        else 
            Xtest(:, i-n) = X(:, ind);
            Ytest(i-n, :) = Y(ind, :);
        end
    end
end