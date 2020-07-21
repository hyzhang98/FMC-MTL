function [Xtrain, Ytrain, Xtest, Ytest] = gen_noised_dataset(X, Y, n, r, is_gaussian)
    [Xtrain, Ytrain, Xtest, Ytest] = select_sample(X, Y, n);
    m = round(n * r);
    li = randperm(n, m);
    [~, k] = size(Y);
    for i = 1: m 
        if is_gaussian
            Ytrain(li(i), :) = Ytrain(li(i), :) + randn(1, k) * 50;
        else
            Ytrain(li(i), :) = Ytrain(li(i), :) .* (rand(1, k) * 10);
        end
    end
end