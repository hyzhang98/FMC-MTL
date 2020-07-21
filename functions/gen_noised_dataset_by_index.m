function [Xtrain, Ytrain, Xtest, Ytest, tr_index, test_index] = gen_noised_dataset_by_index(X, y, train_ratio, r, tasks_index, is_gaussian)
    %% generate training sets and test sets with noises
    % The split of training set and test set is based on each task.
    % In other words, for the i-th task, we select training_ratio samples of X_i
    %% ----------input----------
    % X: d * N, where N = n_1 + n_2 + ... + n_k and k is number of tasks
    % y: N * 1
    % training_ratio: the ratio of training set in the whole dataset
    % r: pollution ratio
    % tasks_index: k * 1, the starting indeces of k tasks
    %% ----------output----------
    % Xtrain: d * n_train, where n_train = n_1_train + ... + n_k_train
    % Ytrain: n_train * k
    % Xtest: d * n_test, where n_test = n_1_test + ... + n_k_test
    % Ytest: n_test * k
    % tr_index: starting indeces of k tasks in training set
    % test_index: starting indeces of k tasks in test set
    %% function body

    [Xtrain, Ytrain, Xtest, Ytest, tr_index, test_index] = select_sample_by_index(X, y, train_ratio, tasks_index);
    training_size = size(Xtrain, 2);
    m = round(training_size * r);
    li = randperm(training_size, m);
    [~, k] = size(Ytrain);
    for i = 1: m 
        if is_gaussian
            Ytrain(li(i), :) = Ytrain(li(i), :) + randn(1, k) * 50;
        else
            Ytrain(li(i), :) = Ytrain(li(i), :) .* (rand(1, k) * 10);
        end
    end
end

function [Xtrain, Ytrain, Xtest, Ytest, tr_index, test_index] = select_sample_by_index(X, y, train_ratio, tasks_index)
    % ----------output----------
    % Xtrain, Xtest: d * n
    % Ytrain, Ytest: n * k
    [d, N] = size(X);
    k = length(tasks_index);
    t1 = [tasks_index(2:end);N+1];
    tasks_size = t1 - tasks_index;
    train_tasks_size = double(int16(tasks_size * train_ratio));
    test_tasks_size = tasks_size - train_tasks_size;
    training_size = sum(train_tasks_size);

    Xtrain = zeros(d, training_size);
    Xtest = zeros(d, N - training_size);
    Ytrain = zeros(training_size, k);
    Ytest = zeros(N - training_size, k);
    tr_index = zeros(k, 1);
    test_index = zeros(k, 1);
    train_start = 1;
    test_start = 1;
    for i = 1:k 
        % generate training sets
        tr_index(i) = train_start;
        train_end = train_start+train_tasks_size(i);
        Xtrain(:, train_start:train_end-1) = X(:, tasks_index(i):tasks_index(i)+train_tasks_size(i)-1);
        Ytrain(train_start:train_end-1, i) = y(tasks_index(i):tasks_index(i)+train_tasks_size(i)-1);
        train_start = train_end;
        % generate test sets
        test_index(i) = test_start;
        test_end = test_start + test_tasks_size(i);
        test_low = tasks_index(i)+train_tasks_size(i);
        test_high = tasks_index(i) + tasks_size(i) - 1;
        Xtest(:, test_start:test_end-1) = X(:, test_low:test_high);
        Ytest(test_start:test_end-1, i) = y(test_low:test_high);
        test_start = test_end;
    end
end