function nmse = cal_nmse(y_pred, y)
    % y_pred & y : n * m, n is the amount of data points, m is the amount of tasks or dimensions
    [n, ~] = size(y);
    assert(isequal(size(y_pred), size(y)));
    res = sum((y_pred - y).^2) / n;
    d = sqrt(sum(y.^2));
    nmse = sum(res ./ d);
end