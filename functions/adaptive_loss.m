function val = adaptive_loss(X, sigma)
    t = sum(X.^2, 2);
    sqrt_t = sqrt(t);
    t = t ./ (sqrt_t + sigma);
    val = sum(t) * (1 + sigma);
end