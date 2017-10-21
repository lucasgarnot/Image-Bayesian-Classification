function [w_hat, variance] = g1_train_linear_regression(data,labels,K )
% initialize parameters
t = clock;
data = double(data);
labels = double(labels);
[N, size_point]= size(data);
X = ones(N,1+size_point*K);

% Compute X for K dimensions using the data
for k = 1:K
    X(:,2+size_point*(k-1):1+size_point*k) = data.^k;
end

% Compute w_hat
w_hat = pinv(X'*X)*X'*labels;

% Compute the value of the variance for the lokelihood optimization
variance = 0;
% variance = (labels-X*w_hat)'*(labels-X*w_hat)/N;

excecution_time =clock - t
end

