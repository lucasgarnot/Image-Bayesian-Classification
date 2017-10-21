function [ predictions, confusion_matrix, accuracy ] = g2_predict_linear_regression(data_test, labels_test, w_hat,variance, K)
% Initialize parameters
data_test = double(data_test);
labels_test = double(labels_test);
[M,size_point] = size(data_test);
max_label = max(labels_test);
min_label = min(labels_test);

% Compute X_test
X_test = ones(M,1+size_point*K);
for k = 1:K
    X_test(:,2+size_point*(k-1):1+size_point*k) = data_test.^k;
end

% Add a gaussian noise generated with the variance (no noise if the
% variance parameter is set to zero
switch variance
    case 0
        noise = zeros(M,1);
    otherwise
        noise = normrnd(0,sqrt(variance),[M 1]);
end

% Compute the predictions
predictions = X_test*w_hat + noise;

% We have observed that the algorithm have difficulties to predict extreme
% values as 0 or 9. Therefore we choose a different method of rounding
% predictions depending on wether the prediction is below 5 or above 5
predictions(predictions>5.5)= ceil(predictions(predictions>5.5));
predictions(predictions<4.5)= floor(predictions(predictions<4.5));
predictions(4.5<predictions & predictions<5.5)= round(predictions(4.5<predictions & predictions<5.5));

% We verify that the labels are in the right range
predictions(predictions>max_label)= max_label; 
predictions(predictions<min_label)= min_label;

% Compute the accuracy
accuracy = sum(predictions == labels_test)/M;

% Compute the confusion matrix
confusion_matrix = f3_compute_confusion_matrix( labels_test, predictions, max_label +1);

end

