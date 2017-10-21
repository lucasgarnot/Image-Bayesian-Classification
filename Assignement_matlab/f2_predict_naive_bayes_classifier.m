function [predicted_labels, confusion_matrix, ratio] = f2_predict_naive_bayes_classifier( means, variances, data, labels,offset)
    % initialization of parameters
    nb_labels = max(labels)+1;
    N = size(data,1);
    
    % To handle the problem of probability set to zero which lead to a
    % logarithm set to -INF, we add a little number to the variance
    variances = variances + offset; %MNIST 0.084 CIFAR 0.0001
    
    % Prepare the matrix of data and parameters to optimize the algorithm
    % (by avoiding for loops)
    data = double(repmat(data,[1 1 nb_labels]));
    means = permute(repmat(means,[1 1 N]),[3 2 1]);
    variances = permute(repmat(variances,[1 1 N]),[3 2 1]);
    
    % Compute the logarithm of the probability
    probability = log(normpdf(data,means,sqrt(variances)));
    
    % Sum the probability for each point and class
    sum_proba = permute(sum(probability, 2),[1 3 2]);
    
    % Normalize the probabilities in order to be able to compare them
    % between the different classes
    m_min = repmat(min(sum_proba),[N 1]);
    m_max = repmat(max(sum_proba), [N 1]);
    normalized_sum_proba = (sum_proba - m_min)./(m_max - m_min);
    
    % Take the class with the highest probability for each picture
    [m,predicted_labels] = max(normalized_sum_proba,[],2);
    predicted_labels = predicted_labels - 1;
    
    % Compute the confusion matrix and the ratio
    confusion_matrix = f3_compute_confusion_matrix( labels, predicted_labels, nb_labels);
    ratio = double(sum(predicted_labels == labels))/N;
end
