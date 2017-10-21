function [ means, variances] = train_naive_bayes_classifier( data, labels )
    %Initialize the parameters
    nb_labels = max(labels)+1;
    nb_param = size(data(1,:),2);
    data = double(data);
    means = zeros(nb_labels,nb_param); 
    variances = zeros(nb_labels,nb_param);
    
    %Computing the Means and the Variances for each class
    for i = 1:nb_labels
         class_i = data(labels == i-1,:); %select the data that is in class
         N = size(class_i,1);             % N is the number of data in the class
         means(i,:) = mean(class_i);
         variances(i,:) = (1/(N -1))*sum((class_i - repmat(means(i,:),N,1)).^2,1)+0.011;
    end
end

