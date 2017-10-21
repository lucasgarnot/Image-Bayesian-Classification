% % % % %       Train the Naive Bayes Classifier        % % % % % 

% With the CIFAR Data
[C_means, C_variances] = f1_train_naive_bayes_classifier( C_data_train./255, C_labels_train);

% With the  MNIST Data
[M_means, M_variances] = f1_train_naive_bayes_classifier( M_data_train, M_labels_train );
