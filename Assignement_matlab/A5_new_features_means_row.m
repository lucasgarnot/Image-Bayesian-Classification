 % Add new features: mean of each row and column

M_new_data_train = reshape(M_data_train,[60000,24,24,1]);
M_new_data_train = [repmat([permute(mean(M_new_data_train,2),[1 3 2]),mean(M_new_data_train,3)],[1 1]), M_data_train];

M_new_data_test = reshape(M_data_test,[10000,24,24,1]);
M_new_data_test = [repmat([permute(mean(M_new_data_test,2),[1 3 2]),mean(M_new_data_test,3)],[1 1]), M_data_test];

% Train the classifier
[M_new_means, M_new_variances] = f1_train_naive_bayes_classifier( M_new_data_train, M_labels_train );

% Test the predictions on the test data for the MNIST dataset
[M_labels_prediction, M_confusion_matrix, M_accuracy] = f2_predict_naive_bayes_classifier( M_new_means, M_new_variances, M_new_data_test, M_labels_test, 0.084);

% Display the confusion matrix and the accuracy
M_confusion_matrix
M_accuracy

% Display the confusion matrix through an image
figure();
colormap hot;
image(M_confusion_matrix*2.5);
title('confusion matrix - naive bayes classifier - MNIST with new features(mean row and column)')
