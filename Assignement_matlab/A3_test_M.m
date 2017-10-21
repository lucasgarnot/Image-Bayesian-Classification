% With the  MNIST Data
[M_means, M_variances] = f1_train_naive_bayes_classifier( M_data_train, M_labels_train );

% Test the predictions on the test data for the MNIST dataset
[M_labels_prediction, M_confusion_matrix, M_accuracy] = f2_predict_naive_bayes_classifier( M_means, M_variances, M_data_test, M_labels_test, 0.084);

% Display the confusion matrix and the accuracy
M_confusion_matrix
M_accuracy

% Display the confusion matrix through an image
figure();
colormap hot;
image(M_confusion_matrix*2.5);
title('confusion matrix - naive bayes classifier - MNIST dataset')
