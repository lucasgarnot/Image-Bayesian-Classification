
% Test the predictions on the test data for the CIFAR dataset
[C_labels_prediction, C_confusion_matrix, C_accuracy] = f2_predict_naive_bayes_classifier( C_means, C_variances , C_data_test(1:3000,:)./255, C_labels_test(1:3000,:), 0.0001);

% Dispay the confusion matrix and the accuracy
C_confusion_matrix
C_accuracy

% Display the confusion matrix through an image
close all
colormap hot
image(C_confusion_matrix)
title('confusion matrix - naive bayes classifier - CIFAR dataset')
