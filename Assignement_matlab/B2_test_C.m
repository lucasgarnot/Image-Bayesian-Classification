% Test the predictions given by the model on the test datas

%CIFAR Dataset
[ C2_labels_predictions, C2_confusion_matrix, C2_accuracy ] = g2_predict_linear_regression(C_data_test(1:5000,:), C_labels_test(1:5000), C2_w_hat,0, 1);

% Display the confusion matrix and the accuracy
C2_confusion_matrix
C2_accuracy

% Display the confusion matrix through an image
colormap hot;
image(C2_confusion_matrix);
title('confusion matrix - naive bayes classifier - CIFAR dataset')

% Scatter plot of the resulting labels versus the true labels
C2_error = g3_plot_error( C_labels_test(1:5000), C2_labels_predictions, 20)