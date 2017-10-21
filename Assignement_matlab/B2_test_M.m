% % %       Test the predictions given by the model on the test datas     % % %

% With the MNIST Dataset
[ M2_labels_predictions, M2_confusion_matrix, M2_accuracy ] = g2_predict_linear_regression(M_data_test, M_labels_test, M2_w_hat, 0, 2);

%display the confusion matrix and the accuracy
M2_confusion_matrix
M2_accuracy

% Display the confusion matrix through an image
colormap hot;
image(M2_confusion_matrix);
title('confusion matrix - naive bayes classifier - MNIST dataset')

% Scatter plot of the resulting labels versus the true labels
M2_error = g3_plot_error( M_labels_test, M2_labels_predictions, 20)