% Add new features by taking the mean of certain areas of each picture

N=60000;M=10000;
M_new_data_train = reshape(M_data_train,[N,24,24]);
M_new_data_train  = permute(imresize(permute(M_new_data_train,[2,3,1]),0.4),[3,1,2]);
M_new_data_train = [repmat(M_new_data_train(:,:), [1 4]) M_data_train];

M_new_data_test = reshape(M_data_test,[M,24,24]);
M_new_data_test  = permute(imresize(permute(M_new_data_test,[2,3,1]),0.4),[3,1,2]);
M_new_data_test = [repmat(M_new_data_test(:,:),[1 4]),M_data_test];

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
title('confusion matrix - naive bayes classifier - MNIST dataset with new features (mean of areas)')
