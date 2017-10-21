% % % % %        Train the Linear Regression model     % % % % %

% With the MNIST Dataset (K=2)
[M2_w_hat, M2_variance] = g1_train_linear_regression(M_data_train,M_labels_train,2);

% With the CIFAR Dataset (K=1)
[C2_w_hat, C2_variance] = g1_train_linear_regression(C_data_train(1:2000,:),C_labels_train(1:2000,:),1);