% Add new features with covnet

N=60000;
M=10000;

% Initialize some filters
sharper1 = [0 -1 0;-1 5 -1; 0 -1 0];%81
sharper2 = [1,1,1;1,-9,1; 1,1,1];%79 or with -7
edge1 = [1 0 -1;0 0 0;-1 0 1];
edge2=[
   0,  0, -1,  0,  0;
   0,  0, -1,  0,  0;
   0,  0,  2,  0,  0;
   0,  0,  0,  0,  0;
   0,  0,  0,  0,  0;
];%68
edge3=[
   0,  0, -1,  0,  0;
   0,  0, -1,  0,  0;
   0,  0,  4,  0,  0;
   0,  0, -1,  0,  0;
   0,  0, -1,  0,  0;
];%80
edge4=[
  -1,  0,  0,  0,  0;
   0, -2,  0,  0,  0;
   0,  0,  6,  0,  0;
   0,  0,  0, -2,  0;
   0,  0,  0,  0, -1;
];%78.9
mean_f = [1 1 1; 1 1 1; 1 1 1]./9;
emboss = [-1 -1 0;-1 0 1;0 1 1];
random1 = 3*(rand(5,5)-0.5);
random2 = 3*(rand(5,5)-0.5);
random3 = 3*(rand(5,5)-0.5);
random4 = 3*(rand(5,5)-0.5);
id = 1;

% Select the filter to use in the convnet
filters = {sharper1,edge3,edge4};
% filters = {id};
filters = {random1, random2, random3};

% Select the number of layers and the ratio used to reduce the image size
% in the pooling layer
nb_layers = 2;
ratio = 0.6;

% Extract the features
[ M_new_data_train ,layer] = h2_extract_feature(M_data_train,filters,nb_layers,ratio);

[ M_new_data_test ,layer1] = h2_extract_feature(M_data_test,filters,nb_layers,ratio);

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
image(M_confusion_matrix*2);
title('confusion matrix - naive bayes classifier - MNIST with convnet features')

clearvars random1 random2 random3 random4 emboss mean_f id filters edge1 edge2 edge3 edge4 sharper1 sharper2 N M
