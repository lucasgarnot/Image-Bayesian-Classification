t = clock;

%Load The MNIST Data
M_data_train = loadMNISTImages('train-images.idx3-ubyte')';
M_labels_train = loadMNISTLabels('train-labels.idx1-ubyte');
M_data_test = loadMNISTImages('t10k-images.idx3-ubyte')';
M_labels_test = loadMNISTLabels('t10k-labels.idx1-ubyte');

%Load the CIFAR Data
d1 = load('data_batch_1.mat');
d2 = load('data_batch_2.mat');
d3 = load('data_batch_3.mat');
d4 = load('data_batch_4.mat');
d5 = load('data_batch_5.mat');
test = load('test_batch.mat');
C_data_train = vertcat(d1.data,d2.data,d3.data,d4.data,d5.data);
C_labels_train = vertcat(d1.labels,d2.labels,d3.labels,d4.labels,d5.labels);
C_data_test = test.data;
C_labels_test = test.labels;

%display the time of excecution
time = clock - t
clearvars t d1 d2 d3 d4 d5 test time