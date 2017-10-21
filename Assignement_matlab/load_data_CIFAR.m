d1 = load('data_batch_1.mat');
d2 = load('data_batch_2.mat');
d3 = load('data_batch_3.mat');
d4 = load('data_batch_4.mat');
d5 = load('data_batch_5.mat');
data_C = vertcat(d1.data,d2.data,d3.data,d4.data,d5.data);
label_C = vertcat(d1.labels,d2.labels,d3.labels.d4.labels.d5.labels);