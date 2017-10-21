function [ error ] = g3_plot_error( labels, predicted_labels,number_to_print)
N = size(labels,1);

% Plot a sample of point's labels among with their prediction
figure();
hold on;
plot(labels(50:50+number_to_print),'b.','MarkerSize',13);
plot(predicted_labels(50:50+number_to_print),'r.','MarkerSize',13);

% Compute the error
error = sum(abs(double(labels) - double(predicted_labels)))/N;
end

