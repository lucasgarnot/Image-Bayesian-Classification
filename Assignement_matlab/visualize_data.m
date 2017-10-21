%display a sample of images
close all;

for i = 1:4
    figure();
    imshow(reshape(M_data_train(i,:),24,24,1)); % Show the first images
end

% %show the distibution of labels
% figure();
% histogram(labels)