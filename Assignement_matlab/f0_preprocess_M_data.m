function [ preprocessed_data ] = preprocess_M_data( data )
% preprocess the MNIST data and remove the border pixels
N= size(data,1);
preprocessed_data = data;
border_pixels = [1:208];
border_pixels(108:208)= 784-100:784;
k=0;

for i=2:25
    for j = [1 2 27 28]
        k = k+1;
        border_pixels(28*2 + k) = 28*i + j;
    end
end

preprocessed_data(:,border_pixels)=[];

end

