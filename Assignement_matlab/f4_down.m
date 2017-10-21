function [ data_down_sampled ] = f4_down( data, ratio )
    % Relu layer
    %data = max(data,0);
    % We don't use this layer because it gives worse results.
    
    % Pooling layer (mean pooling)
    data_down_sampled = permute(imresize(permute(data,[2,3,1]),ratio),[3,1,2]);

end

