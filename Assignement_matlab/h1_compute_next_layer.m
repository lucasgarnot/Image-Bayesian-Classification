function [ layer ] = h1_compute_next_layer(data,filters,ratio)
    depth = size(filters,2);
    depth0 = size(data,4);
    
    %initialize the layer with the right size
    a = f4_down(data(:,:,:,1),ratio);
    [r,s,t] = size(a);
    layer = ones(r,s,t,depth*depth0);
    
    for j = 0:depth0-1
        for i = 1:depth
            a = f4_down(imfilter(data(:,:,:,j+1),filters{i}),ratio);
            layer(:,:,:,j*depth0+i) = a;
        end
    end
    
end

