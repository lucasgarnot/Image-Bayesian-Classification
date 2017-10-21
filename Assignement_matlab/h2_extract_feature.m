function [ new_data, layer_old] = h2_extract_feature( data,filters,number_of_layers,ratio )
    [N,k]=size(data);
    l=sqrt(k);
    data = reshape(data,[N,l,l]);

    layer_old = h1_compute_next_layer(data,filters,ratio);
    layer_new = layer_old;
    
    for i = 1:number_of_layers-1
        layer_new = h1_compute_next_layer(layer_old,filters,ratio);
        layer_old = layer_new;
    end
    new_data = layer_new(:,:);
end

