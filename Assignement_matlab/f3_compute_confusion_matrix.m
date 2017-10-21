function [ confusion_matrix ] = f3_compute_confusion_matrix( labels, predicted_labels, nb_labels)
    % Initialize the parameters
    N = size(labels);
    confusion_matrix = zeros(nb_labels+1);
    
    % Each row is a true labels / Each column is a prediction
    for i=1:N
        confusion_matrix(labels(i,1)+2,predicted_labels(i,1)+2) = confusion_matrix(labels(i)+2,predicted_labels(i)+2) + 1;
    end 
    m_sum = repmat(sum(confusion_matrix,2),[1 nb_labels+1]);
    confusion_matrix = round(confusion_matrix./m_sum*100);
    
    % Put the label number in the first raw and column
    for j=1:nb_labels
        confusion_matrix(1,j+1)=j-1;
        confusion_matrix(j+1,1)=j-1;
    end
end

