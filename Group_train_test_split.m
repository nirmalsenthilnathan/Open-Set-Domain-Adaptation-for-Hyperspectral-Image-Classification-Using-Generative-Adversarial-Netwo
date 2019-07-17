
%Function file recieves input and give multiple outputs:
function [k k_train k_test] = Group_train_test_split(data, gt, t)

%Vectorization to get spectral information:
[m n b]=size(data);
d=[];
for i=1:b;
    d1=data(:,:,i);
    d=[d d1(:)];
end

%declare necessary matrix information:
c=max(max(gt));
l = [];
k = [];
k2 = []
k_train = [];
k_test  = [];
k1  = [];
count = [];

%Grouping and slpit:
for i=[1:c];
    l  = find(gt==i);    %location for each class
    l1 = numel(l);       %total number in each class
    count = [count; l1]; 
    k1  = d(l, :)';      %extract pixel values for each class   
    v = ones(1,l1)*(i-1);
    
    train = round(l1*(t/100));
    test  = l1-train;
    k2 = [k1; v];
    
    %gouping based on class label
    k  = [k k2];
    %separate train and test data for each class
    k_train = [k_train, k1(:, 1:train)];
    k_test  = [k_test, k1(:, train+1:l1)];
    k1 = [];
    k2 = [];
    v = [];
end    
end
