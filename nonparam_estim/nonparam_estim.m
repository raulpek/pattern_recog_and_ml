clc
clearvars
close all

load t024.mat

cv = cvpartition(size(X,1), 'HoldOut', 0.2);
x_train = X(training(cv),:);
y_train = y(training(cv),:);
x_test = X(test(cv),:);
y_test = y(test(cv),:);
unique_class = unique(y_train);
% call the function
probabilities = kd_classification(x_train, y_train, x_test, 3);
[~, predictions] = max(probabilities, [], 2);
% calculate accuracy and show predictions
accuracy = sum(predictions == y_test) / length(y_test)
predicted_classes = unique_class(predictions)

%% Functions
function [probas] = kd_classification(traindata, trainclass, data,h)
% determine the probability of the sample
num_train = size(traindata,1);
num_samples = size(data,1);
dim = size(traindata,2);
unique_classes = unique(trainclass);
num_classes = length(unique_classes);
class_idx = cell(num_classes,1);
probas = zeros(num_samples,num_classes);
multip = 1/h^dim;
gauss_const = 1/sqrt(2*pi);
% compute class idx
for c = 1:num_classes
    current_class = unique_classes(c);
    idx = zeros(num_train,1);
    class_count = 0;
    for k = 1:num_train
        if trainclass(k) == current_class
            class_count = class_count + 1;
            idx(class_count) = k;
        end
    end
    class_idx{c} = idx(1:class_count);
end
% per sample
for i = 1:num_samples
    % per class
    for c = 1:num_classes
        indices  = class_idx{c};
        N_class = length(indices);
        if N_class == 0
            continue
        end
    kernel_sum = 0;
    % sum samples
    for n = 1:N_class
        train_idx = indices(n);
        input = (traindata(train_idx,:) - data(i,:))/h;
        kernel_val = gauss_const*exp(-0.5*sum(input.^2));
        kernel_sum = kernel_sum + kernel_val;
    end
    prob = multip*(1/N_class)*kernel_sum;
    probas(i,c) = prob;
    end
end
end
