function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cvec = [.01 .03 .1 .3 1 3 10 30];
Svec = [.01 .03 .1 .3 1 3 10 30];
E = zeros(length(Cvec),length(Svec));

for idx = 1:numel(Cvec)
    for vdx = 1:numel(Svec)
        model= svmTrain(X, y, Cvec(idx), @(x1, x2) gaussianKernel(x1, x2, Svec(vdx)));
        pred = svmPredict(model,Xval);
        E(idx, vdx) = mean(double(pred ~=yval));
    end 
end 

min_val = min(min(E));
[C_ind, S_ind] = find(E==min_val);

C=Cvec(C_ind);
S=Svec(S_ind);

% =========================================================================

end
