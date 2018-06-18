function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
n = size(Theta1,2);

num_labels = size(Theta2, 1);

%add ones to X matrix 
X = [ones(m, 1) X];

%add ones to Theta1 matrix
%Theta1 = [ones(1,n); Theta1];
%this works but is techincally not correct practice to modify feature
%parameters


% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%theta 1 =[25,401] theta2 = [10, 26]
a1 = sigmoid(X*Theta1');
a1 = [ones(m,1) a1];  %my issue before was adding the ones to theta
%we add the ones to be the bias for each activation layer.
%predict second layer
[i,p] = max(sigmoid(a1*Theta2'),[],2);

%note my test was correct but failed the unit test.
%technically we add the ones to the activation layer not to the theta.
%the theta just holds parameters. We dont add a bias here. Just like in our
%X variable we add the bias to, not the theta. Same effect though.







% =========================================================================


end
