function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%unlike in ex3 we find the COST not just the ACCURACY so we must convert Y
%into a binary matrix
I = eye(num_labels);
Y = I(y,:);

X = [ones(m,1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
J = 1/m*sum(sum(-Y.*log(a3) - (1-Y).*log(1-a3)));
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


%find error=e of each layer i
e3 = a3 - Y;
e2 = e3*Theta2 .*a2.*(1-a2);

%now calculate the gradient of the error found by backpropogation
%remember the exlude the biases. ?WHY THOUGH?
d2 = e3' * a2;
d1 = e2(:,2:end)' * X;

%divide by m training examples
Theta1_grad = d1/m;
Theta2_grad = d2/m;

%?WHY IS THE REGULARIZATION TERM NOT SQUARED?
%remember that we never regularize the bias. would make it way too large
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

%unroll the parameters
grad = [Theta1_grad(:); Theta2_grad(:)];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
