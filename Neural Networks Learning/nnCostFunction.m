function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% part 1
[y_size,useless]= size(y);
recoded_y = zeros(y_size,num_labels);
for i = 1:y_size
    recoded_y(i,y(i))=1;
end
X = [ones(m, 1) X];
hidden = sigmoid(X * Theta1.');
m_hidden = size(hidden,1);
hidden = [ones(m_hidden,1) hidden];
output = sigmoid(hidden * Theta2.');
J = 1/m * trace(-recoded_y.' * log(output) - (ones(y_size,num_labels) - recoded_y).' * log((ones(y_size,num_labels) - output)));

% part 2
for t = 1:m
   % feed forward
   a_1 =  X(t,:);
   z_2 = a_1 * Theta1.';
   a_2 = sigmoid(z_2);
   a_2 = [ones(size(a_2,1),1) a_2];
   z_3 = a_2 * Theta2.';
   a_3 = sigmoid(z_3);
   
   delta = 0;
   delta_3 = a_3 - recoded_y(t,:);
   delta_2 = Theta2(:,2:end).' * delta_3.' .* sigmoidGradient(z_2.');
   
   Theta2_grad = Theta2_grad + delta_3.' * a_2;
   Theta1_grad = Theta1_grad + delta_2 * a_1;
end
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

Theta1_reg = (lambda/m) * Theta1;
Theta2_reg = (lambda/m) * Theta2;

Theta1_grad = Theta1_grad + [zeros(size(Theta1,1),1) Theta1_reg(:,2:end)];
Theta2_grad = Theta2_grad + [zeros(size(Theta2,1),1) Theta2_reg(:,2:end)];
%Theta2_grad+=[zeros(size(Theta2,1),1) reg_theta2(:,2:end)];
%Theta1_grad+=[zeros(size(Theta1,1),1) reg_theta1(:,2:end)];
% part 3
[Theta1_row, Theta1_col] = size(Theta1);
[Theta2_row, Theta2_col] = size(Theta2);
J = J + lambda/(2*m) * (trace(Theta1(:,2:Theta1_col).'*Theta1(:,2:Theta1_col)) + trace(Theta2(:,2:Theta2_col).'*Theta2(:,2:Theta2_col)));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
