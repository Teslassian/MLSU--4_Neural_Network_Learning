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

%Forward Propagation
a1 = [ones(size(X)(1),1),X];				%5000x401
z2 = a1*Theta1';							%5000x25
a2 = sigmoid(z2);							%5000x25
a2 = [ones(size(a2)(1),1),a2];				%5000x26
z3 = a2*Theta2';							%5000x10
a3 = sigmoid(z3);							%5000x10
h = a3;										%5000x10

%Unregularized Cost
yMask = zeros(size(y), num_labels);			%5000x10
for i=1:m						
	yMask(i,y(i)) = 1;						%5000x10
end

for i=1:m
	term1 = -yMask(i,:)*log(h(i,:))';
	term2 = (1-yMask(i,:))*log(1-h(i,:))';
	J = J + term1 - term2;
end
J = J/m;

%Regularization
Theta1_noBias = Theta1(:,2:end);
Theta2_noBias = Theta2(:,2:end);
term1_reg = sum(sum(Theta1_noBias.^2));
term2_reg = sum(sum(Theta2_noBias.^2));
J = J + lambda/2/m*(term1_reg + term2_reg);

%Backpropagation
Delta_2 = zeros(size(Theta2));					%10x26
Delta_1 = zeros(size(Theta1));					%25x401
for t=1:m
	a1 = [1;X(t,:)'];							%401x1
	z2 = Theta1*a1;								%25x1
	a2 = sigmoid(z2);							%25x1
	a2 = [1;a2];								%26x1
	z3 = Theta2*a2;								%10x1
	a3 = sigmoid(z3);							%10x1
	
	yk = zeros(num_labels,1);
	for k=1:num_labels
		yk(k) = y(t)==k;						%10x1
	end
	
	delta3 = a3 - yk;							%10x1
	
	delta2 = Theta2'*delta3.*[1;sigmoidGradient(z2)];		%26x1
	
	Delta_2 += delta3*a2';						%10x26								
	delta2 = delta2(2:end);						%25x1
	Delta_1 += delta2*a1';						%25x401
end

Theta1_grad = 1/m * [Delta_1(:,1) , (Delta_1 + lambda * Theta1)(:,2:end)];	%25x401
Theta2_grad = 1/m * [Delta_2(:,1) , (Delta_2 + lambda * Theta2)(:,2:end)];	%10x26

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
