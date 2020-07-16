function [J, grad] = costFunction(theta, X, y)
m = length(y); 
J = 0;
grad = zeros(size(theta));
prob = sigmoid(X*theta);
J = (1 / m) * ((-y' * log(prob)) - (1 - y)' * log(1 - prob));
grad = (1 / m) * (prob - y)' * X;
end
