function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(grad);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
    J = J + y(i) * log(sigmoid(theta' * X(i, :)')) + (1 - y(i)) * log(1- sigmoid(theta' * X(i, :)'));
end
J = -J ./ m + lambda ./ (2 * m) * (sum(theta .* theta) - theta(1)*theta(1));

for j = 1:n
    s = 0;
    for i = 1:m
        s = s + (sigmoid(theta' * X(i, :)') - y(i)) * X(i, j);
    end
    if j == 1
        s = s ./ m;
        grad(j) = s;
    else
        s = s ./ m + lambda ./ m * theta(j);
        grad(j) = s;
    end
    
end


% =============================================================

end
