function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(theta);
theta_res = theta;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    
%     for j = 1:n
%         dis = 0;
%         
%         for i = 1:m
%             
%             dis = dis + (theta_res' * X(i, :)' - y(i)) * X(i, j);
%         end
%         dis = dis * alpha / m;
%                 
%         theta_res(j) = theta(j) - dis;
%     end
%     theta = theta_res;
    for j = 1 : n
        theta_res(j) = theta(j) - (alpha / m) * sum((X * theta - y) .* X(:, j));
    end
    theta = theta_res;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %computeCost(X, y, theta)
end

end
