function [w] = ridge(X, Y, lambda, m)
  X = X(:,1:m);
  Y = Y(1:m,:);
  d = size(X, 1);
  w = inv(X*(X')+lambda*eye(d,d))*X*Y;
end
