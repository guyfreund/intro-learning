function alpha = softsvmrbf(lambda, sigma, m, d, Xtrain, Ytrain)
  G = make_gauss(Xtrain, sigma, m);
  H=spalloc(m+m,m+m,m*m);
  H(1:m,1:m)=2.0*lambda*G;
  epsilon = 1e-4;
  H = H + epsilon*speye(m+m);
  u = [zeros(1,m),(1.0/m)*ones(1,m)];
  A = spalloc(2*m,m+m,m*m+2*m);
  A(1:m,1:m) = Ytrain .* G; % yi*<alpha,G_row_i>
  A(1:m,m+1:m+m) = speye(m); % xi >= 0
  A(m+1:2*m,m+1:m+m) = speye(m); % +xi
  v = vertcat(ones(m,1),zeros(m,1));
  z = quadprog(H,u,-A,-v);
  alpha = z(1:m,:);
end

function G = make_gauss(X, sigma, m)
  nsq=sum(X.^2,2); % X(i)^2
  nsq=nsq*ones(1,m);
  G=(nsq)+(nsq')-(2*X)*(X');
  G=exp(-1 * G/(2*sigma));
end

function res = h(alpha, Xtrain, sigma, m)
  G = make_gauss(Xtrain, sigma, m);
  res = sign(dot(alpha, G(i:m,:)));
end
