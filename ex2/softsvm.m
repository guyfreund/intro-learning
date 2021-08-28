function w = softsvm(lambda, m, d, Xtrain, Ytrain)
  H=spalloc(m+d,m+d,d);
  H(1:d,1:d)=2.0*lambda*speye(d);
  epsilon = 1e-4;
  H = H + epsilon*speye(m+d);
  u = [zeros(1,d),(1.0/m)*ones(1,m)];
  A = spalloc(2*m,m+d,m*d+2*m);
  A(m+1:2*m,1:d) = diag(Ytrain) * Xtrain;
  A(1:m,d+1:m+d) = speye(m);
  A(m+1:2*m,d+1:m+d) = speye(m);
  v = vertcat(zeros(m,1),ones(m,1));
  z = quadprog(H,u,-A,-v);
  w = z(1:d,:);
end