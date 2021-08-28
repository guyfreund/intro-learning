function []=q1()
  X = 10:100;
  Y = [];
  data = load("regdata.mat");
  Xtrain = data.X;
  Ytrain = data.Y;
  Xtest = data.Xtest;
  Ytest = data.Ytest;
  for m = X
    best_lambda = -1;
    best_err = -1;
    for lambda = 0:30
      [w] = ridge(Xtrain, Ytrain, lambda, m);
      [err] = calc_mse(Xtest, Ytest, w, m);
      if best_err == -1 || err < best_err
        best_err = err;
        best_lambda = lambda;
      end
    end
    Y = [Y best_lambda];
  end
  print_graph(X, Y);
end

function [err] = calc_mse(X, Y, w, m)
  err = norm(X'*w-Y, 2)/m;
end

function []=print_graph(X, Y)
  figure;
  plot(X, Y);
  title('Q1');
  xlabel('Training Set Size (m)');
  ylabel('Best Lambda (minimizes MSE)');
  % legend('');
end
