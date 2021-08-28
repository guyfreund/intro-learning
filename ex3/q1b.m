function [] = q1b()
  data = load("EX3q1_data.mat");
  Xtrain = data.Xtrain;
  Ytrain = data.Ytrain;
  Xtest = data.Xtest;
  Ytest = data.Ytest;
  m = size(Xtrain, 1);
  n = size(Xtest, 1);
  d = size(Xtrain, 2);
  num_folds = 10;
  [xtrain_folds, ytrain_folds, xval_folds, yval_folds] = ...
    gen_folds(Xtrain, Ytrain, m, d, num_folds);
  e_cross_val = [];
  e_test = [];
  ls = [];

  for lambda = [0.01, 0.1, 1]
    for sigma = [0.01, 0.05, 1, 2]
      % Calculate the average cross-validation error.
      e_tmp = [];
      for i = 1:num_folds
        m_train = size(xtrain_folds(:,:,i), 1);
        m_val = size(xval_folds(:,:,i), 1);
        alpha = softsvmrbf(lambda, sigma, m_train, d, xtrain_folds(:,:,i), ytrain_folds(:,:,i));
        e_tmp(i) = calc_err_fast(alpha, sigma, m_train, m_val, xtrain_folds(:,:,i), xval_folds(:,:,i), yval_folds(:,:,i));
      end
      e_cross_val = [e_cross_val mean(e_tmp)];
      ls = [ls; [lambda, sigma]];
      % Calculate the error, as measured on the test set, when using this pair to train
      % a classifier on the entire training set.
      alpha = softsvmrbf(lambda, sigma, m, d, Xtrain, Ytrain);
      e_test = [e_test calc_err_fast(alpha, sigma, m, n, Xtrain, Xtest, Ytest)];
    end
  end
  e_cross_val
  e_test
end

function [xtrain_folds, ytrain_folds, xval_folds, yval_folds] = gen_folds(X, Y, m, d, num_folds)
  fold_size = floor(m / num_folds);
  xtrain_folds = zeros(m-fold_size, d, num_folds);
  ytrain_folds = zeros(m-fold_size, 1, num_folds);
  xval_folds = zeros(fold_size, d, num_folds);
  yval_folds = zeros(fold_size, 1, num_folds);
  for i = 0:num_folds-1
    i1 = i*fold_size+1;
    i2 = fold_size-1;
    xval_folds(:,:,i+1) = X(i1:i1+i2,:);
    yval_folds(:,:,i+1) = Y(i1:i1+i2,:);
    x_temp = [];
    y_temp = [];
    if i1>=2
      x_temp = [x_temp; X(1:i1-1,:)];  
      y_temp = [y_temp; Y(1:i1-1,:)];
    end
    if (i1+i2+1) <= m
      x_temp = [x_temp; X(i1+i2+1:m,:)];
      y_temp = [y_temp; Y(i1+i2+1:m,:)];
    end
    xtrain_folds(:,:,i+1) = x_temp;
    ytrain_folds(:,:,i+1) = y_temp;
  end
end

function err = calc_err_fast(alpha, sigma, m_train, m_val, Xtrain, Xval, Yval)
  nsqt=sum(Xtrain.^2,2);
  nsqv=sum(Xval.^2,2);
  M=(nsqt')-(2*Xval)*(Xtrain');
  M=nsqv+M;
  M=exp(-M/(2*sigma));
  A = M * alpha;
  A = Yval .* A;
  err = mean(A <= 0);
end

function err = calc_err(alpha, sigma, m_train, m_val, Xtrain, Xval, Yval)
  M = [];
  for i=1:m_val
    for j=1:m_train
      M(i, j) = norm(Xtrain(j,:) - Xval(i,:));
    end
  end
  M = exp(-1 * M.^2 / (2 * sigma));
  A = M * alpha;
  A = diag(Yval) * A;
  err = mean(A <= 0);
end