function [] = q4()
  mnist = load("mnist_all.mat");
  [X, Y_train, Y_test] = build_graph(mnist.train3, mnist.train5, mnist.test3, mnist.test5)
  figure;
  plot(X, Y_train, X, Y_test);
  title('Q3');
  xlabel('Lambda');
  ylabel('Avg. Error');
  legend('3/5 train', '3/5 test');
end

function [X,Y] = gensmallm(labelAsample,labelBsample,A, B, samplesize)
  %load('mnist_all.mat') then use this function on two digits
  alldata = double([labelAsample;labelBsample]);
  alllabels = [A* ones(size(labelAsample,1),1);B* ones(size(labelBsample,1),1)];
  [m,d] = size(alldata);
  perm = randperm(m);
  trainind = perm(1:samplesize);
  X = alldata(trainind,:);
  Y = alllabels(trainind);
end

function [X,Y_train,Y_test] = build_graph(train1, train2, test1, test2)
  m = 100;
  d = 784;
  n = size(test1, 1) + size(test2, 1);
  [Xtest,Ytest]=gensmallm(test1, test2, 1, -1, n);
  X = -4:12;
  Y_train = zeros(1,size(-4:12,2));
  Y_test = zeros(1,size(-4:12,2));

  for i=1:10
    [Xtrain,Ytrain]=gensmallm(train1, train2, 1, -1, m);  
    train_errors = [];
    test_errors = [];
    for n=-4:12
      lambda = 3^n
      w = softsvm(lambda, m, d, Xtrain, Ytrain);
      A = diag(Ytrain) * (Xtrain * w);
      B = diag(Ytest) * (Xtest * w);
      train_errors = [train_errors mean(A <= 0)];
      test_errors = [test_errors mean(B <= 0)];
    end
    Y_train = Y_train + train_errors;
    Y_test = Y_test + test_errors;
  end
  Y_train = Y_train / 10;
  Y_test = Y_test / 10;
end