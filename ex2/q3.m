function [] = q3()
  mnist = load("mnist_all.mat");
  [X, Y_train, Y_test] = build_graph(mnist.train3, mnist.train5, mnist.test3, mnist.test5);
  [X, Y2_train, Y2_test] = build_graph(mnist.train5, mnist.train6, mnist.test5, mnist.test6);
  figure;
  plot(X, Y_train, X, Y_test, X, Y2_train, X, Y2_test);
  title('Q3');
  xlabel('Sample Size');
  ylabel('Avg. Error');
  semilogy();
  legend('3/5 train', '3/5 test', '5/6 train', '5/6 test');
end

function [X,Y_train,Y_test] = build_graph(train1, train2, test1, test2)
  n = size(test1, 1) + size(test2, 1);
  [Xtest,Ytest]=gensmallm(test1, test2, 1, -1, n);
  d = 784;
  maxupdates = 10000;
  X = 1:25:1000
  Y_train = [];
  Y_test = [];
  for m=X
    [Xtrain,Ytrain]=gensmallm(train1, train2, 1, -1, m);
    w = perceptron(m,d,Xtrain,Ytrain,maxupdates);
    A = diag(Ytrain) * (Xtrain * w);
    B = diag(Ytest) * (Xtest * w);
    err_train = mean(A <= 0);
    err_test = mean(B <= 0);
    Y_train = [Y_train err_train];
    Y_test = [Y_test err_test];
  end
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

function [w,X,Y] = perceptron_q3f(m, d, Xtrain, Ytrain, maxupdates)
  t = 1;
  w = zeros(d, 1);
  num_updates = 0;
  counter = 0;
  i = 1;
  X = []
  Y = []
  while num_updates < maxupdates && counter < m
    if i > m
      i = 1;
    end
    if (Ytrain(i) * (Xtrain(i,:) * w)) <= 0
      w = w + Ytrain(i) * transpose(Xtrain(i,:));
      counter = 0;
      num_updates = num_updates + 1;
      A = diag(Ytrain) * (Xtrain * w);
      X = [X num_updates];
      Y = [Y mean(A <= 0)];
    else
      counter++;
    end
    i++;
  end
end

function [] = q3f()
  m = 1000;
  d = 784;
  maxupdates = 1000;
  mnist = load("mnist_all.mat");
  [Xtrain,Ytrain]=gensmallm(mnist.train5, mnist.train6, 1, -1, m);
  [w,X,Y] = perceptron_q3f(m,d,Xtrain,Ytrain,maxupdates);
  figure;
  plot(X, Y);
  title('Q3f');
  xlabel('Number of updates');
  ylabel('Avg. Error');
  semilogy();
end