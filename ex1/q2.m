function q2_fun = q2
  q2_fun.q2a=@q2a;
  q2_fun.q2d=@q2d;
  q2_fun.q2f=@q2f;
return

function [] = q2a()
  mnist = load("mnist_all.mat");
  k = 1;
  d = 784;
  X = 1:11:100;
  Y = [];
  # Generate a group of n test samples.
  n = 20;
  [Xtest,Ytest]=gensmallm(mnist.test0, mnist.test3, mnist.test5, mnist.test8, 0, 3, 5, 8, n);
  # Loop through m samples (from list X)
  for m=X
    curr_mean = [];
    # Loop through 10 groups of train samples.
    for i=1:10
      # Generate randomly ordered train set of m samples (using only the digits 0,3,5,8).
      [Xtrain,Ytrain]=gensmallm(mnist.train0, mnist.train3, mnist.train5, mnist.train8, 0, 3, 5, 8, m);
      # Retrieve the classifier.
      classifier = learnknn(k, d, m, Xtrain, Ytrain);
      # Predict the label.
      Ytest_predict = predictknn(classifier, n, Xtest);
      curr_mean(i) = mean(Ytest ~= Ytest_predict);
    end
    disp(curr_mean);
    disp("#####");
    curr_mean = mean(curr_mean);
    Y = [Y curr_mean];
  end
  figure;
  plot(X,Y);
  title('Q2.a');
  xlabel('Sample Size');
  ylabel('Avg. Error');
return

function [] = q2d()
  mnist = load("mnist_all.mat");
  d = 784;
  X = 1:2:11;
  Y = [];
  m = 100;
  for k=X
    curr_mean = [];
    # Generate a group of n test samples.
    n = 200;
    [Xtest,Ytest]=gensmallm(mnist.test0, mnist.test3, mnist.test5, mnist.test8, 0, 3, 5, 8, n);
    # Loop through 10 groups of train samples.
    for i=1:10
      # Generate randomly ordered train set of m samples (using only the digits 0,3,5,8).
      [Xtrain,Ytrain]=gensmallm(mnist.train0, mnist.train3, mnist.train5, mnist.train8, 0, 3, 5, 8, m);
      # Retrieve the classifier.
      classifier = learnknn(k, d, m, Xtrain, Ytrain);
      # Predict the label.
      Ytest_predict = predictknn(classifier, n, Xtest);
      # Average error.
      curr_mean(i) = mean(Ytest ~= Ytest_predict);
    end
    disp(curr_mean);
    disp("#####");
    curr_mean = mean(curr_mean);
    Y = [Y curr_mean];
  end
  figure;
  plot(X,Y);
  title('Q2.d');
  xlabel('Number of Neighbours (k)');
  ylabel('Avg. Error');
return

function [confusion] = q2f()
  mnist = load("mnist_all.mat");
  d = 784;
  k = 3;
  m = 100;
  curr_mean = [];
  # Generate randomly ordered train set of m samples (using only the digits 0,3,5,8).
  [Xtrain,Ytrain]=gensmallm(mnist.train0, mnist.train3, mnist.train5, mnist.train8, 0, 3, 5, 8, m);
  # Retrieve the classifier.
  classifier = learnknn(k, d, m, Xtrain, Ytrain);
  # Generate a group of n test samples.
  n = 200;
  [Xtest,Ytest]=gensmallm(mnist.test0, mnist.test3, mnist.test5, mnist.test8, 0, 3, 5, 8, n);
  # Predict the label.
  Ytest_predict = predictknn(classifier, n, Xtest);
  # Average error.
  curr_mean = mean(Ytest ~= Ytest_predict);
  disp(sprintf("error %d", curr_mean));
  disp("#####");
  confusion = [];
  v = [0,3,5,8];
  for i=1:4
    for j=1:4
      counter = 0;
      for t=1:n
        if Ytest(t) == v(i) && Ytest_predict(t) == v(j)
          counter += 1;
        end
      end
      confusion(i, j) = counter / n;
    end
  end
return
