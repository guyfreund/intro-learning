function[classifier]=learnknn(k, d, m, Xtrain, Ytrain)
  classifier.k = k;
  classifier.d = d;
  classifier.m = m;
  classifier.Xtrain = Xtrain;
  classifier.Ytrain = Ytrain;
end
