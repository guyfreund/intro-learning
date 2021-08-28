function w = perceptron(m, d, Xtrain, Ytrain, maxupdates)
  t = 1;
  w = zeros(d, 1);
  num_updates = 0;
  counter = 0;
  i = 1;
  
  while num_updates < maxupdates && counter < m
    if i > m
      i = 1;
    end
    if (Ytrain(i) * (Xtrain(i,:) * w)) <= 0
      w = w + Ytrain(i) * transpose(Xtrain(i,:));
      counter = 0;
      num_updates = num_updates + 1;
    else
      counter = counter + 1;
    end
    i = i + 1;
  enz
end
