function [Ytestprediction]=predictknn(classifier, n, Xtest)
  Ytestprediction = [];
  # Loop through n tests examples.
  for i=1:n
    dist = [];
    # Loop through m train examples.
    for j=1:classifier.m
      # Calculate the distance between the j'th train example
      # to the i'th test example.
      dist(j) = distance(classifier.Xtrain(j,:), Xtest(i,:));
    end
    # Calculate the k nearest neighbours and their indices.
    [nearest, nearest_i] = mink(dist, classifier.k);
    nearest_ys = [];
    # Loop through the k nearest neighbours indices
    for j=1:size(nearest_i)(1)
      # Calculate their labels by accessing Ytrain matrix.
      nearest_ys(j) = classifier.Ytrain(nearest_i(j, 1), 1);
    end
    # Calculate the mode (most frequent) label.
    Ytestprediction(i) = mode(nearest_ys);
  end
  # Transpose the prediction to a column vector.
  Ytestprediction = transpose(Ytestprediction);
end

function [dist] = distance(z1, z2)
  dist = norm(z1-z2);
end
p
# Input:  a row vector (A), a number k.
# Output: column vectors y,yi representing
#         the k minimal values in A.
function [y, yi] = mink(A,k)
  B = [];
  # Loop through A's elements.
  for j=1:size(A)(2)
    # For each column (element of A),
    # concatenate to B the following row:
    # j, A_1j.
    B = vertcat(B, [j, A(1,j)]);
  end
  # Sort B's rows by the A_1j values.
  B = sortrows(B, 2);
  # Retrieve the k minimal rows from B.
  B = B(1:k, :);
  # Retrieve B's first column (indices).
  yi = B(:, 1);
  # Retrieve B's second column (values).
  y = B(:, 2);
end

function [Ytestprediction]=test()
  k=1;
  m=3;
  d=2;
  Xtrain = [1,2;3,4;5,6];
  Ytrain = [1;0;1];
  classifier = learnknn(k,d,m,Xtrain,Ytrain);
  n=4;
  Xtest = [10,11;3.1,4.2;2.9,4.2;5,6];
  Ytestprediction = predictknn(classifier, n, Xtest);
end
