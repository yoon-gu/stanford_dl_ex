function binary_classifier_accuracy=binary_classifier_accuracy(theta, X,y)
  correct=sum(y == (sigmoid(theta'*X) > 0.5));
  binary_classifier_accuracy = correct / length(y);
