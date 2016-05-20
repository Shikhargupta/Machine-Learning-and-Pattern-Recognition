
Code includes 3 generative bayesian classifiers. Difference in classifiers is the covariance matrices used to distinguish characters.


1. The samples of a given character class are modelled by a separate covariance matrix Σi.
2. The samples across all the characters are pooled to generate a common diagonal covariance matrix Σ. The diagonal entries correspond to the variances of the individual features, that are considered to be independent.
3. The covariance matrix of each class is forced to be identity matrix.


 The mean and the covariance matrices are estimated from the training data using the Maximum Likelihood techniques.
