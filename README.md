# ClassificationUsingScikitLearn
- Learn to learn with scikit-learn. Kaggle is a great place for people interested in machine learning. Many users actively posting useful information in the forum to help people get started. 

- **Kaggle Otto group classification challenge**:
  * This directory contains some of my contributions to the team BoilerUp in the Kaggle otto challenge. nn_otto_ensemble_v8.6.py was one of our best attempts. 
  * Data and description of the problem can be obtained at:  https://www.kaggle.com/c/otto-group-product-classification-challenge/
  * 61878 enetries are provided as the training data. Each entry of the training data is consisted of 93 features, and are labeled one of nine classes. 
  * We used an ensemble of 4 models:
      * a random forest model with 2000 decision trees, no more than 40 features are allowed to be used by each tree, and each decision tree is regulated by constraining their max_depth=45, min_samples_leaf=1, min_samples_split=5; 
      * a gradient boosting model with 6000 boostings at a learning rate of 0.01, the depth of the tree is constrained to be no more than 5; 
      * a neural network model that contains the input layer, two dense layers of 800 and 500 neurons, a dropout layer between the two dense layers with a dropout probability of 0.5; and 
      * a support vector classifier model using RBF kernels with kernel coefficient gamma=0.001 and penalty parameter C=100. 
  * We used nolearn (https://github.com/dnouri/nolearn) and lasagne (https://github.com/Lasagne/Lasagne) for the neural network model, and scikit-learn (http://scikit-learn.org/) for the other three models. 
  * This is our first Kaggle project (and the first machine learning problem for me). We ranked 483th among 3514 teams. Our best log loss score was 0.4392, while the winner score was 0.3824. Not too bad for a first timer, yet a long way to go. It's been a lot of fun, and you can do it too! 
