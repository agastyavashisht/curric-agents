# Machine Learning Basics — Reference Corpus

## Introduction to ML
Machine learning is a subfield of AI where models learn patterns from data rather than following explicit rules. Supervised learning: labeled input-output pairs (classification, regression). Unsupervised learning: unlabeled data, find structure (clustering, dimensionality reduction). Reinforcement learning: agent learns by receiving rewards. Common frameworks: scikit-learn, TensorFlow, PyTorch.

## Data Preprocessing
Raw data must be cleaned before modelling. Missing values: drop rows, mean/median imputation, or model-based imputation. Normalization (min-max): scales to [0,1] — `(x - min)/(max - min)`. Standardization (z-score): `(x - mean)/std`, zero mean and unit variance. Categorical encoding: one-hot encoding for nominal, ordinal encoding for ordered. Train/test split before any preprocessing to avoid data leakage.

## Linear Regression
Models a linear relationship: `y = β₀ + β₁x₁ + ... + βₙxₙ`. Fit by minimizing Mean Squared Error (MSE). Coefficients interpreted as: "one unit increase in xᵢ changes y by βᵢ, holding others constant." R² (coefficient of determination): fraction of variance explained, range [0,1]. Assumptions: linearity, independence, homoscedasticity, normality of residuals.

## Logistic Regression
Binary classification using the sigmoid function: `σ(z) = 1/(1 + e^(-z))`, output ∈ (0,1) interpreted as probability. Decision boundary at P=0.5 by default. Loss function: binary cross-entropy. Coefficients on log-odds scale; exponentiate for odds ratios. Multiclass via one-vs-rest or softmax (multinomial LR).

## Decision Trees
Partition feature space recursively by choosing splits that maximize information gain (entropy-based) or minimize Gini impurity. Leaf nodes give predictions. Prone to overfitting — control with `max_depth`, `min_samples_split`, `min_samples_leaf`. Gini impurity: `1 - Σpᵢ²`. Entropy: `-Σpᵢ log₂(pᵢ)`.

## Model Evaluation
Never evaluate on training data. Hold-out: simple train/test split (e.g., 80/20). k-Fold cross-validation: partition into k folds, train on k-1, test on 1, rotate, average. Metrics — Classification: accuracy, precision, recall, F1-score, ROC-AUC. Regression: MSE, RMSE, MAE, R². Confusion matrix shows TP, FP, FN, TN.

## Overfitting and Regularization
Overfitting: model memorizes training data, poor generalization (high variance). Underfitting: model too simple (high bias). L1 (Lasso): penalty `λΣ|wᵢ|`, drives some weights to zero (sparse model, feature selection). L2 (Ridge): penalty `λΣwᵢ²`, shrinks weights toward zero (dense model). Dropout (neural nets). Early stopping. Bias-variance tradeoff: increasing model complexity reduces bias but increases variance.

## Ensemble Methods
Bagging: train multiple models on bootstrap samples, aggregate predictions (reduces variance). Random Forest: bagged decision trees with random feature subset at each split — highly effective, robust. Boosting: sequential trees, each corrects errors of previous (AdaBoost, Gradient Boosting, XGBoost). Stacking: meta-learner trained on base models' predictions.

## Clustering
K-means: assign points to nearest of k centroids, update centroids, repeat until convergence. Elbow method: plot inertia vs k, choose "elbow" where gain decreases. DBSCAN: density-based, discovers arbitrary shapes, handles noise. Hierarchical clustering: agglomerative (bottom-up merge) or divisive. Evaluation: silhouette score, Davies-Bouldin index (no ground truth needed).

## Neural Network Basics
Feedforward network: layers of neurons, each computing `σ(Wx + b)`. Activation functions: ReLU `max(0,x)` (hidden layers), sigmoid (binary output), softmax (multiclass output). Forward propagation: compute predictions layer by layer. Loss function measures error. Backpropagation: compute gradients via chain rule. Gradient descent: update weights `w = w - α∇L`. Batch vs mini-batch vs stochastic GD.
