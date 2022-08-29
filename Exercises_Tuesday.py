import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# ----------------------------------------------------------------------------
# QUESTION 1 - Regularization
# ----------------------------------------------------------------------------

# (a) Read the data from yesterday’s exercise ”Feature Engineering”
# (./output/polynomials.csv) into a pandasDataFra

POLY = './output/polynomials.csv'
df1_1 = pd.read_csv(POLY)
# check the dataframe:
df1_1.head()
df1_1.describe()


# (b) Use column ”y” as target variable and all other columns as predicting
# variables (named X in class) and split them as usual.

series_y = df1_1['Y']
array_y = series_y.to_numpy()

# drop the columns that won't be included in X:
df1_2 = df1_1.drop('Unnamed: 0', axis=1, inplace=False)
df1_3 = df1_2.drop('Y', axis=1, inplace=False)
print(df1_3.head())
array_X = df1_3.to_numpy()
X = array_X
y = array_y

# Usual split:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# (c) Learn an ordinary OLS model, a Ridge model and a Lasso model using the
# provided data with penalty parameter equal to 0.3. Using the R2 scores,
# which model yields the best prediction?

lm = OLS().fit(X_train, y_train)
ridge = Ridge(alpha=0.3).fit(X_train, y_train)
lasso = Lasso(alpha=0.3).fit(X_train, y_train)

# Comparison
print(f'R^2 for OLS is {lm.score(X_test, y_test):.2}, '
      f'for ridge {ridge.score(X_test, y_test):.2} '
      f'and for lasso {lasso.score(X_test, y_test):.2}')
print('Ridge yields the best prediction')


# (d) Create a new pandas.DataFrame() containing the learned coefficients of
# all models and the feature names as index. In how many rows are the Lasso
# coefficients equal to 0 while the Ridge coefficients are not?

df1_4 = pd.DataFrame(lm.coef_, index=df1_3.columns, columns=['OLS'])
df1_4['Lasso'] = lasso.coef_
df1_4['Ridge'] = ridge.coef_

df1_4_filtered = df1_4[(df1_4['Lasso'] == 0) & (df1_4['Ridge'] != 0)]
print(f'There are {len(df1_4_filtered)} '
      f'rows in which Lasso coef are zero and Ridge coef are not zero')


# (e) Using matplotlib.pyplot, create a horizontal bar plot of dimension 10x30
# showing the coefficient sizes. Save the figure as ./output/polynomials.pdf.

figure1 = df1_4.plot.barh(figsize=(30, 10), title='Coefficient sizes')
figure1.get_figure().savefig('./output/polynomials.pdf')


# -----------------------------------------------------------------------------
# QUESTION 2 - Neural Network Regression
# -----------------------------------------------------------------------------

# (a) Load the diabetes dataset using sklearn.datasets.load_diabetes().
# The data is on health and diabetes of 442 patients. Split the data as usual.

diabetes = load_diabetes()
X = diabetes['data']
y = diabetes['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# (b) Learn a Neural Network Regressor with identity-activation after
# Standard-Scaling with in total nine parameter combinations of your choice.
# Use the best solver for weight optimization for this dataset according to the
# documentation! To keep computational burden low you may use a 3-fold
# Cross-validation and at most 1,000 iterations.

algorithms = [('scaler', StandardScaler()),
              ('nn', MLPRegressor(max_iter=1000, random_state=42,
                                  solver='lbfgs', activation='identity'))]

pipe = Pipeline(algorithms, verbose=True)
param_grid = {'nn__hidden_layer_sizes': [(75, 75), (100, 100), (120, 120)],
              'nn__activation': ['identity'],
              'nn__alpha': [0.001, 0.005, 0.01]}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train, y_train)


# (c) What are your best parameters? How well do they perform in the training
# set? How well does your model generalize?

# Check grid search results
# grid.best_score_)
# grid.cv_results_))
results = pd.DataFrame(grid.cv_results_)
print(f'The combination that yields the best estimator is:\n'
      f'{grid.best_estimator_}')
# sort the results by the rank test score to check the mean test score:
results_sorted = results.sort_values(by=['rank_test_score'],
                                     ignore_index=True)
mean_test_score = results_sorted['mean_test_score']
print('The mean test score is not very high for any of the tests,\n'
      'and they are closed to each other:')
print(mean_test_score)


# (d) Plot a heatmap for the first coefficients matrix of the best model
# (Access the model via .best_estimator_. One of its attributes is
# _final_estimator, which behaves like a normal model object.).
# Be sure to label the correct axis with the feature names.
# Save the heatmap as ./output/nn_diabetes_importances.pdf.

best_model = grid.best_estimator_
fe = best_model._final_estimator
type(fe)
df2_1 = pd.DataFrame(fe.coefs_[0])
heatmap2 = sns.heatmap(df2_1, yticklabels=diabetes['feature_names'])
figure = heatmap2.get_figure()
figure.savefig('./output/nn_diabetes_importances.pdf')


# -----------------------------------------------------------------------------
# QUESTION 3 - Neural Networks Classification
# -----------------------------------------------------------------------------

# (a) Load the breast cancer dataset using
# sklearn.datasets.load_breast_cancer().
# As usual, split the data into test and training set.

breast_cancer = load_breast_cancer()
X3 = breast_cancer['data']
y3 = breast_cancer['target']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3,
                                                        random_state=42)


# (b) Read up about the Area Under the Curve of the Receiver Operating
# Characteristic Curve, the so-called ROC-AUC-metric, e.g. at
# towardsdatascience.com.

# ROC
# the Receiver Operating Characteristic curve, or ROC curve is a plot of the
# false positive rate (x-axis) versus the true positive rate (y-axis) for a
# number of different candidate threshold values between 0.0 and 1.0.
# Put another way, it plots the false alarm rate versus the hit rate.

# AUC
# The area under the curve (AUC) can be used as a summary of the model skill.

# Interpretation:
# Smaller values on the x-axis of the plot indicate lower false positives and
# higher true negatives. Larger values on the y-axis of the plot indicate
# higher true positives and lower false negatives.
# A skillful model will assign a higher probability to a randomly chosen real
# positive occurrence than a negative occurrence on average. This is what we
# mean when we say that the model has skill. Generally, skillful models are
# represented by curves that bow up to the top left of the plot.
# A model with no skill at each threshold is represented by a diagonal line
# from the bottom left of the plot to the top right and has an AUC of 0.5.
# A model with perfect skill is represented by a line that travels from the
# bottom left of the plot to the top left and then across the top to the top
# right.

# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
lr_probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Logistic: ROC AUC=%.3f' % lr_auc)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()


# (c) Learn a Neural Network Classifier after MinMax-Scaling with in total
# four parameter combinations (and 1000 iterations) of your choice using
# 5-fold Cross-Validation. To keep computation burden low, stop after 1,000
# iterations and use the best solver for this dataset.
# Using the ROC-AuC-score metric to pick the best model, what are the best
# parameter combinations, which is its ROC-AuC-score, and how well does it
# generalize in terms of the ROC-AuC-score?

# scaling:
algorithms = [('scaler', MinMaxScaler()),
              ('nn', MLPClassifier(max_iter=1000, random_state=42,
                                   solver='lbfgs', activation='relu'))]

pipe = Pipeline(algorithms, verbose=True)
param_grid = {'nn__hidden_layer_sizes': [(100, 100), (120, 120)],
              'nn__alpha': [0.001, 0.005]}

grid3 = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid3.fit(X3_train, y3_train)

# Check grid search results
results3 = pd.DataFrame(grid3.cv_results_)
print(f'The combination that yields the best estimator is:\n'
      f'{grid3.best_estimator_}')

# Scores:
# sort the results by the rank test score to check the mean test score:
results3_sorted = results3.sort_values(by=['rank_test_score'],
                                       ignore_index=True)
mean_test_score3 = results3_sorted['mean_test_score']
print('The Roc-Auc mean test score is high for all the tests:')
print(mean_test_score3)


# (d) Plot the confusion matrix as a heatmap for the best model and save it as
# ./output/nn_breast_confusion.pdf.

preds3 = grid3.predict(X3_test)

# create confusion_matrix with the predicted (preds) and the true values
# (y_test)?
matrix3 = confusion_matrix(y3_test, preds3)
heatmap3 = sns.heatmap(matrix3, annot=True,
                       xticklabels=breast_cancer['target_names'],
                       yticklabels=breast_cancer['target_names'])
figure3 = heatmap3.get_figure()
figure3.savefig('./output/nn_breast_confusion.pdf')
