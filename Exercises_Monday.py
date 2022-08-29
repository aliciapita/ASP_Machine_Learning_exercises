import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
import seaborn as sns

# -----------------------------------------------------------------------------
# QUESTION 1 - Feature Engineering
# -----------------------------------------------------------------------------

# (a) Load the Breast Cancer dataset using
# sklearn.datasets.load_ breast_cancer() as per usual.

housing = fetch_california_housing()


# (b) Extract polynomial features (without bias!) and interactions up to a
# degree of 2 using PolynomialFeatures().
# How many features do you end up with?

# Define data (X) and target (Y)
X1 = housing['data']
Y1 = housing['target']

# build a dataframe
housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)

# Polynomial features extraction:
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly.fit(housing_df)
X1_poly = poly.transform(housing_df)

# transformation to count the features:
type(X1_poly)
row, col = X1_poly.shape
print(f'We end up with {col} features')


# (c) Create a pandas.DataFrame() using the polynomials. Use the originally
# provided feature names to generate names for the polynomials
# ( .get_feature_names() accepts a parameter) and use them as column names.
# Also add the dependent variable to the object and name the column ”y”.
# Finally save it as comma-separated textfile named ./output/polynomials.csv.

# Dataframe with the polynomials:
df_poly = pd.DataFrame(X1_poly,
                       columns=poly.get_feature_names(housing_df.columns))

# Add the dependent variable Y:
df_poly['Y'] = Y1.tolist()
df_poly.head()

# Save as CSV:
df_poly.to_csv('./output/polynomials.csv')


# -----------------------------------------------------------------------------
# QUESTION 2 - Principal Component Analysis
# -----------------------------------------------------------------------------

# (a) Read the textfile ./data/olympics.csv (in your git repository) into a
# pandas.DataFrame() using the first column as index. The data lists the
# individual performances of 33 male athletes during the Decathlon of the 1988
# Olympic summer games (100m sprint, running long, (broad) jump, shot put,
# high jump, 400m run, 110m hurdles, discus throw, pole vault, javelin throw,
# 1.500m run). Print summary statistics for each of the variables and decide
# (and act accordingly). Does it make sense to drop variable ”score” before
# proceeding?

# CSV to dataframe:
OLY = './data/olympics.csv'
df2_1 = pd.read_csv(OLY, index_col=0)
df2_1.head()

# Summary statistics:
print('Summary statistics - Olympic athletes')
print(df2_1.describe())

# Variable "score":
print("It make sense to drop variable ”score” before proceeding because it's\n"
      "a linear combination of the other variables and we need orthogonality")
df2_2 = df2_1.drop(columns='score')
print(df2_2.head())


# (b) Scale the data such that all variables have unit variance.
# Which pandas.DataFrame() method can you use to assert that all variables
# have unit variance?

# Scaling data:
scaler = StandardScaler().fit_transform(df2_2)
df_scaled = pd.DataFrame(scaler, columns=df2_2.columns)

# Check unit variance:
print(df_scaled.var())


# (c) Fit a plain vanilla PCA model. Store the components in a
# pandas.DataFrame() to display the loadings of each variable. Which variables
# load most prominently on the first component? Which ones on the second?
# Which ones on the third? How would you thus interpret those components?

# PCA model:
pca = PCA()
pca.fit(df_scaled)

# Store in a dataframe
loadings_df = pd.DataFrame(pca.components_.T,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
                                    'PC7', 'PC8', 'PC9', 'PC10'],
                           index=df2_2.columns)

# Analyse components individually:
PC1 = loadings_df['PC1']
PC1_sorted = PC1.sort_values()
print(PC1_sorted)
print("The variables that load most prominently on the first component are:\n"
      "110m hurdles (+), 100m sprint (+), running long (-) and perc (-) but\n"
      "none of them has a correlation (either positive or negative) higher\n"
      "than 0.45 with the first principal component")
PC2 = loadings_df['PC2']
PC2_sorted = PC2.sort_values()
print(PC2_sorted)
print("The variables that load most prominently on the second component are:\n"
      "discus throw (+), poid (+), 1.500m run (+) and running long (-) but\n"
      "only discus throw has a correlation of more than 0.5 with the 2nd PC.")
PC3 = loadings_df['PC3']
PC3_sorted = PC3.sort_values()
print(PC3_sorted)
print("The variables that load most prominently on the third component are:\n"
      "mainly haut (+), that has a correlation of more than 0.85 with the\n"
      " third PC. followed by 100m sprint (+), 1.500m run (-) and javelin\n"
      " throw (+), but all these three with less than 0.3 correlation with\n"
      " the 3rd PC")

# INTERPRETATION:
print("INTERPRETATION OF THE COMPONENTS:\n"
      "The farther from zero the loading is, the more correlated is the\n"
      "original variable with the PC. For example, the third PC increases\n"
      "with 'haut'. Thus, the 3rd PC could be interpreted as a measure of\n"
      "how important the discipline 'haut' is in terms of the performance\n"
      "of the athletes.")


# (d) How many components do you need to explain at least 90% of the data?

OLY = './data/olympics.csv'
df2_1 = pd.read_csv(OLY, index_col=0)
df2_2 = df2_1.drop(columns='score')
scaler = StandardScaler().fit_transform(df2_2)
df_scaled = pd.DataFrame(scaler, columns=df2_2.columns)
pca = PCA(n_components=0.9)
pca.fit(df_scaled)
new = pca.transform(df_scaled)
loadings_df_new = pd.DataFrame(pca.components_.T)
print(loadings_df_new.info())
print("Seven components are needed to explain at least 90% of the data")


# -----------------------------------------------------------------------------
# QUESTION 3 - Clustering
# -----------------------------------------------------------------------------

# (a) Load the iris dataset using sklearn.datasets.load_iris(). The data is on
# classifying flowers.

iris = load_iris()
X3 = iris['data']
# y = iris['target']
type(X3)

# (b) Scale the data such that each variable has unit variance.

scaler = StandardScaler()
scaler.fit(X3)
X3_scaled = scaler.transform(X3)
np.var(X3_scaled)

# (c) Assume there are three clusters. Fit a K-Means model, an Agglomerative
# Model and a DBSCAN model with min sample equal to 2 and ε equal to 1) with
# Euclidean distance. Store only the cluster assignments in a new
# pandas.DataFrame().

kmeans = KMeans(n_clusters=3, random_state=42)
predict1 = kmeans.fit_predict(X3_scaled)
df3_1 = pd.DataFrame(predict1)
df3_1.columns = ['kmeans']

agg = AgglomerativeClustering(n_clusters=3)
predict2 = agg.fit_predict(X3_scaled)
df3_1['Agglomerative'] = pd.DataFrame(predict2)

dbscan = DBSCAN(eps=1, min_samples=2)
predict3 = dbscan.fit_predict(X3_scaled)
df3_1['DBSCAN'] = pd.DataFrame(predict3)
print(df3_1.head())


# (d) Compute the silhouette scores using sklearn.metrics.silhouette_score()
# for each cluster algorithm from c). Why do you have to treat noise
# assignments from DBSCAN differently? Which model has the highest Silhouette
# score?

print(silhouette_score(X3_scaled, kmeans.labels_))
print(silhouette_score(X3_scaled, agg.labels_))
print(silhouette_score(X3_scaled, dbscan.labels_))

print("NOISE under DBSCAN:\n"
      "when there are no other data points around a data point within\n"
      "epsilon radius, it is treated as Noise under DBSCAN --> noise depends\n"
      "on the chosen epsilon: the smaller, the more noise, the bigger, the\n"
      "less noise. The data points that are treated as noise under DBSCAN do\n"
      "not belong to any cluster.")

print("DBSCAN model has the highest Silhouette score.\n"
      "The closer it's to 1, the more apart from each other are the clusters.")


# (e) Add variables ”sepalwidth” & ”petallength” including the corresponding
# column names to the pandas.DataFrame() that contains the cluster assignments.
# (Beware of the dimensionality!)

df3_2 = pd.DataFrame(data=iris.data, columns=iris.feature_names)
extracted_col1 = df3_2['sepal width (cm)']
extracted_col2 = df3_2['petal length (cm)']
df3_3 = df3_1.join(extracted_col1)
df3_4 = df3_3.join(extracted_col2)
df3_4.head()


# (f) Rename noise assignments to ”Noise”.

df3_5 = df3_4.replace(-1, 'Noise')
print(df3_5)


# (g) Plot a three-fold scatter plot using ”sepal width” as x-variable and
# ”petal length” as y-variable, with dots colored by the cluster assignment
# and facets by cluster algorithm. (Melt the pandas.DataFrame() with above
# variables as ID variables.) Save the plot as ./output/cluster_petal.pdf.
# Does the noise assignment make sense intuitively?

id_vars = ["sepal width (cm)", "petal length (cm)"]
value_vars = ["kmeans", "Agglomerative", "DBSCAN"]
melted = df3_5.melt(id_vars=id_vars, value_vars=value_vars)
figure = sns.relplot(x="sepal width (cm)", y="petal length (cm)", data=melted,
                     hue="value", col="variable")
figure.savefig("output/cluster_petal.pdf")
