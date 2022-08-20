# 1. Tips

# (a) Load seaborn’s tips dataset using seaborn.load_dataset("tips") \t
# into a pandas.DataFrame().

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

print(type(tips))

print(tips.head(4))

# (b) Convert the short weekday names to their long version \t
# (e.g.,”Thursday” instead of ”Thur”) using .replace().

days = tips['day'].unique()

print(days)

tips = tips.replace({'day': {'Sun': 'Sunday', 'Thur': 'Thursday', 'Fri': 'Friday', 'Sat': 'Saturday'}})

days_replaced = tips['day'].unique()

print(days_replaced)

# (c) Create a scatterplot of 'tip' vs. 'total bill' colored by 'day' \t
# and facets (either by row or by column) by variable 'sex'. \t
# Label the axis so that the unit becomes apparent. \t
# Save the figure as ./output/tips.pdf

fig1 = sns.FacetGrid(tips, col='sex')

fig1.map_dataframe(sns.scatterplot, x='total_bill', y='tip', hue='day')

fig1.add_legend()

plt.savefig('./output/tips.pdf')

# 2. Occupations

# (a) Import the pipe-separated dataset from \t
# https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user \t
# into a pandas.DataFrame(). The data is on occupations and demographic information.

DAT8 = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user"

demog_df = pd.read_csv(DAT8, sep='|')

print(type(demog_df))

print(demog_df.head(4))

# (b) Print the last 10 entries and then the first 25 entries.

print(demog_df.tail(10))

print(demog_df.head(25))

# (c) What is the type of each column?

print(demog_df.dtypes)

# (d) Count how often each occupation is present! \t
# Store the information in a new object.

series2_1 = demog_df.occupation.value_counts()

series_to_df = pd.DataFrame(series2_1)

occup_df = series_to_df.reset_index()

occup_df.columns = ['occupation', 'counts']

print(occup_df)

# (e) How many different occupations are there in the dataset? \t
# What is the most frequent occupation? ( Try to use a programmatic \t
# solution for these questions using the new object!)

print(len(occup_df))

print(occup_df.loc[occup_df['counts'].idxmax()])

# (f) Sort the new object by index. Then create a figure and an axis. \t
# Plot a histogram for occupations on that axis ( Do not use .hist().). \t
# Add an appropriate label to the x-axis. How does the figure look like \t
# if you don’t sort by index beforehand? Save the figure as \t
# ./output/occupations.pdf!

# I think I did it in the previous exercise but I can order it \t
# alphabetically and plot both versions

occup_sort_alpha = occup_df.sort_values('occupation')

occup_alpha = occup_sort_alpha.reset_index(drop=True)

print(occup_alpha)

fig3_1 = occup_df.plot(x='occupation', kind='bar')

plt.savefig('./output/occupations1.pdf')

fig3_2 = occup_alpha.plot(x='occupation', kind='bar')

plt.savefig('./output/occupations2.pdf')

# 3. Iris

# (a) Read the Iris dataset from \t
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data \t
# as pandas.DataFrame(). The data is on measures of flowers and \t
# the values are on ”sepal length (in cm)”, ”sepal width (in cm)”, \t
# ”petal length (in cm)”, ”petal width (in cm)” and ”class”. \t
# Since the data doesn’t provide the column names, add the column names \t
# after reading in, or alternatively provide while reading in.

Iris = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df_Iris = pd.read_csv(Iris, sep=',', names=['sepal length (cm)',
                                            'sepal width (cm)',
                                            'petal length (cm)',
                                            'petal width (cm)',
                                            'class'])

print(type(df_Iris))

print(df_Iris.head(4))

# (b) Set the values of the rows 10 to 29 of the column \t
# ’petal length (in cm)’ to missing.

df_Iris.iloc[10:30, 2] = 'missing'

print(df_Iris.loc[9:34])

# (c) Replace missing values with 1.0.

df_Iris_new = df_Iris.replace(to_replace='missing', value=1.0)

print(df_Iris_new.loc[9:34])

# (d) Save the data as comma-separated file named \t
# ./output/iris.csv without index.

df_Iris_new.to_csv('./Output/iris.csv', index=False)

# (e) Visualize the distribution of all of the continuous variables \t
# by "class" with a catplot of your choice. Optionally, try to \t
# tilt/rotate the labels using .set_xticklabels(), which accepts \t
# a rotation parameter). Save the figure as ./output/iris.pdf.

df_Iris_new2 = df_Iris_new.replace({'class': {'Iris-setosa': 'Setosa',
                                              'Iris-versicolor': 'Versicolor',
                                              'Iris-virginica': 'Virginica'}})

fig, axes = plt.subplots(1, 4, sharex=True, figsize=(16, 4))

fig.suptitle('Distribution of attributes according to Iris class')

sns.stripplot(ax=axes[0], x="class", y="sepal length (cm)", data=df_Iris_new2)

sns.stripplot(ax=axes[1], x="class", y="sepal width (cm)", data=df_Iris_new2)

sns.stripplot(ax=axes[2], x="class", y="petal length (cm)", data=df_Iris_new2)

sns.stripplot(ax=axes[3], x="class", y="petal width (cm)", data=df_Iris_new2)

fig.savefig('./output/iris.pdf')


# 4. Memory

# I tried to solve number 4 in a different file
