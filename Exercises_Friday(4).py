# 4. Memory

# (a) Load the comma-separated data from \t
# https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx \t
# into a pandas.DataFrame() (large file!).

import pandas as pd

df4_1 = 'https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx'

df4_1 = pd.read_csv(df4_1, sep=',', low_memory=False)

print(type(df4_1))

# (b) Inspect the object using .info() and afterwards \t
# .info(memory_usage="deep"). What is the difference between \t
# the two calls? How much space does the DataFrame require in memory?

print(df4_1.info())

print(df4_1.info(memory_usage="deep"))

print('The second method gives precise info on memory usage.'
      'The DataFrame requires 859.4 MB of memory')

# (c) Create a copy of the object with only columns of type object \t
# by using .select_dtypes(include=['object']).

df4_2 = df4_1.select_dtypes(include=['object'])

print(df4_2.info())


# (d) Look at the summary of this object new (using .describe()).\t
# Which columns have very few unique values compared to the number of observations?

print(df4_2.describe())

column = df4_2.columns

print('The columns with very few unique values are:')

for i in column:
    res = df4_2[i].value_counts(normalize=True) * 100
    if res.iloc[0] <= 50:
        print(df4_1[i])


# (e) Does it make sense to convert a column of type object to type category\t
# if more than 50% of the observations contain unique values? Why/Why not?

print('No, it makes sense to convert from object to category when there\'s a \'category\' that repeats systematically,'
      'because a categorical variable takes on a limited, and usually fixed, number of possible values '
      '--> days of the week, blood type, country, etc. are good examples')


# (f) Convert all columns of type object of the original dataset \t
# to type category where you deem this appropriate.

column = df4_2.columns

for i in column:
    res = df4_1[i].value_counts(normalize=True) * 100
    if res.iloc[0] <= 50:
        df4_1[i] = df4_1[i].astype('category')

print(df4_1.info())

