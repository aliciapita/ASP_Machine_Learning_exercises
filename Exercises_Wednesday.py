import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import pathlib
import pickle
from string import digits, punctuation
import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import locale

# -----------------------------------------------------------------------------
# QUESTION 1 - Regular Expressions
# -----------------------------------------------------------------------------

# (a) Solve the first eight regex practice problems on
# https://regexone.com/problem/matching_html (hover over ”Interactive Tutorial”
# to see them).

# Done online

# -----------------------------------------------------------------------------
# QUESTION 2 - Speeches I
# -----------------------------------------------------------------------------

# (a) Read up about pathlib.Path().glob(). Use it to read the text files in
# ./data/speeches into a corpus (i.e. a list of strings), but only those that
# start with ”R0”. The files represent a non-random selection of speeches of
# central bankers, which have already been stripped off meta information.
# The files are encoded in UTF8, except for two broken ones. Use a
# try-except-statement to skip reading them and print the filename instead.

text_files = list(pathlib.Path('./data/speeches').glob('R0*'))
print(len(text_files))

corpus = []
for i in text_files:
    try:
        e = open(i, mode='r', encoding="utf-8").read()
        corpus.append(e)
    except UnicodeDecodeError:
        print(f'{i} is broken')

len(corpus)


# (b) Vectorize the speeches using tfidf using 1-grams, 2-grams and 3-grams
# while removing English stopwords and proper tokenization (i.e., you create a
# tfidf matrix).

# remove English stopwords:
_stopwords = nltk.corpus.stopwords.words('english')

# word stemmer for English language:
_stemmer = nltk.snowball.SnowballStemmer('english')


# tokenize function (with word stemming):

def tokenize_and_stem(text):
    """Return tokens of text deprived of numbers and punctuation."""
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text.lower())]


tfidf2 = TfidfVectorizer(stop_words=_stopwords, tokenizer=tokenize_and_stem,
                         ngram_range=(1, 3))
type(tfidf2)
# tfidf matrix:
tfidf2_matrix = tfidf2.fit_transform(corpus)
type(tfidf2_matrix)

# To have it as a dataframe:
df_tfidf2_1 = pd.DataFrame(tfidf2_matrix.todense().T,
                           index=tfidf2.get_feature_names_out())


# (c) Pickle the resulting sparse matrix using pickle.dump() as
# ./output/speech_matrix.pk. Save the terms as well as ./output/terms.csv

file = open('./output/speech_matrix.pk', 'wb')
pickle.dump(tfidf2_matrix, file)

# Save only the index (the terms) to CSV:
df_tfidf2_2 = df_tfidf2_1.reset_index()
df_tfidf2_3 = df_tfidf2_2['index']
df_tfidf2_3.to_csv("./output/terms.csv")


# -----------------------------------------------------------------------------
# QUESTION 3 - Speeches II
# -----------------------------------------------------------------------------

# (a) Read the count-matrix from exercise ”Speeches I”
# (./output/speech_matrix.pk) using pickle.load().

file = open('./output/speech_matrix.pk', 'wb')
pickle.dump(tfidf2_matrix, file)
pickle_file = open('./output/speech_matrix.pk', 'rb')
content = pickle.load(pickle_file)
type(content)


# (b) Using the matrix, create a dendrogram of hierarchical clustering using
# the cosine distance and the complete linkage method. Remove the x-ticks from
# the plot. Optionally, set the color threshold such that three clusters are
# shown.

content_array = content.toarray()
type(content_array)

# Perform hierarchical clustering.
clusters = shc.linkage(content_array, method='complete', metric='cosine')

# Dendrogram
plt.figure()
dendrogram = shc.dendrogram(clusters, color_threshold=0.85, no_labels=True)


# (c) Save the dendrogram as ./output/speeches_dendrogram.pdf.

plt.savefig('./output/speeches_dendrogram.pdf')


# -----------------------------------------------------------------------------
# QUESTION 4 - Job Ads
# -----------------------------------------------------------------------------

# (a) Read the text file ./data/Stellenanzeigen.txt and parse the lines such
# that you obtain a pandas.DataFrame() with three columns: ”Newspaper”, ”Date”,
# ”Job Ad”. Make sure to set the Date column as datetime type.

# Read the text file
file4 = open('./data/Stellenanzeigen.txt', 'r')
text4_1 = file4.read()

# Create a list with the different ads as elements:
ad_list = text4_1.split('\n\n')
print(f'There are {len(ad_list)} job ads in the file')

# Create a list separating newspaper, date and the text of the ad:
data = []
for ad in ad_list:
    ad_first_split = ad.split(',', 1)
    newspaper = ad_first_split[0]
    ad_second_split = ad_first_split[1].split('\n', 1)
    date_string = ad_second_split[0]
    ad_text = ad_second_split[1]
    datum = [newspaper, date_string, ad_text]
    data.append(datum)

# Create a dataframe with newspaper, date and job ad as columns:
df_ads = pd.DataFrame(data, columns=['Newspaper', 'Date', 'Job ad'])

# Change the column 'Date' to datetime type:
locale.setlocale(locale.LC_ALL, 'de_DE')
df_ads.Date = pd.to_datetime(df_ads.Date, format=" %d. %B %Y")
print(df_ads.dtypes)


# (b) Create a new column counting the number of words per job ad. Plot the
# average job ad length by decade.

# New column counting the # of words:
df_ads['N words'] = df_ads['Job ad'].apply(lambda x: len(str(x).split(' ')))

# Create a dataframe grouping by decades and calculating average length of ad:
df_decades = df_ads['N words'].groupby(df_ads.Date.dt.year // 10).mean()\
    .rename('Average Length').reset_index()
df_decades.Date = df_decades.Date * 10
df_decades = df_decades.rename({'Date': 'Decade'}, axis=1)

# Plot the average job ad length by decade.
fig_av_length = df_decades.plot.bar(figsize=(12, 8), x='Decade', legend=False,
                                    xlabel='Decade', rot=0,
                                    ylabel='Average number of words',
                                    title='Job ads - Average Length')


# (c) Create a second pandas.DataFrame() that aggregates the job ads by decade,
# keeping just this information. Which are the most often used terms used by
# decade after appropriate cleaning?

# Create a DataFrame that aggregates the job ads by decade
ads_decades = df_ads['Job ad'].groupby(df_ads.Date.dt.year // 10).sum()\
    .rename('Job ads').reset_index()
ads_decades.Date = ads_decades.Date * 10
ads_decades = ads_decades.set_index('Date')

ads_decades["Job ads"] = ads_decades["Job ads"].values.astype('str')

# Convert the sum of ads to a list of strings:
corpus_ads = ads_decades.values.tolist()
corpus_ads_str = []
for item in corpus_ads:
    a = str(item)
    corpus_ads_str.append(a)

# Appropriate cleaning:
corpus_ads_nosym = []
for item in corpus_ads_str:
    b = item.replace('§', '')
    corpus_ads_nosym.append(b)

# remove German and English stopwords:
_stopwords_ger = nltk.corpus.stopwords.words('german')
_stopwords_eng = nltk.corpus.stopwords.words('english')
_stopwords_ger_eng = _stopwords_eng + _stopwords_ger

# word stemmer for German and English language:
_stemmer_2 = nltk.snowball.SnowballStemmer('german')

# tokenize function (with word stemming):


def tokenize_and_stem_2(text):
    """Return tokens of text deprived of numbers and punctuation."""
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer_2.stem(t) for t in nltk.word_tokenize(text.lower())]

# Counting words after cleaning:


count_words = CountVectorizer(stop_words=_stopwords_ger_eng,
                              tokenizer=tokenize_and_stem_2)
count_words.fit(corpus_ads_nosym)
count_matrix = count_words.transform(corpus_ads_nosym)
count_words.get_feature_names_out()
df_count = pd.DataFrame(count_matrix.todense().T,
                        index=count_words.get_feature_names_out(),
                        columns=df_decades["Decade"])

# Get the most used terms for every decade:
df_count.columns = df_count.columns.map(str)
# 1900 Decade
s_1900 = df_count['1900']
mode_1900 = s_1900.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1900s were:\n'
      f'{mode_1900}')
# 1910 Decade
s_1910 = df_count['1910']
mode_1910 = s_1910.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1910s were:\n'
      f'{mode_1910}')
# 1920 Decade
s_1920 = df_count['1920']
mode_1920 = s_1920.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1920s were:\n'
      f'{mode_1920}')
# 1930 Decade
s_1930 = df_count['1930']
mode_1930 = s_1930.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1930s were:\n'
      f'{mode_1930}')
# 1940 Decade
s_1940 = df_count['1940']
mode_1940 = s_1940.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1940s were:\n'
      f'{mode_1940}')
# 1950 Decade
s_1950 = df_count['1950']
mode_1950 = s_1950.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1950s were:\n'
      f'{mode_1950}')
# 1960 Decade
s_1960 = df_count['1960']
mode_1960 = s_1960.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1960s were:\n'
      f'{mode_1960}')
# 1970 Decade
s_1970 = df_count['1970']
mode_1970 = s_1970.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1970s were:\n'
      f'{mode_1970}')
# 1980 Decade
s_1980 = df_count['1980']
mode_1980 = s_1980.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1980s were:\n'
      f'{mode_1980}')
# 1990 Decade
s_1990 = df_count['1990']
mode_1990 = s_1990.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 1990s were:\n'
      f'{mode_1990}')
# 2000 Decade
s_2000 = df_count['2000']
mode_2000 = s_2000.nlargest(n=10)
print(f'The most used (stem) words for the selected job ads in 2000s were:\n'
      f'{mode_2000}')
