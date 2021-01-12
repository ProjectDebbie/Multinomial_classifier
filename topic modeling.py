# Importing modules
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer


os.chdir('..')
# Read data into papers
papers = pd.read_csv(r'C:\Users\archi\Documents\Multinomial_classifier-master\3d_printing_literature.csv')
# to print the data head
print(papers.head())

from nltk.corpus import stopwords
stop = stopwords.words('english')
newstopwords = ['p']
stop.extend(newstopwords)
papers['textwithoutstopwords'] = papers['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

papers['text_tokenized'] = papers.apply(lambda row: nltk.word_tokenize(row['textwithoutstopwords']), axis=1)
print(papers)

lemmatizer = WordNetLemmatizer()
papers['text_lemmatized'] = papers['text_tokenized'].apply(lambda lst:[lemmatizer.lemmatize(word) for word in lst])
print(papers)

# to change the list format to string
papers['text_lemmatized'] = papers['text_lemmatized'].apply(', '.join)
print(papers)

# Import the wordcloud library
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Join everything together
long_string = ','.join(list(papers['text_lemmatized'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
plt.imshow(wordcloud)


# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')


# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))
    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(papers['text_lemmatized'])
# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)

import warnings

warnings.simplefilter("ignore", DeprecationWarning)
# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


# Tweak the two parameters below
number_topics = 6
number_words = 6
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis

LDAvis_data_filepath = os.path.join(r'E:\multinomial_classifier\pubmed\3d printing literature\ldavis_prepared_' + str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, r'E:\multinomial_classifier\pubmed\3d printing literature\ldavis_prepared_' + str(number_topics) + '.html')