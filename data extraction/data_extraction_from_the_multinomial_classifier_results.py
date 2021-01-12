# master thesis project - Clarence Lépine (UPC Barcelona)
# data extraction from the multinomial classifier results using term frequency and tf-idf values


# import packages
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# load the data
with open(r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_clinical.txt', 'w') as output_file:
    with open(r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_clinical.csv', 'r') as input_file:
        [output_file.write("".join(row)+'clinical ;') for row in input_file]

with open(r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_mixed.txt', 'w') as output_file:
    with open(r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_mixed.csv', 'r') as input_file:
        [output_file.write("".join(row)+'mixed ;') for row in input_file]

with open(r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_non_biomaterials.txt', 'w') as output_file:
    with open(r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_non_biomaterials.csv', 'r') as input_file:
        [output_file.write("".join(row)+'non biomaterials ;') for row in input_file]


# convert the data for each category into pandas dataframe
input = r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_clinical.txt'
input2 = r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_mixed.txt' # mixed corresponds to the in vivo/in vitro category
input3 = r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_non_biomaterials.txt'
input4 = r'E:\multinomial_classifier\Pubmed\3d printing literature\classification_results_SGD_all.txt' # file with the results from each category all together

def make_dataframe():
    with open(input4) as f:
        content = f.readlines()
        labels, texts = ([], [])
        for line in content:
            if line.count(';') == 0:
                continue
            if line:
                label, text = line.split(';', 1)
                labels.append(label)
                texts.append(text)

        df = pd.DataFrame()
        df['label'] = labels
        df['text'] = texts
        df.to_csv(r'E:\multinomial_classifier\pubmed\3d printing literature\data_extraction_all.csv', index=False, header=True, encoding='utf-8')


# make_dataframe()

# to remove the last row of the dataframe with all categories (i.e. the entire 3d printing literature) that contains a NaN
data_corpus = pd.read_csv(r'E:\multinomial_classifier\pubmed\3d printing literature\data_extraction_all.csv')
data_corpus_final = data_corpus.head(-1)

# to get a new column in the dataframe with the text without stopwords
stop = stopwords.words('english')
newstopwords = ['0','1','2','p']
stop.extend(newstopwords)
data_corpus_final['textwithoutstopwords'] = data_corpus_final['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# to get the 20 most frequent terms from the 3d printing literature & to represent the results in a bar diagram
results_corpus = pd.Series(' '.join(data_corpus_final.textwithoutstopwords).split()).value_counts()[:20]
print('Most frequent words in 3d printing literature:\n',results_corpus)
results_corpus.plot.bar()
plt.title('Term frequencies in 3d printing literature')
plt.show()

# other option using a word cloud representation
# to transform the text into tokens & get the 20 most common tokens
def my_tokenizer(text):
    return text.split()

tokens = data_corpus_final.textwithoutstopwords.map(my_tokenizer).sum()
counter = Counter(tokens)
counter.most_common(20)

# to represent the results in a word cloud
def wordcloud(counter):
    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(counter)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# wordcloud(counter)


# to transform the text into tf-idf values & get a big tf-idf matrix with the tf-idf values of words in each document
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data_corpus_final['textwithoutstopwords'])
new_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())

# 1) to extract the results of the tf-idf matrix from each category, 2) get the 20 most common terms in each category regarding their tf-idf scores, and 3) to plot the results in a bar diagram
# for the clinical category
clinical_df = new_df.iloc[0:2174,:]
clinical_df.loc['tf idf average'] = clinical_df.mean()
tf_idf_sorted = clinical_df.sort_values(by='tf idf average', axis=1)
# print(tf_idf_sorted)
tf_idf_sorted = tf_idf_sorted[tf_idf_sorted.columns[-20:]]
print(tf_idf_sorted)
tf_idf_sorted.loc['tf idf average'].plot(kind='bar')
plt.title('TF-IDF values for the clinical category')
plt.show()

# for the in vivo/in vitro category
in_vivo_in_vitro_df = new_df.iloc[2175:6781,:]
in_vivo_in_vitro_df.loc['tf idf average'] = in_vivo_in_vitro_df.mean()
tf_idf_sorted = in_vivo_in_vitro_df.sort_values(by='tf idf average', axis=1)
# print(tf_idf_sorted)
tf_idf_sorted = tf_idf_sorted[tf_idf_sorted.columns[-20:]]
tf_idf_sorted.loc['tf idf average'].plot(kind='bar')
plt.title('TF-IDF values for the in vivo/in vitro category')
plt.show()

# for the non biomaterials category
non_biomaterials_df = new_df.iloc[6782:11153,:]
non_biomaterials_df.loc['tf idf average'] = non_biomaterials_df.mean()
tf_idf_sorted = non_biomaterials_df.sort_values(by='tf idf average', axis=1)
# print(tf_idf_sorted)
tf_idf_sorted = tf_idf_sorted[tf_idf_sorted.columns[-20:]]
tf_idf_sorted.loc['tf idf average'].plot(kind='bar')
plt.title('TF-IDF values for the non biomaterials category')
plt.show()

# extraction of the 20 most frequent terms in each category (without tf-idf)
# for the clinical category
# to remove the last row of the dataframe that contains a NaN
data_clinical = pd.read_csv(r'E:\multinomial_classifier\pubmed\3d printing literature\data_extraction_clinical.csv')
data_clinical_final = data_clinical.head(-1)

# to define the stopwords
stop = stopwords.words('english')
newstopwords = ['0','1','2','p']
stop.extend(newstopwords)
# to get a new column in the dataframe with text without stopwords
data_clinical_final['textwithoutstopwords'] = data_clinical_final['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# to get the 20 most frequent terms from the category & to represent the results in a bar diagram
results_clinical_class = pd.Series(' '.join(data_clinical_final.textwithoutstopwords).split()).value_counts()[:20]
print('Most frequent words in clinical studies:\n',results_clinical_class)
results_clinical_class.plot.bar()
plt.title('Term frequencies in clinical articles regarding 3D printing')
plt.show()

# other option using a word cloud representation
# to transorm the text into tokens
def my_tokenizer(text):
    return text.split()

# to get the 20 most common tokens
tokens = data_clinical_final.textwithoutstopwords.map(my_tokenizer).sum()
counter = Counter(tokens)
counter.most_common(20)

# to represent the results in a word cloud
def wordcloud(counter):
    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(counter)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

wordcloud(counter)


# same steps with the in vivo/in vitro category (called mixed here)

data_mixed = pd.read_csv(r'E:\multinomial_classifier\pubmed\3d printing literature\data_extraction_mixed.csv')
data_mixed_final = data_mixed.head(-1) # to remove the last row of the dataframe that contains a NaN
data_mixed_final['textwithoutstopwords'] = data_mixed_final['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

results_mixed_class = pd.Series(' '.join(data_mixed_final.textwithoutstopwords).split()).value_counts()[:20]
print('Most frequent words in vivo/in vitro studies:\n',results_mixed_class)
results_mixed_class.plot.bar()
plt.title('Term frequencies in "in vivo/in vitro" articles regarding 3D printing')
plt.show()

tokens = data_mixed_final.textwithoutstopwords.map(my_tokenizer).sum()
counter = Counter(tokens)
counter.most_common(20)
wordcloud(counter)


# same steps with the non biomaterials category

data_non_biomaterials = pd.read_csv(r'E:\multinomial_classifier\pubmed\3d printing literature\data_extraction_non_biomaterials.csv')
data_non_biomaterials_final = data_non_biomaterials.head(-1) # to remove the last row of the dataframe that contains a NaN
data_non_biomaterials_final['textwithoutstopwords'] = data_non_biomaterials_final['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

results_non_biomaterials_class = pd.Series(' '.join(data_non_biomaterials_final.textwithoutstopwords).split()).value_counts()[:20]
print('Most frequent words in non_biomaterials studies:\n',results_non_biomaterials_class)
results_non_biomaterials_class.plot.bar()
plt.title('Term frequencies in non biomaterials articles regarding 3D printing')
plt.show()

tokens = data_non_biomaterials_final.textwithoutstopwords.map(my_tokenizer).sum()
counter = Counter(tokens)
counter.most_common(20)
wordcloud(counter)


