# import packages
import numpy as np
import os

training = r"..." # put your training set here
test = r"..." # put your testing set here
training_set = [os.path.join(training, f) for f in os.listdir(training)]
test_set = [os.path.join(test, f) for f in os.listdir(test)]


# make labels:
# 1 -> non_relevant
# 0 -> relevant
def make_labels(data):
    files = data
    file_list = []
    train_labels = np.zeros(len(files))  # to get a new array filled with zeros for the future labels 
    count = 0
    docID = 0
    for fil in files:
        file_list.append(fil)
        train_labels[docID] = 0
        filepathTokens = fil.split('/')
        lastToken = filepathTokens[len(filepathTokens) - 1]
        print(lastToken)
        if lastToken.startswith(("train+test\\abstracts_final", "new_test\\abstracts_final"), ): # change according to your problem
            train_labels[docID] = 1
            count = count + 1
        docID = docID + 1
    return train_labels, file_list


# (Convert text into the corresponding numerical features using bag of words method)
# make a dictionary from the training set:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(input='filename', ngram_range=(1, 2), max_df=0.9, min_df=0.0, stop_words='english')
X_train_counts = count_vect.fit_transform(training_set)


# (Convert values obtained using the bag of words model into TFIDF values)
# Term Frequency (TF)= (Number of Occurrences of a word)/(Total words in the document)
# IDF(word) = Log((Total number of documents)/(Number of documents containing the word))
# TF-IDF(word) = TF(word) * IDF(word)
# tf_idf calculation for the words in the training_set
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# make labels for training and test set
training_set_labels, file_list = make_labels(training_set)
test_set_labels, test_file_list = make_labels(test_set)
print(training_set_labels)
print(test_set_labels)


# train the model
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier().fit(X_train_tfidf, training_set_labels)

from sklearn.ensemble import RandomForestClassifier
rd_clf = RandomForestClassifier().fit(X_train_tfidf, training_set_labels)


# bag of word and tf_idf of test set
X_test_count = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_test_count)


predicted = sgd_clf.predict(X_test_tfidf)
predicted_2 = rd_clf.predict(X_test_tfidf)

results = dict(zip(test_set, predicted))
results_2 = dict(zip(test_set, predicted_2))

import pandas as pd
df_results = pd.DataFrame.from_dict(results, orient='index')
df_results_2 = pd.DataFrame.from_dict(results_2, orient='index')
print("df_results(SGD):", df_results)
print("df_results(RandomForest):", df_results_2)
df_results.to_csv(r"... .csv", sep='\t')
df_results.to_csv(r"... .csv", sep='\t')

# to count the number of abstracts in each class/category
relevant_abstracts_SGD = []
not_relevant_abstracts_SGD = []

for key, value in results.items():
    if value == 0.0:
        relevant_abstracts_SGD.append(key)
    elif value == 1.0:
        not_relevant_abstracts_SGD.append(key)


relevant_abstracts_RDF = []
not_relevant_abstracts_RDF = []

for key, value in results_2.items():
    if value == 0.0:
        relevant_abstracts_RDF.append(key)
    elif value == 1.0:
        not_relevant_abstracts_RDF.append(key)


# print the results (number of abstracts in each category)
print('number of relevant abstracts (SGD):', len(relevant_abstracts_SGD))
print('number of non-relevant abstracts (SGD):', len(not_relevant_abstracts_SGD))

print('number of relevant abstracts (RDF):', len(relevant_abstracts_RDF))
print('number of non-relevant abstracts (RDF):', len(not_relevant_abstracts_RDF))


# working (warning if we run this several times the results are added each time to the folder)
from shutil import copy2
for filename in os.listdir(r"..."): # put your testing set here
    file_to_copy = os.path.join(r"...", filename) # same
    if str(file_to_copy) in relevant_abstracts_SGD:
        copy2(file_to_copy, r"...") # put the direction where you want to store the biomaterials abstracts
    else:
        copy2(file_to_copy, r"...") # put the direction where you want to store the non biomaterials abstracts


# check performance
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_set_labels, predicted)
print('accuracy(SGD):', accuracy)

accuracy_2 = accuracy_score(test_set_labels, predicted_2)
print('accuracy(RandomForest):', accuracy_2)

from sklearn.metrics import average_precision_score
precision = average_precision_score(test_set_labels, predicted)
print('precision(SGD):', precision)

precision_2 = average_precision_score(test_set_labels, predicted_2)
print('precision(RandomForest):', precision_2)

from sklearn.metrics import recall_score
recall = recall_score(test_set_labels, predicted)
print("recall(SGD):", recall)

recall_2 = recall_score(test_set_labels, predicted_2)
print("recall(RandomForest):", recall_2)

# f1 performance score (working)
from sklearn.metrics import f1_score
f1 = f1_score(test_set_labels, predicted, average="binary")
print("f1(SGD):", f1)

f1_2 = f1_score(test_set_labels, predicted_2, average="binary")
print("f1(RandomForest):", f1_2)

# other option to get the performance scores (quicker)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("with SGD classifier")
print(confusion_matrix(test_set_labels, predicted))
print(classification_report(test_set_labels, predicted))
print(accuracy_score(test_set_labels, predicted))

print("with RandomForest classifier")
print(confusion_matrix(test_set_labels, predicted_2))
print(classification_report(test_set_labels, predicted_2))
print(accuracy_score(test_set_labels, predicted_2))



