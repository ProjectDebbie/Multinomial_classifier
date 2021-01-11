# master thesis project - Clarence Lépine (UPC Barcelona)
# multinomial classifier with 3 class: clinical studies, in vivo/in vitro studies, non biomaterials studies
# 4 classification models tested: Multinomial NB, SGDClassifier, Random Forest and k-Nearest-Neighbors

# import packages
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# load the data (training set and testing set -> csv files with one abstract in each line)
# new_train = extension of the DEBBIE gold standard and background datasets with ~1000 clinical abstracts
# new_test = 3d printing testing set
data_train = pd.read_csv(r'E:\multinomial_classifier\pubmed\new_train.csv')
data_test = pd.read_csv(r'E:\multinomial_classifier\pubmed\new_test.csv')

# to visualize the training set
print("Training data shape = {}".format(data_train.shape))
print('training set header\n', data_train.head())

# to visualize the testing set
print("Testing data shape = {}".format(data_test.shape))
print('testing set header\n', data_test.head())

# to represent the distribution of the training set in a circular diagram
labels = ['non biomaterials','in vivo-in vitro','clinical']
sizes = data_train["label"].value_counts()
colours = {'non biomaterials':'b','in vivo-in vitro':'g','clinical':'r'}
plt.pie(sizes,colors=[colours[key] for key in labels],autopct='%1.1f%%')
plt.legend(labels)
plt.title('Overview of the training set')
plt.show()

# to represent the distribution of the testing set in a circular diagram
labels = ['in vivo-in vitro','clinical', 'non biomaterials']
sizes = data_test["label"].value_counts()
colours = {'non biomaterials':'b','in vivo-in vitro':'g','clinical':'r'}
plt.pie(sizes,colors=[colours[key] for key in labels],autopct='%1.1f%%')
plt.legend(labels)
plt.title('Overview of the testing set')
plt.show()

# to make sure that there is not any multiple spaces or high letters in the training set and the testing set
def preprocess(txt):
  txt = re.sub('\s+',' ', txt)
  txt = txt.lower()
  return txt


data_train['text'] = data_train['text'].map(lambda x :preprocess(x))
data_test['text'] = data_test['text'].map(lambda x :preprocess(x))

# to convert the labels from the training set into numbers & to convert the text into tf-idf values
# the first label encountered will be assign to 0, the second to 1, the third to 2...
# here, we have 3 categories: 0 <-> clinical, 1 <-> in vivo/in vitro, 2 <-> non biomaterials (look at the dataset to know that)
encoder = LabelEncoder()
vectorizer = TfidfVectorizer()

train_x = vectorizer.fit_transform(data_train['text'])
train_y = encoder.fit_transform(data_train['label'])

# to fit the model parameters for Multinomial NB (NB) using grid search cross validation (with 10 iterations)
NB_param_grid = dict({"alpha":[1, 10, 100], 'fit_prior':[True, False]})
NB_grid = GridSearchCV(MultinomialNB(), param_grid=NB_param_grid, cv=10)
NB_grid.fit(train_x, train_y)

# to fit the model parameters for SGDClassifier (SGD) using grid search cross validation (with 10 iterations)
SGD_param_grid = dict({'loss': ['log'], 'penalty': ['l2'], "alpha": [0.01, 0.01, 1]})
SGD_grid = GridSearchCV(SGDClassifier(), param_grid=SGD_param_grid, cv=10)
SGD_grid.fit(train_x, train_y)

# to fit the model parameters for Random Forest (RF) using grid search cross validation (with 10 iterations)
RF_param_grid = dict({'max_features':[2,3], 'min_samples_leaf':[1]})
RF_grid = GridSearchCV(RandomForestClassifier(), param_grid=RF_param_grid, cv=10)
RF_grid.fit(train_x, train_y)

# to get the classification results for NB (confusion matrix, accuracy)
x = vectorizer.transform(data_test['text'])
y = encoder.transform(data_test['label'])
score = NB_grid.score(x, y)
print("Accuracy of Model is {}".format(score))
y_pred = NB_grid.predict(x)
print(classification_report(y, y_pred=y_pred))

# to put the classification results from NB in a file
results = dict(zip(data_test['text'], y_pred))
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['label'])
print('NB results:\nnumber of articles in each class\n', df_results['label'].value_counts()) # also done later maybe needs to be deleted
print("df_results(NB):", df_results)
df_results.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_NB.csv", sep='\t')

# print the number of abstracts in each class for NB
clinical_abstracts_NB = []
in_vivo_in_vitro_abstracts_NB = []
non_biomaterials_abstracts_NB = []

for key, value in results.items():
    if value == 0.0:
        clinical_abstracts_NB.append(key)
    elif value == 1.0:
        in_vivo_in_vitro_abstracts_NB.append(key)
    elif value == 2.0:
        non_biomaterials_abstracts_NB.append(key)

print('number of clinical abstracts (NB):', len(clinical_abstracts_NB))
print('number of in vivo-in vitro abstracts (NB):', len(in_vivo_in_vitro_abstracts_NB))
print('number of non biomaterials abstracts (NB):', len(non_biomaterials_abstracts_NB))

# to get the classification results for SGD (confusion matrix, accuracy)
x = vectorizer.transform(data_test['text'])
y = encoder.transform(data_test['label'])
score = SGD_grid.score(x, y)
print("Accuracy of Model is {}".format(score))
y_pred = SGD_grid.predict(x)
print(classification_report(y, y_pred=y_pred))

# to put the classification results from SGD in a file
results = dict(zip(data_test['text'], y_pred))
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['label'])
print('SGD results:\nnumber of articles in each class\n', df_results['label'].value_counts())
print("df_results(SGD):", df_results)
df_results.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_SGD.csv", sep='\t')

#  to print the number of abstracts in each class for SGD
clinical_abstracts_SGD = []
in_vivo_in_vitro_abstracts_SGD = []
non_biomaterials_abstracts_SGD = []

for key, value in results.items():
    if value == 0.0:
        clinical_abstracts_SGD.append(key)
    elif value == 1.0:
        in_vivo_in_vitro_abstracts_SGD.append(key)
    elif value == 2.0:
        non_biomaterials_abstracts_SGD.append(key)

print('number of clinical abstracts (SGD):', len(clinical_abstracts_SGD))
print('number of in vivo-in vitro abstracts (SGD):', len(in_vivo_in_vitro_abstracts_SGD))
print('number of non biomaterials abstracts (SGD):', len(non_biomaterials_abstracts_SGD))

# put the results for each category in a dedicate file (these files will be used for data extraction later)
# only done for SGD because it is the best classification model, so we will only extract information from its results
clinical = df_results[df_results['label'] == 0]
clinical.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_SGD_clinical.csv", sep='\t')
in_vivo_in_vitro = df_results[df_results['label'] == 1]
in_vivo_in_vitro.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_SGD_in_vivo_in_vitro.csv", sep='\t')
non_biomaterials = df_results[df_results['label'] == 2]
non_biomaterials.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_SGD_non_biomaterials.csv", sep='\t')

# to get the classification results for Random Forest (confusion matrix, accuracy)
x = vectorizer.transform(data_test['text'])
y = encoder.transform(data_test['label'])
score = RF_grid.score(x, y)
print("Accuracy of Model is {}".format(score))
y_pred = RF_grid.predict(x)
print(classification_report(y, y_pred=y_pred))

# to put the classification results from Random Forest in a file
results = dict(zip(data_test['text'], y_pred))
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['label'])
print('RF results:\nnumber of articles in each class\n', df_results['label'].value_counts())
print("df_results(RF):", df_results)
df_results.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_RF.csv", sep='\t')

#  to print the number of abstracts in each class for Random Forest
clinical_abstracts_RF = []
in_vivo_in_vitro_abstracts_RF = []
non_biomaterials_abstracts_RF = []

for key, value in results.items():
    if value == 0.0:
        clinical_abstracts_RF.append(key)
    elif value == 1.0:
        in_vivo_in_vitro_abstracts_RF.append(key)
    elif value == 2.0:
        non_biomaterials_abstracts_RF.append(key)

print('number of clinical abstracts (RF):', len(clinical_abstracts_RF))
print('number of in vivo-in vitro abstracts (RF):', len(in_vivo_in_vitro_abstracts_RF))
print('number of non biomaterials abstracts (RF):', len(non_biomaterials_abstracts_RF))

# to find the best number of neighbours for k-Nearest-Neighbors (kNN) using grid search cross validation (with 10 iterations)
knn_param_grid = dict({'n_neighbors':[1, 10, 11, 12, 13, 14, 15]})
knn_grid = GridSearchCV(KNeighborsClassifier(), cv=10, param_grid=knn_param_grid)
knn_grid.fit(train_x, train_y)

# to print the best number of neighbours & plot the cross validation results
print('knn best parameters:', knn_grid.best_params_)
score = knn_grid.cv_results_
neighbours = knn_param_grid['n_neighbors']
mean_score = score['mean_test_score']
plt.plot(neighbours, mean_score)
plt.xlabel('Neighbours')
plt.ylabel('mean score')
plt.title("Cross validation result")
plt.show()

# to get the classification results for kNN (confusion matrix, accuracy)
x = vectorizer.transform(data_test['text'])
y = encoder.transform(data_test['label'])
score = knn_grid.score(x, y)
print("Accuracy of Model is {}".format(score))
y_pred = knn_grid.predict(x)
print(classification_report(y, y_pred=y_pred))

# to put the classification results from kNN in a file
results = dict(zip(data_test['text'], y_pred))
df_results = pd.DataFrame.from_dict(results, orient='index', columns=['label'])
print('kNN results:\nnumber of articles in each class\n', df_results['label'].value_counts())
print("df_results(kNN):", df_results)
df_results.to_csv(r"E:\multinomial_classifier\Pubmed\classification_results_kNN.csv", sep='\t')

#  to print the number of abstracts in each class for kNN
clinical_abstracts_kNN = []
in_vivo_in_vitro_abstracts_kNN = []
non_biomaterials_abstracts_kNN = []

for key, value in results.items():
    if value == 0.0:
        clinical_abstracts_kNN.append(key)
    elif value == 1.0:
        in_vivo_in_vitro_abstracts_kNN.append(key)
    elif value == 2.0:
        non_biomaterials_abstracts_kNN.append(key)

print('number of clinical abstracts (kNN):', len(clinical_abstracts_kNN))
print('number of in vivo-in vitro abstracts (kNN):', len(in_vivo_in_vitro_abstracts_kNN))
print('number of non biomaterials abstracts (kNN):', len(non_biomaterials_abstracts_kNN))



