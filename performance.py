# check performance

p = pd.DataFrame(columns=['run_description' ,'accuracy_score', 'precision', 'recall' , 'F1'])
performance = ['loss=log and only classifying laboratory random set improved labels']

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels, predicted)
performance.append(accuracy)

from sklearn.metrics import average_precision_score
precision = average_precision_score(test_labels, predicted)
performance.append(precision)

from sklearn.metrics import recall_score
recall = recall_score(test_labels, predicted)
performance.append(recall)

from sklearn.metrics import f1_score
f1 = f1_score(test_labels, predicted, average="binary")
print("f1:", f1)
performance.append(f1)

p.loc[len(df)] = performance
p.to_csv("location/classification_results/performance_SVM14.csv" , sep='\t')
