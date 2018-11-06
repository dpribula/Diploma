import pandas as pd
import evaluation_helper

# Read in data

features = pd.read_csv('../data/answer.csv', sep=';')

# Parsing labels
import numpy as np
#features = features[1000000:1100000]
questions = np.array(features['place_asked'])
answers = np.array(features['place_answered'])
labels = np.equal(questions, answers)
labels = list(map(lambda x: 1 if x else 0, labels))
features['correct'] = np.array(labels)

ids = [i for i in range(1473)]
filtered = features[features['place_asked'].isin(ids)]
grouped = filtered.groupby('place_asked')['correct'].agg([['averages', 'mean']])

correct_labels = features['correct']
prediction_labels = []
skip = True
for index, row in features.iterrows():
    question_id = int(row['place_asked'])
    temp = grouped['averages']
    prediction_labels.append(temp[question_id])


print(evaluation_helper.rmse(correct_labels,prediction_labels))
print(evaluation_helper.auc(correct_labels, prediction_labels))
print(evaluation_helper.pearson(correct_labels, prediction_labels))
print(evaluation_helper.accuracy(correct_labels, prediction_labels))






