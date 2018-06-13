import pandas as pd
import evaluation_helper as eval

# Read in data

features = pd.read_csv('../data/answer.csv', sep=';', nrows=10000, skiprows=[i for i in range(1, 500000)])

print(features.head(5))

# Parsing labels
import numpy as np
questions = np.array(features['place_asked'])
answers = np.array(features['place_answered'])
labels = np.equal(questions, answers)
labels = list(map(lambda x: 1 if x else 0, labels))
labels = np.array(labels)

# Removing answers from features
features = features.drop('place_answered', axis=1)
# TODO parse date
features = features.drop('inserted', axis=1)
features = features.drop('place_map', axis=1)
features = features.drop('ip_country', axis=1)
features = features.drop('ip_id', axis=1)
print(features.describe())


# Parsing options to number of options
features['options'] = features['options'].apply(lambda x: len(x[1:-1].split(",")))
features = features['response_time']
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
averages = np.array([1 for x in range(0, len(test_labels))])
# Baseline errors, and display average baseline error
baseline_errors = abs(averages - test_labels)
print("##############################BASELINE###################################")
print('Average baseline error: ', round(np.mean(baseline_errors), 2))
print('Accuracy: ', eval.accuracy(test_labels, averages))
print('RMSE: ', eval.rmse(averages, test_labels))
print('Pearson: ', eval.pearson(averages, test_labels))



# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print("##############################TEST SET###################################")
print('Average baseline error:', round(np.mean(errors), 2))
print('Accuracy: ', eval.accuracy(test_labels, averages))
print('RMSE: ', eval.rmse(predictions, test_labels))
print('Pearson: ', eval.pearson(predictions, test_labels))
print('AUC: ', eval.auc(list(map(int,test_labels)), predictions))

print("##############################FEATURES IMPORTANCE###################################")
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
