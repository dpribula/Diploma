from __future__ import print_function, division
from math import sqrt
from sklearn import metrics
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np
import datetime
### my imports
import data_helper
import output_writer
import graph_helper
import evaluation_helper
import random

test_path = "/home/dave/projects/diploma/datasets/world_test2.csv"
train_path = "/home/dave/projects/diploma/datasets/world_train2.csv"

STUDENTS_COUNT_MAX = 30000
BATCH_SIZE = 10
num_steps = 100

train_set = data_helper.SlepeMapyData(train_path, STUDENTS_COUNT_MAX, BATCH_SIZE, False)
test_set = data_helper.SlepeMapyData(test_path, STUDENTS_COUNT_MAX, BATCH_SIZE, True)

def run_evaluation(questions, labels):
    correct_labels = []
    prediction_labels = []
    for label in labels:
        correct_labels += label
        prediction_labels += [1 for _ in range(0, len(label))]

    rmse = sqrt(metrics.mean_squared_error(correct_labels, prediction_labels))
    auc = metrics.roc_auc_score(correct_labels, prediction_labels)
    fpr, tpr, thresholds = metrics.roc_curve(correct_labels, prediction_labels, pos_label=1)
    auc2 = metrics.auc(fpr, tpr)
    pearson = pearsonr(correct_labels, prediction_labels)
    r2 = r2_score(correct_labels, prediction_labels)
    accuracy = evaluation_helper.accuracy(correct_labels, prediction_labels)
    # pearson = pearson * pearson

    print("RMSE is: ", rmse)
    print("AUC is: ", auc)
    print("AUC is: ", auc2)
    print("Pearson coef is:", pearson)
    print("Pearson coef is:", r2)
    print("Accuracy is:", accuracy)
# TODO test
def run_evaluation_test(questions, labels):
    correct_labels = []
    prediction_labels = []

    test_batch_X, test_batch_Y, test_batch_target_X, test_batch_target_Y, test_batch_seq =  test_set.next(BATCH_SIZE, 100)
    correct_labels += (evaluation_helper.get_questions(test_batch_target_X))


    rmse = evaluation_helper.rmse(correct_labels, prediction_labels)
    auc = evaluation_helper.auc(correct_labels, prediction_labels)
    pearson = evaluation_helper.pearson(correct_labels, prediction_labels)
    accuracy = evaluation_helper.accuracy(correct_labels, prediction_labels)

    print("RMSE is: ", rmse)
    print("AUC is: ", auc)
    print("Pearson coef is:", pearson)
    print("Accuracy is:", accuracy)

print("TRAIN SET\n")
run_evaluation(train_set.questions, train_set.labels)

print("TEST_SET")
run_evaluation(test_set.questions, test_set.labels)

