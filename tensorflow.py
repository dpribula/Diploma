from __future__ import print_function, division

from comet_ml import Experiment
from comet_ml import Optimizer

import tensorflow as tf
import datetime
import time
import data_helper
import graph_helper
import nn_model_tensorflow
import evaluation_helper

start_time = time.time()

### Params for running parts of the nn
LOG_COMET = False
RESTORE_MODEL = False
RUN_TRAIN = True
RUN_TEST = True
RUN_MAPS = False


### Params for nn setup
STUDENTS_COUNT_MAX = 1000
TIMESTAMP = str(datetime.datetime.now())
STATE_SIZE = 100  # number of hidden neurons
LEARNING_RATE = 0.001
BATCH_SIZE = 10
NUM_EPOCHS = 50

# Create an experiment with for the comet
if LOG_COMET:
    experiment = Experiment(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5", project_name="slepemapy-test")
    optimizer = Optimizer(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5")
    params = """  
    learning_rate real [0, 100] [10] 
    state_size integer [2,500] [100] 
    """
    # optimizer.set_params(params)
    hyper_params = {"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
                    "state_size": STATE_SIZE}
    experiment.log_multiple_params(hyper_params)

test_path = "/home/dave/projects/diploma/datasets/world_test2.csv"
train_path = "/home/dave/projects/diploma/datasets/world_train2.csv"

train_set = data_helper.SlepeMapyData(train_path, STUDENTS_COUNT_MAX, BATCH_SIZE, False)
test_set = data_helper.SlepeMapyData(test_path, STUDENTS_COUNT_MAX, BATCH_SIZE, True)
num_batches_train = len(train_set.questions) // BATCH_SIZE
num_batches_test = len(test_set.questions) // BATCH_SIZE

num_steps = max(train_set.max_seq_len, test_set.max_seq_len) + 1
print(test_set.max_seq_len)
train_test_set_ratio = len(train_set.questions) // len(test_set.questions)

num_classes = max(train_set.num_questions, test_set.num_questions) + 1  # number of classes
print(num_batches_train)

graph_loss = []
graph_rmse_train = []
graph_rmse_test = []


def run_train(model, sess):
    print("New data, epoch", epoch_idx)
    prediction_labels = []
    correct_labels = []
    for step in range(num_batches_train):
        questions, answers, questions_target, answers_target, batch_seq = train_set.next(BATCH_SIZE, num_steps)

        _total_loss, _train_step, _predictions_series = model.run_model_train(questions, answers, questions_target, answers_target, batch_seq, sess)

        questions = evaluation_helper.get_questions(questions_target)
        prediction_labels += evaluation_helper.get_predictions(_predictions_series, questions)
        correct_labels += evaluation_helper.get_labels(answers_target, questions)

        # EVALUATION
        if step + 1 >= num_batches_train:
            time1 = time.time()
            rmse = evaluation_helper.rmse(correct_labels, prediction_labels)
            auc = evaluation_helper.auc(correct_labels, prediction_labels)
            pearson = evaluation_helper.pearson(correct_labels, prediction_labels)
            accuracy = evaluation_helper.accuracy(correct_labels, prediction_labels)
            print("Time of the run was %s seconds" % (time.time() - time1))

            loss_list.append(_total_loss)
            graph_loss.append(_total_loss)
            graph_rmse_train.append(rmse)
            print("Step", step, "Loss", _total_loss)
            print("RMSE is: ", rmse)
            print("AUC is: ", auc)
            print("Accuracy is: ", accuracy)
            print("Pearson coef is:", pearson)

            if LOG_COMET:
                experiment.log_metric("rmse_train", rmse)
                experiment.log_metric("auc_train", auc)
                experiment.log_metric("pearson_train", pearson)
                experiment.log_metric("accuracy_train", accuracy)


def run_test(model):
    print("--------------------------------")
    print("Calculating test set predictions")
    questions = []
    prediction_labels = []
    correct_labels = []

    for i in range(num_batches_test):
        test_batch_x, test_batch_y, test_batch_target_x, test_batch_target_y, test_batch_seq = test_set.next(BATCH_SIZE, num_steps)

        test_predictions = model.run_model_test(test_batch_x, test_batch_y, test_batch_seq)

        questions = (evaluation_helper.get_questions(test_batch_target_x))
        prediction_labels += (evaluation_helper.get_predictions(test_predictions, questions))
        correct_labels += (evaluation_helper.get_labels(test_batch_target_y, questions))


    # EVALUATION
    rmse_test = evaluation_helper.rmse(correct_labels, prediction_labels)
    auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
    pearson_test = evaluation_helper.pearson(correct_labels, prediction_labels)
    accuracy_test = evaluation_helper.accuracy(correct_labels, prediction_labels)

    if LOG_COMET:
        experiment.log_metric("rmse_test", rmse_test)
        experiment.log_metric("auc_test", auc_test)
        experiment.log_metric("pearson_test", pearson_test)
        experiment.log_metric("accuracy_test", accuracy_test)

    graph_rmse_test.append(rmse_test)

    print("RMSE for test set:%.5f" % rmse_test)
    print("AUC for test set:%.5f" % auc_test)
    print("Pearson coef is:", pearson_test)
    print("Accuracy is:", accuracy_test)

    print("--------------------------------")

    return rmse_test, auc_test, pearson_test, accuracy_test


with tf.Session() as sess:

    model = nn_model_tensorflow.Model(num_classes, num_steps, sess, RESTORE_MODEL)
    loss_list = []

    #
    # Training
    #
    for epoch_idx in range(NUM_EPOCHS):
        if LOG_COMET:
            experiment.set_step(epoch_idx)

        ###
        ### TRAINING DATASET
        ###
        if RUN_TRAIN:
            run_train(model, sess)
            model.save_model()

        ###
        ### TESTING DATASET
        ###
        if RUN_TEST:
            rmse, auc, pearson, accuracy = run_test(model)
        #
        # if RUN_MAPS:
        #      run_test_for_map()

        ###
        ### SAVING NETWORK
        ###

        file_writer = tf.summary.FileWriter('~', sess.graph)

    print("------------------------------")
    print("Time of the run was %s seconds" % (time.time() - start_time))
    print("------------------------------")

###
### SHOWING GRAPH
###
graph_helper.show_graph(graph_rmse_train, graph_rmse_test)
