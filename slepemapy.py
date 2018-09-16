from __future__ import print_function, division

from comet_ml import Experiment
from comet_ml import Optimizer

import tensorflow as tf
import datetime

import data_helper
import nn_model
import output_writer
import graph_helper
import evaluation_helper

### Params for running parts of the nn
LOG_COMET = False
RESTORE_MODEL = True
RUN_TRAIN = True
RUN_TEST = True
RUN_MAPS = True


### Params for nn
STUDENTS_COUNT_MAX = 3000
BATCH_SIZE = 10
# TODO fix if there is less data then STUDENTS_COUNT_MAX
NUM_BATCHES = STUDENTS_COUNT_MAX // BATCH_SIZE
TIMESTAMP = str(datetime.datetime.now())
assert(STUDENTS_COUNT_MAX % BATCH_SIZE == 0)
num_steps = 0
num_epochs = 5
state_size = 100 # number of hidden neurons
learning_rate = 100



# Create an experiment with your api key
if LOG_COMET:
    experiment = Experiment(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5", project_name="slepemapy")
    optimizer = Optimizer(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5")

train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
train_path= "/home/dave/projects/GoingDeeperWithDKT/data/0910_c_train.csv"
test_path = "/home/dave/projects/GoingDeeperWithDKT/data/0910_c_test.csv"
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
train_path = "/home/dave/projects/diploma/datasets/world_train.csv"
test_path = "/home/dave/projects/diploma/datasets/world_test.csv"

train_set = data_helper.SlepeMapyData(train_path, STUDENTS_COUNT_MAX, BATCH_SIZE, False)
test_set = data_helper.SlepeMapyData(test_path, STUDENTS_COUNT_MAX, BATCH_SIZE, True)
# TODO check padding as it will not work if test set has more data
num_steps = max(train_set.max_seq_len, test_set.max_seq_len) + 1
print(train_set.max_seq_len)
print(test_set.max_seq_len)

# TODO get this from dataset
num_classes = max(train_set.num_questions, test_set.num_questions) + 1  # number of classes
print(NUM_BATCHES)
# TODO think if needed --> if not delete
num_skills = num_classes
# # COMET hyperparams
# if LOG_COMET:
#     params = """
#     learning_rate real [0, 100] [10]
#     state_size integer [2,500] [100]
#     """
#     optimizer.set_params(params)
#     hyper_params = {"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": BATCH_SIZE, "state_size": state_size}
#     experiment.log_multiple_params(hyper_params)
#
# ##########
# #  MODEL
# ##########
#
# # PLACEHOLDERS
# x = tf.placeholder(tf.int32, [None, num_steps])
# y = tf.placeholder(tf.float32, [None, num_steps])
# seqlen = tf.placeholder(tf.int32, [None])
# target_x = tf.placeholder(tf.int32, [None, num_steps])
# target_y = tf.placeholder(tf.float32, [None, num_steps])
# #
# # SHAPING OF DATA
# #
# inputs_series = tf.split(x, num_steps, 1)
# target_label_s = tf.reshape(target_y, [-1, num_steps, 1])
# target_label_s = tf.tile(target_label_s, [1, 1, num_classes])
# target_one_hot = tf.one_hot(target_x, num_skills)
# rnn_inputs = tf.one_hot(x, num_skills)
# y_2 = tf.reshape(y, [-1, num_steps, 1])
# rnn_inputs = tf.concat([rnn_inputs, y_2], axis=2)
#
# # labels_series = tf.unstack(target_x, axis=1)
# ###
# ### FORWARD PASS
# ###
# cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
# # possible to add initial state initial_state=init_state
# output, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, sequence_length=seqlen, dtype=tf.float32)
# # TODO embedding one how vector tf.nn.embedding lookup check possible improvements
# with tf.variable_scope('softmax'):
#     W = tf.get_variable('W', [state_size, num_classes])
#     b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
#
# logits = tf.reshape(tf.matmul(tf.reshape(output, [-1, state_size]), W) + b,
#             [-1, num_steps, num_classes])
# predictions_series = tf.sigmoid(logits)
#
# ###
# ### BACKPROPAGATION
# ###
# target_label_s = target_label_s * target_one_hot
# logits = logits * target_one_hot
# losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label_s, logits=logits)
# total_loss = tf.reduce_mean(losses)
# train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

#experiment.set_model_graph()
# ##### END OF MODEL



graph_loss = []
graph_rmse_train = []
graph_rmse_test = []


def run_train(model,sess):
    print("New data, epoch", epoch_idx)
    questions = []
    prediction_labels = []
    correct_labels = []
    for step in range(NUM_BATCHES):
        batch_X, batch_Y, batch_target_X, batch_target_Y, batch_seq = train_set.next(BATCH_SIZE, num_steps)

        _total_loss, _train_step, _predictions_series = model.run_model_train(batch_X, batch_Y, batch_target_X, batch_target_Y, batch_seq, sess)

        questions = evaluation_helper.get_questions(batch_target_X)
        prediction_labels += evaluation_helper.get_predictions(_predictions_series, questions)
        correct_labels += evaluation_helper.get_labels(batch_target_Y, questions)

        # OUTPUT
        output_writer.output_visualization('visualization/data.txt',
                                           batch_target_X, batch_target_Y, batch_seq, _predictions_series, step)

        # EVALUATION
        if step + 1 >= NUM_BATCHES:
            rmse = evaluation_helper.rmse(correct_labels, prediction_labels)
            auc = evaluation_helper.auc(correct_labels, prediction_labels)
            pearson = evaluation_helper.pearson(correct_labels, prediction_labels)
            accurracy = evaluation_helper.accuracy(correct_labels, prediction_labels)

            loss_list.append(_total_loss)
            graph_loss.append(_total_loss)
            graph_rmse_train.append(rmse)
            print("Step", step, "Loss", _total_loss)
            print("RMSE is: ", rmse)
            print("AUC is: ", auc)
            print("Accuracy is: ", accurracy)
            print("Pearson coef is:", pearson)

            output_writer.output_results('results/results' + TIMESTAMP + '.txt', step, _total_loss, rmse, auc)
            output_writer.output_predictions('results/predictions_train.txt', questions, prediction_labels,
                                            correct_labels)


def run_test(model):
    print("--------------------------------")
    print("Calculating test set predictions")
    questions = []
    prediction_labels = []
    correct_labels = []

    # TODO MAKE TO WORK IN GENERAL
    for i in range(NUM_BATCHES // 10):
        test_batch_X, test_batch_Y, test_batch_target_X, test_batch_target_Y, test_batch_seq = test_set.next(BATCH_SIZE, num_steps)

        test_predictions = model.run_model_test(test_batch_X, test_batch_Y, test_batch_seq)

        questions = (evaluation_helper.get_questions(test_batch_target_X))
        prediction_labels += (evaluation_helper.get_predictions(test_predictions, questions))
        correct_labels += (evaluation_helper.get_labels(test_batch_target_Y, questions))
        # OUTPUT
        output_writer.output_visualization('visualization/test_data.txt',
                                           test_batch_target_X, test_batch_target_Y, test_batch_seq, test_predictions, i)
        output_writer.output_for_map('maps/nn_output/', test_batch_target_X, test_batch_target_Y, test_batch_seq, test_predictions)
        with open('results/results_test' + str(i) + '.txt', 'a') as f:
            rmse_test = evaluation_helper.rmse(prediction_labels, correct_labels)
            auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
            f.write("Epoch test RMSE is: %.3f \n" % rmse_test)
            f.write("Epoch test AUC is: %.3f \n\n" % auc_test)
            f.write("-------------------------------------------------------------------------------------- \n")


    # EVALUATION
    rmse_test = evaluation_helper.rmse(correct_labels, prediction_labels)
    auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
    pearson_test = evaluation_helper.pearson(correct_labels, prediction_labels)
    accurracy_test = evaluation_helper.accuracy(correct_labels, prediction_labels)

    if LOG_COMET:
        experiment.log_metric("rmse", rmse_test)
        experiment.log_metric("auc", auc_test)
        experiment.log_metric("pearson", pearson_test)
        experiment.log_metric("accuracy", accurracy_test)

    graph_rmse_test.append(rmse_test)
    with open('results/results_test' + TIMESTAMP + '.txt', 'a') as f:
        f.write("Epoch test RMSE is: %.3f \n" % rmse_test)
        f.write("Epoch test AUC is: %.3f \n\n" % auc_test)
        f.write("-------------------------------------------------------------------------------------- \n")

    print("RMSE for test set:%.5f" % rmse_test)
    print("AUC for test set:%.5f" % auc_test)
    print("Pearson coef is:", pearson_test)
    print("Accuracy is:", accurracy_test)

    print("--------------------------------")
    output_writer.output_predictions('results/predictions_test' + TIMESTAMP + '.txt', questions, prediction_labels,
                                     correct_labels)
    return rmse_test, auc_test, pearson_test, accurracy_test


def run_test_for_map(model):
    print("--------------------------------")
    print("Calculating test set predictions")
    questions = []
    prediction_labels = []
    correct_labels = []

    # TODO MAKE TO WORK IN GENERAL
    for i in range(NUM_BATCHES // 10):
        test_batch_X, test_batch_Y, test_batch_target_X, test_batch_target_Y, test_batch_seq = data_helper.get_data_for_map()

        test_predictions = model.run_model_test(test_batch_X, test_batch_Y, test_batch_seq)


        questions = (evaluation_helper.get_questions(test_batch_target_X))
        prediction_labels += (evaluation_helper.get_predictions(test_predictions, questions))
        correct_labels += (evaluation_helper.get_labels(test_batch_target_Y, questions))
        # OUTPUT
        output_writer.output_visualization('visualization/test_data.txt',
                                           test_batch_target_X, test_batch_target_Y, test_batch_seq, test_predictions,
                                           i)
        output_writer.output_for_map('maps/nn_output/', test_batch_target_X, test_batch_target_Y, test_batch_seq,
                                     test_predictions)
        with open('results/results_test' + str(i) + '.txt', 'a') as f:
            rmse_test = evaluation_helper.rmse(prediction_labels, correct_labels)
            auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
            f.write("Epoch test RMSE is: %.3f \n" % rmse_test)
            f.write("Epoch test AUC is: %.3f \n\n" % auc_test)
            f.write("-------------------------------------------------------------------------------------- \n")

    # EVALUATION
    rmse_test = evaluation_helper.rmse(correct_labels, prediction_labels)
    auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
    pearson_test = evaluation_helper.pearson(correct_labels, prediction_labels)
    accurracy_test = evaluation_helper.accuracy(correct_labels, prediction_labels)

    if LOG_COMET:
        experiment.log_metric("rmse", rmse_test)
        experiment.log_metric("auc", auc_test)
        experiment.log_metric("pearson", pearson_test)
        experiment.log_metric("accuracy", accurracy_test)

    graph_rmse_test.append(rmse_test)
    with open('results/results_test' + TIMESTAMP + '.txt', 'a') as f:
        f.write("Epoch test RMSE is: %.3f \n" % rmse_test)
        f.write("Epoch test AUC is: %.3f \n\n" % auc_test)
        f.write("-------------------------------------------------------------------------------------- \n")

    print("RMSE for test set:%.5f" % rmse_test)
    print("AUC for test set:%.5f" % auc_test)
    print("Pearson coef is:", pearson_test)
    print("Accuracy is:", accurracy_test)

    print("--------------------------------")
    output_writer.output_predictions('results/predictions_test' + TIMESTAMP + '.txt', questions, prediction_labels,
                                     correct_labels)
    return rmse_test, auc_test, pearson_test, accurracy_test



with tf.Session() as sess:
    model = nn_model.Model(num_classes, num_steps, sess, RESTORE_MODEL)

    loss_list = []
    #while True:
    if LOG_COMET:
        suggestion = optimizer.get_suggestion()
        experiment = Experiment(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5", project_name="slepemapy")
    #
    # Training
    #
    for epoch_idx in range(num_epochs):
        if LOG_COMET:
            experiment.set_step(epoch_idx)

        ###
        ### TRAINING DATASET
        ###
        if RUN_TRAIN:
            run_train(model,sess)

        ###
        ### TESTING DATASET
        ###
        if RUN_TEST:
            rmse, auc, pearson, accuracy = run_test(model)

        # if RUN_MAPS:
        #     run_test_for_map()

    ###
    ### LOGGING RESULTS
    ###
    if LOG_COMET:
        suggestion.report_score("auc", auc)
        suggestion.report_score("rmse", rmse)

    model.save_model()



###
### SHOWING GRAPH
###
#graph_helper.show_graph(graph_rmse_train, graph_rmse_test)
