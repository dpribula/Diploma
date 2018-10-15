from __future__ import print_function, division

from comet_ml import Experiment
from comet_ml import Optimizer

import tensorflow as tf
import datetime

import data_helper
import nn_model_tensorflow
import output_writer
import evaluation_helper
from maps import map_helper

### Params for running parts of the nn
LOG_COMET = False
RESTORE_MODEL = True
RUN_TRAIN = True
RUN_TEST = True
RUN_MAPS = False


### Params for nn setup
STUDENTS_COUNT_MAX = 4800
TIMESTAMP = str(datetime.datetime.now())
STATE_SIZE = 100  # number of hidden neurons
LEARNING_RATE = 1000
BATCH_SIZE = 10
NUM_EPOCHS = 10

# Create an experiment with your api key
if LOG_COMET:
    experiment = Experiment(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5", project_name="slepemapy")
    optimizer = Optimizer(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5")
    params = """  
    learning_rate real [0, 100] [10] 
    state_size integer [2,500] [100] 
    """
    optimizer.set_params(params)
    hyper_params = {"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
                    "state_size": STATE_SIZE}
    experiment.log_multiple_params(hyper_params)

train_path= "/home/dave/projects/GoingDeeperWithDKT/data/0910_c_train.csv"
test_path = "/home/dave/projects/GoingDeeperWithDKT/data/0910_c_test.csv"
train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
train_path = "/home/dave/projects/diploma/datasets/world_train2.csv"
test_path = "/home/dave/projects/diploma/datasets/world_test2.csv"

train_set = data_helper.SlepeMapyData(train_path, STUDENTS_COUNT_MAX, BATCH_SIZE, False)
test_set = data_helper.SlepeMapyData(test_path, STUDENTS_COUNT_MAX, BATCH_SIZE, True)
num_batches_train =  len(train_set.questions) // BATCH_SIZE
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
        batch_X, batch_Y, batch_target_X, batch_target_Y, batch_seq = train_set.next(BATCH_SIZE, num_steps)

        _total_loss, _train_step, _predictions_series = model.run_model_train(batch_X, batch_Y, batch_target_X, batch_target_Y, batch_seq, sess)

        questions = evaluation_helper.get_questions(batch_target_X)
        prediction_labels += evaluation_helper.get_predictions(_predictions_series, questions)
        correct_labels += evaluation_helper.get_labels(batch_target_Y, questions)

        # OUTPUT
        output_writer.output_visualization('visualization/data.txt',
                                           batch_target_X, batch_target_Y, batch_seq, _predictions_series, step)

        # EVALUATION
        if step + 1 >= num_batches_train:
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
    for i in range(num_batches_test):
        test_batch_x, test_batch_y, test_batch_target_x, test_batch_target_y, test_batch_seq = test_set.next(BATCH_SIZE, num_steps)

        test_predictions = model.run_model_test(test_batch_x, test_batch_y, test_batch_seq)

        questions = (evaluation_helper.get_questions(test_batch_target_x))
        prediction_labels += (evaluation_helper.get_predictions(test_predictions, questions))
        correct_labels += (evaluation_helper.get_labels(test_batch_target_y, questions))
        # OUTPUT
        output_writer.output_visualization('visualization/test_data.txt',
                                           test_batch_target_x, test_batch_target_y, test_batch_seq, test_predictions, i)
        output_writer.output_for_map('maps/nn_output/', test_batch_target_x, test_batch_target_y, test_batch_seq, test_predictions)
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
    accuracy_test = evaluation_helper.accuracy(correct_labels, prediction_labels)

    if LOG_COMET:
        experiment.log_metric("rmse", rmse_test)
        experiment.log_metric("auc", auc_test)
        experiment.log_metric("pearson", pearson_test)
        experiment.log_metric("accuracy", accuracy_test)

    graph_rmse_test.append(rmse_test)
    with open('results/results_test' + TIMESTAMP + '.txt', 'a') as f:
        f.write("Epoch test RMSE is: %.3f \n" % rmse_test)
        f.write("Epoch test AUC is: %.3f \n\n" % auc_test)
        f.write("-------------------------------------------------------------------------------------- \n")

    print("RMSE for test set:%.5f" % rmse_test)
    print("AUC for test set:%.5f" % auc_test)
    print("Pearson coef is:", pearson_test)
    print("Accuracy is:", accuracy_test)

    print("--------------------------------")
    output_writer.output_predictions('results/predictions_test' + TIMESTAMP + '.txt', questions, prediction_labels,
                                     correct_labels)
    return rmse_test, auc_test, pearson_test, accuracy_test


def run_maps(model):
    questions = []
    answers = []
    for i in range(10):
        questions.append(159)
        answers.append(1)
    ques, ans, seq_len = map_helper.create_data_for_nn(answers, questions, 0)

    questions, prediction_labels, correct_labels, test_predictions = map_helper.run_data_on_nn(model, ans, ques, seq_len)
    output_writer.output_predictions('results/predictions_test' + TIMESTAMP + '.txt', questions, prediction_labels,
                                     correct_labels)
    map_helper.get_map_from_data()



with tf.Session() as sess:

    model = nn_model_tensorflow.Model(num_classes, num_steps, sess, RESTORE_MODEL)
    if RUN_MAPS:
        run_maps(model)
    loss_list = []

    #
    # Training
    #
    i = 0
    for epoch_idx in range(NUM_EPOCHS):
        if LOG_COMET:
            experiment.set_step(i)
            suggestion = optimizer.get_suggestion()
            i += 1

        ###
        ### TRAINING DATASET
        ###
        if RUN_TRAIN:
            run_train(model,sess)
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
        ### LOGGING RESULTS
        ###
        if LOG_COMET:
            print("COMET LOG")
            suggestion.report_score("rmse", rmse)



###
### SHOWING GRAPH
###
#graph_helper.show_graph(graph_rmse_train, graph_rmse_test)
