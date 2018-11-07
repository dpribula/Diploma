from comet_ml import Experiment
from comet_ml import Optimizer

import torch
import torch.nn as nn
import torch.optim as optim
import data_helper
import evaluation_helper
import numpy as np
from typing import List

import graph_helper

SHOW_GRAPH = True
LOG_COMET = True

## Neural net params
LEARNING_RATE = 0.001
BATCH_SIZE = 100
DROPOUT = 0.5
NUM_EPOCHS = 50
HIDDEN_SIZE = 100

# Create an experiment with your api key
if LOG_COMET:
    experiment = Experiment(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5", project_name="slepemapy-test")
    optimizer = Optimizer(api_key="LNWEZpzWIqUYH9X3s6D3n7Co5")
    params = """  
    learning_rate real [0.0001, 0.01] [0.001] 
    state_size integer [10,500] [100] 
    """
    optimizer.set_params(params)
    hyper_params = {"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
                    "state_size": HIDDEN_SIZE}
    experiment.log_multiple_params(hyper_params)

# Torch settings
torch.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# DATA INITIALIZATION
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
train_path = "/home/dave/projects/diploma/datasets/simulated_train.csv"
test_path = "/home/dave/projects/diploma/datasets/simulated_test.csv"
test_path = "/home/dave/projects/diploma/datasets/world_test2.csv"
train_path = "/home/dave/projects/diploma/datasets/world_train2.csv"
# RNN PARAMETERS

MAX_COUNT = 30000
NUM_BATCH_TRAIN = MAX_COUNT // BATCH_SIZE
NUM_BATCH_TEST = MAX_COUNT // BATCH_SIZE // 5
train_data = data_helper.SlepeMapyData(train_path, MAX_COUNT, BATCH_SIZE, False)
test_data = data_helper.SlepeMapyData(test_path, MAX_COUNT // 5, BATCH_SIZE, False)
num_classes = train_data.num_questions + 1
input_size = num_classes + 1  # correct/incorrect or any other additional params
num_layers = 1
sequence_length = train_data.max_seq_len
# MODEL


def detach_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)

class Model(nn.Module):
    def __init__(self, batch_size, hidden_dim, input_size, sequence_length):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True, dropout=DROPOUT)

        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(num_layers, self.batch_size, self.hidden_dim))

    def forward(self, questions, answers, hidden):
        # questions to Tensor
        questions = torch.tensor(questions)
        questions = torch.unsqueeze(questions, 2)
        x = torch.Tensor(BATCH_SIZE, sequence_length, num_classes)
        x.zero_()
        x.scatter_(2, questions, 1)
        answers = torch.tensor(answers, dtype=torch.float)
        answers = torch.unsqueeze(answers, 2)
        input_concatenation = torch.cat((x, answers), 2)
        input_concatenation = input_concatenation.view(self.batch_size, self.sequence_length , -1)
        lstm_out, self.hidden = self.lstm(input_concatenation, hidden)
        ### Not neccessity just to be sure to have exact dimensions
        output = self.hidden2tag(lstm_out.view(BATCH_SIZE, self.sequence_length, -1))
        return output, self.hidden

    def run_train(self, hidden):
        prediction_labels = []
        correct_labels = []
        for i in range(NUM_BATCH_TRAIN):
            ### Forward pass
            questions, answers, questions_target, answers_target, batch_seq_len = train_data.next(BATCH_SIZE, sequence_length)
            ### Detach hidden layer from history so we don't backpropagate through every step
            ### and reuse the same weights
            hidden = detach_hidden(hidden)
            optimizer.zero_grad()
            tag_scores, hidden = model.forward(questions, answers, hidden)

            ### Preparing data for backpropagation
            target: List[int] = []
            for batch_num in range(len(questions_target)):
                for seq_num, target_id in enumerate(questions_target[batch_num]):
                    target.append(batch_num * num_classes * sequence_length + seq_num * num_classes + int(target_id))

            # Preparing logits
            targets = torch.tensor(target, dtype=torch.int64)
            tag_scores2 = tag_scores.view(-1)
            logits = torch.gather(tag_scores2, 0, targets)

            # Preparing correct answers
            answers_target = torch.tensor(answers_target, dtype=torch.int64)
            target_correctness = answers_target.view(-1)
            target_correctness = torch.tensor(target_correctness, dtype=torch.float)

            ### Backpropagation ###
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits, target_correctness)
            loss.backward()
            optimizer.step()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            ### EVALUATION
            questions = (evaluation_helper.get_questions(np.asarray(questions_target)))
            preds = np.asarray(torch.sigmoid(tag_scores).detach())
            prediction_labels += (evaluation_helper.get_predictions(preds, questions))
            correct_labels += (evaluation_helper.get_labels(np.asarray(answers_target), questions))

        # EVALUATION
        rmse_train = evaluation_helper.rmse(correct_labels, prediction_labels)
        auc_train = evaluation_helper.auc(correct_labels, prediction_labels)
        pearson_train = evaluation_helper.pearson(correct_labels, prediction_labels)
        accuracy_train = evaluation_helper.accuracy(correct_labels, prediction_labels)
        print("==========================TRAIN SET==========================")
        evaluation_helper.print_results(rmse_train, auc_train, pearson_train, accuracy_train)

        # Comet reporting
        if LOG_COMET:
            experiment.log_metric("rmse_train", rmse_train)
            experiment.log_metric("auc_train", auc_train)
            experiment.log_metric("pearson_train", pearson_train)
            experiment.log_metric("accuracy_train", accuracy_train)

        return rmse_train

    def run_test(self, hidden):
        with torch.no_grad():
            # TODO make work in general
            prediction_labels = []
            correct_labels = []
            for i in range(NUM_BATCH_TEST):
                ### Forward pass
                questions, answers, questions_target, answers_target, batch_seq_len = test_data.next(BATCH_SIZE, sequence_length)
                ### Detach hidden layer from history so we don't backpropagate through every step
                ### and reuse the same weights
                hidden = detach_hidden(hidden)
                optimizer.zero_grad()
                tag_scores, hidden = model.forward(questions, answers, hidden)

                answers_target = torch.tensor(answers_target, dtype=torch.int64)


                ### EVALUATION
                questions = (evaluation_helper.get_questions(np.asarray(questions_target)))
                preds = np.asarray(torch.sigmoid(tag_scores).detach())
                prediction_labels += (evaluation_helper.get_predictions(preds, questions))
                correct_labels += (evaluation_helper.get_labels(np.asarray(answers_target), questions))

            # EVALUATION
            rmse_test = evaluation_helper.rmse(correct_labels, prediction_labels)
            auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
            pearson_test = evaluation_helper.pearson(correct_labels, prediction_labels)
            accuracy_test = evaluation_helper.accuracy(correct_labels, prediction_labels)
            print("################ TEST SET ##################")
            evaluation_helper.print_results(rmse_test, auc_test, pearson_test, accuracy_test)

            # Comet reporting
            if LOG_COMET:
                experiment.log_metric("rmse_test", rmse_test)
                experiment.log_metric("auc_test", auc_test)
                experiment.log_metric("pearson_test", pearson_test)
                experiment.log_metric("accuracy_test", accuracy_test)


            return rmse_test, prediction_labels

    def save_model(self):
        return


model = Model(BATCH_SIZE, HIDDEN_SIZE, input_size, sequence_length)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

graph_rmse_train = []
graph_rmse_test = []

for epoch in range(NUM_EPOCHS):

    if LOG_COMET:
        experiment.set_step(epoch)

    hidden = model.init_hidden()
    rmse_train = model.run_train(hidden)
    hidden = model.init_hidden()
    rmse_test, preds = model.run_test(hidden)

    if SHOW_GRAPH:
        graph_rmse_train.append(rmse_train)
        graph_rmse_test.append(rmse_test)




if SHOW_GRAPH:
    graph_helper.show_graph(graph_rmse_train, graph_rmse_test)
