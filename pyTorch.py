import time
import torch
import torch.nn as nn
import torch.optim as optim
import data_helper
import evaluation_helper
import numpy as np
from typing import List

import graph_helper

SHOW_GRAPH = True

start_time = time.time()
# Torch settings
torch.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# DATA INITIALIZATION
test_path = "/home/dave/projects/diploma/datasets/world_test2.csv"
train_path = "/home/dave/projects/diploma/datasets/world_train2.csv"
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
# RNN PARAMETERS
batch_size = 10
MAX_COUNT = 1000
NUM_BATCH_TRAIN = MAX_COUNT // batch_size
NUM_BATCH_TEST = MAX_COUNT // batch_size // 5
train_data = data_helper.SlepeMapyData(train_path, MAX_COUNT, batch_size, False)
test_data = data_helper.SlepeMapyData(test_path, MAX_COUNT//5, batch_size, False)
num_classes = train_data.num_questions + 1
input_size = num_classes + 1  # correct/incorrect or any other additional params
hidden_size = 100
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
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(num_layers, self.batch_size, self.hidden_dim))

    def forward(self, questions, answers, hidden):
        # questions to Tensor
        questions = torch.tensor(questions)
        questions = torch.unsqueeze(questions, 2)
        x = torch.Tensor(batch_size, sequence_length, num_classes)
        x.zero_()
        x.scatter_(2, questions, 1)
        answers = torch.tensor(answers, dtype=torch.float)
        answers = torch.unsqueeze(answers, 2)
        input_concatenation = torch.cat((x, answers), 2)
        input_concatenation = input_concatenation.view(self.batch_size, self.sequence_length, -1)
        lstm_out, self.hidden = self.lstm(input_concatenation, hidden)
        ### Not neccessity just to be sure to have exact dimensions
        output = self.hidden2tag(lstm_out.view(batch_size, self.sequence_length, -1))
        return output, self.hidden

    def run_train(self, hidden):
        #TODO make work in general
        predictions = []
        prediction_targets = []
        prediction_labels = []
        correct_labels = []
        for i in range(NUM_BATCH_TRAIN):
            ### Forward pass
            questions, answers, questions_target, answers_target, batch_seq_len = train_data.next(batch_size, sequence_length)
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
        rmse_test = evaluation_helper.rmse(correct_labels, prediction_labels)
        auc_test = evaluation_helper.auc(correct_labels, prediction_labels)
        pearson_test = evaluation_helper.pearson(correct_labels, prediction_labels)
        accuracy_test = evaluation_helper.accuracy(correct_labels, prediction_labels)
        print("==========================TRAIN SET==========================")
        print("RMSE for test set:%.5f" % rmse_test)
        print("AUC for test set:%.5f" % auc_test)
        # print("Pearson coef is:", pearson_test)
        # print("Accuracy is:", accuracy_test)
        print("======================================================================")
        return rmse_test

    def run_test(self, hidden):
        with torch.no_grad():
            # TODO make work in general
            prediction_labels = []
            correct_labels = []
            for i in range(NUM_BATCH_TEST):
                ### Forward pass
                questions, answers, questions_target, answers_target, batch_seq_len = test_data.next(batch_size, sequence_length)
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

                targets = torch.tensor(target, dtype=torch.int64)
                tag_scores2 = tag_scores.view(-1)
                logits = torch.gather(tag_scores2, 0, targets)
                answers_target = torch.tensor(answers_target, dtype=torch.int64)
                target_correctness = answers_target.view(-1)
                target_correctness = torch.tensor(target_correctness, dtype=torch.float)

                ### Backpropagation ###
                # loss_function = nn.BCEWithLogitsLoss()
                # loss = loss_function(logits, target_correctness)
                # loss.backward()
                # optimizer.step()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

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
            print("RMSE for test set:%.5f" % rmse_test)
            print("AUC for test set:%.5f" % auc_test)
            # print("Pearson coef is:", pearson_test)
            # print("Accuracy is:", accuracy_test)
            print("####################################################")
            return rmse_test

    def save_model(self):
        return


model = Model(batch_size, hidden_size, input_size, sequence_length)
optimizer = optim.Adam(model.parameters(), lr=0.001)

graph_rmse_train = []
graph_rmse_test = []

for epoch in range(100):
    hidden = model.init_hidden()
    rmse_train = model.run_train(hidden)
    hidden = model.init_hidden()
    rmse_test = model.run_test(hidden)
    if SHOW_GRAPH:
        graph_rmse_train.append(rmse_train)
        graph_rmse_test.append(rmse_test)

print("------------------------------")
print("Time of the run was %s seconds" % (time.time() - start_time))
print("------------------------------")
if SHOW_GRAPH:
    graph_helper.show_graph(graph_rmse_train, graph_rmse_test)
