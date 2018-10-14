import torch
import torch.nn as nn
import torch.optim as optim
import data_helper
import evaluation_helper
import numpy as np
from typing import List



# Torch settings
torch.manual_seed(1)
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

# DATA INITIALIZATION
train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
# RNN PARAMETERS
batch_size = 10
MAX_COUNT = 100000
data = data_helper.SlepeMapyData(train_path, MAX_COUNT, batch_size, False)
num_classes = data.num_questions + 1
input_size = num_classes + 1 # correct/incorrect
hidden_size = 50
num_layers = 1
sequence_length = data.max_seq_len
# MODEL


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

    def forward(self, questions, answers, target, hidden):
        # questions to Tensor
        questions = torch.tensor(questions)
        questions = torch.unsqueeze(questions, 2)
        x = torch.Tensor(batch_size, sequence_length, num_classes)
        x.zero_()
        x.scatter_(2, questions, 1)
        answers = torch.tensor(answers, dtype=torch.double)
        answers = torch.unsqueeze(answers, 2)
        y = torch.cat((x, answers), 2)
        y = y.view(self.batch_size, self.sequence_length, -1)
        lstm_out, self.hidden = self.lstm(y, hidden)
        output = self.hidden2tag(lstm_out.view(batch_size, self.sequence_length, -1))
        return output, self.hidden

    def run_train(self, hidden):
        #TODO make work in general
        predictions = []
        prediction_targets = []
        for i in range(100):
            questions, answers, questions_target, answers_target, batch_seq_len = data.next(batch_size, sequence_length)

            tag_scores, hidden = model.forward(questions, answers, questions_target, hidden)
            target: List[int] = []
            for batch_num in range(len(questions_target)):
                for seq_num, target_id in enumerate(questions_target[batch_num]):
                    target.append(batch_num * num_classes * sequence_length + seq_num * num_classes + int(target_id) )

            targets = torch.tensor(target, dtype=torch.int64)
            tag_scores = tag_scores.view(-1)
            logits = torch.gather(tag_scores, 0, targets)
            answers_target = torch.tensor(answers_target, dtype=torch.int64)
            target_correctness = answers_target.view(-1)
            target_correctness = torch.tensor(target_correctness, dtype=torch.double)
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits, target_correctness)
            loss.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            a = np.asarray(torch.sigmoid(logits).detach())
            b = np.asarray(target_correctness)
            predictions.append(a)
            prediction_targets.append(b)

        print("RMSE: ", evaluation_helper.rmse(prediction_targets, predictions))
        return loss.item()

    def run_test(self):
        return

    def save_model(self):
        return


model = Model(batch_size, hidden_size, input_size, sequence_length)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    hidden = model.init_hidden()
    optimizer.zero_grad()
    loss = model.run_train(hidden)
    print(loss)

