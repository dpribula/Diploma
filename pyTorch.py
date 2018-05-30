import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_helper

# Torch settings
torch.manual_seed(1)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

# RNN PARAMETERS
num_classes = 1
input_size = num_classes
hidden_size = num_classes
batch_size = 1
num_layers = 1
sequence_length = 20
# DATA INITIALIZATION
train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
data = data_helper.SlepeMapyData(train_path, 1000, batch_size, False)
# MODEL


class LSTM1(nn.Module):
    def __init__(self, batch_size, hidden_dim, input_size, sequence_length):
        super(LSTM1, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(num_layers, self.batch_size, self.hidden_dim))

    def forward(self, questions):
        # questions to Tensor
        questions = torch.Tensor(questions)
        questions = questions.view(self.batch_size, self.sequence_length, -1)
        lstm_out, self.hidden = self.lstm(questions, self.hidden)
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return lstm_out


model = LSTM1(batch_size, hidden_size, input_size, sequence_length)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0)

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss_function(input, target)

for epoch in range(100):
    loss = 0
    optimizer.zero_grad()
    hidden = model.init_hidden()
    for questions, labels in zip(data.data, data.labels):
        labels = torch.Tensor(labels)
        labels = labels.view(1, 20)
        labels = torch.empty(1, 1, dtype=torch.long).random_(5)
        output = model(questions)
        loss += loss_function(output, labels)

    # add my data

    loss.backward(retain_graph=True)
    optimizer.step()
