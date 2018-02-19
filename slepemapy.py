from __future__ import print_function, division
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# python 2 compatibility for deepcopy
from copy import deepcopy
import time
import datetime
import matplotlib.pyplot as plt

# Limiting number of student for performance
STUDENTS_COUNT = 100
TIMESTAMP = str(datetime.datetime.now())
#
# Parsing dataset into data, labels and their lengths
#
def read_dataset(path):
    data = []
    labels = []
    seq_len = []
    max_seq_len = 0
    file = open(path,'r')
    count = 0
    for line in file:
        questions = []
        correct = []
        line_data = line.split(",")
        max_seq_len = max(max_seq_len, len(line_data))
        if (count % 3) == 0:
            count += 1
            continue
        elif (count % 3) == 1:
            data.append(list(map(lambda x:int(x), line_data)))
            seq_len.append(len(line_data)-1)
            count += 1
        elif (count % 3) == 2:
            labels.append(list(map(lambda x:int(x), line_data)))
            count += 1
            
        if count >= STUDENTS_COUNT*3:
            break
    add_padding(data, max(max_seq_len,num_steps + 1))
    add_padding(labels, max(max_seq_len,num_steps + 1))

    return data, labels, seq_len, max_seq_len -1

#
# Adding zeros to max length to data 
# We need to have matrices of the same size
#
def add_padding(data, length):
    for entry in data:
        while(len(entry)<length):
            entry.append(int(0))

class SlepeMapyData(object):
    def __init__(self,path):
        self.data, self.labels, self.seqlen, self.max_seq_len = read_dataset(path)
        self.batch_id = 0
    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        questions = (self.data[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        answers = (self.labels[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        #TODO refactor
        questions_target = questions
        answers_target = answers
        for i in range(batch_size):
            temp = questions[i].copy() #python3
            temp2 = deepcopy(answers[i]) #python3
            questions[i] = deepcopy(questions[i][:-1])
            questions_target[i] = temp[1:]
            answers[i] = answers[i][:-1]
            answers_target[i] = temp2[1:]
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return questions, answers, questions_target, answers_target, batch_seqlen

#train_path = "./datasets/generated_train.txt"
#test_path = "./datasets/generated_test.txt"
### DEBUG 
train_path = "/home/dave/projects/diploma/datasets/generated_train.txt"
test_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
num_steps = 0
train_set = SlepeMapyData(train_path)
num_steps = train_set.max_seq_len 
test_set = SlepeMapyData(test_path)

num_epochs = 100
total_series_length = num_steps * STUDENTS_COUNT 
state_size = 64 # number of hidden neurons
num_classes = 3 # number of classes
batch_size = 50
num_batches = total_series_length//batch_size//num_steps
learning_rate = 1
#TODO think if needed --> if not delete 
num_skills = num_classes
    
##########
### MODEL 
##########

# PLACEHOLDERS
x = tf.placeholder(tf.int32, [None, num_steps])
y = tf.placeholder(tf.float32, [None, num_steps])
seqlen = tf.placeholder(tf.int32, [None])
target_x = tf.placeholder(tf.int32, [None, num_steps])
target_y = tf.placeholder(tf.float32, [None, num_steps])
#
# SHAPING OF DATA
#
inputs_series = tf.split(x, num_steps, 1)
target_label_s = tf.reshape(target_y, [-1, num_steps, 1])
target_label_s = tf.tile(target_label_s, [1, 1, num_classes])
target_one_hot = tf.one_hot(target_x, num_skills)
rnn_inputs = tf.one_hot(x, num_skills)
y_2 = tf.reshape(y, [-1, num_steps, 1])
rnn_inputs = tf.concat([rnn_inputs, y_2], axis=2)
#labels_series = tf.unstack(target_x, axis=1)
# 
# FORWARD PASS
#
cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
# possible to add initial state initial_state=init_state
output, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, sequence_length=seqlen, dtype=tf.float32)
# TODO embedding one how vector tf.nn.embeddinglookup check possible improvements
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

logits = tf.reshape(tf.matmul(tf.reshape(output, [-1, state_size]), W) + b,
            [-1, num_steps, num_classes])
predictions_series = tf.sigmoid(logits)
#msqrt = tf.sqrt(tf.reduce_sum(tf.square(predictions_series - target_label_s) * target_one_hot)

#
#BACKPROPAGATION
#
# selected_logits = tf.reshape(logits, [-1, num_classes])
# selected_logits2 = tf.gather(selected_logits, target_x, axis=1)
# #test1 = tf.gather(selected_logits, target_x, axis=2)
# test2 = tf.gather(selected_logits, target_x, axis=0)
target_label_s = target_label_s * target_one_hot
logits = logits * target_one_hot 
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label_s, logits=logits)
#losses = tf.reduce_sum(losses)
#losses = tf.reduce_sum(losses * target_one_hot, axis=-1)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

###### END OF MODEL

graph_loss = []
graph_rmse_train = []
graph_rmse_test = []
def get_questions(batch_questions):
    questions = []
    for question in batch_questions:
        questions.extend(question)
    return questions

def get_predictions(_predictions_series,questions):
    pred_labels = []
    pred_labels_without0 = []
    i = 0
    for predictions in _predictions_series: 
        j = 0
        for prediction in predictions:
            pred_labels.append(prediction[questions[i*num_steps + j]]) 
            j+=1
        i+=1
    for i in range(len(pred_labels)):
        #CUTTING PADDING
        if questions[i] != 0:
            pred_labels_without0.append(pred_labels[i])

    return pred_labels_without0    

def get_labels(labels, questions):
    correct_labels = []
    correct_labels_without0 = []
    for label in labels:
        correct_labels.extend(label)
    for i in range(len(correct_labels)):
        #CUTTING PADDING
        if questions[i] != 0:
            correct_labels_without0.append(correct_labels[i])  
    return correct_labels_without0
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    for epoch_idx in range(num_epochs):
        print("New data, epoch", epoch_idx)

        for step in range(num_batches):
            batch_X, batch_Y, batch_target_X, batch_target_Y, batch_seq = train_set.next(batch_size)

            _total_loss, _train_step, _predictions_series = sess.run(
                [total_loss, train_step,  predictions_series],
                feed_dict={
                    x:batch_X,
                    y:batch_Y,
                    target_x:batch_target_X,
                    target_y:batch_target_Y,
                    seqlen:batch_seq                   
                })
           
            if step % 100 == 0:
                questions = get_questions(batch_target_X)
                pred_labels_without0 = get_predictions(_predictions_series, questions)       
                correct_labels_without0 = get_labels(batch_target_Y, questions)
                
                loss_list.append(_total_loss)
                graph_loss.append(_total_loss)
                
                print("Step",step, "Loss", _total_loss)
                rmse = sqrt(mean_squared_error(pred_labels_without0, correct_labels_without0))
                graph_rmse_train.append(rmse)
                print("Epoch train RMSE is: ", rmse)
                with open('results'+TIMESTAMP +'.txt','a') as f:
                    f.write("Step %.2f Loss %.2f \n" % (step, _total_loss))
                    f.write("Epoch train RMSE is: %.2f \n" % rmse)
                
                with open('predictions_train.txt','w') as f:
                    for i in range(len(correct_labels_without0)):
                        if questions[i] != 0:
                            f.write('question ID is: %d' % questions[i])
                            f.write("pred:%.2f " % pred_labels_without0[i])
                            f.write("correct:%s" % correct_labels_without0[i])
                            f.write("\n")
            # TODO 
            # fpr, tpr, thresholds = metrics.roc_curve(batchY[0], pred_labels, pos_label=1)
            # auc = metrics.auc(fpr, tpr)

            # #calculate r^2
            # r2 = r2_score(batchY[0], pred_labels)#
        
        ###
        ### TESTING DATASET
        ###
        ### TODO do in batches for memory efficiency
        test_question = np.array(test_set.data)[:,:-1]
        test_labels = np.array(test_set.labels)[:,:-1]
        prediction_question = np.array(test_set.data)[:,1:]
        prediction_labels =  np.array(test_set.labels)[:,1:]
        # We do not need target as we are not learning on test dataset
        test_predictions = sess.run(predictions_series,
                    feed_dict={
                            x:test_question,
                            y:test_labels,
                            seqlen:test_set.seqlen})
        print("--------------------------------")
        print("Calculating test set predictions")
        
        questions = get_questions(test_question)
        # Cutting out padding data from predictions
        pred_labels = get_predictions(test_predictions, questions)       
        correct_labels = get_labels(prediction_labels, questions)
        
        rmse_test = sqrt(mean_squared_error(pred_labels, correct_labels))
        graph_rmse_test.append(rmse_test)
        with open('results_test'+TIMESTAMP +'.txt','a') as f:
                    f.write("Epoch test RMSE is: %.2f \n" % rmse_test)
        print("RMSE for test set:%.5f" % rmse_test)
        print("--------------------------------")
        print(len(correct_labels))
        with open('predictions_test' +TIMESTAMP +'.txt','w') as f:

            for i in range(len(correct_labels)):
                if questions[i] != 0:
                    f.write('%d. ' % i)
                    f.write('question ID is: %d' % questions[i])
                    f.write("pred:%.2f " % pred_labels[i])
                    f.write("correct:%s" % correct_labels[i])
                    f.write("\n")
                if((i+1)%7 ==0):
                    f.write("\n")
    
#plt.plot(graph_loss)
plt.plot(graph_rmse_test, label='test_RMSE',color='blue')
plt.plot(graph_rmse_train, label='train_RMSE',color='green')
plt.plot([0 for _ in range(0,len(graph_rmse_test))],color='red',linewidth=4)
plt.ylabel('Training')
plt.gca().set_ylim([0,1])
plt.show()
