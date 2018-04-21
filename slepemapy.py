from __future__ import print_function, division
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import datetime
### my imports
import data_helper
import output_writer
import graph_helper
import evaluation_helper


# Limiting number of student for performance
STUDENTS_COUNT = 10000
TIMESTAMP = str(datetime.datetime.now())


### DEBUG 
train_path = "/home/dave/projects/diploma/datasets/generated_test.txt"
test_path = "/home/dave/projects/diploma/datasets/generated_train.txt"

num_steps = 0
train_set = data_helper.SlepeMapyData(train_path)
num_steps = train_set.max_seq_len 
test_set = data_helper.SlepeMapyData(test_path)

num_epochs = 10
total_series_length = num_steps * STUDENTS_COUNT 
state_size = 64 # number of hidden neurons
num_classes = 100 # number of classes
batch_size = 10
num_batches = total_series_length//batch_size//num_steps
learning_rate = 0.1
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
# labels_series = tf.unstack(target_x, axis=1)
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
#m sqrt = tf.sqrt(tf.reduce_sum(tf.square(predictions_series - target_label_s) * target_one_hot)

#
# BACKPROPAGATION
target_label_s = target_label_s * target_one_hot
logits = logits * target_one_hot
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label_s, logits=logits)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

# ##### END OF MODEL

graph_loss = []
graph_rmse_train = []
graph_rmse_test = []

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
                questions = evaluation_helper.get_questions(batch_target_X)
                pred_labels_without0 = evaluation_helper.get_predictions(_predictions_series, questions)       
                correct_labels_without0 = evaluation_helper.get_labels(batch_target_Y, questions)
                
                rmse = sqrt(mean_squared_error(pred_labels_without0, correct_labels_without0))

                loss_list.append(_total_loss)
                graph_loss.append(_total_loss)
                graph_rmse_train.append(rmse)
                
                print("Step",step, "Loss", _total_loss)
                print("Epoch train RMSE is: ", rmse)
                
                output_writer.output_results('results/results'+TIMESTAMP +'.txt', step, _total_loss,rmse)    
                output_writer.output_predictions('results/predictions_train.txt', questions, pred_labels_without0, correct_labels_without0)
               
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
        
        questions = evaluation_helper.get_questions(prediction_question)
        # Cutting out padding data from predictions
        pred_labels = evaluation_helper.get_predictions(test_predictions, questions)       
        correct_labels = evaluation_helper.get_labels(prediction_labels, questions)
        
        rmse_test = sqrt(mean_squared_error(pred_labels, correct_labels))
        graph_rmse_test.append(rmse_test)
        with open('results/results_test'+TIMESTAMP +'.txt','a') as f:
                    f.write("Epoch test RMSE is: %.2f \n" % rmse_test)
        print("RMSE for test set:%.5f" % rmse_test)
        print("--------------------------------")
        print(len(correct_labels))
        output_writer.output_predictions('results/predictions_test' +TIMESTAMP +'.txt',questions,pred_labels,correct_labels)

graph_helper.show_graph(graph_rmse_train, graph_rmse_test)
