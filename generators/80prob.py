import csv
import random
import numpy as np

PROBABILITY = 80
def generate_list():
    return [((np.random.randint(1,100) // PROBABILITY) +1)%2 for _ in range(0,20)]
        

train_data = []
test_data = [[],[]]
question_list = [1+i for i in range(0,20)]
for i in range(0,1000):
    train_data.insert(i,[[],[]])
    train_data[i][0]= question_list
    train_data[i][1] = generate_list()

writer_test = csv.writer(open('./datasets/generated_train.txt', "w"))
writer_train = csv.writer(open('./datasets/generated_test.txt', "w"))

for i in range(0,1000):
    writer_test.writerow([len(train_data[i][0])])
    writer_test.writerow(train_data[i][0])
    writer_test.writerow(train_data[i][1])
    writer_train.writerow([len(train_data[i][0])])
    writer_train.writerow(train_data[i][0])
    writer_train.writerow(train_data[i][1])