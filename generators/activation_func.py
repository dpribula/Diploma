import csv
import numpy as np

PROBABILITY = 90
STUDENTS = 1000
LENGTH = 10

def generate_list(length):
    activated = False
    temp_list = []
    for i in range(0,length):
        if np.random.randint(1, 100) >= PROBABILITY:
            activated = True
        temp_list.append(1 if activated else 0)
    return temp_list


question_list = [i for i in range(0, 10)]

train_data = []
test_data = []
for i in range(0,STUDENTS):
    train_data.insert(i,[[],[]])
    train_data[i][0]= question_list
    train_data[i][1] = generate_list(LENGTH)
    test_data.insert(i,[[],[]])
    test_data[i][0]= question_list
    test_data[i][1] = generate_list(LENGTH)
writer_test = csv.writer(open('../datasets/generated_train.txt', "w"))
writer_train = csv.writer(open('../datasets/generated_test.txt', "w"))

for i in range(0,STUDENTS):
    writer_train.writerow([len(train_data[i][0])])
    writer_train.writerow(train_data[i][0])
    writer_train.writerow(train_data[i][1])
    writer_test.writerow([len(test_data[i][0])])
    writer_test.writerow(test_data[i][0])
    writer_test.writerow(test_data[i][1])