import csv
import numpy as np

PROBABILITY = 80
STUDENTS = 1000
LENGTH = 10


def generate_list():
    return [((np.random.randint(1, 100) // PROBABILITY) + 1) % 2 for _ in range(0, LENGTH)]
        

train_data = []
test_data = [[],[]]
question_list = [x for x in range(LENGTH)]

for i in range(0, STUDENTS):
    train_data.insert(i, [[], []])
    train_data[i][0]= question_list
    train_data[i][1] = generate_list()

writer_train = csv.writer(open('.././datasets/generated_train.txt', "w"))
writer_test = csv.writer(open('.././datasets/generated_test.txt', "w"))

for i in range(0,STUDENTS):
    writer_train.writerow([len(train_data[i][0])])
    writer_train.writerow(train_data[i][0])
    writer_train.writerow(train_data[i][1])
    writer_test.writerow([len(train_data[i][0])])
    writer_test.writerow(train_data[i][0])
    writer_test.writerow(train_data[i][1])