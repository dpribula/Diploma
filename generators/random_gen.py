import csv
import numpy as np

PROBABILITY = 50
STUDENTS = 1000
SEQUENCE_LENGTH = 10


def generate_random_list():
    random_list = []
    for i in range(0, SEQUENCE_LENGTH):
        random_list.append(1 if np.random.randint(1,100) >= PROBABILITY else 0)
    return random_list


questions = [i for i in range(0, SEQUENCE_LENGTH)]

train_data = []
test_data = []

for i in range(0,STUDENTS):
    train_data.insert(i,[[],[]])
    train_data[i][0]= questions
    train_data[i][1] = generate_random_list()
    test_data.insert(i,[[],[]])
    test_data[i][0]= questions
    test_data[i][1] = generate_random_list()

writer_train = csv.writer(open('../datasets/generated_train.txt', "w"))
writer_test = csv.writer(open('../datasets/generated_test.txt', "w"))

for i in range(0, STUDENTS):
    writer_train.writerow([len(train_data[i][0])])
    writer_train.writerow(train_data[i][0])
    writer_train.writerow(train_data[i][1])

    writer_test.writerow([len(test_data[i][0])])
    writer_test.writerow(test_data[i][0])
    writer_test.writerow(test_data[i][1])