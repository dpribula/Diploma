import csv
import numpy as np

PROBABILITY = 50
STUDENTS = 1000


def generate_list():
    answer_list = []
    question_list = []
    x = np.random.randint(1, 10) + 10
    for i in range(0, x):
        answer_list.append(i % 4 / 4)
        question_list.append(i)
    return question_list, answer_list


train_data = []
test_data = []

for i in range(0, STUDENTS):
    train_data.insert(i, [[], []])
    train_data[i][0], train_data[i][1] = generate_list()
    test_data.insert(i, [[], []])
    test_data[i][0], test_data[i][1] = generate_list()

writer_train = csv.writer(open('../datasets/generated_train.txt', "w"))
writer_test = csv.writer(open('../datasets/generated_test.txt', "w"))

for i in range(0, STUDENTS):
    writer_train.writerow([len(train_data[i][0])])
    writer_train.writerow(train_data[i][0])
    writer_train.writerow(train_data[i][1])
    writer_test.writerow([len(test_data[i][0])])
    writer_test.writerow(test_data[i][0])
    writer_test.writerow(test_data[i][1])