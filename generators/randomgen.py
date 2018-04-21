import csv
import numpy as np

PROBABILITY = 50
STUDENTS = 10000
def generate_list():
    temp_list = []
    for i in range(0,20):
        temp_list.append(np.random.randint(1,2) if np.random.randint(1,100) >= PROBABILITY else 0)
    return temp_list

def generate_random_list():
    return [1 for _ in range(0,20)]     

train_data = []
test_data = []
questions1 = generate_random_list()
questions2 = generate_random_list()
for i in range(0,STUDENTS):
    train_data.insert(i,[[],[]])
    train_data[i][0]= questions1 #generate_random_list()
    train_data[i][1] = generate_list() #generate_list()
    test_data.insert(i,[[],[]])
    test_data[i][0]= questions2#generate_random_list()
    test_data[i][1] = generate_list()
writer_test = csv.writer(open('../datasets/generated_train.txt', "w"))
writer_train = csv.writer(open('../datasets/generated_test.txt', "w"))

for i in range(0,STUDENTS):
    writer_test.writerow([len(train_data[i][0])])
    writer_test.writerow(train_data[i][0])
    writer_test.writerow(train_data[i][1])
    writer_train.writerow([len(test_data[i][0])])
    writer_train.writerow(test_data[i][0])
    writer_train.writerow(test_data[i][1])