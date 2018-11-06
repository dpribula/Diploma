import csv
train_data = []
test_data = [[],[]]
length = 0
question_list = [1 for i in range(0, length+10)]
answer_list = [i % 2 for i in range(0, length+10)]
student_count = 1000
for i in range(0, student_count):
    train_data.insert(i,[[],[]])
    train_data[i][0] = question_list
    train_data[i][1] = answer_list

writer_test = csv.writer(open('../datasets/generated_train.txt', "w"))
writer_train = csv.writer(open('../datasets/generated_test.txt', "w"))

for i in range(0, student_count):
    writer_test.writerow([len(train_data[i][0])])
    writer_test.writerow(train_data[i][0])
    writer_test.writerow(train_data[i][1])
    writer_train.writerow([len(train_data[i][0])])
    writer_train.writerow(train_data[i][0])
    writer_train.writerow(train_data[i][1])

    
    
