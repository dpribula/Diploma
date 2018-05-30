import csv
import argparse
import sys
FLAGS = None
TRAIN_LIMIT = 8000000
ANSWER_LIMIT = 500000


def main():
    testReader = csv.DictReader(
        open('../datasets/answer.csv'), delimiter=';', fieldnames=['id', 'user', 'place_asked', 'place_answered', 'type'])
    user_count = -1
    answer_count = 0
    users = []
    users_id = set()
    for row in testReader:
        if user_count == -1 :
            user_count += 1
            continue
        users_id.add(row['user'])
        correct_answer = 1 if row['place_answered'] == row['place_asked'] else 0

        users.append( [row['user'],row['place_asked'], correct_answer])
        answer_count += 1
        if(answer_count > ANSWER_LIMIT):
            break

    # print(users)
    print("Total number of users: " + str(user_count))
    print("Total number of answers: " + str(answer_count))

    writer_test = open('../datasets/builder_test.txt', "w")
    writer_train = csv.writer(
        open('../datasets/builder_train.txt', "w"), delimiter=' ')
    train_count = 0
    for user in users:
        writer_train.writerow(user)
        train_count+=1
    for i in range(0,len(users_id)):
        if(i < len(users_id) * 0.8):
            writer_test.write('1 ')
        else:
            writer_test.write('0 ')


if __name__ == '__main__':
    main()
