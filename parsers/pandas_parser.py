import pandas as pd
import csv
from sklearn.utils import shuffle

FLAGS = None
PARSE_OPTIONS = False

def main():

    features = pd.read_csv('../data/answer.csv', sep=';')
    # features = pd.read_csv('../data/answer.csv', sep=';')

    # Parsing options
    if PARSE_OPTIONS:
        features['options'] = features['options'].apply(lambda x: len(x[1:-1].split(",")))
        features['options'] = features['options'].apply(lambda x: 0 if int(x) == 1 else 1 / int(x))

    user_count = -1
    answer_count = 0
    users = {}
    max_question = 0
    bigger = 0
    for index, row in features.iterrows():

        if user_count == -1:
            user_count += 1
            continue
        else:
            if int(row['place_asked']) > 224 or int(row['place_asked']) < 51:
                continue
        if row['user'] not in users:
            users[row['user']] = [[], [], []]

            add_row(row, users)
            user_count += 1
        elif len(users[row['user']][0]) < 500:
            add_row(row, users)


        max_question = max(max_question, int(row['place_asked']))
        answer_count += 1

    # print(users)
    print("Total number of users: " + str(user_count))
    print("Total number of answers: " + str(answer_count))

    writer_test = csv.writer(open('../datasets/world_train2.csv', "w")
                             , delimiter=',')
    writer_train = csv.writer(open('../datasets/world_test2.csv', "w")
                              , delimiter=',')

    train_count = 0
    for user_id, user_answers in users.items():
        user_answers[0], user_answers[1] = shuffle(user_answers[0], user_answers[1])
        if (train_count + 3) % 5 != 0:
            writer_test.writerow([len(user_answers[0])])
            writer_test.writerow(user_answers[0])
            writer_test.writerow(user_answers[1])
            writer_test.writerow(user_answers[2])
        else:
            writer_train.writerow([len(user_answers[0])])
            writer_train.writerow(user_answers[0])
            writer_train.writerow(user_answers[1])
            writer_train.writerow(user_answers[2])
        train_count += 1

    print(max_question)
    print(bigger)


def add_row(row, users):
    users[row['user']][0] += [row['place_asked']]
    users[row['user']][1] += [1 if (row['place_answered'] == row['place_asked']) else 0]
    if PARSE_OPTIONS:
        users[row['user']][2] += [row['options']]


if __name__ == '__main__':
    main()
