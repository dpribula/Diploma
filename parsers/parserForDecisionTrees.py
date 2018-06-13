import csv
from sklearn.utils import shuffle
import datetime

FLAGS = None
USERS_LIMIT = 1000
TRAIN_LIMIT = USERS_LIMIT * 0.8


def main():
    testReader = csv.DictReader(
        open('../data/matej_answer_europe.csv'), delimiter=';',
        fieldnames=['id', 'user', 'place_asked', 'place_answered', 'type', 'inserted', 'response_time', 'place_map', 'language', 'options'])
    user_count = -1
    answer_count = 0
    users = {}
    max_question = 0
    bigger = 0
    for row in testReader:

        if user_count == -1:
            user_count += 1
            continue
        else:
            if int(row['place_asked']) > 224 or int(row['place_asked']) < 51:
                continue
        if row['user'] not in users:

            users[row['user']] = [[], [], [], [], [], []]

            users[row['user']][0] = [row['place_asked']]
            users[row['user']][1] = [1 if (row['place_answered'] == row['place_asked']) else 0]
            users[row['user']][2] = [row['type']]
            users[row['user']][3] = parse_time([row['inserted']])
            users[row['user']][4] = [row['response_time']]
            users[row['user']][5] = parse_options([row['options']])
            user_count += 1
        elif len(users[row['user']][0]) < 100:
            users[row['user']][0] += [row['place_asked']]
            users[row['user']][1] += [1 if (row['place_answered'] == row['place_asked']) else 0]
            users[row['user']][2] += [row['type']]
            users[row['user']][3] += parse_time([row['inserted']])
            users[row['user']][4] += [row['response_time']]
            users[row['user']][5] += parse_options([row['options']])
        max_question = max(max_question, int(row['place_asked']))

        answer_count += 1
        if (user_count > USERS_LIMIT):
            break

    # print(users)
    print("Total number of users: " + str(user_count))
    print("Total number of answers: " + str(answer_count))

    writer_test = csv.writer(open('../datasets/trees_train.csv', "w")
                             , delimiter=',')
    writer_train = csv.writer(open('../datasets/trees_test.csv', "w")
                              , delimiter=',')

    train_count = 0
    for user_id, user_answers in users.items():
        #user_answers[0], user_answers[1], user_answers[2],user_answers[3], user_answers[4], user_answers[5] = shuffle( user_answers[0], user_answers[1], user_answers[2],user_answers[3], user_answers[4], user_answers[5])
        if train_count % 5 != 0:
            write_data(writer_test, user_answers)
        else:
            write_data(writer_train, user_answers)
        train_count += 1

    print(max_question)
    print(bigger)


def write_data(writer, data):
    writer.writerow([len(data[0])])
    for i in range(0, len(data)):
        writer.writerow(data[i])


def parse_options(options):
    parsed_options = options[0]
    return [len(parsed_options[1:-1].split(","))]


def parse_time(time):
    return time

if __name__ == '__main__':
    main()
