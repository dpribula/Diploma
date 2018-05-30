import csv
from sklearn.utils import shuffle

FLAGS = None
TRAIN_LIMIT = 8000000
USERS_LIMIT = 1000000000

def main():
    testReader = csv.DictReader(
        open('../data/matej_answer_europe.csv'), delimiter=';',fieldnames=['id', 'user', 'place_asked', 'place_answered','type'])
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
            
            users[row['user']] = [[],[]]
            
            users[row['user']][0] = [row['place_asked']]
            users[row['user']][1] = [1 if (row['place_answered'] == row['place_asked']) else 0]
            user_count += 1
        elif len(users[row['user']][0]) < 100:
            users[row['user']][0] += [row['place_asked']]
            users[row['user']][1] += [1 if (row['place_answered'] == row['place_asked']) else 0]
        max_question = max(max_question, int(row['place_asked']))
        
        answer_count += 1
        if(user_count > USERS_LIMIT):
            break
        
    #print(users)
    print("Total number of users: " + str(user_count))
    print("Total number of answers: " + str(answer_count))
    
    writer_test = csv.writer(open('../datasets/world_train.csv', "w")
    , delimiter=',')
    writer_train = csv.writer(open('../datasets/world_test.csv', "w")
    , delimiter=',')

    train_count = 0
    for user_id, user_answers in users.items():
        user_answers[0], user_answers[1] = shuffle(user_answers[0], user_answers[1])
        if (train_count + 3) % 5 != 0:
            writer_test.writerow([len(user_answers[0])])
            writer_test.writerow(user_answers[0])
            writer_test.writerow(user_answers[1])
        else:
            writer_train.writerow([len(user_answers[0])])
            writer_train.writerow(user_answers[0])
            writer_train.writerow(user_answers[1])
        train_count += 1

    print(max_question)
    print(bigger)


if __name__ == '__main__':
    main()
