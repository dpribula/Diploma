#
import csv

def output_for_map(path, questions, answers, seq_len, predictions,number):
    map_data = csv.writer(open(path + 'questions' + str(number) +'.csv', "w"), delimiter=';')
    map_data.writerow(questions[0])
    map_data.writerow(answers[0])
    map_data2 = csv.writer(open(path + 'probabilities' + str(number) + '.csv', "w"), delimiter=';')

    map_data2.writerow(["id", "prediction"])
    length_of_input_array = seq_len[0] - 1
    for i in range(225):
        temp = []
        temp.append(i)
        temp.append((predictions[0][length_of_input_array][i])*100)
        map_data2.writerow(temp)

