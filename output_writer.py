#
import csv


def output_results(path, step, loss, rmse, auc):
    with open(path, 'a') as file:
        file.write("Step %.2f Loss %.2f \n" % (step, loss))
        file.write("Epoch train RMSE is: %.2f \n" % rmse)
        file.write("Epoch train AUC is %.2f \n" % auc)
        file.write("----------------------------------------------------------------------------------------------------------------- \n")


def output_predictions(path, questions, pred_labels, correct_labels):
    return
    #check if they have the same size
    # with open(path, 'w') as file:
    #     for i, _ in enumerate(correct_labels):
    #         if questions[i] != 0:
    #             file.write('%d. ' % i)
    #             file.write('%d. ' % i)
    #             file.write('question ID is: %d' % questions[i])
    #             file.write("pred:%.2f " % pred_labels[i])
    #             file.write("correct:%s" % correct_labels[i])
    #             file.write("\n")


def output_visualization(path, questions, answers, seq_len, predictions, batch_count):
    with open(path + str(batch_count), 'w') as file:
        count = 0
        for question in questions:
            for i in range(0, seq_len[count]):
                file.write('%6d, ' % question[i])
            file.write('\n')
            for i in range(0, seq_len[count]):
                file.write('%6d, ' % answers[count][i])
            file.write('\n')
            for i in range(0, seq_len[count]):
                file.write('%5f, ' % predictions[count][i][question[i]])
            file.write('\n\n')
            count += 1



def output_for_map(path, questions, answers, seq_len, predictions,):
    map_data = csv.writer(open(path + 'something.csv', "w"), delimiter=';')
    map_data.writerow(questions[0])
    map_data.writerow(answers[0])
    map_data2 = csv.writer(open(path + 'something2.csv', "w"), delimiter=';')

    map_data2.writerow(["id", "prediction"])
    #TODO make prediction for the length of arrayS
    length_of_input_array = seq_len[0] - 1
    for i in range(0, 225):
        temp = []
        temp.append(i)
        temp.append((predictions[0][length_of_input_array][i])*100)
        map_data2.writerow(temp)

