#
def output_results(path, step, loss, rmse):
    with open(path, 'a') as file:
        file.write("Step %.2f Loss %.2f \n" % (step, loss))
        file.write("Epoch train RMSE is: %.2f \n" % rmse)

def output_predictions(path, questions, pred_labels, correct_labels):
    #check if they have the same size
    with open(path, 'w') as file:
        for i,_ in enumerate(correct_labels):
            if questions[i] != 0:
                file.write('%d. ' % i)
                file.write('question ID is: %d' % questions[i])
                file.write("pred:%.2f " % pred_labels[i])
                file.write("correct:%s" % correct_labels[i])
                file.write("\n")
            