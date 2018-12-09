from sklearn import metrics
from math import sqrt


def get_questions(batch_questions):
    questions = []
    for question in batch_questions:
        questions.extend(question)
    return questions


def get_predictions(predictions_series, questions):
    pred_labels = []
    pred_labels_without0 = []
    i = 0
    for predictions in predictions_series:
        num_steps = len(predictions)
        j = 0
        for prediction in predictions:
            pred_labels.append(prediction[questions[i*num_steps + j]])
            j += 1
        i += 1
    for i in range(len(pred_labels)):
        # CUTTING PADDING
        if questions[i] != 0:
            pred_labels_without0.append(pred_labels[i])

    return pred_labels_without0    


def get_labels(labels, questions):
    correct_labels = []
    correct_labels_without0 = []
    for label in labels:
        correct_labels.extend(label)
    for i in range(len(correct_labels)):
        # CUTTING PADDING
        if questions[i] != 0:
            correct_labels_without0.append(correct_labels[i])  
    return correct_labels_without0


def rmse(correct, predictions):
    return sqrt(metrics.mean_squared_error(correct, predictions))


def auc(correct, predictions):
    return metrics.roc_auc_score(correct, predictions)


def pearson(correct, predictions):
    return metrics.r2_score(correct, predictions)


def accuracy(correct, predictions):
    correct_count = 0
    prediction_over = 0
    correct_1 = 0
    correct_0 = 0
    for i in range(0, len(correct)):
        if correct[i] == 0:
            correct_0 += 1
        else:
            correct_1 += 1
        if predictions[i] > 0.5:
            prediction_over += 1
        if correct[i] == 1 and predictions[i] > 0.5:
            correct_count += 1
        elif correct[i] == 0 and predictions[i] < 0.5:
            correct_count += 1

    print("Correct 0  ", correct_0)
    print("Correct 1  ", correct_1)
    print("Predictions > 0.5  ", prediction_over)

    return correct_count/len(correct)


def print_results(rmse, auc, pearson, accuracy):
    print("RMSE for test set:%.5f" % rmse)
    print("AUC for test set:%.5f" % auc)
    print("Pearson coef is:", pearson)
    print("Accuracy is:", accuracy)
    print("####################################################")

