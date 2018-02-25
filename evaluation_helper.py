num_steps = 19

def get_questions(batch_questions):
    questions = []
    for question in batch_questions:
        questions.extend(question)
    return questions

def get_predictions(_predictions_series,questions):
    pred_labels = []
    pred_labels_without0 = []
    i = 0
    for predictions in _predictions_series: 
        j = 0
        for prediction in predictions:
            pred_labels.append(prediction[questions[i*num_steps + j]]) 
            j+=1
        i+=1
    for i in range(len(pred_labels)):
        #CUTTING PADDING
        if questions[i] != 0:
            pred_labels_without0.append(pred_labels[i])

    return pred_labels_without0    

def get_labels(labels, questions):
    correct_labels = []
    correct_labels_without0 = []
    for label in labels:
        correct_labels.extend(label)
    for i in range(len(correct_labels)):
        #CUTTING PADDING
        if questions[i] != 0:
            correct_labels_without0.append(correct_labels[i])  
    return correct_labels_without0