from sklearn.utils import shuffle
#
# Parsing dataset into data, labels and their lengths
#
def read_dataset(path, max_count, batch_size, test_set):
    questions = []
    labels = []
    seq_len = []
    max_length = 0
    num_questions = 0
    file = open(path,'r')
    count = 0
    for line in file:
        line_data = line.split(",")
        max_length = max(max_length, len(line_data))
        if (count % 3) == 0:
            count += 1
            continue
        elif (count % 3) == 1:
            num_questions = max(num_questions, max(map(int, line_data)))
            questions.append(list(map(int, line_data)))
            seq_len.append(len(line_data)-1)
            count += 1
        elif (count % 3) == 2:
            labels.append(list(map(int, line_data)))
            count += 1
            
        if count >= max_count*3:
            break
        if count*5 >= max_count*3 and test_set:
            break
    questions, labels, seq_len = shuffle(questions, labels, seq_len)
    print("Max dataset sequence: ", max_length)
    print("Max question id: ", num_questions)
    print("Dataset count: ", len(questions))
    print("Dataset count: ", len(labels))
    print("Dataset count: ", len(seq_len))
    return questions, labels, seq_len, max_length - 1, num_questions


def get_global_variables_from_dataset():
    return 0


#
# Adding zeros to max length to data 
# We need to have matrices of the same size
#
def add_padding(data, length):
    for entry in data:
        while len(entry) < length:
            entry.append(int(0))


#
# Adding zeros to max length to data
# We need to have matrices of the same size
#
def add_padding_array(data, length):
    while len(data) < length:
        data.append(int(0))
    return data


def get_data_for_map(questions, batch_size, padding_size):
    questions = questions
    labels = []
    seq_len = []

    questions.append(1)
    labels.append(53)
    seq_len.append(1)
    return questions, labels, seq_len, 10 - 1,


class SlepeMapyData(object):
    def __init__(self,path, max_count, batch_size, test_set):
        self.questions, self.labels, self.seqlen, self.max_seq_len , self.num_questions = read_dataset(path, max_count, batch_size, test_set)
        self.batch_id = 0
    def next(self, batch_size, padding_size):
        if self.batch_id == len(self.questions):
            self.batch_id = 0
        questions = (self.questions[self.batch_id:self.batch_id + batch_size])
        answers = (self.labels[self.batch_id:self.batch_id + batch_size])
        batch_seq_len = (self.seqlen[self.batch_id:self.batch_id + batch_size])
        #TODO refactor
        questions_target = questions.copy()
        answers_target = answers.copy()
        for i in range(batch_size):
            temp = questions[i] 
            temp2 = answers[i] 
            questions[i] = temp[:-1]
            questions_target[i] = temp[1:]
            answers[i] = temp2[:-1]
            answers_target[i] = temp2[1:]
        self.batch_id = min(self.batch_id + batch_size, len(self.questions))
        add_padding(questions, padding_size)
        add_padding(answers, padding_size)
        add_padding(questions_target, padding_size)
        add_padding(answers_target, padding_size)
        return questions, answers, questions_target, answers_target, batch_seq_len
