
STUDENTS_COUNT = 100
max_seq_len = 0
#
# Parsing dataset into data, labels and their lengths
#
def read_dataset(path):
    data = []
    labels = []
    seq_len = []
    max_seq_len = 0
    file = open(path,'r')
    count = 0
    for line in file:
        line_data = line.split(",")
        max_seq_len = max(max_seq_len, len(line_data))
        if (count % 3) == 0:
            count += 1
            continue
        elif (count % 3) == 1:
            data.append(list(map(lambda x:int(x), line_data)))
            seq_len.append(len(line_data)-1)
            count += 1
        elif (count % 3) == 2:
            labels.append(list(map(lambda x:int(x), line_data)))
            count += 1
            
        if count >= STUDENTS_COUNT*3:
            break
    add_padding(data, max_seq_len)
    add_padding(labels,max_seq_len)

    return data, labels, seq_len, max_seq_len -1

#
# Adding zeros to max length to data 
# We need to have matrices of the same size
#
def add_padding(data, length):
    for entry in data:
        while(len(entry)<length):
            entry.append(int(0))


class SlepeMapyData(object):
    def __init__(self,path):
        self.data, self.labels, self.seqlen, self.max_seq_len = read_dataset(path)
        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        questions = (self.data[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        answers = (self.labels[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                batch_size, len(self.data))])
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
            temp = []
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return questions, answers, questions_target, answers_target, batch_seqlen