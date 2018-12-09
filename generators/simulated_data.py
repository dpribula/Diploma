import csv
import numpy as np

STUDENT_COUNT = 500
CONCEPTS = 5
EXERCISES = 10
SEQUENCE_LENGTH= 500

DECAY = 0.01
CORRECT_CONCEPT = 0.2
CORRECT_EXERCISE = 0.2


def generate_students():
    students = []
    for i in range(STUDENT_COUNT):
        student = []
        for k in range(CONCEPTS):
            exercises = []
            concept_strength = np.random.uniform(0, 1)
            for k in range(EXERCISES):
                exercise_strength = get_value(concept_strength + np.random.uniform(-0.2, 0.2))
                exercises.append(exercise_strength)

            student.append(exercises)
        students.append(student)
    return students



def get_value(value):
    value = min(value, 1)
    value = max(value, 0)
    return value


def update_student(students, student_id, concept_id, exercise_id, correct):
    student = students[student_id]

    change = CORRECT_CONCEPT if correct else -CORRECT_CONCEPT/2
    updated_student = []

    for i, concept in enumerate(student):
        concept_new = []
        for j, exercise in enumerate(student[i]):
            if i == concept_id:
                if i == exercise_id and change > 0:
                    exercise = get_value(exercise+CORRECT_EXERCISE)
                else:
                    exercise = get_value(exercise+change)
            else:
                exercise = get_value(exercise - DECAY)
            concept_new.append(exercise)
        updated_student.append(concept)

    students[student_id] = updated_student


def get_answer(students, concept_id, question_id, student_id):
    student = students[student_id]
    concept = student[concept_id]
    knowledge = concept[question_id]

    return 1 if knowledge > np.random.uniform(0,1) else 0


def generate_question():
    return np.random.randint(0, CONCEPTS), np.random.randint(0, EXERCISES)


def generate_data():
    writer_train = csv.writer(open('../datasets/generated_train.txt', "w"))
    writer_test = csv.writer(open('../datasets/generated_test.txt', "w"))

    students = generate_students()
    questions = []
    answers = []
    for student_id in range(STUDENT_COUNT):
        question = []
        answer = []
        for _ in range(SEQUENCE_LENGTH):
            concept_id, question_id = generate_question()
            answer_temp = get_answer(students, concept_id, question_id, student_id)
            update_student(students, student_id, concept_id,question_id, answer_temp)

            question.append(concept_id*EXERCISES + question_id)
            answer.append(answer_temp)

        questions.append(question)
        answers.append(answer)

    for i in range(STUDENT_COUNT):
        if i % 5 != 0:
            writer_train.writerow([SEQUENCE_LENGTH])
            writer_train.writerow(questions[i])
            writer_train.writerow(answers[i])
        else:
            writer_test.writerow([SEQUENCE_LENGTH])
            writer_test.writerow(questions[i])
            writer_test.writerow(answers[i])

generate_data()




