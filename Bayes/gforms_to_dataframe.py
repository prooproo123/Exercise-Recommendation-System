import pandas as pd


def process_data(data, skill_names):
    df = pd.read_csv(data, index_col=False)

    no_skills = len(skill_names)
    no_questions = int(
        (len(list(df.columns)) - 2) / 3)
    no_students = len(df.index)
    no_questions_per_skill = int(no_questions / no_skills)
    student_list = []

    for j in range(no_skills):
        for i in range(no_students):
            student_list += [i] * int(no_questions / no_skills)

    question_id_list = list(range(1, int(
        no_questions / no_skills) + 1)) * no_students * no_skills

    correctness_list = []
    list_of_gform_columns = [column for column in list(df.columns.values) if '[rezultat]' in column]

    answers = df[list_of_gform_columns]
    for index, row in answers.iterrows():
        row.apply(lambda cell: correctness_list.append(1 if cell == '1.00 / 1' else 0))

    print(len(question_id_list), len(student_list), len(correctness_list))
    output_data = {'num': question_id_list, 'student': student_list,
                   'skill': [skill for skill in skill_names for i in range(0, no_students * no_questions_per_skill)],
                   'right': correctness_list}
    output_dataframe = pd.DataFrame(output_data)

    return output_dataframe

