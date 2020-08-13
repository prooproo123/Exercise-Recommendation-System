from __future__ import division

import logging
import math
# import os
import pickle

import numpy as np
import random
import pandas as pd

# from collections import defaultdict

logging.getLogger().setLevel(logging.DEBUG)

history_path = '/home/zvonimir/PycharmProjects/lentil/data/skill_builder_data.csv'
# history_path = '/home/zvonimir/PycharmProjects/lentil/data/skill_builder_pickle.pkl'
# history_path = '/home/zvonimir/PycharmProjects/lentil/data/skill_builder_model.pkl'

df = pd.read_csv(history_path,
                 dtype={'order_id': int, 'assignment_id': int, 'user_id': int, 'assistment_id': int, 'problem_id': int,
                        'original': int, 'correct': int, 'attempt_count': int, 'ms_first_response': int,
                        'tutor_mode': 'string', 'answer_type': 'string', 'sequence_id': int, 'student_class_id': int,
                        'position': int, 'type': 'string', 'base_sequence_id': int, 'skill_id': float,
                        'skill_name': 'string',
                        'teacher_id': int, 'school_id': int, 'hint_count': int, 'hint_total': int, 'overlap_time': int,
                        'template_id': int, 'answer_id': int, 'answer_text': 'string', 'first_action': int,
                        'bottom_hint': int, 'opportunity': int, 'opportunity_original': int
                        },
                 usecols=['order_id', 'assignment_id', 'user_id', 'assistment_id', 'problem_id', 'original', 'correct',
                          'attempt_count', 'ms_first_response', 'tutor_mode', 'answer_type', 'sequence_id',
                          'student_class_id', 'position', 'type', 'base_sequence_id', 'skill_id', 'skill_name',
                          'teacher_id', 'school_id', 'hint_count', 'hint_total', 'overlap_time', 'template_id',
                          'first_action', 'opportunity', ])

# unfiltered_history = interaction_history_from_assistments_data_set(
#     df,
#     module_id_column='problem_id',
#     duration_column='ms_first_response')
#
# REPEATED_FILTER = 3  # number of times to repeat filtering
# history = reduce(
#     lambda acc, _: filter_history(acc, min_num_ixns=75, max_num_ixns=1000),
#     range(REPEATED_FILTER), unfiltered_history)
#
#
# history_path = '/home/zvonimir/PycharmProjects/lentil/data/skill_builder_pickle.pkl'
#
# # serialize history
# with open(history_path, 'wb') as f:
#     pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

# load history from file
history_path = '/home/zvonimir/PycharmProjects/lentil/data/skill_builder_pickle.pkl'
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# history = datatools.InteractionHistory(filtered_history)
df = history.data
duration = history.duration()

num_students = history.num_students()
num_left_out_students = int(0.3 * num_students)
left_out_user_ids = {history.id_of_user_idx(
    user_idx) for user_idx in random.sample(
    range(num_students), num_left_out_students)}

left_out_ixns = df['user_id'].isin(left_out_user_ids)
left_in_ixns = ~left_out_ixns
left_in_modules = set(df[left_in_ixns]['module_id'].unique())

print
"Number of unique students = %d" % (len(df['user_id'].unique()))
print
"Number of unique modules = %d" % (len(df['module_id'].unique()))

# these constraints speed up bubble collection
MIN_BUBBLE_LENGTH = 2  # number of interactions
MAX_BUBBLE_LENGTH = 20  # number of interactions

grouped = history.data.groupby('user_id')

# dict[(str, str), dict[tuple, int]]
# (start lesson id, final assessment id) ->
# dict[(lesson id, lesson id, ...) ->
# number of students who took this path]
bubble_paths = {}

# dict[(str, str, (str, str, ...)), list[str]]
# (start lesson id, final assessment id, lesson sequence) ->
# [ids of students who took this path]
bubble_students = defaultdict(list)

# dict[(str, str, (str, str, ...)), set[str]]
# (start lesson id, final assessment id, lesson sequence) ->
# [outcomes for students who took this path]
bubble_outcomes = defaultdict(list)

for user_id in left_out_user_ids:
    group = grouped.get_group(user_id)
    module_ids = list(group['module_id'])
    module_types = list(group['module_type'])
    outcomes = list(group['outcome'])
    for i, start_lesson_id in enumerate(module_ids):
        if module_types[i] != datatools.LessonInteraction.MODULETYPE:
            continue
        if start_lesson_id not in left_in_modules:
            continue
        for j, (final_assessment_id, module_type, outcome) in enumerate(
                zip(module_ids[(i + MIN_BUBBLE_LENGTH):(i + MAX_BUBBLE_LENGTH)],
                    module_types[(i + MIN_BUBBLE_LENGTH):(i + MAX_BUBBLE_LENGTH)],
                    outcomes[(i + MIN_BUBBLE_LENGTH):(i + MAX_BUBBLE_LENGTH)])):
            if final_assessment_id not in left_in_modules:
                break
            if module_type == datatools.AssessmentInteraction.MODULETYPE:
                lesson_seq = [x for x, y in zip(module_ids[i:(i + MIN_BUBBLE_LENGTH + j)],
                                                module_types[i:(i + MIN_BUBBLE_LENGTH + j)]) if
                              y == datatools.LessonInteraction.MODULETYPE]
                path = tuple(lesson_seq)
                if any(m not in left_in_modules for m in path):
                    break

                try:
                    bubble_paths[(start_lesson_id, final_assessment_id)][path] += 1
                except KeyError:
                    bubble_paths[(start_lesson_id, final_assessment_id)] = defaultdict(int)
                    bubble_paths[(start_lesson_id, final_assessment_id)][path] += 1

                bubble_students[(start_lesson_id, final_assessment_id, path)].append(user_id)
                bubble_outcomes[(start_lesson_id, final_assessment_id, path)].append(1 if outcome else 0)

MIN_NUM_STUDENTS_ON_PATH = 1

# dict[(str, str, (str, str, ...), (str, str, ...)]
# (start lesson id, final assessment id, path, other_path) ->
# ([ids of students who took path], [ids of students who took other_path])
my_bubble_students = {}

# dict[(str, str, (str, str, ...), (str, str, ...)]
# (start lesson id, final assessment id, path, other_path) ->
# ([outcomes for students who took path], [outcomes for students who took other_path])
#
my_bubble_outcomes = {}

for (start_lesson_id, final_assessment_id), d in bubble_paths.iteritems():
    paths = [path for path, num_students_on_path in d.iteritems() if num_students_on_path >= MIN_NUM_STUDENTS_ON_PATH]
    for i, path in enumerate(paths):
        for other_path in paths[(i + 1):]:
            #    if len(path) != len(other_path):
            # paths must have the same number of lesson interactions
            # in order to be part of a bubble
            #        continue
            my_bubble_students[(start_lesson_id, final_assessment_id, path, other_path)] = (
                bubble_students[(start_lesson_id, final_assessment_id, path)],
                bubble_students[(start_lesson_id, final_assessment_id, other_path)])
            my_bubble_outcomes[(start_lesson_id, final_assessment_id, path, other_path)] = (
                bubble_outcomes[(start_lesson_id, final_assessment_id, path)],
                bubble_outcomes[(start_lesson_id, final_assessment_id, other_path)])

a = 0
