import pickle

import numpy as np
from gym import spaces

# from new_rs import self.num_concepts, self.num_questions, self.candidate_exercises
from rllab.envs.gym_env import *


class MyGymEnv(GymEnv):
    def __init__(self, env, record_video=False, video_schedule=None, log_dir=None, record_log=False,
                 force_reset=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        self.env = env
        self.env_id = ''

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = self.env.n_steps
        self._log_dir = log_dir
        self._force_reset = force_reset


class StudentEnv(gym.Env):

    def __init__(self, n_items=10, n_steps=100, discount=1., reward_func='likelihood', num_questions=0, num_concepts=0,
                 candidate_exercises=None):

        if candidate_exercises is None:
            candidate_exercises = []

        self.right = []

        self.curr_step = None
        self.n_steps = n_steps
        self.n_items = n_items
        self.now = 0
        self.curr_item = 0
        self.curr_outcome = None
        self.curr_delay = None
        self.discount = discount
        self.reward_func = reward_func
        self.action_space = spaces.Discrete(n_items)
        self.observation_space = spaces.Box(np.zeros(2), np.array([n_items - 1, 1]))

        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.candidate_exercises = candidate_exercises

    # def tempinit(self,num_questions,num_concepts,candidate_exercises):
    #     self.num_questions=num_questions
    #     self.num_concepts=num_concepts
    #     self.candidate_exercises=candidate_exercises

    def _recall_likelihoods(self):
        raise NotImplementedError

    def _recall_log_likelihoods(self, eps=1e-9):
        return np.log(eps + self._recall_likelihoods())

    def predict(self, q):
        raise NotImplementedError

    def _update_model(self, curr_item, curr_outcome):
        raise NotImplementedError

    def _obs(self):
        return np.array([self.curr_item, self.curr_outcome], dtype=int)

    def _rew(self):
        if self.reward_func == 'likelihood':
            return self._recall_likelihoods().mean()
        elif self.reward_func == 'log_likelihood':
            return self._recall_log_likelihoods().mean()
        else:
            raise ValueError

    def step(self, action: int):
        if self.curr_step is None or self.curr_step >= self.n_steps:
            raise ValueError

        if action < 0 or action >= self.n_items:
            raise ValueError

        # student model do the exercise and update model
        self.curr_item = action
        self.curr_outcome = 1 if np.random.random() < self.predict(self.candidate_exercises[action]) else 0

        self._update_model(self.candidate_exercises[self.curr_item], self.curr_outcome)
        self.curr_step += 1

        # if the exercise which student used to answer correctly, the reward is 0
        if self.curr_item in self.right:
            r = 0
        else:
            r = self._rew()
        if self.curr_outcome == 1 and self.curr_item not in self.right:
            self.right.append(action)

        obs = self._obs()
        done = self.curr_step == self.n_steps
        info = {}

        return obs, r, done, info

    def actualStep(self, action, answer):
        self.curr_item = action
        self.curr_outcome = answer
        self._update_model(self.candidate_exercises[self.curr_item], self.curr_outcome)
        obs = self._obs()
        return obs

    def reset(self):
        self.curr_step = 0
        self.now = 0
        return self.step(np.random.choice(range(self.n_items)))[0]

    def recomreset(self):
        self.curr_step = 0
        self.now = 0


class DKVEnv(StudentEnv):
    def __init__(self, **kwargs):

        super(DKVEnv, self).__init__(**kwargs)

        # self._init_params(params)

    def init_params(self):
        """
        Init DKVMN-CA student model
        """

        # the parameters of trained DKVMN-CA model
        # with open('../data/skill_builder/kt_params.pkl', 'rb') as f:
        #     params = pickle.load(f)

        with open('old/checkpoint/skill_builder0_10batch_2epochs/kt_params', 'rb') as f:
            params = pickle.load(f)

        # Knowledge self.num_concepts Corresponding to the exercise
        # Knowledge Concepts Corresponding to the exercise
        with open('data/skill_builder/chunk_exercise_concepts_mapping.pkl', 'rb') as f:
            self.e2c = pickle.load(f)

        # contains the exercise which has already been answered correctly
        self.right = []

        self.q_embed_mtx = params['Embedding/q_embed:0']

        self.qa_embed_mtx = params['Embedding/qa_embed:0']

        self.key_matrix = params['Memory/key:0']

        self.value_matrix = params['Memory/value:0']

        self.summary_w = params['Summary_Vector/weight:0']

        self.summary_b = params['Summary_Vector/bias:0']

        self.predict_w = params['Prediction/weight:0']

        self.predict_b = params['Prediction/bias:0']

        self.erase_w = params['DKVMN_value_matrix/Erase_Vector/weight:0']

        self.erase_b = params['DKVMN_value_matrix/Erase_Vector/bias:0']

        self.add_w = params['DKVMN_value_matrix/Add_Vector/weight:0']

        self.add_b = params['DKVMN_value_matrix/Add_Vector/bias:0']

    def softmax(self, num):
        return np.exp(num) / np.sum(np.exp(num), axis=0)

    def cor_weight(self, embedded, q):
        """
            Calculate the KCW of the exercise
        Args:
            embedded: the embedding of exercise q
            q: exercise ID

        Returns:
            the KCW of the exercise
        """
        concepts = self.e2c[q]
        corr = self.softmax([np.dot(embedded, self.key_matrix[i]) for i in concepts])
        correlation = np.zeros(self.num_concepts)
        for j in range(len(concepts)):
            correlation[concepts[j]] = corr[j]
        return correlation

    def read(self, value_matrix, correlation_weight):
        """
            Calculate master level of concepts related to the exercise
        Args:
            value_matrix: master level of different knowledge concepts
            correlation_weight: KCW of exercise

        Returns:
            master level of concepts related to the exercise
        """
        read_content = []
        for dim in range(value_matrix.shape[0]):
            r = np.multiply(correlation_weight[dim], value_matrix[dim])
            read_content.append(r)
        read_content = np.sum(np.array(read_content), axis=0)
        return read_content

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def linear_op(self, w, b, x):
        return np.matmul(x, w) + b

    def predict(self, q):
        """
            Probability of answer exercise q correctly
        Args:
            q: Exercise ID

        Returns:
            Probability
        """
        cor = self.cor_weight(self.q_embed_mtx[q], q)
        read_content = self.read(self.value_matrix, cor)
        mastery_level_prior_difficulty = np.append(read_content, self.q_embed_mtx[q])
        summary_vector = np.tanh(
            self.linear_op(self.summary_w, self.summary_b, mastery_level_prior_difficulty)
        )
        pred_logits = self.linear_op(self.predict_w, self.predict_b, summary_vector)
        return self.sigmoid(pred_logits)

    def write(self, correlation_weight, qa_embed):
        """
            Update the Value_matrix
        Args:
            correlation_weight: KCW of exercise
            qa_embed: the embedding of answering result

        Returns:
            new Value_matrix
        """
        erase_vector = self.linear_op(self.erase_w, self.erase_b, qa_embed)
        erase_signal = self.sigmoid(erase_vector)
        add_vector = self.linear_op(self.add_w, self.add_b, qa_embed)
        add_signal = np.tanh(add_vector)
        for dim in range(self.value_matrix.shape[0]):
            self.value_matrix[dim] = self.value_matrix[dim] * (1 - correlation_weight[dim] * erase_signal) + \
                                     correlation_weight[
                                         dim] * add_signal
        return self.value_matrix

    def _recall_likelihoods(self):
        """
        The average probability of doing all the test exercises correctly
        """
        return np.array(list(map(self.predict, self.candidate_exercises)))

    def _update_model(self, item, outcome):
        """
        Update student model
        :param item: action(recommended exercise)
        :param outcome: answer result
        """
        ans = self.num_questions * outcome + item
        cor = self.cor_weight(self.q_embed_mtx[item], item)
        self.value_matrix = self.write(cor, self.qa_embed_mtx[ans])

    def reset(self):
        """
        Reset for training agent
        :return:
        """
        self.init_params()
        return super(DKVEnv, self).reset()

    def recomreset(self):
        """
        Reset for recommendation(for example, agent recommend to a new student)
        :return:
        """
        self.init_params()
        return super(DKVEnv, self).recomreset()
