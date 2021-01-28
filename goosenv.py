from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
from random import choice

import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import *

tf.compat.v1.enable_v2_behavior()


def get_board(env):
    config = env.configuration
    columns = config.columns
    rows = config.rows

    numeric_board = np.full([columns * rows], 10, dtype=int)

    food_number = 5

    for pos in env.state[0].observation.food:
        numeric_board[pos] = food_number

    for index, goose in enumerate(env.state[0].observation.geese):
        for position in goose:
            numeric_board[position] = index

    numeric_board = numeric_board.reshape((columns, rows))

    return numeric_board


class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):

        self._env = make("hungry_geese")
        # The number of agents
        self._NUM_AGENTS = 2

        # Reset environment
        observations = self._env.reset(num_agents=self._NUM_AGENTS)

        self._state = get_board(self._env)

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, 1, self._state.shape[0], self._state.shape[1]), dtype=np.int32, minimum=0, maximum=10,
            name='observation')
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        observations = self._env.reset(num_agents=self._NUM_AGENTS)
        self._state = [[get_board(self._env)]]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._state = get_board(self._env)

        choices = ['NORTH', 'SOUTH', 'WEST', 'EAST']

        actions = [choices[action], choice(choices)]

        self._env.step(actions)

        reward = self._env.steps[len(self._env.steps) - 1][0].reward

        if self._env.done:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(np.array([[self._state]], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array([[self._state]], dtype=np.int32), reward=0.0, discount=1.0)
