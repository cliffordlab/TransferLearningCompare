#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
from six.moves import xrange

def discount_episode_rewards(rewards=[], gamma=0.99, mode=0):
    """ Take 1D float array of rewards and compute discounted rewards for an
    episode. When encount a non-zero value, consider as the end a of an episode.

    Parameters
    ----------
    rewards : numpy list
        a list of rewards
    gamma : float
        discounted factor
    mode : int
        if mode == 0, reset the discount process when encount a non-zero reward (Ping-pong game).
        if mode == 1, would not reset the discount process.

    Examples
    ----------
    >>> rewards = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    >>> gamma = 0.9
    >>> discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma)
    >>> print(discount_rewards)
    ... [ 0.72899997  0.81        0.89999998  1.          0.72899997  0.81
    ... 0.89999998  1.          0.72899997  0.81        0.89999998  1.        ]
    >>> discount_rewards = tl.rein.discount_episode_rewards(rewards, gamma, mode=1)
    >>> print(discount_rewards)
    ... [ 1.52110755  1.69011939  1.87791049  2.08656716  1.20729685  1.34144104
    ... 1.49048996  1.65610003  0.72899997  0.81        0.89999998  1.        ]
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if mode == 0:
            if rewards[t] != 0: running_add = 0

        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


def cross_entropy_reward_loss(logits, actions, rewards, name=None):
    """ Calculate the loss for Policy Gradient Network.

    Parameters
    ----------
    logits : tensor
        The network outputs without softmax. This function implements softmax
        inside.
    actions : tensor/ placeholder
        The agent actions.
    rewards : tensor/ placeholder
        The rewards.

    Examples
    ----------
    >>> states_batch_pl = tf.compat.v1.placeholder(tf.float32, shape=[None, D])   # observation for training
    >>> network = tl.layers.InputLayer(states_batch_pl, name='input_layer')
    >>> network = tl.layers.DenseLayer(network, n_units=H, act = tf.nn.relu, name='relu1')
    >>> network = tl.layers.DenseLayer(network, n_units=3, act = tl.activation.identity, name='output_layer')
    >>> probs = network.outputs
    >>> sampling_prob = tf.nn.softmax(probs)
    >>> actions_batch_pl = tf.compat.v1.placeholder(tf.int32, shape=[None])
    >>> discount_rewards_batch_pl = tf.compat.v1.placeholder(tf.float32, shape=[None])
    >>> loss = cross_entropy_reward_loss(probs, actions_batch_pl, discount_rewards_batch_pl)
    >>> train_op = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(loss)
    """

    try: # TF 1.0
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits, name=name)
    except:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, targets=actions)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)

    try: ## TF1.0
        loss = tf.reduce_sum(tf.multiply(cross_entropy, rewards))
    except: ## TF0.12
        loss = tf.reduce_sum(tf.multiply(cross_entropy, rewards))   # element-wise mul
    return loss
