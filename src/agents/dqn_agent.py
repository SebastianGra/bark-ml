import tensorflow as tf

# tfa
from tf_agents.networks import q_network

from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils.common import Checkpointer
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from src.agents.tfa_agent import TFAAgent

class DQNAgent(TFAAgent):
  """DQN-Agent
     This agent is based on the tf-agents library.
  """
  def __init__(self,
               environment=None,
               replay_buffer=None,
               checkpointer=None,
               dataset=None,
               params=None):
    TFAAgent.__init__(self,
                      environment=environment,
                      params=params)
    self._replay_buffer = self.get_replay_buffer()
    self._dataset = self.get_dataset()
    self._collect_policy = self.get_collect_policy()
    self._eval_policy = self.get_eval_policy()

  def get_agent(self, env, params):
    """Returns a TensorFlow DQN-Agent
    
    Arguments:
        env {TFAPyEnvironment} -- Tensorflow-Agents PyEnvironment
        params {ParameterServer} -- ParameterServer from BARK
    
    Returns:
        agent -- tf-agent
    """

    # q network
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=tuple(
          self._params["ML"]["DQNAgent"]["categorical_fc_layer_params", "", [300, 300, 300, 300]]))
    
    # agent
    tf_agent = dqn_agent.DqnAgent(
      env.time_step_spec(),
      env.action_spec(),
      q_network=q_net,
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=self._params["ML"]["DQNAgent"]["learning_rate", "", 5e-5]),
      epsilon_greedy=self._params["ML"]["DQNAgent"]["epsilon_greedy", "", 0.3],
      target_update_tau=self._params["ML"]["DQNAgent"]["target_update_tau", "", 0.01],
      target_update_period=self._params["ML"]["DQNAgent"]["target_update_period", "", 1],
      td_errors_loss_fn=common.element_wise_squared_loss,
      gamma=self._params["ML"]["DQNAgent"]["gamma", "", 0.995],
      reward_scale_factor=self._params["ML"]["DQNAgent"]["reward_scale_factor", "", 1.],
      gradient_clipping=self._params["ML"]["DQNAgent"]["gradient_clipping", "", 10.],
      train_step_counter=self._ckpt.step,
      name=self._params["ML"]["DQNAgent"]["agent_name"],
      debug_summaries=self._params["ML"]["DQNAgent"]["debug_summaries", "", False])
    tf_agent.initialize()
    return tf_agent

  def get_replay_buffer(self):
    """Replay buffer
    
    Returns:
        ReplayBuffer -- tf-agents replay buffer
    """
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self._agent.collect_data_spec,
      batch_size=self._env.batch_size,
      max_length=self._params["ML"]["DQNAgent"]["replay_buffer_capacity", "", 200000])

  def get_dataset(self):
    """Dataset generated of the replay buffer
    
    Returns:
        dataset -- subset of experiences
    """
    dataset = self._replay_buffer.as_dataset(
      num_parallel_calls=self._params["ML"]["DQNAgent"]["parallel_buffer_calls", "", 1],
      sample_batch_size=self._params["ML"]["DQNAgent"]["batch_size", "", 1024],
      num_steps=self._params["ML"]["DQNAgent"]["n_step_update", "", 2]) \
        .prefetch(self._params["ML"]["DQNAgent"]["buffer_prefetch", "", 2])
    return dataset

  def get_collect_policy(self):
    """Returns the collection policy of the agent
    
    Returns:
        CollectPolicy -- Samples from the agent's distribution
    """
    return self._agent.collect_policy

  def get_eval_policy(self):
    """Returns the greedy policy of the agent
    
    Returns:
        GreedyPolicy -- Always returns best suitable action
    """
    return self._agent.policy

  def reset(self):
    pass

  @property
  def collect_policy(self):
    return self._collect_policy

  @property
  def eval_policy(self):
    return self._eval_policy

  def act(self, state):
    """ see base class
    """
    action_step = self.eval_policy.action(
      ts.transition(state, reward=0.0, discount=1.0))
    return action_step.action.numpy()
