import sys
import logging
import time
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

from modules.runtime.commons.parameters import ParameterServer

from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts

from src.runners.tfa_runner import TFARunner

logger = logging.getLogger()
# this will print all statements
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class CDQNRunner(TFARunner):
  """Runner that takes the runtime and agent
     and runs the training and evaluation as specified.
  """
  def __init__(self,
               runtime=None,
               agent=None,
               params=ParameterServer(),
               unwrapped_runtime=None):
    TFARunner.__init__(self,
                       runtime=runtime,
                       agent=agent,
                       params=params,
                       unwrapped_runtime=unwrapped_runtime)
  
  @tf.function
  def _inference(self, input):
      q_logits, _ = self.model.call(input)
      q_probabilities = tf.nn.softmax(q_logits)
      q_values = tf.reduce_sum(self._agent._agent._support * q_probabilities, axis = 1)
      return q_values

  def _train(self):
    self.model = self._agent._agent._q_network
    #obs = np.array([[[0.68890405, 0.07822049, 0.5780419 , 1., 0.5767162 ,
    #    0.14886844, 0.65039325, 0.52042484]]]) 
    #self.model(obs)
    #print(self.model(obs))

    num_state_dims = np.shape(self._runtime._observation_spec)[1]
    inference = self._inference.get_concrete_function(input=tf.TensorSpec([1, 1, num_state_dims], tf.float32))
    self.model.save('./model', save_format='tf', include_optimizer=False, signatures=inference)

    """Trains the agent as specified in the parameter file
    """
    iterator = iter(self._agent._dataset)
    for _ in range(0, self._params["ML"]["Runner"]["number_of_collections"]):
      global_iteration = self._agent._agent._train_step_counter.numpy()
      self._collection_driver.run()
      experience, _ = next(iterator)
      self._agent._agent.train(experience)
      if global_iteration % self._params["ML"]["Runner"]["evaluate_every_n_steps"] == 0:
        self.evaluate()
        self._agent.save()