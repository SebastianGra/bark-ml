

import numpy as np
from abc import ABC, abstractmethod

from bark.core.models.behavior import BehaviorMotionPrimitives, \
  BehaviorDynamicModel
from bark.core.models.dynamic import SingleTrackModel
from bark.runtime.commons.parameters import ParameterServer


class ActionWrapper(ABC):
  """Wrapper that transforms actions
  """
  def __init__(self, params):
    self._params = params

  @abstractmethod
  def action_to_behavior(self, world, actions):
    """Sets the action in the bark.BehaviorModel
    
    Arguments:
        world {bark.world} -- World containing all objects
        actions {any} -- can be of any type

    Returns:
        bark.world -- Updated world
    """
    pass

  @abstractmethod
  def reset(self, world, agents_to_act):
    """Resets the ActionWrapper
    
    Arguments:
        world {bark.world} -- World containing all objects
        agents_to_act {list(int)} -- Agents that an action
                                     is specified for
                                     
    Returns:
        bark.world -- Updated world
    """
    pass
  
  @property
  def action_space(self):
    """Information about the action space
    """
    pass


