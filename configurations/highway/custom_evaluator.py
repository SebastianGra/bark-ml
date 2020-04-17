import numpy as np
from bark.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionAgents, \
  EvaluatorStepCount, EvaluatorDrivableArea, EvaluatorCollisionEgoAgent
from modules.runtime.commons.parameters import ParameterServer
from bark.geometry import *
from bark.models.dynamic import StateDefinition

from src.evaluators.goal_reached import GoalReached

class CustomEvaluator(GoalReached):
  """Shows the capability of custom elements inside
     a configuration.
  """
  def __init__(self,
               params=ParameterServer(),
               eval_agent=None):
    GoalReached.__init__(self,
                         params,
                         eval_agent)

  def _add_evaluators(self):
    self._evaluators["goal_reached"] = EvaluatorGoalReached()
    self._evaluators["drivable_area"] = EvaluatorDrivableArea()
    self._evaluators["collision"] = EvaluatorCollisionEgoAgent()
    self._evaluators["step_count"] = EvaluatorStepCount()

  def reverse_norm(self, value, range):
    return (value * (range[1]-range[0])) + range[0]

  def calculate_reward(self, observed_world, eval_results, action, observed_state):  # NOLINT
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    
    ego_agent = observed_world.ego_agent
    goal_def = ego_agent.goal_definition
    goal_center_line = goal_def.sequential_goals[0].center_line
    ego_agent_state = ego_agent.state
    lateral_offset = Distance(goal_center_line,
                              Point2d(ego_agent_state[1], ego_agent_state[2]))

    bb = observed_world.bounding_box
    x_range = [bb[0].x(), bb[1].x()]
    y_range = [bb[0].y(), bb[1].y()]
    nearest_state_x = self.reverse_norm(observed_state[4], x_range)
    nearest_state_y = self.reverse_norm(observed_state[5], y_range)
    minimal_dist = (nearest_state_x - ego_agent_state[1])**2 + \
                    (nearest_state_y - ego_agent_state[2])**2
    if minimal_dist - 30 > 0: minimal_dist = 30
    
    actions = np.reshape(action, (-1, 2))
    accs = actions[:, 0]
    delta = actions[:, 1]
    # TODO(@hart): use parameter server
    inpt_reward = np.sum((4/0.15*delta)**2 + (accs)**2)
    reward = collision * self._collision_penalty + \
      success * self._goal_reward + \
      drivable_area * self._collision_penalty - \
      0.01*lateral_offset**2 + 0.01*inpt_reward - 0.01*(30 - minimal_dist)
    return reward

  def _evaluate(self, observed_world, eval_results, action, observed_state):
    """Returns information about the current world state
    """
    done = False
    success = eval_results["goal_reached"]
    collision = eval_results["collision"]
    drivable_area = eval_results["drivable_area"]
    step_count = eval_results["step_count"]
    reward = self.calculate_reward(observed_world, eval_results, action, observed_state)    
    if success or collision or step_count > self._max_steps or drivable_area:
      done = True
    return reward, done, eval_results
    
