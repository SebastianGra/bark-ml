from absl import app
from absl import flags
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment

from modules.runtime.scenario.scenario_generation.uniform_vehicle_distribution \
  import UniformVehicleDistribution
from modules.runtime.scenario.scenario_generation.deterministic \
  import DeterministicScenarioGeneration
from modules.runtime.scenario.scenario_generation.configurable_scenario_generation import \
  ConfigurableScenarioGeneration

from modules.runtime.commons.parameters import ParameterServer
from modules.runtime.viewer.matplotlib_viewer import MPViewer
from modules.runtime.viewer.video_renderer import VideoRenderer
from modules.runtime.viewer.pygame_viewer import PygameViewer


from src.rl_runtime import RuntimeRL
from src.observers.nearest_state_observer import ClosestAgentsObserver
from src.wrappers.dynamic_model import DynamicModel
from src.wrappers.tfa_wrapper import TFAWrapper
from src.evaluators.goal_reached import GoalReached
from src.agents.sac_agent import SACAgent
from src.agents.ppo_agent import PPOAgent
from src.runners.sac_runner import SACRunner
from src.runners.ppo_runner import PPORunner
from configurations.base_configuration import BaseConfiguration

# configuration specific evaluator
from configurations.highway.custom_evaluator import CustomEvaluator
from bark_ml.observers import NearestObserver
#from bark_ml.observers import NearestStateObserver

class HighwayConfiguration(BaseConfiguration):
  """Hermetic and reproducible configuration class
  """
  def __init__(self,
               params):
    BaseConfiguration.__init__(
      self,
      params)

  def _build_configuration(self):
    """Builds a configuration using an SAC agent
    """
    # TODO(@hart): increase scenario number
    self._scenario_generator = \
      ConfigurableScenarioGeneration(num_scenarios=100,
                                     params=self._params)

    # USE THIS FOR DETERMINISTIC SCENARIO GEN.
    # self._scenario_generator = \
    #   DeterministicScenarioGeneration(num_scenarios=100,
    #                                   params=self._params)
    #self._observer = NearestObserver(self._params)
    #self._observer = ClosestAgentsObserver(self._params)
    self._observer = NearestStateObserver(self._params)

    self._behavior_model = DynamicModel(params=self._params)
    self._evaluator = CustomEvaluator(params=self._params)
    sim_step_time = 0.2
    real_time_factor = 5

    resolution = (1920, 1080 )
    dpi = 300
    fig_env = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi), dpi=dpi)
    gs = gridspec.GridSpec(1, 1, left=0.0, right=1, bottom=0, top=0.9)
    axis = plt.subplot(gs[0])
    viewer = MPViewer(
                  params=self._params,
                  y_length = 80,
                  enforce_y_length=True,
                  enforce_x_length=False,
                  follow_agent_id=True,
                  axis=axis)
    # self._viewer = VideoRenderer(renderer=viewer, world_step_time=0.2)
    self._viewer = viewer
    self._runtime = RuntimeRL(action_wrapper=self._behavior_model,
                              observer=self._observer,
                              evaluator=self._evaluator,
                              step_time=0.05,
                              viewer=self._viewer,
                              scenario_generator=self._scenario_generator)
    tfa_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    eval_tf_env = tf_py_environment.TFPyEnvironment(TFAWrapper(self._runtime))
    # self._agent = PPOAgent(tfa_env, params=self._params)
    # self._runner = PPORunner(tfa_env,
    #                          eval_tf_env,
    #                          self._agent,
    #                          params=self._params,
    #                          unwrapped_runtime=self._runtime)
    self._agent = SACAgent(tfa_env, params=self._params)
    self._runner = SACRunner(tfa_env,
                             self._agent,
                             params=self._params,
                             unwrapped_runtime=self._runtime)
