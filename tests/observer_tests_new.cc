// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#include "gtest/gtest.h"
#include "modules/commons/params/params.hpp"
#include "modules/geometry/geometry.hpp"
#include "modules/commons/params/default_params.hpp"
//#include "src/observers/nearest_observer.hpp"
#include "src/observers/nearest_observer_new.hpp"
//#include "src/observers/nn_observer.hpp"
#include "modules/world/tests/make_test_world.hpp"

#include "modules/geometry/commons.hpp"
#include "modules/models/behavior/constant_velocity/constant_velocity.hpp"
#include "modules/models/behavior/idm/idm_classic.hpp"
#include "modules/models/dynamic/single_track.hpp"
#include "modules/models/execution/interpolation/interpolate.hpp"


using namespace modules::models::dynamic;
using namespace modules::models::execution;
using namespace modules::commons;
using namespace modules::models::behavior;
using namespace modules::world::map;
using namespace modules::world;
using namespace modules::geometry;
using namespace modules::world::tests;

// observer
using observers::NearestObserver;
//using observers::NearestStateObserver;
using observers::ObservedState;


TEST(observes, observes_NearestObserver_Test0) {  

  Polygon polygon(
    Pose(1, 1, 0),
    std::vector<Point2d>{
      Point2d(0, 0),
      Point2d(0, 2),
      Point2d(2, 2),
      Point2d(2, 0),
      Point2d(0, 0)});

  std::shared_ptr<Polygon> goal_polygon(std::dynamic_pointer_cast<Polygon>(polygon.Translate(Point2d(50, -2))));  // < move the goal polygon into the driving
                           // corridor in front of the ego vehicle
  auto goal_definition_ptr = std::make_shared<GoalDefinitionPolygon>(*goal_polygon);
  float ego_velocity = 15.0, rel_distance = 7.0, velocity_difference = 0.0;

  ObservedWorld observed_world = make_test_observed_world(1, rel_distance, ego_velocity, velocity_difference, goal_definition_ptr);
  WorldPtr world = make_test_world(1, rel_distance, ego_velocity, velocity_difference, goal_definition_ptr);
  ObservedWorldPtr obs_world_ptr = std::make_shared<ObservedWorld>(observed_world);

  // Observer
  auto params = std::make_shared<DefaultParams>();
  //NearestStateObserver TestObserver(params); //Intance TestObersver of Constructor NearestStateObserver
  NearestObserver TestObserver(params);

  // Observe
  ObservedState res = TestObserver.observe(obs_world_ptr);
  //std::cout << res << std::endl;

  // Reset
  std::vector<int> agent_ids{0};
  TestObserver.Reset(world, agent_ids);  

}

TEST(observers, observes_NearestObserver_Test1){
  auto params = std::make_shared<DefaultParams>();
  
  ExecutionModelPtr exec_model(new ExecutionModelInterpolate(params));
  DynamicModelPtr dyn_model(new SingleTrackModel(params));
  BehaviorModelPtr beh_model_const(new BehaviorConstantVelocity(params));
  BehaviorModelPtr beh_model_idm(new BehaviorIDMClassic(params));
  Polygon polygon(Pose(1.25, 1, 0), std::vector<Point2d>{Point2d(0, 0), Point2d(0, 2), Point2d(4, 2), Point2d(4, 0), Point2d(0, 0)});

  State init_state0(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state1(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state2(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state3(static_cast<int>(StateDefinition::MIN_STATE_SIZE));
  State init_state4(static_cast<int>(StateDefinition::MIN_STATE_SIZE));

  //initialize agent states:
  //Time[s], X_Pos[m], Y_Pos[m], Theata[rad], Vel[m/s]
  init_state0 << 0.0, 0.0, 0.0, 0.0, 5.0;
  init_state1 << 0.0, 10.0, 20.0, -1.5708, 5.0;
  init_state2 << 0.0, -10, 5, 1.5708, 10;       //90deg rotated, would crash ego agent
  init_state3 << 0.0, 0.02, -9999.95, 0.0, 5.0; //check computational limits
  init_state4 << 0.0, 10000, 10000, 0.0, 5.0;   //add to verify that oberver only uses defined maximum number of agents

  //create agents
  AgentPtr agent0(new Agent(init_state0, beh_model_idm, dyn_model, exec_model, polygon, params)); //ego
  AgentPtr agent1(new Agent(init_state1, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent2(new Agent(init_state2, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent3(new Agent(init_state3, beh_model_const, dyn_model, exec_model, polygon, params));
  AgentPtr agent4(new Agent(init_state4, beh_model_const, dyn_model, exec_model, polygon, params));
 
  //create world
  WorldPtr world(new World(params));
  world->AddAgent(agent0);
  world->AddAgent(agent1);
  world->AddAgent(agent2);
  world->AddAgent(agent3);
  //world->AddAgent(agent4);
  world->UpdateAgentRTree();
  WorldPtr world1 = world->Clone();


  ObservedWorld observed_world1(world1, world1->GetAgents().begin()->second->GetAgentId());
  ObservedWorldPtr obs_world_ptr1 = std::make_shared<ObservedWorld>(observed_world1);

  //create instance of Observer and pass observed world
  NearestObserver TestObserver1(params);
  ObservedState res = TestObserver1.observe(obs_world_ptr1);
  //std::cout << res << std::endl;

  // Reset
  std::vector<int> agent_ids1{0};
  TestObserver1.Reset(world, agent_ids1);  
}





