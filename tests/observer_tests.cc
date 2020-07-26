// Copyright (c) 2019 fortiss GmbH, Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.


#include "gtest/gtest.h"
#include "bark/commons/params/params.hpp"
#include "bark/geometry/geometry.hpp"
//#include "bark/commons/params/default_params.hpp"
#include "bark/commons/params/setter_params.hpp"
#include "src/observers/nearest_observer.hpp"
#include "bark/world/tests/make_test_world.hpp"

TEST(observes, nearest_observer) {
  // ugly imports
  using namespace bark::models::dynamic;
  using namespace bark::models::execution;
  using namespace bark::commons;
  using namespace bark::models::behavior;
  using namespace bark::models::dynamic;
  using namespace bark::world;
  using namespace bark::geometry;
  using namespace bark::world::tests;

  // observer
  using observers::NearestObserver;
  using observers::ObservedState;

  // Create world
  Polygon polygon(
    Pose(1, 1, 0),
    std::vector<Point2d>{
      Point2d(0, 0),
      Point2d(0, 2),
      Point2d(2, 2),
      Point2d(2, 0),
      Point2d(0, 0)});

  std::shared_ptr<Polygon> goal_polygon(
    std::dynamic_pointer_cast<Polygon>(polygon.Translate(
      Point2d(50, -2))));  // < move the goal polygon into the driving
                           // corridor in front of the ego vehicle
  auto goal_definition_ptr =
    std::make_shared<GoalDefinitionPolygon>(*goal_polygon);
  float ego_velocity = 15.0, rel_distance = 7.0,
    velocity_difference = 0.0;
  auto observed_world = make_test_observed_world(
    1, rel_distance, ego_velocity, velocity_difference, goal_definition_ptr);

  WorldPtr world = make_test_world(
    1, rel_distance, ego_velocity, velocity_difference, goal_definition_ptr);
  
  ObservedWorldPtr obs_world_ptr =
    std::make_shared<ObservedWorld>(observed_world);

  // Observer
  ParamsPtr params = std::make_shared<SetterParams>();
  NearestObserver observer(params);

  // Observe
  ObservedState res = observer.Observe(obs_world_ptr);
  std::cout << res << std::endl;

  // Reset
  std::vector<int> agent_ids{0};
  observer.Reset(world, agent_ids);

}
