#ifndef SRC_OBSERVERS_NEAREST_STATE_OBSERVER_HPP_
#define SRC_OBSERVERS_NEAREST_STATE_OBSERVER_HPP_

#include <memory>
#include <vector>
#include <tuple>
#include <map>
#include <functional>
#include <Eigen/Dense>


#include "bark/commons/params/params.hpp"
#include "bark/world/world.hpp"
#include "bark/world/observed_world.hpp"
#include "bark/world/goal_definition/goal_definition_state_limits_frenet.hpp"
#include "bark/models/dynamic/dynamic_model.hpp"
#include "src/commons/spaces.hpp"
#include "src/commons/commons.hpp"
#include "bark/geometry/angle.hpp"

namespace observers {
using bark::commons::ParamsPtr;
using bark::world::Agent;
using spaces::Box;
using commons::Norm;
using spaces::Matrix_t;
using bark::world::AgentMap;
using bark::world::AgentPtr;
using bark::world::WorldPtr;
using bark::world::goal_definition::GoalDefinitionStateLimitsFrenet;
using bark::world::ObservedWorldPtr;
using bark::geometry::Point2d;
using bark::geometry::Line;
using bark::geometry::Distance;
using bark::models::dynamic::StateDefinition;
using ObservedState = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>; 
using bark::commons::transformation::FrenetPosition;
using State = Eigen::Matrix<float, Eigen::Dynamic, 1>;


class StateObserver{
public:
    StateObserver():state_size_(){} //default constructor
    StateObserver(const ParamsPtr& params); //params_(params),state_size_() {};
   
    //State _select_state_by_index(const State, StateDefinition);
    ParamsPtr params_; 
    
protected:
    const float B_PI = 3.14159265358979323846; //using angle.hpp lib later
    const float B_2PI = B_PI * 2.0; // 2 * Pi;    
    float min_vel_, max_vel_, min_theta_, max_theta_;
    bool normalization_enabled_;
    int max_num_vehicles_, observation_len_;
    const int state_size_;
    float world_x_range [2] = {-10000, 10000};
    float world_y_range [2] = {-10000, 10000};    
};

StateObserver::StateObserver(const ParamsPtr& params):params_(params),state_size_(4){   //Konstruktor von Klasse StateObserver
    min_vel_ = params_->GetReal("ML:Observer:min_vel","", 0);
    max_vel_ = params_->GetReal("ML:Observer:max_vel","", 100);
    min_theta_ = params_->GetReal("ML::Observer::min_theta", "", 0);
    max_theta_ = params_->GetReal("ML::Observer::max_theta", "", B_2PI);
    normalization_enabled_= params_->GetBool("ML::Observer::norm::normalization_enabled_","", true);
    max_num_vehicles_ = params_->GetInt("ML::Observer::max_num_vehicles","", 2);
    observation_len_ = max_num_vehicles_* state_size_;
}



//void StateObserver::observe(const ObservedWorldPtr world){}

class NearestStateObserver : public StateObserver{ //Vererbt von Klasse StateObserver
public:
    NearestStateObserver() : StateObserver(params_){};
    NearestStateObserver(const ParamsPtr& params_);        
    ObservedState observe (const ObservedWorldPtr& world);
    int _len_state(State state_);
    State _norm(State);
    float _norm_to_range(float, float, float);   
    WorldPtr Reset(const WorldPtr& world, const std::vector<int>& agent_ids);
    Box<float> ObservationSpace() const; 

private:
    //StateObserver StateObserver_;
    StateDefinition state_definition_;
    State ego_state_; //Methode GetCurrentState nur für Point2D (X,Y Pos), s. agent.hpp/
    ObservedWorldPtr ego_observed_world_;

    float _max_distance_other_agents;
    int num_other_agents_, nearest_agent_num_;
    int max_dist_ = 75; //Parameter to define 
    ObservedState concatenated_state;
};

NearestStateObserver::NearestStateObserver(const ParamsPtr& params_){
    StateObserver StateObserver(params_); //s.Python Observer, Konstruktor von StateObserver hier aufrufen
    state_definition_= (StateDefinition::X_POSITION,
                        StateDefinition::Y_POSITION,
                        StateDefinition::THETA_POSITION,
                        StateDefinition::VEL_POSITION);
    _max_distance_other_agents = params_->GetReal("ML::Observer::max_dist", "", 30);
}

ObservedState NearestStateObserver::observe(const ObservedWorldPtr& world){        
  
    std::shared_ptr<const Agent> ego_agent_= world->GetEgoAgent(); //Datatype std::shared_ptr??

    ego_observed_world_ = world;   
    num_other_agents_ = sizeof(ego_observed_world_->GetOtherAgents())/sizeof((ego_observed_world_->GetOtherAgents())[0]);
    ego_state_ = ego_agent_->GetCurrentState();

    std::vector<int> nearest_agent_ids; // <- find before
    
    // find near agents (n)
    const Point2d ego_pos_ = ego_agent_->GetCurrentPosition(); //wird rot wenn in private
    AgentMap nearest_agents = world->GetNearestAgents(ego_pos_, num_other_agents_);
    
    //loop over nearest agents to get their IDs
    uint current_agent_idx = 0;
    for (const auto& agent : nearest_agents) {        
        nearest_agent_ids.push_back(agent.second->GetAgentId()); //agent.first alternativ?
        current_agent_idx++;
    }

    std::map<int, float> nearest_distances_  = {}; //use map
    //loop over other other agents and calc their relative position to ego agent
    for (const auto& agent_id : nearest_agent_ids) {
        if (agent_id == ego_agent_->GetAgentId()){
            continue; // this is yourself
        }
        const auto& agent_pos_ = ego_observed_world_->GetAgent(agent_id)
                                    ->GetCurrentPosition();              
        float dist = Distance(ego_pos_, agent_pos_);
        if (dist < max_dist_){ //to remove agents far away
            //add element to map Format: {<int>agent_id, <float>dist}
            //nearest_distances_[agent_id] = dist; //option1
            nearest_distances_.insert({agent_id, dist}); //option2
        }        
    } 

    //init vector with 0
    //std::vector<StateDefinition> concatenated_state
    ObservedState concatenated_state(1, observation_len_);
    concatenated_state.setZero();
    //uint vector_pos_idx = 0;
    uint agent_idx = 0;
    std::vector<int> position_indices{  StateDefinition::X_POSITION,
                                        StateDefinition::Y_POSITION,
                                        StateDefinition::THETA_POSITION,
                                        StateDefinition::VEL_POSITION};    
    
    //ego agent on first position of vector
    const auto& ego_state_normalized = _norm(ego_state_);
    /*for (const auto& state_position_idx : position_indices) {
        //concatenated_state[agent_idx + vector_pos_idx] = ego_state_normalized(state_position_idx);
        concatenated_state.block(agent_idx, state_position_idx, 1, 1) = ego_state_normalized.block(state_position_idx, )
        //vector_pos_idx++;
    }*/
    
    concatenated_state.block(agent_idx*state_size_, 0, 1, state_size_) = ego_state_normalized;
    agent_idx++;

    //insert other agents
    for (const auto& agent_id : nearest_agent_ids) {
        State agent_state = (ego_observed_world_->GetAgent(agent_id))->GetCurrentState();
        const auto& agent_state_normalized = _norm(agent_state);
        /*        
        for (const auto& state_position_idx : position_indices) {
            concatenated_state[agent_idx + vector_pos_idx] = agent_state(state_position_idx);
            //vector_pos_idx++;
        }*/
        concatenated_state.block(agent_idx*state_size_, 0, 1, state_size_) = agent_state_normalized;
        agent_idx++;
    }   
    return concatenated_state;
}

int NearestStateObserver::_len_state(State state_){
    return sizeof(state_)/sizeof(state_[0]);
}

State NearestStateObserver::_norm(State state_){
    if (normalization_enabled_ == false){
        return state_;
    }
    else
    {
        state_[StateDefinition::X_POSITION] =
        _norm_to_range(state_(StateDefinition::X_POSITION), world_x_range[0], world_x_range[1]);
        state_[StateDefinition::Y_POSITION] = 
        _norm_to_range(state_(StateDefinition::Y_POSITION), world_y_range[0], world_y_range[1]);
        state_[StateDefinition::THETA_POSITION] = 
        _norm_to_range(state_(StateDefinition::THETA_POSITION), min_theta_, max_theta_);
        state_[StateDefinition::VEL_POSITION] = 
        _norm_to_range(state_(StateDefinition::VEL_POSITION), min_vel_, max_vel_);        
        return state_;
    }   
}

float NearestStateObserver::_norm_to_range(float value, float min_val, float max_val){
    float norm = (value - min_val)/(max_val - min_val);
    return norm;
}

WorldPtr NearestStateObserver::Reset(const WorldPtr& world, const std::vector<int>& agent_ids) {
    return world;
}

Box<float> NearestStateObserver::ObservationSpace() const {
    Matrix_t<float> low(1, observation_len_);
    low.setZero();
    Matrix_t<float> high(1, observation_len_);
    high.setOnes();
    std::vector<int> shape{1, observation_len_};
    return Box<float>(low, high, shape);
}

}
#endif




    /*
    State StateObserver::_select_state_by_index(const State State, StateDefinition StateDefinition){
    bark::models::dynamic::StateDefinition
    state(StateDefinition::X_POSITION),
                             StateDefinition::Y_POSITION,
                             StateDefinition::THETA_POSITION,
                             StateDefinition::VEL_POSITION);
    return reduced_state_definion;
    }
    /*


      /*alt*
    //num_other_agents = ego_observed_world
    ego_agent_ = ego_observed_world_->GetEgoAgent();
    ego_state_ = ego_agent_->GetCurrentState();
    const Point2d ego_pos_ = ego_agent_->GetCurrentPosition(); //warum hier Datentyp notwendig??
   
    //calculate nearest agents distances
    nearest_agent_num_ = params_->GetInt("ML::Observer::n_nearest_agents", "Nearest agents number", 4);
    AgentMap nearest_agents = world->GetNearestAgents(ego_pos_, nearest_agent_num_);
    std::map<float, AgentPtr, std::greater<float>> distance_agent_map;
    for (const auto& agent : nearest_agents) {
      const auto& agent_state = agent.second->GetCurrentPosition();
      float distance = Distance(ego_pos_, agent_state);
      distance_agent_map[distance] = agent.second;
    *********/




    //preallocate array and add ego state
    //was macht das?
    // reserve the array //from Julian
    
    //std::vector<StateDefinition> concatenated_state(agent_state_size * max_num_agents);
    /*
    current_agent_idx = 0;
    for (const auto& agent_id : nearest_agent_ids ) {
        auto agent_state = observed_world->GetAgent(agent_id);
        const auto& vector_pos_idx = 0;
        
        for (const auto& state_position_index : position_indices) {
            concatenated_state(current_agent_idx + vector_pos_idx) = agent_state[state_position_index];
            vector_pos_idx++;
        }
        current_agent_idx++;
    }   
    */