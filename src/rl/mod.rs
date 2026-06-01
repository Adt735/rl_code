//! This crate holds the implementation of RL algorithms and environments, 
//! as well as the definition of its framework
use serde::{Serialize, de::DeserializeOwned};

pub mod algorithms;
pub mod environments;
pub mod utils;


/// Defines a reinforcement learning environment.
///
/// The environment exposes the current state, available actions,
/// and supports stepping through the environment.
pub trait EnvironmentTrait<S: State, A: Action> {
    /// Returns the current state of the environment.
    fn get_state(&self) -> S;
    
    /// Applies an action to the environment.
    ///
    /// Returns a tuple `(reward, terminated)` where:
    /// - `reward`: scalar feedback signal
    /// - `terminated`: whether the episode has ended
    fn step(&mut self, action: A) -> (f32, bool);

    /// Resets the environment to an initial state.
    fn reset(&mut self);

    /// Plots the environment and last trajectory
    /// 
    /// Just for visualization and validation purposes
    fn plot(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;

    /// Plots the environment and a real-time video of the evolution
    /// 
    /// Just for visualization and validation purposes
    fn real_time_video(&self, frames_path: &str, video_path: &str) -> Result<(), Box<dyn std::error::Error>>;
}


/// Defines a generic reinforcement learning algorithm.
///
/// The algorithm interacts with an environment and learns a policy
/// that maps states to actions.
pub trait RLAlgorithmTrait<S: State, A: Action>: Serialize + DeserializeOwned {
    /// Executes one training epoch over the environment.
    fn train_epoch(&mut self, environment: &mut Box<dyn EnvironmentTrait<S, A>>);

    /// Selects the best action following the learned policy for a given state from a set of possible actions.
    ///
    /// Returns `None` if no valid action is available.
    fn best_action(&self, state: &S, actions: &[A]) -> Option<A>;

    /// Get memory usage of the algorithm in `kb`. Just for research purposes
    fn get_memory_usage(&self) -> f32 { 0.0 }
}

/// All the methods every `State` must implement
pub trait State: Clone {

}

/// All the methodws every `Action` must implement
pub trait Action: Clone {
    fn get_all_actions() -> Vec<Self>;
}
