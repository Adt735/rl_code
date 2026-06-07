//! This crate holds the implementation of RL algorithms and environments, 
//! as well as the definition of its framework
use std::hash::Hash;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tch::Tensor;

use crate::rl::{algorithms::prelude::{DeepQLearning, EvolutionaryAlgorithm, PPO, TabularQLearning}, utils::prelude::Statistics};

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
pub trait RLAlgorithmTrait<S: State, A: Action>
{
    /// Executes one training epoch over the environment.
    fn train_epoch(&mut self, environment: &mut Box<dyn EnvironmentTrait<S, A>>, rng: &mut dyn rand::rand_core::Rng);

    /// Selects the best action following the learned policy for a given state from a set of possible actions.
    ///
    /// Returns `None` if no valid action is available.
    fn best_action(&self, state: &S, actions: &[A]) -> Option<A>;

    /// Get memory usage of the algorithm in `kb`. Just for research purposes
    fn get_memory_usage(&self) -> f32 { 0.0 }

    /// Get statistics
    fn get_statistics(&self) -> &Statistics;

    fn get_type(&self) -> RlAlgoType;
    fn to_json(&self) -> String;
}

/// All the methods every `State` must implement
pub trait State: Default + Clone + ToTensor {}

/// All the methodws every `Action` must implement
pub trait Action: Default + Clone + ToTensor {
    fn get_all_actions() -> Vec<Self>;
}


pub trait SerializableAlgorithm: Serialize + DeserializeOwned
{
    fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn load(info: String) -> Self
    where
        Self: Sized,
    {
        serde_json::from_str(&info).unwrap()
    }
}


pub trait ToTensor {
    /// Get the number of elements of a single struct
    fn len(&self) -> usize;
    /// Convert a state to a Vec of f32 (for converting it to a tensor later)
    fn to_vec(&self) -> Vec<f32>;

    fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(&self.to_vec())
    }
}

pub trait ToTensorVecExt {
    fn to_tensor(&self) -> Tensor;
}
impl<T: ToTensor> ToTensorVecExt for Vec<T> {
    fn to_tensor(&self) -> Tensor {
        // if self.is_empty() {
        //     // Returns an empty 2D tensor [[0, 0]] if the vector is empty
        //     return Tensor::empty([0, 0], (tch::Kind::Float, tch::Device::Cpu));
        // }

        let data: Vec<f32> = self
            .iter()
            .flat_map(|v| v.to_vec())
            .collect();

        // self.len() is the batch size, self[0].len() is the features per struct
        Tensor::from_slice(&data).view([self.len() as i64, self[0].len() as i64])
    }
}


#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub enum RlAlgoType {
    TabularQTable,
    DeepQ,
    PPO,
    Evolutionary,
}
#[derive(Serialize, Deserialize)]
pub struct AlgoHeader {
    pub algo_type: RlAlgoType,
}