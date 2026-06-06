use std::{collections::HashMap, hash::Hash};

use serde::{Serialize, Deserialize};
use serde_with::serde_as;

use crate::rl::{utils::prelude::*, *};




/**************************************************************
===============================================================
                        Structs
===============================================================
**************************************************************/
/// Simplest algorithm:
/// * Q Values are stored in a table (Dictionnary)
/// * Values are updated with Bellman equation
#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct TabularQLearning<S: State + Hash + Eq + Serialize + DeserializeOwned, A: Action + Hash + Eq + Serialize + DeserializeOwned> {
    pub current_e: f32,
    pub min_e: f32,
    pub decay_rate_e: f32,
    pub learning_rate: f32,
    pub reward_discount_factor: f32,
    pub max_steps_per_epoch: usize,

    #[serde_as(as = "Vec<(_, _)>")]
    pub q_mat: HashMap<(S, A), f32>,

    #[serde(skip, default)]
    pub statistics: Statistics,

    algo_type: RlAlgoType,
}
impl<S: State + Hash + Eq + Serialize + DeserializeOwned, A: Action + Hash + Eq + Serialize + DeserializeOwned> TabularQLearning<S, A> {
    pub fn new(min_e: f32, decay_rate_e: f32, learning_rate: f32, reward_discount_factor: f32, max_steps_per_epoch: usize) -> Self {
        Self {
            current_e: 1.0,
            min_e, decay_rate_e,
            learning_rate, reward_discount_factor,
            max_steps_per_epoch,
            q_mat: HashMap::default(),
            statistics: Statistics::default(),
            algo_type: RlAlgoType::TabularQTable,
        }
    }
}
impl<S: State + Hash + Eq + Clone + Send + Sync + Serialize + DeserializeOwned, A: Action + Hash + Eq + Copy + Send + Sync + Serialize + DeserializeOwned> RLAlgorithmTrait<S, A> for TabularQLearning<S, A> {
    fn train_epoch(&mut self, environment: &mut Box<dyn EnvironmentTrait<S, A>>) {
        let mut rewards = Vec::with_capacity(self.max_steps_per_epoch);
        let mut steps = 0;
        loop {
            let agent_pos = environment.get_state();
            let possible_actions = A::get_all_actions();
    
            // Greedy epsilon
            let action = if fastrand::f32() < self.current_e {
                None
            } else {
                self.best_action(&agent_pos, &possible_actions)
            };
            let action = action.unwrap_or_else(|| *fastrand::choice(&possible_actions).unwrap());
    
            let (reward, terminated) = environment.step(action);
            rewards.push(reward);
    
            if terminated || steps > self.max_steps_per_epoch {
                let value = self.q_mat.entry((agent_pos, action)).or_insert(0.0);
                *value = *value + self.learning_rate * (reward - *value);

                // Update explotation - exploration
                self.current_e = (self.current_e * self.decay_rate_e).max(self.min_e);

                environment.reset();
    
                // Update statistics
                self.statistics.push(&rewards, terminated, self.current_e);
                rewards.clear();
                break;
            } else {
                // Find expected reward of new state
                let new_state = environment.get_state();
                let max_value = possible_actions.iter().map(|a| {
                    *self.q_mat.get(&(new_state.clone(), *a)).unwrap_or(&0.0)
                }).fold(f32::NEG_INFINITY, f32::max);

                // Update table
                let value = self.q_mat.entry((agent_pos, action)).or_insert(0.0);
                *value = *value + self.learning_rate * (reward + self.reward_discount_factor * max_value - *value);
                steps += 1;
            }
        }
    }

    fn best_action(&self, state: &S, actions: &[A]) -> Option<A> {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_actions = Vec::new();

        for &a in actions {
            let v = *self.q_mat.get(&(state.clone(), a)).unwrap_or(&0.0);

            if (v - best_value).abs() < 1e-6 {
                best_actions.push(a);
            } else if v > best_value {
                best_value = v;
                best_actions.clear();
                best_actions.push(a);
            } 
        }

        if best_actions.is_empty() {
            None
        } else {
            Some(*fastrand::choice(&best_actions).unwrap())
        }
    }

    fn get_memory_usage(&self) -> f32 {
        let buckets = self.q_mat.len();
        let entry_size = size_of::<((S, A), f32)>();

        (buckets * entry_size) as f32 / 1024.0
    }

    fn get_statistics(&self) -> &Statistics {
        &self.statistics
    }

    fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }

    fn get_type(&self) -> RlAlgoType {
        self.algo_type
    }
}

