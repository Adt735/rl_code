
use clap::Parser;
use rapier3d::math::Vec2;
use std::path::PathBuf;
use serde::Deserialize;

use crate::rl::environments::prelude::{GridRewardType, PlaneRewardType};


/**************************************************************
===============================================================
                            General
===============================================================
**************************************************************/
#[derive(Parser)]
pub struct Cli {
    #[arg(short, long)]
    pub config: PathBuf,
}

#[derive(Deserialize)]
pub struct Config {
    pub algo: AlgoInitialization,
    pub env: EnvironmentType,
    pub training: TrainingConfig,
    pub action: Action,
}


/**************************************************************
===============================================================
                        Algorithm
===============================================================
**************************************************************/
#[derive(Deserialize)]
pub enum AlgoInitialization {
    Load { path: String },
    FromConfig(AlgoType)
}
#[derive(Deserialize)]
pub enum AlgoType {
    TabularQLearning(TabularQConfig),
    DeepQLearning(DeepQConfig),
    PPO(PPOConfig),
    Evolutionary(EvolutionaryConfig),
}

#[derive(Deserialize)]
pub struct TabularQConfig {
    pub min_e: f32,
    pub decay_rate_e: f32,
    pub learning_rate: f32,
    pub reward_discount_factor: f32,
}
#[derive(Deserialize)]
pub struct DeepQConfig {
    pub min_e: f32,
    pub decay_rate_e: f32,
    pub learning_rate: f32,
    pub reward_discount_factor: f32,

    pub batch_size: usize, 
    pub n_rollouts: usize, 
    pub n_epochs: usize, 
    pub n_epochs_to_update_target: usize,
    pub replay_memory_capacity: usize, 
    pub q_network_layers: Vec<i64>,
}
#[derive(Deserialize)]
pub struct PPOConfig {
    pub actor_layers: Vec<i64>,
    pub critic_layers: Vec<i64>,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub lmbda: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub mini_batch_size: usize,
    pub epsilon: f32,
    pub gamma: f32,
    pub current_entropy_weight: f32,
    pub min_entropy: f32,
    pub decay_rate_entropy: f32,
}
#[derive(Deserialize)]
pub struct EvolutionaryConfig {
    pub nn_layers: Vec<usize>,
    pub population_size: usize,
    pub elite_count: usize,
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub mutation_strength_decay: f32,
    pub min_mutation_strength: f32,    
}


/**************************************************************
===============================================================
                        Environment
===============================================================
**************************************************************/
#[derive(Deserialize)]
pub enum EnvironmentType {
    Grid(GridEnvConfig),
    Plane(PlaneEnvConfig),
}
#[derive(Deserialize)]
pub struct GridEnvConfig {
    pub width: usize,
    pub height: usize,

    pub start_x: i32,
    pub start_y: i32,

    pub goal_x: i32,
    pub goal_y: i32,

    pub maze_difficulty: Option<f32>,

    pub reward_type: GridRewardType,
}

#[derive(Deserialize)]
pub struct PlaneEnvConfig {
    pub width: f32,
    pub height: f32,

    pub start_x: f32,
    pub start_y: f32,

    pub goal_x: f32,
    pub goal_y: f32,

    pub obstacles: Vec<Vec2>,
    pub n_rays: usize, 
    pub length_ray: f32, 
    pub ray_span: f32,

    pub reward_type: PlaneRewardType,
}


/**************************************************************
===============================================================
                        Training
===============================================================
**************************************************************/
#[derive(Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub max_steps_per_epoch: usize,
}


/**************************************************************
===============================================================
                        Actions
===============================================================
**************************************************************/
#[derive(Deserialize)]
pub enum Action {
    SolutionVisualization(OutputConfig),
    CompareEnvironmentRewardType(CompareEnvironmentRewardTypeConfig),
    CheckHyperparametersEffect(HyperparametersEffectConfig),
    GetResourceUsageStatistics(ResourceUsageStatisticsConfig)
}

#[derive(Deserialize)]
pub struct OutputConfig {
    pub saving_path: String,
    pub show_info_in_plot: bool,
    pub create_video: bool,
}

#[derive(Deserialize)]
pub struct CompareEnvironmentRewardTypeConfig {
    pub saving_path: String,
    pub n_iterations: usize,
}

#[derive(Deserialize)]
pub struct HyperparametersEffectConfig {
    pub saving_path: String,
    pub learning_rates: Vec<f32>,
    pub reward_discount_factors: Vec<f32>,
    pub epsilon_decays: Vec<f32>,
}

#[derive(Deserialize)]
pub struct ResourceUsageStatisticsConfig {
    pub tries: usize,
}
