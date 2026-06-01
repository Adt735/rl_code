
use clap::Parser;
use std::path::PathBuf;
use serde::Deserialize;

use crate::rl::environments::prelude::GridRewardType;

#[derive(Parser)]
pub struct Cli {
    #[arg(short, long)]
    pub config: PathBuf,
}


#[derive(Deserialize)]
pub struct Config {
    pub algo: AlgoInitialization,
    pub env: EnvConfig,
    pub training: TrainingConfig,
    pub action: Action,
}


#[derive(Deserialize)]
pub enum AlgoInitialization {
    Load { path: String },
    FromConfig(AlgoConfig)
}
#[derive(Deserialize)]
pub struct AlgoConfig {
    pub r#type: String,

    pub min_e: f32,
    pub decay_rate_e: f32,
    pub learning_rate: f32,
    pub reward_discount_factor: f32,
    pub max_steps_per_epoch: usize,
}

#[derive(Deserialize)]
pub struct EnvConfig {
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
pub struct TrainingConfig {
    pub epochs: usize,
}


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
