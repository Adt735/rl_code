
use std::collections::{HashMap, HashSet};

use clap::Parser;
use indicatif::ProgressIterator;
use nalgebra::Vector2;
use rl::{*, algorithms::prelude::*, environments::prelude::*};

use crate::rl::utils::{cli::*, plots, prelude::Statistics};

mod rl;


fn main() {
    // Parser for a Command Line App
    let cli = Cli::parse();
    let content = std::fs::read_to_string(cli.config).unwrap();
    let config: Config = toml::from_str(&content).unwrap();


    match config.action {
        utils::cli::Action::SolutionVisualization(output_config) => {
            // ----------------------- Initialization ----------------------------------------
            let mut algo = load_algo(&config.algo);
            let mut environment: Box<dyn EnvironmentTrait<GridState, GridActions>> = Box::new(generate_environment(&config.env));

            // ----------------------- Training ----------------------------------------
            println!("Training...");
            for _ in (0..config.training.epochs).progress() {
                algo.train_epoch(&mut environment);
            }

            // ----------------------- Visualization ----------------------------------------
            environment.reset();
            for _ in 0..250 {
                let action = algo.best_action(&environment.get_state(), &GridActions::ALL);
                if let Some(action) = action {
                    let (_, terminated) = environment.step(action);
                    if terminated { break }
                } else {
                    break
                }
            }

            // ---------------------- Saving Results ---------------------------------------
            std::fs::create_dir_all(output_config.saving_path.clone()).unwrap();
            std::fs::create_dir_all(output_config.saving_path.clone() + "frames/").unwrap();

            algo.statistics.plot(&(output_config.saving_path.clone() + "rewards_plot.png"), "TabularQLearning over Discrete Grid", "epsilon", true, false, output_config.show_info_in_plot).unwrap();
            environment.plot(&(output_config.saving_path.clone() + "solution.png")).unwrap();
            if output_config.create_video {
                environment.real_time_video(&(output_config.saving_path.clone() + "frames/"), &(output_config.saving_path.clone() + "solution.mp4")).unwrap();
            }
            std::fs::write(
                &(output_config.saving_path.clone() + "model.json"), 
                serde_json::to_string_pretty(&algo).unwrap()
            ).unwrap();

            std::fs::remove_dir_all(output_config.saving_path.clone() + "frames/").unwrap();
        },

        utils::cli::Action::CompareEnvironmentRewardType(compare_config) => {
            let statistics_series = GridRewardType::ALL.map(|reward_type| {
                // ----------------------- Initialization ----------------------------------------
                let mut environment: Box<dyn EnvironmentTrait<GridState, GridActions>> = Box::new(generate_environment(&config.env));

                // ----------------------- Training ----------------------------------------
                // this trains n algorithms and unifies their statistics
                let statistics = Statistics::mean_statistics(
                (0..compare_config.n_iterations).map(|_| {
                        let algo_config = &config.algo;

                        let mut algo = load_algo(&config.algo);

                        for _ in 0..config.training.epochs {
                            algo.train_epoch(&mut environment);
                        }

                        algo.statistics
                    }).collect(),
                );

                (format!("{:?}", reward_type), statistics)
            });

            // ---------------------- Saving Results ---------------------------------------
            std::fs::create_dir_all(compare_config.saving_path.clone()).unwrap();
            Statistics::plot_multiple(
                &(compare_config.saving_path.clone() + "comparison_plot.png"), 
                "TabularQLearning over Discrete Grid RewardType comparison", 
                &statistics_series
            ).unwrap()
        }
    
        utils::cli::Action::CheckHyperparametersEffect(hyperparams_config) => {
            let mut results = Vec::new();
            let mut decay_results = Vec::new();
            let algo_config = match config.algo {
                AlgoInitialization::Load { .. } => panic!("Only accepts FromConfig"),
                AlgoInitialization::FromConfig(algo_config) => algo_config,
            };
            let environment = generate_environment(&config.env);
            let goal_pos = environment.goal_pos;
            let mut environment: Box<dyn EnvironmentTrait<GridState, GridActions>> = Box::new(environment);

            for alpha in &hyperparams_config.learning_rates {
                for gamma in &hyperparams_config.reward_discount_factors {
                    println!("Running Q-Learning for Alpha: {}, Gamma: {}", alpha, gamma);

                    // ----------------------- Initialization ----------------------------------------
                    let mut algo = TabularQLearning::new(
                        algo_config.min_e, algo_config.decay_rate_e, 
                        *alpha, *gamma, 
                        algo_config.max_steps_per_epoch
                    );

                    for _ in 0..config.training.epochs {
                        algo.train_epoch(&mut environment);
                    }
                    
                    results.push((alpha.clone(), gamma.clone(), algo.q_mat.clone()));
                }
            }

            for e_decay in hyperparams_config.epsilon_decays {
                println!("Running Q-Learning for decay: {}", e_decay);

                // ----------------------- Initialization ----------------------------------------
                let mut algo = TabularQLearning::new(
                    algo_config.min_e, e_decay, 
                    algo_config.learning_rate, algo_config.reward_discount_factor, 
                    algo_config.max_steps_per_epoch
                );

                for _ in 0..config.training.epochs {
                    algo.train_epoch(&mut environment);
                }
                
                decay_results.push((format!("{e_decay}"), algo.statistics.clone()));
            }

            // ---------------------- Saving Results ---------------------------------------
            std::fs::create_dir_all(hyperparams_config.saving_path.clone()).unwrap();
            plots::visualize_alpha_gamma_impact(
                hyperparams_config.learning_rates, hyperparams_config.reward_discount_factors, 
                config.env.width, goal_pos,  &(hyperparams_config.saving_path.clone() + "hyper_check.png"), results
            );
            Statistics::plot_multiple(
                &(hyperparams_config.saving_path.clone() + "hyper_e_check.png"), 
                "Epsilon decay rate comparison", 
                &decay_results
            ).unwrap();
        }
    
        utils::cli::Action::GetResourceUsageStatistics(usage_config) => {
            let mut reward = Vec::new();
            let mut success = Vec::new();
            let mut elapsed = Vec::new();
            let mut memory = Vec::new();
            println!("Starting training");
            for _ in (0..usage_config.tries).progress() {
                // ----------------------- Initialization ----------------------------------------
                let mut algo = load_algo(&config.algo);
                let mut environment: Box<dyn EnvironmentTrait<GridState, GridActions>> = Box::new(generate_environment(&config.env));

                // ----------------------- Training ----------------------------------------
                for _ in 0..config.training.epochs {
                    algo.train_epoch(&mut environment);
                }
                
                // ----------------------- Saving ----------------------------------------
                reward.push(algo.statistics.history.last().unwrap().reward_sum);
                success.push(algo.statistics.history.last().unwrap().success as usize);
                elapsed.push(algo.statistics.history.last().unwrap().elapsed);
                memory.push(algo.get_memory_usage());
            }

            println!("Mean reward: {:.2}", reward.iter().sum::<f32>() / reward.len() as f32);
            println!("Pct success: {}%", success.iter().sum::<usize>() * 100 / success.len());
            println!("Elapsed: {:.4} s", elapsed.iter().sum::<f32>() / elapsed.len() as f32);
            println!("Memory used: {} kb", memory.iter().sum::<f32>() / memory.len() as f32);
        }
    }
}


fn load_algo(algo_initialization: &AlgoInitialization) -> TabularQLearning<GridState, GridActions> {
    match algo_initialization {
        AlgoInitialization::Load { path } => {
            let json = std::fs::read_to_string(path).unwrap();
            serde_json::from_str(&json).unwrap()
        },
        AlgoInitialization::FromConfig(algo_config) => {
            TabularQLearning::new(
                algo_config.min_e, algo_config.decay_rate_e, 
                algo_config.learning_rate, algo_config.reward_discount_factor, 
                algo_config.max_steps_per_epoch
            )
        },
    }
}

fn generate_environment(env_config: &EnvConfig) -> SimpleGridEnvironment {
    let agent_start = Vector2::new(env_config.start_x, env_config.start_y);
    let (goal, walls) = match env_config.maze_difficulty {
        Some(difficulty) => {
            let walls = generate_maze(env_config.width as i32, env_config.height as i32, difficulty, agent_start);
            let goal = find_furthest_cell(agent_start, env_config.width as i32, env_config.height as i32, &walls);
            (goal, walls)
        },
        None => (Vector2::new(env_config.goal_x,env_config.goal_y), HashSet::default()),
    };

    SimpleGridEnvironment::new(
        env_config.width, env_config.height,
        agent_start, goal,
        walls, env_config.reward_type,
    )
}
