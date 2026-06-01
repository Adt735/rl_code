use clap::Parser;
use indicatif::ProgressIterator;
use rapier3d::math::Vec2;

use crate::rl::{Action, EnvironmentTrait, RLAlgorithmTrait, algorithms::prelude::*, environments::prelude::*, utils::{self, cli::*}};



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
            let mut environment: Box<dyn EnvironmentTrait<Environment2dState, Environment2dActions>> = Box::new(generate_environment(&config.env));

            // ----------------------- Training ----------------------------------------
            println!("Training...");
            for _ in (0..config.training.epochs).progress() {
                algo.train_epoch(&mut environment);
            }

            // ----------------------- Visualization ----------------------------------------
            environment.reset();
            for _ in 0..algo.max_steps_per_epoch {
                let action = algo.best_action(&environment.get_state(), &Environment2dActions::get_all_actions());
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
        }

        _ => ()
    }
}


fn load_algo(algo_initialization: &AlgoInitialization) -> DeepQLearning<Environment2dState, Environment2dActions> {
    match algo_initialization {
        AlgoInitialization::Load { path } => {
            let json = std::fs::read_to_string(path).unwrap();
            serde_json::from_str(&json).unwrap()
        },
        AlgoInitialization::FromConfig(algo_type) => {
            match algo_type {
                AlgoType::DeepQLearning(algo_config) => {
                    DeepQLearning::new(
                        algo_config.min_e, algo_config.decay_rate_e, algo_config.learning_rate, algo_config.reward_discount_factor, algo_config.max_steps_per_epoch,
                        algo_config.batch_size, algo_config.n_rollouts, algo_config.n_epochs, algo_config.n_epochs_to_update_target,
                        tch::Device::Cpu, algo_config.replay_memory_capacity, algo_config.q_network_layers.clone()
                    )
                }
                _ => panic!("Only accepts DeepQ")
            }
        },
    }
}


fn generate_environment(env_type: &EnvironmentType) -> Simple2dEnvironment {
    match env_type {
        EnvironmentType::Plane(env_config) => {
            let agent_start = Vec2::new(env_config.start_x, env_config.start_y);
            let goal_pos = Vec2::new(env_config.goal_x,env_config.goal_y);

            Simple2dEnvironment::new(
                env_config.width, env_config.height, agent_start, goal_pos,
                env_config.obstacles.clone(), env_config.reward_type, 
                env_config.n_rays, env_config.length_ray, env_config.ray_span
            )
        }
        _ => panic!("This executable only accepts Plain Environment")
    }
}
