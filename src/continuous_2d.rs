use clap::Parser;
use indicatif::ProgressIterator;
use rapier3d::math::Vec2;

use crate::rl::{Action, AlgoHeader, EnvironmentTrait, RLAlgorithmTrait, algorithms::prelude::*, environments::prelude::*, utils::{self, cli::*}};



mod rl;

fn main() {
    // Parser for a Command Line App
    let cli = Cli::parse();
    let content = std::fs::read_to_string(cli.config).unwrap();
    let config: Config = toml::from_str(&content).unwrap();

    match config.action {
        utils::cli::Action::SolutionVisualization(output_config) => {
            let mut rng = rand::rng();

            // ----------------------- Initialization ----------------------------------------
            let mut algo = load_algo(&config.algo, config.training.max_steps_per_epoch);
            let mut environment: Box<dyn EnvironmentTrait<Environment2dState, Environment2dActions>> = Box::new(generate_environment(&config.env));

            println!("Algo type loaded: {:?}", algo.get_type());
            println!("Epochs: {:?}", config.training.epochs);

            // ----------------------- Training ----------------------------------------
            println!("Training...");
            for _ in (0..config.training.epochs).progress() {
                algo.train_epoch(&mut environment, &mut rng);
            }

            // ----------------------- Visualization ----------------------------------------
            environment.reset();
            for _ in 0..config.training.max_steps_per_epoch {
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

            for (i, plot) in output_config.plots.iter().enumerate() {
                algo.get_statistics().plot(&format!("{}metric_{i}_plot.png", output_config.saving_path.clone()), &plot.0, &plot.1, plot.2).unwrap();
            }
            environment.plot(&(output_config.saving_path.clone() + "solution.png")).unwrap();
            if output_config.create_video {
                environment.real_time_video(&(output_config.saving_path.clone() + "frames/"), &(output_config.saving_path.clone() + "solution.mp4")).unwrap();
            }
            std::fs::write(
                &(output_config.saving_path.clone() + "model.json"), 
                algo.to_json()
            ).unwrap();

            std::fs::remove_dir_all(output_config.saving_path.clone() + "frames/").unwrap();
        }

        _ => ()
    }
}


fn load_algo(algo_initialization: &AlgoInitialization, max_steps_per_epoch: usize) -> Box<dyn RLAlgorithmTrait<Environment2dState, Environment2dActions>> {
    match algo_initialization {
        AlgoInitialization::Load { path } => {
            let json = std::fs::read_to_string(path).unwrap();
            let header: AlgoHeader = serde_json::from_str(&json).unwrap();
            
            match header.algo_type {
                rl::RlAlgoType::TabularQTable => {panic!("Does not accept Tabular Learning")},
                rl::RlAlgoType::DeepQ => {let a: DeepQLearning<Environment2dState, Environment2dActions> = serde_json::from_str(&json).unwrap(); Box::new(a)},
                rl::RlAlgoType::PPO => {let a: PPO<Environment2dState, Environment2dActions> = serde_json::from_str(&json).unwrap(); Box::new(a)},
                rl::RlAlgoType::Evolutionary => {let a: EvolutionaryAlgorithm<Environment2dState, Environment2dActions> = serde_json::from_str(&json).unwrap(); Box::new(a)},
            }
        },
        AlgoInitialization::FromConfig(algo_type) => {
            match algo_type {
                AlgoType::DeepQLearning(algo_config) => {
                    Box::new(DeepQLearning::new(
                        algo_config.min_e, algo_config.decay_rate_e, algo_config.learning_rate, algo_config.reward_discount_factor, max_steps_per_epoch,
                        algo_config.batch_size, algo_config.n_rollouts, algo_config.n_epochs, algo_config.n_epochs_to_update_target,
                        tch::Device::Cpu, algo_config.replay_memory_capacity, algo_config.q_network_layers.clone()
                    ))
                }
                AlgoType::PPO(algo_config) => {
                    Box::new(PPO::new(
                        tch::Device::Cpu, algo_config.actor_lr, algo_config.critic_lr, 
                        algo_config.lmbda, algo_config.epochs, algo_config.batch_size, algo_config.mini_batch_size, 
                        algo_config.epsilon, algo_config.gamma, 
                        algo_config.current_entropy_weight, algo_config.min_entropy, algo_config.decay_rate_entropy, 
                        max_steps_per_epoch, algo_config.actor_layers.clone(), algo_config.critic_layers.clone()
                    ))
                }
                AlgoType::Evolutionary(algo_config) => {
                    Box::new(EvolutionaryAlgorithm::new(
                        &algo_config.nn_layers, algo_config.population_size, algo_config.elite_count, 
                        algo_config.mutation_rate, algo_config.mutation_strength, algo_config.min_mutation_strength, algo_config.mutation_strength_decay, 
                        max_steps_per_epoch,
                    ))
                }
                _ => panic!("Does not accept Tabular Learning")
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
