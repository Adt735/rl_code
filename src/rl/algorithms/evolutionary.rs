use rand::{RngExt, seq::SliceRandom};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::rl::{
    Action, EnvironmentTrait, RLAlgorithmTrait, RlAlgoType, State, algorithms::simple_nn::SimpleNN, utils::prelude::Statistics 
};

#[derive(Serialize, Deserialize)]
/// Evolutionary / neuroevolution agent.
///
/// It does not use gradients.
/// Instead it:
/// 1) evaluates a population of small networks,
/// 2) keeps the best one,
/// 3) mutates clones of it to create the next generation.
pub struct EvolutionaryAlgorithm<S, A> {
    /// How many candidate networks are evaluated per generation.
    pub population_size: usize,

    /// How many of the best networks survive unchanged.
    /// For your use-case, 1 is usually enough.
    pub elite_count: usize,

    /// Probability that a weight / bias gets mutated.
    pub mutation_rate: f32,

    /// Maximum absolute mutation added to a parameter.
    pub mutation_strength: f32,

    pub mutation_strength_decay: f32,
    pub min_mutation_strength: f32,    

    /// Maximum number of environment steps per evaluation episode.
    pub max_steps_per_episode: usize,

    pub nn_layers: Vec<usize>,

    /// Current population.
    pub population: Vec<SimpleNN>,

    /// Best network found so far.
    pub champion: SimpleNN,

    #[serde(skip, default)]
    pub statistics: Statistics,
    #[serde(skip, default)]
    pub return_list: Vec<f32>,

    algo_type: RlAlgoType,

    _phantom_s: std::marker::PhantomData<S>,
    _phantom_a: std::marker::PhantomData<A>,
}

impl<S, A> EvolutionaryAlgorithm<S, A>
where
    S: State,
    A: Action,
{
    pub fn new(
        layers: &[usize],
        population_size: usize,
        elite_count: usize,
        mutation_rate: f32,
        mutation_strength: f32,
        min_mutation_strength: f32,
        mutation_strength_decay: f32,
        max_steps_per_episode: usize,
    ) -> Self {
        assert!(population_size > 0, "population_size must be > 0");
        assert!(elite_count > 0, "elite_count must be > 0");
        assert!(elite_count <= population_size, "elite_count must be <= population_size");

        let population = (0..population_size)
            .map(|_| SimpleNN::new_random(layers))
            .collect::<Vec<_>>();

        let champion = population[0].clone();

        Self {
            population_size,
            elite_count,
            mutation_rate,
            mutation_strength,
            min_mutation_strength,
            mutation_strength_decay,
            max_steps_per_episode,
            nn_layers: layers.try_into().unwrap(),
            population,
            champion,
            statistics: Statistics::default(),
            return_list: Vec::new(),
            _phantom_s: std::marker::PhantomData,
            _phantom_a: std::marker::PhantomData,
            algo_type: RlAlgoType::Evolutionary,
        }
    }

    fn action_index_from_network(&self, network: &SimpleNN, state: &S) -> usize {
        let input = state.to_vec();
        let output = network.forward(&input);

        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn evaluate_network(
        &self,
        environment: &mut Box<dyn EnvironmentTrait<S, A>>,
        network: &SimpleNN,
    ) -> (f32, usize, bool) {
        environment.reset();
        let mut total_reward = 0.0_f32;
        let mut steps = 0_usize;
        let mut _done = false;

        loop {
            let state = environment.get_state();
            let possible_actions = A::get_all_actions();

            let action_idx = self.action_index_from_network(network, &state);
            let action = possible_actions
                .get(action_idx)
                .cloned()
                .unwrap_or_else(|| possible_actions[0].clone());

            let (reward, done) = environment.step(action);
            total_reward += reward;
            steps += 1;

            if done || steps >= self.max_steps_per_episode {
                _done |= done;
                break;
            }
        }

        (total_reward, steps, _done)
    }

    fn evolve_population(&mut self, best_network: &SimpleNN) {
        let mut new_population = Vec::with_capacity(self.population_size);
        let mut rng = rand::rng();

        // Keep a few elite copies unchanged.
        for _ in 0..self.elite_count {
            new_population.push(best_network.clone());
        }

        // Fill the rest with mutated clones of the best network.
        while new_population.len() < self.population_size {
            let mut child = best_network.clone();

            // Global mutation gate: if true, mutate parameters.
            // The actual per-parameter mutation is handled by SimpleNN::mutate.
            if rng.random::<f32>() < 1.0 {
                child.mutate(self.mutation_strength, self.mutation_rate as f64);
            }

            new_population.push(child);
        }

        // Shuffle so the elite is not always first.
        new_population.shuffle(&mut rng);
        self.population = new_population;
        self.mutation_strength = self.min_mutation_strength.max(self.mutation_strength * self.mutation_strength_decay);
    }
}

impl<S, A> RLAlgorithmTrait<S, A> for EvolutionaryAlgorithm<S, A>
where
    S: State + Serialize + DeserializeOwned,
    A: Action + Serialize + DeserializeOwned,
{
    /// One epoch = evaluate the whole population once and produce a new generation.
    fn train_epoch(&mut self, environment: &mut Box<dyn EnvironmentTrait<S, A>>) {
        let mut best_reward = f32::NEG_INFINITY;
        let mut best_network = self.population[0].clone();
        let mut generation_rewards = Vec::with_capacity(self.population_size);
        let mut done = false;
        for network in &self.population {
            let (reward, steps, _done) = self.evaluate_network(environment, network);
            generation_rewards.push(reward / steps as f32);

            if reward > best_reward {
                best_reward = reward;
                best_network = network.clone();
            }

            done |= _done;
        }

        self.champion = best_network.clone();
        self.return_list.push(best_reward);

        self.statistics.push(&generation_rewards, done, self.mutation_strength);

        self.evolve_population(&best_network);
    }

    /// Greedy action from the current best network.
    fn best_action(&self, state: &S, actions: &[A]) -> Option<A> {
        if actions.is_empty() {
            return None;
        }

        let action_idx = self.action_index_from_network(&self.champion, state);
        actions.get(action_idx).cloned()
    }

    fn get_memory_usage(&self) -> f32 {
        ( self.population_size * self.champion.get_bytes_used() ) as f32 / 1024.0
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
