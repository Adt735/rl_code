use std::hash::Hash;

use ndarray::Array1;
use rand::{RngExt, seq::{IndexedRandom, index::sample}};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use tch::{Device, Tensor, nn::{self, AdamW, Module, OptimizerConfig}};

use crate::rl::{Action, EnvironmentTrait, RLAlgorithmTrait, RlAlgoType, State, ToTensorVecExt, algorithms::tch_wrapper::{DeviceDef, NeuralNetwork}, utils::prelude::Statistics};



/**************************************************************
===============================================================
                        Structs
===============================================================
**************************************************************/
#[derive(Serialize, Deserialize)]
pub struct DeepQLearning<S: Default + State, A: Default + Action> {
    pub current_e: f32,
    pub min_e: f32,
    pub decay_rate_e: f32,
    pub reward_discount_factor: f32,
    pub max_steps_per_epoch: usize,

    /// Start at 0
    pub n_updates: usize,
    pub batch_size: usize,
    pub n_rollouts: usize,
    pub n_epochs: usize,
    pub n_epochs_to_update_target: usize,

    #[serde(with = "DeviceDef")]
    pub device: Device,
    pub replay_memory: ReplayMemory<S, A>,
    pub q_network: NeuralNetwork,
    pub target_network: NeuralNetwork,

    #[serde(skip)]
    pub optimizer: Option<nn::Optimizer>,
    pub optimizer_learning_rate: f32,

    #[serde(skip, default)]
    pub statistics: Statistics,

    algo_type: RlAlgoType,
}
impl<S, A> DeepQLearning<S, A>
where
    S: Clone + Default + State,
    A: Clone + Default + Hash + Copy + Action,
{
    pub fn new(
        min_e: f32, decay_rate_e: f32, learning_rate: f32, reward_discount_factor: f32, max_steps_per_epoch: usize,
        batch_size: usize, n_rollouts: usize, n_epochs: usize, n_epochs_to_update_target: usize,
        device: Device, replay_memory_capacity: usize, q_network_layers: Vec<i64>, nn_file_path: String,
    ) -> Self {
        let q_network = NeuralNetwork::new(device, q_network_layers.clone(), false, nn_file_path.clone() + "q_net.ot");
        let mut target_network = NeuralNetwork::new(device, q_network_layers, false, nn_file_path.clone() + "target_net.ot");
        target_network.vs.copy(&q_network.vs).unwrap();

        Self {
            current_e: 1.0, min_e , decay_rate_e, reward_discount_factor, max_steps_per_epoch,
            n_updates: 0, batch_size, n_rollouts, n_epochs, n_epochs_to_update_target,
            device, replay_memory: ReplayMemory::new(replay_memory_capacity, device),
            optimizer: Some(AdamW::default().build(&q_network.vs, learning_rate as f64).unwrap()),
            q_network, target_network, optimizer_learning_rate: learning_rate,
            statistics: Statistics::default(), algo_type: RlAlgoType::DeepQ
        }
    }

    pub fn rollout(&mut self, environment: &mut dyn EnvironmentTrait<S, A>, rng: &mut dyn rand::rand_core::Rng) {
        environment.reset();
        let mut rewards = Vec::with_capacity(self.max_steps_per_epoch);
        let mut current_state = environment.get_state();
        let mut n_steps_in_epoch = 0;

        let mut k = 0;
        while k < self.n_rollouts {
            let possible_actions = A::get_all_actions();

            let action = if rng.random_bool(self.current_e as f64) {
                None
            } else {
                self._best_action(&current_state, &possible_actions)
            };
            let action = action.unwrap_or_else(|| *possible_actions.choose(rng).unwrap());

            let (reward, terminated) = environment.step(action);
            rewards.push(reward);
            let next_state = environment.get_state();
            self.replay_memory.push(current_state.clone(), action, next_state.clone(), reward, terminated, n_steps_in_epoch > self.max_steps_per_epoch);
            n_steps_in_epoch += 1;

            current_state = next_state;
            if terminated || n_steps_in_epoch >= self.max_steps_per_epoch {
                environment.reset();
                current_state = environment.get_state();
                n_steps_in_epoch = 0;

                self.statistics.push(
                    terminated,
                    vec![
                        ("Reward".to_string(), rewards.iter().sum::<f32>()),
                        ("Epsilon".to_string(), self.current_e),
                    ]
                );
                rewards.clear();
            }

            k += 1
        }
        self.current_e = self.min_e.max(self.decay_rate_e * self.current_e);
    }

    pub fn learn(&mut self, rng: &mut dyn rand::rand_core::Rng) {   
        let mut losses = Vec::with_capacity(self.n_epochs);

        for _ in 0..self.n_epochs {
            let (obs, action, next_obs, reward, terminated, truncated) =
                self.replay_memory.sample(self.batch_size, rng);
            let q_values = self.q_network.forward(&obs);
    
            // Forward pass target network without gradients
            let next_q_values = tch::no_grad(|| {
                self.target_network.forward(&next_obs)
            });
    
            // Q(s,a) for chosen actions
            let q_value = q_values
                .gather(1, &action, false)
                .squeeze_dim(1);
    
            // max_a' Q_target(s', a')
            // let next_q_value = next_q_values.max_dim(1, false).0;
            // FIX - Double DQM
            let next_actions = tch::no_grad(|| {
                self.q_network.forward(&next_obs).argmax(1, true)
            });
            let next_q_value = next_q_values
                .gather(1, &next_actions, false)
                .squeeze_dim(1);
    
            // target = r + gamma * maxQ * (1-done) * (1-truncated)
            let target = &reward
                + self.reward_discount_factor
                    * next_q_value
                    * (1.0 - &terminated);
                    // * (1.0 - &truncated);
    
            // Huber loss (smooth_l1_loss)
            let loss = q_value.smooth_l1_loss(
                &target,
                tch::Reduction::Mean,
                1.0, // beta
            );
    
            // Optimize
            let optimizer = self.optimizer.as_mut().unwrap();
            optimizer.zero_grad();
            loss.backward();
    
            // Gradient clipping
            optimizer.clip_grad_norm(10.0);
            optimizer.step();
    
            self.n_updates += 1;
    
            // Periodic target network update
            if self.n_updates % self.n_epochs_to_update_target == 0 {
                self.update_target();
            }

            losses.push(loss.double_value(&[]));
        }

        self.statistics.add_metric_to_previous("Loss", (losses.iter().sum::<f64>() / losses.len() as f64) as f32);
    }

    fn update_target(&mut self) {
        self.target_network.vs.copy(&self.q_network.vs).unwrap();
    }

    fn _best_action(&self, state: &S, actions: &[A]) -> Option<A> {
        let state = state.to_tensor().to_device(self.device).unsqueeze(0);

        let q_values = tch::no_grad(|| {
            self.q_network.model.forward(&state)
        });
        let best_idx = q_values.argmax(-1, false).int64_value(&[0]) as usize;
        actions.get(best_idx).cloned()
    }
}
impl<S, A> RLAlgorithmTrait<S, A> for DeepQLearning<S, A> 
where
    S: Clone + Default + State + DeserializeOwned + Serialize,
    A: Clone + Default + Hash + Eq + Copy + Action + DeserializeOwned + Serialize,
    // Vec<S>: ToTensor,
    // Vec<A>: ToTensor,
{
    fn train_epoch(&mut self, environment: &mut dyn EnvironmentTrait<S, A>, rng: &mut dyn rand::rand_core::Rng) {
        if self.optimizer.is_none() {
            self.optimizer = Some(nn::AdamW::default().build(&self.q_network.vs, self.optimizer_learning_rate as f64).unwrap());
        }

        self.rollout(environment, rng);
        self.learn(rng);
    }

    fn best_action(&self, state: &S, actions: &[A]) -> Option<A> {
        self._best_action(state, actions)
    }

    fn get_memory_usage(&self) -> f32 {
        let replay_memory_mem = self.replay_memory.capacity * (
            2*size_of::<S>() + size_of::<A>() + 3*size_of::<f32>()
        );

        ( replay_memory_mem + 2*self.q_network.varstore_size_bytes() ) as f32 / 1024.0
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


/// Important because NN learns bad on correlated data
pub struct ReplayMemory<S, A> {
    capacity: usize,
    position: usize,
    size: usize,
    device: Device,

    states: Array1<S>,
    actions: Array1<A>,
    rewards: Array1<f32>,
    next_states: Array1<S>,
    terminated: Array1<f32>,
    truncated: Array1<f32>,
}

impl<S, A> ReplayMemory<S, A>
where
    S: Clone + Default + State,
    A: Clone + Default + Action,
    // Vec<S>: ToTensor,
    // Vec<A>: ToTensor,
{
    pub fn new(capacity: usize, device: Device) -> Self {
        ReplayMemory {
            capacity,
            position: 0,
            size: 0,
            device,

            states: Array1::from_elem(capacity, S::default()),
            actions: Array1::from_elem(capacity, A::default()),
            rewards: Array1::zeros(capacity),
            next_states: Array1::from_elem(capacity, S::default()),
            terminated: Array1::zeros(capacity),
            truncated: Array1::zeros(capacity),
        }
    }

    pub fn push(
        &mut self,
        state: S,
        action: A,
        next_state: S,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) {
        let idx = self.position;

        self.states[idx] = state;
        self.actions[idx] = action;
        self.next_states[idx] = next_state;
        self.rewards[idx] = reward;
        self.terminated[idx] = terminated as u8 as f32;
        self.truncated[idx] = truncated as u8 as f32;

        self.position = (self.position + 1) % self.capacity;
        self.size = self.size.min(self.capacity - 1) + 1;
    }

    pub fn sample(&self, batch_size: usize, rng: &mut dyn rand::rand_core::Rng) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let indices = sample(rng, self.size, batch_size);

        let mut states_vec = Vec::with_capacity(batch_size);
        let mut next_states_vec = Vec::with_capacity(batch_size);
        let mut actions_vec = Vec::with_capacity(batch_size);
        let mut rewards_vec = Vec::with_capacity(batch_size);
        let mut terminated_vec = Vec::with_capacity(batch_size);
        let mut truncated_vec = Vec::with_capacity(batch_size);

        for i in indices.iter() {
            states_vec.push(self.states[i].clone());
            next_states_vec.push(self.next_states[i].clone());
            actions_vec.push(self.actions[i].clone());
            rewards_vec.push(self.rewards[i]);
            terminated_vec.push(self.terminated[i]);
            truncated_vec.push(self.truncated[i]);
        }

        let states = states_vec.to_tensor()
            .to_device(self.device);

        let next_states = next_states_vec.to_tensor()
            .to_device(self.device);

        let actions = actions_vec.to_tensor()
            .to_kind(tch::Kind::Int64)
            .to_device(self.device);

        let rewards = Tensor::from_slice(&rewards_vec)
            .to_device(self.device);

        let terminated = Tensor::from_slice(&terminated_vec)
            .to_device(self.device);

        let truncated = Tensor::from_slice(&truncated_vec)
            .to_device(self.device);

        (states, actions, next_states, rewards, terminated, truncated)
    }

    pub fn len(&self) -> usize {
        self.size
    }
}



/**************************************************************
===============================================================
                        Serializing
===============================================================
**************************************************************/
#[derive(Serialize, Deserialize)]
struct ReplayMemoryMetadata {
    capacity: usize,
    #[serde(with = "DeviceDef")]
    device: Device,
}
impl<S, A> Serialize for ReplayMemory<S, A> {
    fn serialize<SSer>(&self, serializer: SSer) -> Result<SSer::Ok, SSer::Error>
    where
        SSer: Serializer,
    {
        let proxy = ReplayMemoryMetadata {
            capacity: self.capacity,
            device: self.device,
        };
        proxy.serialize(serializer)
    }
}
impl<'de, S: Default + State, A: Default + Action> Deserialize<'de> for ReplayMemory<S, A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let proxy = ReplayMemoryMetadata::deserialize(deserializer)?;
        
        Ok(ReplayMemory::new(proxy.capacity, proxy.device))
    }
}
