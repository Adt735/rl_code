use std::hash::Hash;

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tch::{nn::{self, OptimizerConfig}, Device, Kind, Tensor};

use crate::rl::{
    Action, EnvironmentTrait, RLAlgorithmTrait, RlAlgoType, State, ToTensorVecExt,
    algorithms::tch_wrapper::{DeviceDef, NeuralNetwork},
    utils::prelude::Statistics,
};


/**************************************************************
===============================================================
                        PPO
===============================================================
**************************************************************/
#[derive(Serialize, Deserialize)]
pub struct PPO<S, A> {
    pub actor: NeuralNetwork,
    pub critic: NeuralNetwork,

    #[serde(skip)]
    pub actor_optimizer: Option<nn::Optimizer>,
    #[serde(skip)]
    pub critic_optimizer: Option<nn::Optimizer>,
    pub actor_optimizer_learning_rate: f64,
    pub critic_optimizer_learning_rate: f64,

    pub current_entropy_weight: f32,
    pub min_entropy: f32,
    pub decay_rate_entropy: f32,

    pub gamma: f32,
    pub lmbda: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub mini_batch_size: usize,
    pub epsilon: f32,

    #[serde(with = "DeviceDef")]
    pub device: Device,
    pub max_steps_per_epoch: usize,

    #[serde(skip, default)]
    pub return_list: Vec<f32>,
    #[serde(skip, default)]
    pub statistics: Statistics,
    algo_type: RlAlgoType,

    _phantom_s: std::marker::PhantomData<S>,
    _phantom_a: std::marker::PhantomData<A>,
}

/// One trajectory collected with the current policy.
///
/// Keeping the rollout episode-by-episode makes the GAE computation much more stable
/// than flattening transitions from multiple episodes without preserving boundaries.
struct EpisodeRollout<S> {
    states: Vec<S>,
    action_indices: Vec<i64>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    old_log_probs: Vec<f32>,
    values: Vec<f32>,
    next_values: Vec<f32>,
    episode_reward: f32,
    success: bool,
}

impl<S, A> PPO<S, A>
where
    S: Clone + Default + State,
    A: Clone + Default + Action,
{
    pub fn new(
        device: Device,
        actor_lr: f64,
        critic_lr: f64,
        lmbda: f32,
        epochs: usize,
        batch_size: usize,
        mini_batch_size: usize,
        epsilon: f32,
        gamma: f32,
        current_entropy_weight: f32,
        min_entropy: f32,
        decay_rate_entropy: f32,
        max_steps_per_epoch: usize,
        actor_layers: Vec<i64>,
        critic_layers: Vec<i64>,
        nn_file_path: String,
    ) -> Self {
        let actor = NeuralNetwork::new(device, actor_layers, true, nn_file_path.clone() + "actor.ot");
        let critic = NeuralNetwork::new(device, critic_layers, false, nn_file_path + "critic.ot");

        let actor_optimizer = nn::Adam::default()
            .build(&actor.vs, actor_lr)
            .expect("failed to build PPO actor optimizer");
        let critic_optimizer = nn::Adam::default()
            .build(&critic.vs, critic_lr)
            .expect("failed to build PPO critic optimizer");

        Self {
            actor,
            critic,
            actor_optimizer: Some(actor_optimizer),
            critic_optimizer: Some(critic_optimizer),
            actor_optimizer_learning_rate: actor_lr,
            critic_optimizer_learning_rate: critic_lr,
            current_entropy_weight,
            min_entropy,
            decay_rate_entropy,
            gamma,
            lmbda,
            epochs,
            batch_size,
            mini_batch_size,
            epsilon,
            device,
            max_steps_per_epoch,
            return_list: Vec::new(),
            statistics: Statistics::default(),
            algo_type: RlAlgoType::PPO,
            _phantom_s: std::marker::PhantomData,
            _phantom_a: std::marker::PhantomData,
        }
    }

    fn ensure_optimizers(&mut self) {
        if self.actor_optimizer.is_none() {
            self.actor_optimizer = Some(
                nn::AdamW::default()
                    .build(&self.actor.vs, self.actor_optimizer_learning_rate)
                    .unwrap(),
            );
        }
        if self.critic_optimizer.is_none() {
            self.critic_optimizer = Some(
                nn::AdamW::default()
                    .build(&self.critic.vs, self.critic_optimizer_learning_rate)
                    .unwrap(),
            );
        }
    }

    fn state_tensor(&self, state: &S) -> Tensor {
        state.to_tensor().to_device(self.device).unsqueeze(0)
    }

    /// Returns the policy probabilities and the critic value for a single state.
    fn policy_and_value(&self, state: &S) -> (Tensor, Tensor) {
        let input = self.state_tensor(state);
        tch::no_grad(|| {
            let probs = self.actor.forward(&input);
            let value = self.critic.forward(&input);
            (probs, value)
        })
    }

    /// Sample an action from the current policy.
    ///
    /// The policy outputs a categorical distribution over `A::get_all_actions()`.
    fn sample_action(&self, state: &S, actions: &[A]) -> (A, i64, f32, f32) {
        let (probs, value) = self.policy_and_value(state);

        let action_idx_tensor = probs.multinomial(1, true);
        let action_idx = action_idx_tensor.int64_value(&[0, 0]);

        let action = actions[action_idx as usize].clone();
        let action_tensor = Tensor::from_slice(&[action_idx])
            .to_kind(Kind::Int64)
            .to_device(self.device)
            .view([1, 1]);

        let log_prob = probs
            .gather(1, &action_tensor, false)
            .clamp(1e-8, 1.0)
            .log()
            .double_value(&[0, 0]) as f32;

        let value = value.double_value(&[0, 0]) as f32;
        (action, action_idx, log_prob, value)
    }

    /// Collect one complete trajectory.
    ///
    /// This keeps episode boundaries intact, which makes the GAE computation
    /// correct and avoids mixing bootstrapping across episodes.
    fn rollout_episode(
        &mut self,
        environment: &mut dyn EnvironmentTrait<S, A>,
    ) -> EpisodeRollout<S> {
        environment.reset();
        let mut state = environment.get_state();

        let mut states = Vec::new();
        let mut action_indices = Vec::new();
        let mut rewards = Vec::new();
        let mut dones = Vec::new();
        let mut old_log_probs = Vec::new();
        let mut values = Vec::new();
        let mut next_values = Vec::new();

        let mut episode_reward = 0.0_f32;
        let mut success = false;

        for _ in 0..self.max_steps_per_epoch {
            let possible_actions = A::get_all_actions();
            let (action, action_idx, log_prob, value) = self.sample_action(&state, &possible_actions);

            let (reward, terminated) = environment.step(action);
            let next_state = environment.get_state();
            let next_value = tch::no_grad(|| {
                self.critic
                    .forward(&self.state_tensor(&next_state))
                    .double_value(&[0, 0]) as f32
            });

            states.push(state.clone());
            action_indices.push(action_idx);
            rewards.push(reward);
            dones.push(terminated);
            old_log_probs.push(log_prob);
            values.push(value);
            next_values.push(next_value);

            episode_reward += reward;
            success = terminated;
            state = next_state;

            if terminated {
                break;
            }
        }

        EpisodeRollout {
            states,
            action_indices,
            rewards,
            dones,
            old_log_probs,
            values,
            next_values,
            episode_reward,
            success,
        }
    }

    /// Generalized Advantage Estimation (GAE).
    ///
    /// This is the key change versus a naive TD-target update.
    /// It reduces variance while keeping the bootstrap bias controlled.
    fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_values: &[f32],
        dones: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let n = rewards.len();
        let mut advantages = vec![0.0_f32; n];
        let mut gae = 0.0_f32;

        for t in (0..n).rev() {
            let mask = if dones[t] { 0.0 } else { 1.0 };
            let delta = rewards[t] + self.gamma * next_values[t] * mask - values[t];
            gae = delta + self.gamma * self.lmbda * mask * gae;
            advantages[t] = gae;
        }

        let returns = advantages
            .iter()
            .zip(values.iter())
            .map(|(adv, v)| adv + v)
            .collect::<Vec<f32>>();

        (advantages, returns)
    }

    fn flatten_rollouts(
        &self,
        rollouts: &[EpisodeRollout<S>],
    ) -> (
        Vec<S>,
        Vec<i64>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<bool>,
    ) {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut old_log_probs = Vec::new();
        let mut advantages = Vec::new();
        let mut returns = Vec::new();
        let mut rewards_for_stats = Vec::new();
        let mut dones = Vec::new();

        for rollout in rollouts {
            let (adv, ret) = self.compute_gae(
                &rollout.rewards,
                &rollout.values,
                &rollout.next_values,
                &rollout.dones,
            );

            states.extend(rollout.states.iter().cloned());
            actions.extend(rollout.action_indices.iter().copied());
            old_log_probs.extend(rollout.old_log_probs.iter().copied());
            advantages.extend(adv);
            returns.extend(ret);
            rewards_for_stats.extend(rollout.rewards.iter().copied());
            dones.extend(rollout.dones.iter().copied());
        }

        (states, actions, old_log_probs, advantages, returns, rewards_for_stats, dones)
    }

    fn tensor_from_f32_slice(&self, data: &[f32]) -> Tensor {
        Tensor::from_slice(data).to_device(self.device).view([-1, 1])
    }

    fn tensor_from_i64_slice(&self, data: &[i64]) -> Tensor {
        Tensor::from_slice(data)
            .to_kind(Kind::Int64)
            .to_device(self.device)
            .view([-1, 1])
    }

    fn update_policy(
        &mut self,
        states: Vec<S>,
        action_indices: Vec<i64>,
        old_log_probs: Vec<f32>,
        advantages: Vec<f32>,
        returns: Vec<f32>,
    ) -> (f32, f32) {
        if states.is_empty() {
            return (0.0, 0.0);
        }

        let states = states.to_tensor().to_device(self.device);
        let actions = self.tensor_from_i64_slice(&action_indices);
        let old_log_probs = self.tensor_from_f32_slice(&old_log_probs);
        let mut advantages = self.tensor_from_f32_slice(&advantages);
        let returns = self.tensor_from_f32_slice(&returns);

        // Advantage normalization is very important for PPO stability.
        let adv_mean = advantages.mean(Kind::Float);
        let adv_std = advantages.std(true);
        advantages = (advantages - adv_mean) / (adv_std + 1e-8);

        let n_samples = states.size()[0] as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let mut actor_losses = Vec::with_capacity(self.epochs * indices.len() * self.mini_batch_size);
        let mut critic_losses = Vec::with_capacity(self.epochs * indices.len() * self.mini_batch_size);

        for _ in 0..self.epochs {
            indices.shuffle(&mut rand::rng());

            for chunk in indices.chunks(self.mini_batch_size.max(1)) {
                let idx_tensor = Tensor::from_slice(
                    &chunk.iter().map(|&i| i as i64).collect::<Vec<_>>()
                )
                .to_kind(Kind::Int64)
                .to_device(self.device);

                let b_states = states.index_select(0, &idx_tensor);
                let b_actions = actions.index_select(0, &idx_tensor);
                let b_old_log_probs = old_log_probs.index_select(0, &idx_tensor);
                let b_advantages = advantages.index_select(0, &idx_tensor);
                let b_returns = returns.index_select(0, &idx_tensor);

                let probs = self.actor.forward(&b_states);
                let probs_clamped = probs.clamp(1e-8, 1.0);
                let log_probs = probs_clamped
                    .gather(1, &b_actions, false)
                    .log();

                let ratio = (log_probs - &b_old_log_probs).exp();
                let clipped_ratio = ratio.clamp(
                    1.0 - self.epsilon as f64,
                    1.0 + self.epsilon as f64,
                );

                let surr1 = &ratio * &b_advantages;
                let surr2 = &clipped_ratio * &b_advantages;
                let surrogate = Tensor::stack(&[surr1, surr2], 1).min_dim(1, false).0;

                let entropy = -(&probs_clamped * probs_clamped.log())
                    .sum_dim_intlist([-1].as_slice(), false, Kind::Float)
                    .mean(Kind::Float);

                let actor_loss = -surrogate.mean(Kind::Float)
                    - self.current_entropy_weight * entropy;

                let value_pred = self.critic.forward(&b_states);
                let critic_loss = value_pred.mse_loss(&b_returns, tch::Reduction::Mean);

                let actor_optimizer = self.actor_optimizer.as_mut().unwrap();
                let critic_optimizer = self.critic_optimizer.as_mut().unwrap();

                actor_optimizer.zero_grad();
                critic_optimizer.zero_grad();

                actor_loss.backward();
                critic_loss.backward();

                // Clip gradients to keep the update well-behaved.
                actor_optimizer.clip_grad_norm(10.0);
                critic_optimizer.clip_grad_norm(10.0);

                actor_optimizer.step();
                critic_optimizer.step();

                actor_losses.push(actor_loss.double_value(&[]));
                critic_losses.push(critic_loss.double_value(&[]));
            }
        }

        // Exponential entropy decay, but never below the configured floor.
        self.current_entropy_weight = self
            .min_entropy
            .max(self.current_entropy_weight * self.decay_rate_entropy);

        (
            (actor_losses.iter().sum::<f64>() / actor_losses.len() as f64) as f32,
            (critic_losses.iter().sum::<f64>() / critic_losses.len() as f64) as f32,
        )
    }
}

impl<S, A> RLAlgorithmTrait<S, A> for PPO<S, A>
where
    S: Clone + Default + State + Serialize + DeserializeOwned,
    A: Clone + Default + Hash + Action + Eq + Copy + Serialize + DeserializeOwned,
{
    fn train_epoch(&mut self, environment: &mut dyn EnvironmentTrait<S, A>, _rng: &mut dyn rand::rand_core::Rng) {
        self.ensure_optimizers();

        let mut rollouts = Vec::with_capacity(self.batch_size.max(1));
        let mut success = false;
        for _ in 0..self.batch_size.max(1) {
            let rollout = self.rollout_episode(environment);
            success |= rollout.success;
            self.return_list.push(rollout.episode_reward);
            rollouts.push(rollout);
        }

        let (states, actions, old_log_probs, advantages, returns, rewards_for_stats, _dones) =
            self.flatten_rollouts(&rollouts);

        let (actor_loss, critic_loss) = self.update_policy(states, actions, old_log_probs, advantages, returns);

        // Keep the existing statistics API, but record the actual rollout rewards.
        // The `info` field is used here to track the current entropy coefficient.
        self.statistics.push(
            success,
            vec![
                ("Reward".to_string(), rewards_for_stats.iter().sum::<f32>() / self.batch_size as f32),
                ("Entropy".to_string(), self.current_entropy_weight),
                ("Actor Loss".to_string(), actor_loss),
                ("Critic Loss".to_string(), critic_loss),
            ]
        );
    }

    fn best_action(&self, state: &S, actions: &[A]) -> Option<A> {
        let state = state.to_tensor().to_device(self.device).unsqueeze(0);

        let probs = tch::no_grad(|| self.actor.forward(&state));
        let best_idx = probs.argmax(-1, false).int64_value(&[0]) as usize;
        actions.get(best_idx).cloned()
    }

    fn get_memory_usage(&self) -> f32 {
        (self.actor.varstore_size_bytes() + self.critic.varstore_size_bytes()) as f32 / 1024.0
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
