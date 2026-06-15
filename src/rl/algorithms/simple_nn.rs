use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::mem::size_of;


#[derive(Clone, Serialize, Deserialize)]
pub enum Activation {
    None,
    Relu,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>, // flattened: output_dim × input_dim
    pub weight_mask: Vec<bool>,
    pub bias: Vec<f32>,
    pub bias_mask: Vec<bool>,
    pub activation: Activation,
}


impl Layer {
    pub fn new_random(input_dim: usize, output_dim: usize, is_last: bool) -> Self {
        let mut rng = rand::rng();

        let weights = (0..input_dim * output_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let bias = (0..output_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let weight_mask = vec![false; input_dim * output_dim];
        let bias_mask = vec![false; output_dim];

        Self {
            input_dim,
            output_dim,
            weights,
            bias,
            weight_mask,
            bias_mask,
            activation: if is_last { Activation::None } else { Activation::Relu },
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.input_dim,
            "Input size does not match layer input_dim"
        );

        let mut output = vec![0.0; self.output_dim];

        for o in 0..self.output_dim {
            let mut sum = self.bias[o];

            for i in 0..self.input_dim {
                let w = self.weights[o * self.input_dim + i];
                sum += w * input[i];
            }

            // ReLU activation
            output[o] = match self.activation {
                Activation::None => sum,
                Activation::Relu => sum.max(0.0),
            }
        }

        output
    }

    pub fn mutate(&mut self, mutation_strength: f32, mutation_rate: f64, only_masked: bool) {
        let mut rng = rand::rng();

        for (i, w) in self.weights.iter_mut().enumerate() {
            if only_masked && !self.weight_mask[i] {
                continue;
            }
            if rng.random_bool(mutation_rate) {
                *w += rng.random_range(-mutation_strength..mutation_strength);
            }
        }

        for (i, b) in self.bias.iter_mut().enumerate() {
            if only_masked && !self.bias_mask[i] {
                continue;
            }
            if rng.random_bool(mutation_rate) {
                *b += rng.random_range(-mutation_strength..mutation_strength);
            }
        }
    }

    pub fn expand(&mut self, new_input_dim: usize, new_output_dim: usize) {
        assert!(
            new_input_dim >= self.input_dim && new_output_dim >= self.output_dim,
            "New dimensions must be greater than or equal to current dimensions"
        );

        let mut new_weights = vec![0.0; new_input_dim * new_output_dim];
        let mut new_weight_mask = vec![true; new_input_dim * new_output_dim];

        let mut new_bias = vec![0.0; new_output_dim];
        let mut new_bias_mask = vec![true; new_output_dim];

        for o in 0..self.output_dim {
            new_bias[o] = self.bias[o];
            new_bias_mask[o] = self.bias_mask[o];

            for i in 0..self.input_dim {
                let old_idx = o * self.input_dim + i;
                let new_idx = o * new_input_dim + i;
                
                new_weights[new_idx] = self.weights[old_idx];
                new_weight_mask[new_idx] = self.weight_mask[old_idx];
            }
        }

        self.input_dim = new_input_dim;
        self.output_dim = new_output_dim;
        self.weights = new_weights;
        self.weight_mask = new_weight_mask;
        self.bias = new_bias;
        self.bias_mask = new_bias_mask;
    }

    pub fn get_bytes_used(&self) -> usize {
        (self.weights.len() + self.bias.len()) * size_of::<f32>()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleNN {
    pub layers: Vec<Layer>,
}

impl SimpleNN {
    /// Example:
    /// SimpleNN::new_random(&[4, 8, 8, 2])
    ///
    /// Creates:
    /// 4 -> 8 -> 8 -> 2
    pub fn new_random(layer_sizes: &[usize]) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Network must have at least input and output layer"
        );

        let mut layers = Vec::new();

        for (i, window) in layer_sizes.windows(2).enumerate() {
            let input_dim = window[0];
            let output_dim = window[1];

            layers.push(Layer::new_random(input_dim, output_dim, i==layer_sizes.len()-2));
        }

        Self { layers }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut activations = input.to_vec();

        for layer in &self.layers {
            activations = layer.forward(&activations);
        }

        activations
    }

    pub fn mutate(&mut self, mutation_strength: f32, mutation_rate: f64, only_masked: bool) {
        for layer in &mut self.layers {
            layer.mutate(mutation_strength, mutation_rate, only_masked);
        }
    }

    pub fn expand(&mut self, new_layer_sizes: &[usize]) {
        assert_eq!(
            new_layer_sizes.len(),
            self.layers.len() + 1,
            "New layer sizes must match the current number of layers + 1"
        );

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let new_input_dim = new_layer_sizes[i];
            let new_output_dim = new_layer_sizes[i + 1];
            layer.expand(new_input_dim, new_output_dim);
        }
    }

    pub fn get_bytes_used(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.get_bytes_used())
            .sum()
    }
}
