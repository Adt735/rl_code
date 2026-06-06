use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::mem::size_of;


#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>, // flattened: output_dim × input_dim
    pub bias: Vec<f32>,
}


impl Layer {
    pub fn new_random(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();

        let weights = (0..input_dim * output_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let bias = (0..output_dim)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        Self {
            input_dim,
            output_dim,
            weights,
            bias,
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
            output[o] = sum.max(0.0);
        }

        output
    }

    pub fn mutate(&mut self, mutation_strength: f32, mutation_rate: f64) {
        let mut rng = rand::rng();

        for w in &mut self.weights {
            if rng.random_bool(mutation_rate) {
                *w += rng.random_range(-mutation_strength..mutation_strength);
            }
        }

        for b in &mut self.bias {
            if rng.random_bool(mutation_rate) {
                *b += rng.random_range(-mutation_strength..mutation_strength);
            }
        }
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

        for window in layer_sizes.windows(2) {
            let input_dim = window[0];
            let output_dim = window[1];

            layers.push(Layer::new_random(input_dim, output_dim));
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

    pub fn mutate(&mut self, mutation_strength: f32, mutation_rate: f64) {
        for layer in &mut self.layers {
            layer.mutate(mutation_strength, mutation_rate);
        }
    }

    pub fn get_bytes_used(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.get_bytes_used())
            .sum()
    }
}
