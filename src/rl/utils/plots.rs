use std::collections::HashMap;

use nalgebra::Vector2;
use plotters::prelude::*;

use crate::rl::environments::prelude::{GridActions, GridState};


/// Visualize the results for varying alpha and gamma
pub fn visualize_alpha_gamma_impact(
    alphas: Vec<f32>,
    gammas: Vec<f32>,
    grid_size: usize,
    goal_pos: Vector2<i32>,
    output_path: &str,
    results: Vec<(f32, f32, HashMap<(GridState, GridActions), f32>)>,
) {
    let root = BitMapBackend::new(output_path, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let areas = root.split_evenly((alphas.len(), gammas.len())); // Each cell corresponds to an alpha-gamma combination

    for (area, (alpha, gamma, q_values)) in areas.iter().zip(results.iter()) {
        let mut chart = ChartBuilder::on(area)
            .caption(format!("Alpha: {}, Gamma: {}", alpha, gamma), ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(20)
            .y_label_area_size(20)
            .build_cartesian_2d(0..grid_size as i32, 0..grid_size as i32)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        // ----------------- Get State Values (Max) -------------------------------------
        let mut state_values: HashMap<GridState, f32> = HashMap::new();
        for ((state, _action), q) in q_values {
            state_values
                .entry(state.clone())
                .and_modify(|v| *v = v.max(*q))
                .or_insert(*q);
        }
        let min_q = state_values.values().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_q = state_values.values().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        for (state, value) in state_values.iter() {
            let x = goal_pos.x - state.direction_to_goal.x;
            let y = goal_pos.y - state.direction_to_goal.y;
            let intensity = ((*value - min_q) / (max_q - min_q)) as f64;
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x, y), ((x + 1) as i32, (y + 1) as i32)],
                    ShapeStyle::from(&HSLColor(0.6, 1.0, 1.0 - intensity)).filled(),
                )))
                .unwrap();
        }
    }

    root.present().unwrap();
    println!("Visualization completed. Saved to {}", output_path);
}
