

use std::{collections::{HashMap, HashSet, VecDeque}, process::Command};

use plotters::prelude::*;
use nalgebra::{SimdPartialOrd, Vector2};
use rand::{RngExt as _, seq::IndexedRandom};
use serde::{Serialize, Deserialize};

use crate::rl::*;


/**************************************************************
===============================================================
                        Environment
===============================================================
**************************************************************/
/// Wether the reward at each step is:
#[derive(Clone, Default, Debug, PartialEq, Eq, Deserialize, Copy)]
pub enum GridRewardType {
    #[default]
    /// -1.0
    Flat,
    /// Proportional to distance to goal
    Gradient,
}
impl GridRewardType {
    pub const ALL: [GridRewardType; 2] = [GridRewardType::Flat, GridRewardType::Gradient];
}

/// # Simplest Environment, just a grid
/// 
/// It just has to know:
/// + Agent position
/// + Goal position
/// + Walls
/// + Trajectory followed (for plotting purposes)
#[derive(Clone)]
pub struct SimpleGridEnvironment {
    pub agent_pos: Vector2<i32>,
    pub goal_pos: Vector2<i32>,
    pub initial_agent_pos: Vector2<i32>,

    pub width: i32,
    pub height: i32,
    pub weight: f32,
    pub walls: HashSet<(Vector2<i32>, GridActions)>,
    pub trajectory: Vec<Vector2<i32>>,

    pub reward_type: GridRewardType,
}
impl SimpleGridEnvironment {
    pub fn new(width: usize, height: usize, agent: Vector2<i32>, goal: Vector2<i32>, walls: HashSet<(Vector2<i32>, GridActions)>, reward_type: GridRewardType) -> Self {
        let mut s = Self {
            agent_pos: agent,
            goal_pos: goal,
            initial_agent_pos: agent,

            width: width as i32,
            height: height as i32,
            weight: 0.0,
            walls,
            trajectory: Vec::with_capacity(width * height),

            reward_type,
        };
        s.weight = s.compute_reward_weight();
        s
    }

    /// Helper function to know if the agent bumped into a wall
    pub fn collides_with_wall(&self, state: Vector2<i32>, action: GridActions) -> bool {
        if self.walls.contains(&(state, action)) { return true }
        if self.walls.contains(&(state + action.to_vec(), action.opposite())) { return true }

        false
    }

    fn compute_reward_weight(&self) -> f32 {
        match self.reward_type {
            GridRewardType::Flat => {
                let total_steps = (self.width * self.height - 1) as f32;
                100.0 / total_steps
            }
            GridRewardType::Gradient => {
                let mut total_unweighted_penalty = 0.0;
                
                for x in 0..self.width {
                    for y in 0..self.height {
                        // Skip the initial position since no step reward happens there
                        if x as i32 == self.initial_agent_pos.x && y as i32 == self.initial_agent_pos.y {
                            continue;
                        }
                        let dx = (x as i32 - self.goal_pos.x).abs();
                        let dy = (y as i32 - self.goal_pos.y).abs();
                        total_unweighted_penalty += (dx + dy) as f32;
                    }
                }
                100.0 / total_unweighted_penalty
            }
        }
    }

    /// Useful for creating the real-time video
    fn interpolate_trajectory(
        &self,
        steps_per_cell: usize,
    ) -> Vec<(f32, f32)> {

        let mut result = Vec::new();

        for window in self.trajectory.windows(2) {
            let start = &window[0];
            let end = &window[1];

            for step in 0..steps_per_cell {
                let t = step as f32 / steps_per_cell as f32;

                let x =
                    start.x as f32 +
                    (end.x as f32 - start.x as f32) * t;

                let y =
                    start.y as f32 +
                    (end.y as f32 - start.y as f32) * t;

                result.push((x, y));
            }
        }

        if let Some(last) = self.trajectory.last() {
            result.push((last.x as f32, last.y as f32));
        }

        result
    }

    fn plot(&self, path: &str, until: Option<usize>, steps_per_cell: usize) -> Result<(), Box<dyn std::error::Error>> {    
        let cell_size = 50;
        
        // ---------------------------------------------------------
        // Draw config
        // ---------------------------------------------------------
        let root = BitMapBackend::new(
            path,
            (
                self.width as u32 * cell_size,
                self.height as u32 * cell_size,
            ),
        )
        .into_drawing_area();
        root.fill(&WHITE)?;
    
        let mut chart = ChartBuilder::on(&root)
            .margin(0)
            .build_cartesian_2d(
                0f32..self.width as f32,
                0f32..self.height as f32,
            )?;
    
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;
    
        // ---------------------------------------------------------
        // Draw grid
        // ---------------------------------------------------------
        for y in 0..self.height {
            for x in 0..self.width {
                let x0 = x as f32;
                let y0 = y as f32;
    
                // white cell
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x0, y0), (x0 + 1.0, y0 + 1.0)],
                    WHITE.filled(),
                )))?;
    
                // border
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x0, y0), (x0 + 1.0, y0 + 1.0)],
                    ShapeStyle {
                        color: BLACK.mix(0.2),
                        filled: false,
                        stroke_width: 1,
                    },
                )))?;
            }
        }

        // ---------------------------------------------------------
        // Draw walls
        // ---------------------------------------------------------
        for (pos, action) in &self.walls {
            let x = pos.x as f32;
            let y = pos.y as f32;
        
            let wall = match action {
                GridActions::Up => vec![
                    (x, y + 1.0),
                    (x + 1.0, y + 1.0),
                ],
        
                GridActions::Right => vec![
                    (x + 1.0, y),
                    (x + 1.0, y + 1.0),
                ],
        
                GridActions::Down => vec![
                    (x, y),
                    (x + 1.0, y),
                ],
        
                GridActions::Left => vec![
                    (x, y),
                    (x, y + 1.0),
                ],
            };
        
            chart.draw_series(std::iter::once(PathElement::new(
                wall,
                BLACK.stroke_width(4),
            )))?;
        }
    
        // ---------------------------------------------------------
        // Initial position
        // ---------------------------------------------------------
        chart
            .draw_series(std::iter::once(Circle::new(
                (
                    self.initial_agent_pos.x as f32 + 0.5,
                    self.initial_agent_pos.y as f32 + 0.5,
                ),
                8,
                RED,
            )))?
            .label("Start")
            .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));
    
        // ---------------------------------------------------------
        // Goal position
        // ---------------------------------------------------------
        chart
            .draw_series(std::iter::once(Circle::new(
                (
                    self.goal_pos.x as f32 + 0.5,
                    self.goal_pos.y as f32 + 0.5,
                ),
                20,
                GREEN.mix(0.5).filled(),
            )))?
            .label("Goal")
            .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));
    
        // ---------------------------------------------------------
        // Trajectory
        // ---------------------------------------------------------
        let mut trajectory = self.interpolate_trajectory(steps_per_cell);
        if let Some(until) = until {
            trajectory = trajectory[..until+1].to_vec();
        }
        if trajectory.len() >= 2 { 
            let points: Vec<(f32, f32)> = trajectory
                .iter()
                .map(|p| {
                    (
                        p.0 + 0.5,
                        p.1 + 0.5,
                    )
                })
                .collect();
            
            for segment in points.windows(2) {
                chart.draw_series(std::iter::once(
                    PathElement::new(
                        vec![segment[0], segment[1]],
                        RED.mix(0.9).stroke_width(3),
                    )
                ))?;
            }
            // chart.draw_series(LineSeries::new(points, RED.mix(0.9).stroke_width(3)))?;
            // chart.draw_series(std::iter::once(PathElement::new(
            //     points,
            //     RED.stroke_width(3),
            // )))?;
        }

        // ---------------------------------------------------------
        // Current position
        // ---------------------------------------------------------
        if until.is_some() {
            chart
                .draw_series(std::iter::once(Circle::new(
                    (
                        trajectory.last().unwrap().0 as f32 + 0.5,
                        trajectory.last().unwrap().1 as f32 + 0.5,
                    ),
                    8,
                    RED.filled(),
                )))?
                .label("Current pos")
                .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));
        }

        chart
            .configure_series_labels()
            .label_font(("sans-serif", 24))
            .position(SeriesLabelPosition::LowerRight)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;

        root.present()?;
        Ok(())
    }
}

impl EnvironmentTrait<GridState, GridActions> for SimpleGridEnvironment {
    fn get_state(&self) -> GridState {
        let state = GridState {
            direction_to_goal: self.goal_pos - self.agent_pos,
        };

        assert!(
            state.direction_to_goal.x.abs() <= self.width,
            "Invalid dx={} goal={:?} agent={:?}",
            state.direction_to_goal.x,
            self.goal_pos,
            self.agent_pos
        );

        assert!(
            state.direction_to_goal.y.abs() <= self.height,
            "Invalid dy={} goal={:?} agent={:?}",
            state.direction_to_goal.y,
            self.goal_pos,
            self.agent_pos
        );

        state
    }

    /// Simply return Agent position to it initial position
    fn reset(&mut self) {
        self.agent_pos = self.initial_agent_pos;
        self.trajectory.clear();
        self.trajectory.push(self.initial_agent_pos);
    }

    /// Advance the agent in the given direction (except if it jumps into a wall)
    fn step(&mut self, action: GridActions) -> (f32, bool) {
        let collides_with_wall = self.collides_with_wall(self.agent_pos, action);

        let checked_target_pos = if collides_with_wall { self.agent_pos } else { 
            let target_pos = self.agent_pos + action.to_vec();
            Vector2::new(
                target_pos.x.clamp(0, self.width - 1),
                target_pos.y.clamp(0, self.height - 1),
            )
        };

        let mut reward = if checked_target_pos == self.goal_pos { // Won round
            let initial_diff = self.goal_pos - self.initial_agent_pos;
            let initial_dist = (initial_diff.x.abs() + initial_diff.y.abs()) as f32;

            match self.reward_type {
                GridRewardType::Flat => initial_dist - 1.0,
                GridRewardType::Gradient => (initial_dist * (initial_dist - 1.0)) / 2.0,
            }
        } else { // Goal is to minimize route, so every step taken is "bad"
            match self.reward_type {
                GridRewardType::Flat => -1.0,
                GridRewardType::Gradient => {
                    let diff = checked_target_pos - self.goal_pos;
                    -(diff.x.abs() + diff.y.abs()) as f32
                }
            }
        };
        // Scale the reward using the computed weight
        reward *= self.weight;

        self.trajectory.push(checked_target_pos);
        self.agent_pos = checked_target_pos;
        (reward, self.agent_pos == self.goal_pos)
    }

    fn plot(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {    
        self.plot(path, None, 1)
    }

    fn real_time_video(&self, frames_path: &str, video_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let steps_per_cell = 600 / self.trajectory.len() + 1;
        for i in 0..(self.trajectory.len()-1)*steps_per_cell {
            self.plot(&format!("{frames_path}frame_{i:05}.png"), Some(i), steps_per_cell)?;
        }

        let status = Command::new("ffmpeg")
            .args([
                "-loglevel", "error",
                "-hide_banner",
                "-y",
                "-framerate", "60",
                "-i", &format!("{frames_path}frame_%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_path,
            ])
            .status()?;
        if !status.success() {
            println!("Failed to convert to video")
        }

        Ok(())
    }

    fn clone_box(&self) -> Box<dyn EnvironmentTrait<GridState, GridActions>> {
        Box::new(self.clone())
    }
}


/**************************************************************
===============================================================
                        Actions
===============================================================
**************************************************************/
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum GridActions {
    #[default]
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3
}
impl GridActions {
    pub fn to_vec(&self) -> Vector2<i32> {
        match self {
            GridActions::Up => Vector2::new(0, 1),
            GridActions::Right => Vector2::new(1, 0),
            GridActions::Down => Vector2::new(0, -1),
            GridActions::Left => Vector2::new(-1, 0),
        }
    }
    
    pub fn opposite(&self) -> GridActions {
        match self {
            GridActions::Up => GridActions::Down,
            GridActions::Right => GridActions::Left,
            GridActions::Down => GridActions::Up,
            GridActions::Left => GridActions::Right,
        }
    }

    pub const ALL: [GridActions; 4] = [
        GridActions::Up,
        GridActions::Right,
        GridActions::Down,
        GridActions::Left,
    ];
}
impl Action for GridActions {
    fn get_all_actions() -> Vec<Self> {
        vec![Self::Up, Self::Right, Self::Down, Self::Left]
    }
}
impl ToTensor for GridActions {
    fn len(&self) -> usize {
        1
    }

    fn to_vec(&self) -> Vec<f32> {
        vec![*self as u8 as f32]
    }
}


/**************************************************************
===============================================================
                        State
===============================================================
**************************************************************/
#[derive(Clone, Default, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridState {
    pub direction_to_goal: Vector2<i32>,
}
impl State for GridState {}
impl ToTensor for GridState {
    fn len(&self) -> usize {
        2
    }

    fn to_vec(&self) -> Vec<f32> {
        vec![self.direction_to_goal.x as f32, self.direction_to_goal.y as f32]
    }
}



/**************************************************************
===============================================================
                        Utils
===============================================================
**************************************************************/
///
/// difficulty:
/// 0.0 => very easy
/// 1.0 => hard
///
/// Harder means:
/// - fewer loops
/// - longer corridors
///
pub fn generate_maze(
    width: i32,
    height: i32,
    difficulty: f32,
    start: Vector2<i32>,
) -> HashSet<(Vector2<i32>, GridActions)> {
    let mut rng = rand::rng();

    // ------------------------------------------------------------------------
    // PASSAGES (graph edges)
    // ------------------------------------------------------------------------

    let mut visited = HashSet::<Vector2<i32>>::new();

    // passages[a] contains all connected neighbors
    let mut passages: HashMap<
        Vector2<i32>,
        HashSet<Vector2<i32>>
    > = HashMap::new();

    visited.insert(start);

    let mut stack = vec![start];

    // Used for corridor straightness
    let mut previous_direction: Option<GridActions> = None;

    // ------------------------------------------------------------------------
    // RECURSIVE BACKTRACKER (DFS)
    // ------------------------------------------------------------------------
    while let Some(current) = stack.last().copied() {
        let mut neighbors = vec![];

        for action in GridActions::ALL {
            let next = current + action.to_vec();

            if next.x < 0
                || next.x >= width
                || next.y < 0
                || next.y >= height
            {
                continue;
            }

            if !visited.contains(&next) {
                neighbors.push((action, next));
            }
        }

        if neighbors.is_empty() {
            stack.pop();
            previous_direction = None;
            continue;
        }

        // --------------------------------------------------------------------
        // Difficulty bias:
        // harder => prefer continuing straight
        // --------------------------------------------------------------------
        let chosen = if let Some(prev_dir) = previous_direction {
            if rng.random_range(0.0..1.0) < difficulty {
                neighbors
                    .iter()
                    .find(|(dir, _)| *dir == prev_dir)
                    .copied()
                    .unwrap_or_else(|| *neighbors.choose(&mut rng).unwrap())
            } else {
                *neighbors.choose(&mut rng).unwrap()
            }
        } else {
            *neighbors.choose(&mut rng).unwrap()
        };

        let (direction, next) = chosen;

        // Carve passage
        passages.entry(current).or_default().insert(next);
        passages.entry(next).or_default().insert(current);

        visited.insert(next);

        stack.push(next);

        previous_direction = Some(direction);
    }

    // ------------------------------------------------------------------------
    // Add loops for easier mazes
    // ------------------------------------------------------------------------

    let loop_factor = 1.0 - difficulty;

    let extra_connections =
        ((width * height) as f32 * loop_factor * 0.15) as usize;

    for _ in 0..extra_connections {
        let x = rng.random_range(0..width);
        let y = rng.random_range(0..height);

        let pos = Vector2::new(x, y);

        let action = *GridActions::ALL.choose(&mut rng).unwrap();

        let next = pos + action.to_vec();

        if next.x < 0 || next.x >= width || next.y < 0 || next.y >= height {
            continue;
        }

        passages.entry(pos).or_default().insert(next);
        passages.entry(next).or_default().insert(pos);
    }

    // ------------------------------------------------------------------------
    // Convert passages -> walls
    // ------------------------------------------------------------------------

    let mut walls = HashSet::<(Vector2<i32>, GridActions)>::new();

    for y in 0..height {
        for x in 0..width {
            let pos = Vector2::new(x, y);

            for action in GridActions::ALL {
                let next = pos + action.to_vec();

                // Out of bounds = wall
                if next.x < 0
                    || next.x >= width
                    || next.y < 0
                    || next.y >= height
                {
                    walls.insert((pos, action));
                    continue;
                }

                // If no passage exists => wall
                let connected = passages
                    .get(&pos)
                    .map(|s| s.contains(&next))
                    .unwrap_or(false);

                if !connected {
                    walls.insert((pos, action));
                }
            }
        }
    }

    walls
}

pub fn find_furthest_cell(
    start: Vector2<i32>,
    width: i32,
    height: i32,
    walls: &HashSet<(Vector2<i32>, GridActions)>,
) -> Vector2<i32> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((start, 0));
    visited.insert(start);

    let mut best_cell = start;
    let mut best_distance = 0;

    while let Some((current, dist)) = queue.pop_front() {
        if dist > best_distance {
            best_distance = dist;
            best_cell = current;
        }

        for action in GridActions::ALL {
            // Wall blocks movement
            if walls.contains(&(current, action)) {
                continue;
            }

            let next = current + action.to_vec();

            if next.x < 0
                || next.x >= width
                || next.y < 0
                || next.y >= height
            {
                continue;
            }

            if visited.insert(next) {
                queue.push_back((next, dist + 1));
            }
        }
    }

    best_cell
}
