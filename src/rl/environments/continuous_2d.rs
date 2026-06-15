use std::{f32::consts::PI, process::Command};

use ::nalgebra::UnitQuaternion;
use rand::RngExt;
// use ::nalgebra::{UnitQuaternion, Vector3};
use rapier3d::{glamx::{Quat, Vec3Swizzles}, prelude::*};
use serde::{Deserialize, Serialize};
use plotters::prelude::*;
use tch::Tensor;

use crate::rl::{Action, EnvironmentTrait, State, ToTensor, environments::rapier_simulation::RapierSim};


const OBSTACLES_GROUP: InteractionGroups = InteractionGroups::new(Group::GROUP_1, Group::all(), InteractionTestMode::And);


#[derive(Clone)]
pub struct RayObservation2d {
    pub length: f32,
    pub angle: f32,
}


/**************************************************************
===============================================================
                        Environment
===============================================================
**************************************************************/
/// Wether the reward at each step is:
#[derive(Clone, Default, Debug, PartialEq, Eq, Deserialize, Copy)]
pub enum PlaneRewardType {
    #[default]
    /// -1.0
    Flat,
    /// Proportional to distance to goal
    Gradient,
    /// Positive when moving towards the goal
    Additive,
}
impl PlaneRewardType {
    pub const ALL: [PlaneRewardType; 2] = [PlaneRewardType::Flat, PlaneRewardType::Gradient];
}

#[derive(Deserialize, Clone)]
pub enum PlanePositionType {
    Fixed { pos: Vec<Vec2> },
    RandomInside { n: usize, max_distance: f32 },
    RandomBorder { n: usize },
}

#[derive(Clone)]
pub struct TrajectoryPoints {
    pub position: Vec2,

    /// (angle, detection)
    pub rays: Vec<(f32, f32)>,
    pub rays_length: f32
}

pub struct Simple2dEnvironment {
    pub rays: Vec<RayObservation2d>,

    pub goal_pos: Vec2,
    pub initial_agent_pos: Vec2,
    pub goal_pos_config: PlanePositionType,
    pub initial_agent_pos_config: PlanePositionType,

    pub width: f32,
    pub height: f32,
    pub obstacles_config: PlanePositionType,
    pub obstacles: Vec<Vec2>,

    pub simulation: Option<RapierSim>,
    pub agent_handle: Option<RigidBodyHandle>,
    pub agent_collider_handle: Option<ColliderHandle>,

    pub reward_type: PlaneRewardType,
    pub trajectory: Vec<TrajectoryPoints>
}
impl Simple2dEnvironment {
    pub fn new(
        width: f32, height: f32, initial_agent_pos_config: &PlanePositionType, goal_pos_config: &PlanePositionType, obstacles_config: &PlanePositionType, reward_type: PlaneRewardType,
        n_rays: usize, length_ray: f32, ray_span: f32
    ) -> Self {
        let goal_pos = Self::generate_position(goal_pos_config, width, height, Vec2::ZERO, Vec2::ZERO).iter().next().unwrap_or(&Vec2::ZERO).clone();
        let agent_pos = Self::generate_position(initial_agent_pos_config, width, height, Vec2::ZERO, Vec2::ZERO).iter().next().unwrap_or(&Vec2::ZERO).clone();

        Self {
            rays: (0..n_rays).map(|v| RayObservation2d{ length: length_ray, angle: -ray_span/2.0 + v as f32 / (n_rays-1) as f32 * ray_span }).collect(),

            goal_pos_config: goal_pos_config.clone(),
            goal_pos,
            initial_agent_pos_config: initial_agent_pos_config.clone(),
            initial_agent_pos: agent_pos,

            width,
            height,
            obstacles: Self::generate_position(&obstacles_config, width, height, agent_pos, goal_pos),
            obstacles_config: obstacles_config.clone(),

            simulation: None,
            agent_handle: None,
            agent_collider_handle: None,

            reward_type,
            trajectory: Vec::new(),
        }.generate_environment_sim()
    }

    pub fn regenerate(&mut self) {
        self.initial_agent_pos = Self::generate_position(&self.initial_agent_pos_config, self.width, self.height, self.initial_agent_pos, self.goal_pos).iter().next().unwrap_or(&Vec2::ZERO).clone();
        self.goal_pos = Self::generate_position(&self.goal_pos_config, self.width, self.height, self.initial_agent_pos, self.goal_pos).iter().next().unwrap_or(&Vec2::ZERO).clone();
        self.obstacles = Self::generate_position(&self.obstacles_config, self.width, self.height, self.initial_agent_pos, self.goal_pos);

        *self = self.clone().generate_environment_sim();
    }

    fn generate_position(config: &PlanePositionType, width: f32, height: f32, agent_pos: Vec2, goal_pos: Vec2) -> Vec<Vec2> {
        match config {
            PlanePositionType::RandomInside { n, max_distance } => {
                 let mut rng = rand::rng();

                let direction = goal_pos - agent_pos;
                let length = direction.length();

                // Agent and goal are basically the same point.
                if length < 1e-6 {
                    return Vec::new();
                }

                let dir = direction / length;

                // Perpendicular vector
                let normal = Vec2::new(-dir.y, dir.x);

                let mut obstacles = Vec::new();

                for _ in 0..*n {
                    let mut found = false;

                    // Try multiple times before giving up
                    for _ in 0..100 {
                        // Position along the line
                        let t = rng.random_range(0.0..1.0);

                        // Offset from the line
                        let offset = rng.random_range(-*max_distance..*max_distance);

                        let candidate =
                            agent_pos
                            + dir * (t * length)
                            + normal * offset;

                        // Keep inside arena
                        if candidate.x < 0.0
                            || candidate.x > width
                            || candidate.y < 0.0
                            || candidate.y > height
                        {
                            continue;
                        }

                        // Minimum distance between src and dest
                        if candidate.distance(agent_pos) < 1.1
                            || candidate.distance(goal_pos) < 1.1
                        {
                            continue;
                        }

                        obstacles.push(candidate);
                        found = true;
                        break;
                    }

                    // No valid position found for this obstacle:
                    // simply skip it.
                    if !found {
                        continue;
                    }
                }

                obstacles
            },
            PlanePositionType::RandomBorder { n } => {
                let mut rng = rand::rng();
                (0..*n)
                    .map(|_| {
                        match rng.random_range(0..4) {
                            // Bottom border
                            0 => Vec2::new(
                                rng.random_range(0.0..width),
                                0.0,
                            ),

                            // Top border
                            1 => Vec2::new(
                                rng.random_range(0.0..width),
                                height,
                            ),

                            // Left border
                            2 => Vec2::new(
                                0.0,
                                rng.random_range(0.0..height),
                            ),

                            // Right border
                            _ => Vec2::new(
                                width,
                                rng.random_range(0.0..height),
                            ),
                        }
                    })
                    .collect()
            }
            PlanePositionType::Fixed { pos } => pos.clone(),
        }
    }

    fn generate_environment_sim(mut self) -> Self {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let center = vector![self.width/2.0, 0.0, self.height/2.0];

        // --- Ground (plane in XZ) ---
        let ground_body = RigidBodyBuilder::fixed()
            .translation(Vector::new(center.x, center.y, center.z))
            .build();
        let ground_handle = bodies.insert(ground_body);

        let ground_collider = ColliderBuilder::cuboid(
                self.width / 2.0,
                0.1,
                self.height / 2.0,
            )
            .build();
        colliders.insert_with_parent(ground_collider, ground_handle, &mut bodies);


        // --- Obstacles (cylinders) ---
        for pos in &self.obstacles {
            let body = RigidBodyBuilder::fixed()
                .translation(Vector::new(pos.x, 0.5, pos.y))
                .build();
            let handle = bodies.insert(body);

            let collider = ColliderBuilder::cylinder(0.5, 0.5)
                // .collision_groups(OBSTACLES_GROUP)
                .sensor(true)
                .active_events(ActiveEvents::COLLISION_EVENTS)
                .build();
            colliders.insert_with_parent(collider, handle, &mut bodies);
        }

        // --- Agent ---
        let agent_body = RigidBodyBuilder::kinematic_velocity_based()
            .translation(Vec3::new(self.initial_agent_pos.x, 0.5, self.initial_agent_pos.y))
            .rotation(Vector::new(0.0, 0.0, 0.0))
            .build();
        let agent_handle = bodies.insert(agent_body);

        let agent_collider = ColliderBuilder::ball(0.25)
            .active_collision_types(
                ActiveCollisionTypes::default() | ActiveCollisionTypes::KINEMATIC_FIXED,
            )
            .build();
        let agent_collider_handle = colliders.insert_with_parent(agent_collider, agent_handle, &mut bodies);


        // Init sim
        let sim = RapierSim::new(bodies, colliders);
        self.simulation = Some(sim);
        self.agent_handle = Some(agent_handle);
        self.agent_collider_handle = Some(agent_collider_handle);
    
        self
    }

    fn get_rays_observation(&self) -> Vec<(f32, f32)> {
        let simulation = self.simulation.as_ref().unwrap();
        let agent = &simulation.bodies[self.agent_handle.unwrap()];

        // ---------- Agent position -------------
        let agent_pos = agent.position();
        let agent_translation = agent_pos.translation;
        let agent_rotation = agent_pos.rotation;

        self.rays.iter().map(|observation_ray| {
            let rays_rotation = (agent_rotation * Quat::from_axis_angle(Vector3::Y, observation_ray.angle)) * Vector3::Z;
            
            let ray = Ray::new(
                agent_translation, 
                rays_rotation,
            );
            let max_toi = observation_ray.length;
            let solid = true;
            let filter = QueryFilter::default().exclude_rigid_body(self.agent_handle.unwrap());

            let query_pipeline = simulation.broad_phase.as_query_pipeline(
                simulation.narrow_phase.query_dispatcher(),
                &simulation.bodies,
                &simulation.colliders,
                filter,
            );

            let detection = if let Some((_, toi)) = query_pipeline.cast_ray(
                &ray, max_toi, solid
            ) {
                toi / observation_ray.length
            } else {
                1.0
            };
            (rays_rotation.x.atan2(rays_rotation.z), detection)
        }).collect()
    }

    fn downsample_trajcetory(&self, max_points: usize) -> Vec<TrajectoryPoints> {
        let len = self.trajectory.len();

        if len <= max_points || max_points < 2 {
            return self.trajectory.clone();
        }

        let mut reduced = Vec::with_capacity(max_points);
        for i in 0..max_points {
            let fraction = i as f32 / (max_points - 1) as f32;
            let original_index = (fraction * (len - 1) as f32).round() as usize;
            
            reduced.push(self.trajectory[original_index].clone());
        }

        reduced
    }

    fn plot(&self, path: &str, until: Option<usize>, n_trajectory_points: usize) -> Result<(), Box<dyn std::error::Error>> {    
        let cell_size = 50;
        let pixels_per_unit = self.width * cell_size as f32 * 1.5 / (self.width + 4.0);
        
        // ---------------------------------------------------------
        // Draw config
        // ---------------------------------------------------------
        let root = BitMapBackend::new(
            path,
            (
                (self.width * 1.5) as u32 * cell_size,
                (self.height * 1.5) as u32 * cell_size,
            ),
        )
        .into_drawing_area();
        root.fill(&WHITE)?;
    
        let mut chart = ChartBuilder::on(&root)
            .margin(0)
            .build_cartesian_2d(
                -2f32..(self.width + 2.0) as f32,
                -2f32..(self.height + 2.0) as f32,
            )?;
    
        chart
            .configure_mesh()
            // .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;

        // ---------------------------------------------------------
        // Obstacles
        // ---------------------------------------------------------
        for obstacle in &self.obstacles {
            chart.draw_series(std::iter::once(Circle::new(
                (obstacle.x, obstacle.y),
                0.5 * pixels_per_unit,
                full_palette::GREY.filled(),
            )))?;
        }
    
        // ---------------------------------------------------------
        // Initial position
        // ---------------------------------------------------------
        chart
            .draw_series(std::iter::once(Circle::new(
                (
                    self.initial_agent_pos.x as f32,
                    self.initial_agent_pos.y as f32,
                ),
                0.25 * pixels_per_unit,
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
                    self.goal_pos.x as f32,
                    self.goal_pos.y as f32,
                ),
                0.2 * pixels_per_unit,
                GREEN.mix(0.5).filled(),
            )))?
            .label("Goal")
            .legend(|(x, y)| Circle::new((x, y), 5, GREEN.filled()));
    
        // ---------------------------------------------------------
        // Trajectory
        // ---------------------------------------------------------
        let mut trajectory = self.downsample_trajcetory(n_trajectory_points);
        if let Some(until) = until {
             let end = (until + 1).min(trajectory.len());
            trajectory = trajectory[..end].to_vec();
        }
        if trajectory.len() >= 2 { 
            let points: Vec<(f32, f32)> = trajectory
                .iter()
                .map(|p| {
                    (
                        p.position.x,
                        p.position.y,
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
        }

        // ---------------------------------------------------------
        // Current position
        // ---------------------------------------------------------
        if until.is_some() {
            chart
                .draw_series(std::iter::once(Circle::new(
                    (
                        trajectory.last().unwrap().position.x as f32,
                        trajectory.last().unwrap().position.y as f32,
                    ),
                    0.25 * pixels_per_unit,
                    RED.filled(),
                )))?
                .label("Current pos")
                .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));
        }

        // ---------------------------------------------------------
        // Rays
        // --------------------------------------------------------
        if until.is_some() {
            let last_point = trajectory.last().unwrap();
            let origin = (
                last_point.position.x as f32,
                last_point.position.y as f32,
            );

            for &(angle, detection) in &last_point.rays {
                let length = detection * last_point.rays_length;

                let end = (
                    origin.0 + angle.sin() * length,
                    origin.1 + angle.cos() * length,
                );

                chart.draw_series(std::iter::once(
                    PathElement::new(
                        vec![origin, end],
                        BLUE.mix(0.9).stroke_width(1),
                    )
                ))?;
            }
        }


        root.present()?;
        Ok(())
    }
}
impl EnvironmentTrait<Environment2dState, Environment2dActions> for Simple2dEnvironment {
    fn regenerate(&mut self) { self.regenerate(); }

    fn get_state(&self) -> Environment2dState {
        let simulation = self.simulation.as_ref().unwrap();
        let agent = &simulation.bodies[self.agent_handle.unwrap()];

        // ---------- Raycasting -------------
        let rays_info = self.get_rays_observation();

        // ---------- Agent rotation -------------
        let rot = *agent.rotation();
        let forward = rot * Vector3::Z;
        let to_goal = Vector3::new(self.goal_pos.x, 0.5, self.goal_pos.y) - agent.translation();
        let f = Vector3::new(forward.x, 0.0, forward.z).normalize();
        let g = Vector3::new(to_goal.x, 0.0, to_goal.z).normalize();
        let cross = f.cross(g);
        let dot = f.dot(g);
        let angle = cross.y.atan2(dot); // radians

        Environment2dState {
            distance_to_goal: to_goal.xz().distance(Vector2::ZERO) / (self.width / self.height),
            sin_to_goal: angle.sin(),
            cos_to_goal: angle.cos(),
            rays: rays_info.iter().map(|r| r.1).collect()
        }
    }

    fn step(&mut self, action: Environment2dActions) -> (f32, bool) {
        let simulation = self.simulation.as_mut().unwrap();
        let agent = simulation.bodies.get_mut(self.agent_handle.unwrap()).unwrap();
        agent.set_linvel(Vector3::ZERO, true);
        agent.set_angvel(Vector3::ZERO, true);

        let rot = *agent.rotation();
        let forward = rot * Vector3::Z;

        match action {
            Environment2dActions::Forward => agent.set_linvel(forward, true),
            Environment2dActions::TurnLeft => agent.set_angvel(Vector::new(0.0, -2.0*PI, 0.0), true),
            Environment2dActions::TurnRight => agent.set_angvel(Vector::new(0.0, 2.0*PI, 0.0), true),
        }

        self.simulation.as_mut().unwrap().step();

        // -------- Compute distance --------------
        let simulation = self.simulation.as_mut().unwrap();
        let agent = simulation.bodies.get_mut(self.agent_handle.unwrap()).unwrap();
        let goal_pos = Vector3::new(self.goal_pos.x, 0.5, self.goal_pos.y);
        let distance = agent.translation().xz().distance(goal_pos.xz());
        let agent_pos = agent.translation().xz();

        // -------- Compute collision with obstacles --------------
        let mut intersected = false;
        for (_, _, intersecting) in
            simulation.narrow_phase.intersection_pairs_with(self.agent_collider_handle.unwrap())
        {
            if intersecting {
                intersected = true;
                break;
            }
        }

        // ------------- Add trajectory ----------------------------
        let rays_info = self.get_rays_observation();
        self.trajectory.push(TrajectoryPoints {
            position: agent_pos,
            rays: rays_info,
            rays_length: self.rays.first().and_then(|r| Some(r.length)).unwrap_or(1.0),
        });

        let reward = match self.reward_type {
            PlaneRewardType::Flat => 
                -1.0 
                - if intersected { 10.0 } else { 0.0 } 
                + if distance < 0.2 { 50.0 } else { 0.0 }
            ,
            PlaneRewardType::Gradient => 
                (-distance
                - if intersected { 10.0 * (self.width*self.height).sqrt() } else { 0.0 } 
                + if distance < 0.2 { 50.0 * (self.width*self.height).sqrt() } else { 0.0 }) / (self.width*self.height).sqrt()
            ,
            PlaneRewardType::Additive => {
                let mut last_points = self.trajectory.iter().rev().take(2);

                let distance_advanced = if let (Some(pos_1), Some(pos_2)) = (last_points.next(), last_points.next()) {
                    // Use pos_1 and pos_2 here
                    pos_2.position.distance(self.goal_pos) - pos_1.position.distance(self.goal_pos)
                } else {
                    0.0
                };

                distance_advanced - 0.01
                - if intersected { 10.0 } else { 0.0 } 
                + if distance < 0.2 { 50.0 } else { 0.0 }
            },
        };
        
        (
            reward, 
            distance < 0.2,
        )
    }

    fn reset(&mut self) {       
        let simulation = self.simulation.as_mut().unwrap();
        let agent = simulation.bodies.get_mut(self.agent_handle.unwrap()).unwrap();
        agent.set_translation(Vec3::new(self.initial_agent_pos.x, 0.5, self.initial_agent_pos.y), true);
        agent.set_rotation(Quat::from_axis_angle(Vector3::Y, 0.0), true);
        self.trajectory.clear();
    }
    
    fn plot(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.plot(path, None, usize::MAX)
    }
    
    fn real_time_video(&self, frames_path: &str, video_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let n_frames = 60 * 10; //frames/s * s
        for i in 0..n_frames {
            self.plot(&format!("{frames_path}frame_{i:05}.png"), Some(i), n_frames)?;
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

    fn clone_box(&self) -> Box<dyn EnvironmentTrait<Environment2dState, Environment2dActions>> {
        Box::new(self.clone())
    }
}

impl Clone for Simple2dEnvironment {
    fn clone(&self) -> Self {
        Self {
            rays: self.rays.iter().cloned().collect(),
            goal_pos: self.goal_pos.clone(), 
            initial_agent_pos: self.initial_agent_pos.clone(), 
            goal_pos_config: self.goal_pos_config.clone(), 
            initial_agent_pos_config: self.initial_agent_pos_config.clone(), 
            width: self.width.clone(), 
            height: self.height.clone(), 
            obstacles_config: self.obstacles_config.clone(),
            obstacles: self.obstacles.clone(), 
            simulation: None,
            agent_handle: None,
            agent_collider_handle: None,
            reward_type: self.reward_type.clone(),
            trajectory: self.trajectory.clone(),
        }.generate_environment_sim()
    }
}

/**************************************************************
===============================================================
                        Actions
===============================================================
**************************************************************/
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum Environment2dActions {
    #[default]
    Forward = 0,
    TurnLeft = 1,
    TurnRight = 2,
}
impl Action for Environment2dActions {
    fn get_all_actions() -> Vec<Self> {
        vec![Self::Forward, Self::TurnLeft, Self::TurnRight]
    }
}
impl ToTensor for Environment2dActions {
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
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Environment2dState {
    pub distance_to_goal: f32,

    pub sin_to_goal: f32,
    pub cos_to_goal: f32,
    
    /// Each contains a number between 0 and 1
    pub rays: Vec<f32>,
}
impl State for Environment2dState {}

impl ToTensor for Environment2dState {
    fn len(&self) -> usize {
        self.rays.len() + 3
    }

    fn to_vec(&self) -> Vec<f32> {
        let mut data: Vec<f32> = Vec::with_capacity(self.len());
        // data.extend(&[self.agent_pos.x, self.agent_pos.y]);
        data.push(self.distance_to_goal);
        data.push(self.sin_to_goal);
        data.push(self.cos_to_goal);
        data.extend(&self.rays);
        data
    }
}
